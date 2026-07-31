# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


import torch as th
import torch.nn as nn
from .diffusion_utils import add_shape_channels, resolve_loss_target_type
from utils.print_utils import bold_red
from typing import Literal, Optional
from logging import getLogger
from .interface import ModelMeanType, LossType, StepOutput
from .gaussian_diffusion import masked_mean_flat
logger = getLogger(__name__)

from collections import namedtuple
ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_v", "pred_x_start", "model_out", "additional_output"]
)


class FlowMatching(nn.Module):
    r"""
    Built on top of a discrete time diffusion setup, so ensure forward passes scale the time correctly. Do this by relying on model_predictions.
    
    NOTE: These equations are built to integrate with diffusion code, so it diverges from typical flow matching equations.
    Specifically, 
     - x_0 is the data, and x_1 is the noise, matching diffusion notation
     - k = 0 is the data, and k = 1 is the noise
     - we define v = - d x_k / dk instead of d x_k / dk 
     - we convert continuous timesteps to discrete timesteps to match code for noise schedules, etc.
    
    The fundamental equations

        Definitions
            x_k = (1-k) x_0 + k x_1
            v = x_0 - x_1

        Given x_k, \hat{v}, and k
            \hat{x}_0 = x_k + k \hat{v}
        
        Given x_k, \hat{x}_0 and k
            \hat{v} = (\hat{x}_0 - x_k)/ k
        
    """
    def __init__(
        self,
        *,
        betas,
        num_sampling_steps,
        model_mean_type,
        loss_type,
        cfg,
        logger, 
        **kwargs,
    ):
        super().__init__()
        betas = th.as_tensor(betas, dtype=th.float32)
        self.num_noise_levels = int(betas.shape[0]) # just for scaling
        self.register_buffer("betas", betas)# just in case
        self.model_mean_type = model_mean_type
        self.loss_target_type = resolve_loss_target_type(cfg, self.model_mean_type)
        self.loss_type = loss_type

        self.clip_noise = cfg.clip_noise
        self.loss_weighting = cfg.loss_weighting
        # Sampling convention:
        # - num_sampling_steps: number of denoising steps
        # - num_sampling_noise_levels: number of endpoints = num_sampling_steps + 1
        self.num_sampling_steps = int(num_sampling_steps)
        self.num_sampling_noise_levels = int(self.num_sampling_steps) + 1
        self.guidance_scale = cfg.guidance_scale
        self.history_guidance_scale = float(getattr(cfg, "history_guidance_scale", 0.0))
        
        self.use_causal_mask = cfg.use_causal_mask
        self.logger = logger
        self.cfg = cfg

    def q_sample(self, x_start, k, noise=None) -> th.Tensor:
        r"""
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_k | x_0).
        :param x_start: the initial data batch.
        :param k: continuous diffusion/flow noise level in [0,1]
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        
        Math: x_k = (1 - k) x_0 + k noise
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        if x_start.ndim in [4, 5]:
            # 4: images: (bs, c, h, w)
            # 5: video: (bs, f, c, h, w)
            return (
                (1 - add_shape_channels(k, x_start.shape)) * x_start
                + add_shape_channels(k, x_start.shape) * noise
            )
        else:
            raise ValueError(f"x_start.ndim must be 4 or 5, but got {x_start.ndim}")

    def _select_k_model(
        self,
        k: th.Tensor,
        show_k: Optional[th.Tensor],
    ) -> th.Tensor:
        # In deciding what noise level to show the model, we use the following hierarchy:
        # 1. If we hard-passed in a specific noise level using "show_k", use it.
        # 2. Original behavior, show the true noise level
        if show_k is not None:
            return show_k * (self.num_noise_levels - 1)
        return k * (self.num_noise_levels - 1)

    def model_predictions(
        self,
        x,
        k,
        conditions=None,
        conditions_mask=None,
        model=None,
        model_kwargs=None,
        show_k: Optional[th.Tensor] = None,
    ):
        # k is continuous in [0, 1]; scale only for model input, keep continuous for conversions
        model_kwargs = model_kwargs or {}
        k_cont = k
        k_model = self._select_k_model(k_cont, show_k=show_k)

        model_output = model(x, k_model, conditions, **model_kwargs)
        additional_output = None
        if isinstance(model_output, tuple):
            model_output, additional_output = model_output

        if self.model_mean_type == ModelMeanType.START_X:
            x_start = model_output
            pred_v = self._predict_v_from_xstart(x, k_cont, x_start)
        elif self.model_mean_type == ModelMeanType.VELOCITY:
            pred_v = model_output
            x_start = self._predict_xstart_from_v(x, k_cont, pred_v)
        else:
            raise NotImplementedError

        model_pred = ModelPrediction(pred_v, x_start, model_output, additional_output)

        return model_pred

    def _build_null_history_inputs(
        self,
        x: th.Tensor,
        k_cont: th.Tensor,
        history_mask: th.Tensor,
        history_noise: th.Tensor,
    ):
        """
        Construct the "null history" branch inputs: context (history) tokens have
        their content replaced by pure noise and their continuous noise level
        raised to the maximum (k=1). Active tokens are left untouched.
        """
        mask_x = history_mask[(...,) + (None,) * (x.ndim - history_mask.ndim)]
        x_null = th.where(mask_x, history_noise.to(dtype=x.dtype), x)
        k_null = th.where(history_mask.to(th.bool), th.ones_like(k_cont), k_cont)
        return x_null, k_null

    def guided_model_predictions(
        self,
        x,
        k,
        conditions=None,
        conditions_mask=None,
        model=None,
        model_kwargs=None,
        show_k: Optional[th.Tensor] = None,
        history_mask: Optional[th.Tensor] = None,
        history_noise: Optional[th.Tensor] = None,
        text_on: bool = False,
    ):
        # k is continuous in [0, 1]; scale only for model input, keep continuous for conversions
        model_kwargs = model_kwargs or {}
        k_cont = k
        k_model = self._select_k_model(k_cont, show_k=show_k)

        hist_on = history_mask is not None and history_noise is not None

        def _run(x_in, k_in, prompt_key=None, want_hidden=True):
            branch_kwargs = dict(model_kwargs)
            if prompt_key is not None:
                branch_kwargs["prompt_embed_key"] = prompt_key
            if not want_hidden:
                branch_kwargs["return_hidden_states"] = False
            out = model(x_in, k_in, conditions, **branch_kwargs)
            extra = None
            if isinstance(out, tuple):
                out, extra = out
            return out, extra

        # Primary branch: clean history + positive text. Also the source of the
        # returned readout/hidden states.
        out_cc, additional_output_cc = _run(
            x, k_model, prompt_key="positive" if text_on else None, want_hidden=True
        )

        out = out_cc

        # Text CFG direction: amplify positive-vs-null text.
        if text_on:
            out_cn, _ = _run(x, k_model, prompt_key="null", want_hidden=False)
            out = out + (self.guidance_scale - 1.0) * (out_cc - out_cn)

        # History guidance direction: amplify clean-vs-null history.
        if hist_on:
            x_null, k_null_cont = self._build_null_history_inputs(
                x, k_cont, history_mask, history_noise
            )
            k_null_model = self._select_k_model(k_null_cont, show_k=show_k)
            out_nc, _ = _run(
                x_null,
                k_null_model,
                prompt_key="positive" if text_on else None,
                want_hidden=False,
            )
            out = out + (self.history_guidance_scale - 1.0) * (out_cc - out_nc)

        # Assume velocity prediction. The fused model output is the guided velocity.
        if self.model_mean_type == ModelMeanType.START_X:
            raise NotImplementedError
        elif self.model_mean_type == ModelMeanType.VELOCITY:
            pred_v = out
            x_start = self._predict_xstart_from_v(x, k_cont, pred_v)
        else:
            raise NotImplementedError

        model_pred = ModelPrediction(pred_v, x_start, pred_v, additional_output_cc)

        return model_pred

    def _sampling_model_predictions(
        self,
        *,
        x,
        k,
        conditions=None,
        conditions_mask=None,
        model=None,
        model_kwargs=None,
        show_k: Optional[th.Tensor] = None,
    ):
        model_kwargs = dict(model_kwargs or {})
        history_mask = model_kwargs.pop("history_guidance_mask", None)
        history_noise = model_kwargs.pop("history_guidance_noise", None)
        text_on = self.guidance_scale > 0.0
        hist_on = (
            self.history_guidance_scale > 0.0
            and history_mask is not None
            and history_noise is not None
            and bool(history_mask.any())
        )
        if text_on or hist_on:
            return self.guided_model_predictions(
                x=x,
                k=k,
                conditions=conditions,
                conditions_mask=conditions_mask,
                model=model,
                model_kwargs=model_kwargs,
                show_k=show_k,
                history_mask=history_mask if hist_on else None,
                history_noise=history_noise if hist_on else None,
                text_on=text_on,
            )
        return self.model_predictions(
            x=x,
            k=k,
            conditions=conditions,
            conditions_mask=conditions_mask,
            model=model,
            model_kwargs=model_kwargs,
            show_k=show_k,
        )

    def discrete_idx_to_noise_level(self, indices: th.Tensor):
        """
        Convert discrete timestep indices to continuous noise levels in [0, 1].
        Assumes indices are in [0, num_sampling_noise_levels - 1].
        """
        denom = max(int(self.num_sampling_noise_levels - 1), 1)
        return indices.float() / float(denom)

    @staticmethod
    def _resolve_sampling_noise_levels(
        curr_noise_level: th.Tensor,
        next_noise_level: th.Tensor,
        sample_kwargs: Optional[dict] = None,
    ):
        """
        Resolve decoupled noise levels for sampling:
        - model_*: levels presented to the denoiser.
        - sampler_*: levels used for integration / valid-step gating.
        """
        sample_kwargs = sample_kwargs or {}
        show_curr = sample_kwargs.get(
            "model_current_noise_levels",
            sample_kwargs.get("show_model_current_noise_levels", curr_noise_level),
        ).clamp(min=0.0)
        show_next = sample_kwargs.get(
            "model_next_noise_levels",
            sample_kwargs.get("show_model_next_noise_levels", next_noise_level),
        ).clamp(min=0.0)
        true_curr = sample_kwargs.get(
            "sampler_current_noise_levels",
            sample_kwargs.get("true_current_noise_levels", curr_noise_level),
        ).clamp(min=0.0)
        true_next = sample_kwargs.get(
            "sampler_next_noise_levels",
            sample_kwargs.get("true_next_noise_levels", next_noise_level),
        ).clamp(min=0.0)
        return show_curr, show_next, true_curr, true_next

    def sample_step(
        self,
        x: th.Tensor,
        curr_noise_level: th.Tensor,
        next_noise_level: th.Tensor,
        conditions: Optional[th.Tensor],
        conditions_mask: Optional[th.Tensor] = None,
        model = None,
        model_kwargs = None,
        sample_kwargs = None,
        **kwargs,
    ):
        if sample_kwargs is None:
            sample_kwargs = {}
        scheduler_type = sample_kwargs.get("scheduler_type", "euler")
        if conditions is not None and conditions.dtype != x.dtype:
            x = x.to(conditions.dtype) # x dtype is likely to be affected as we need to apply noise to x, while conditions are just loaded from dataloader
        if scheduler_type == "euler":
            return self.euler_sample_step(
                x=x,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                conditions=conditions,
                conditions_mask=conditions_mask,
                model=model,
                model_kwargs=model_kwargs,
                sample_kwargs=sample_kwargs,
            )
        if scheduler_type == "heun":
            return self.heun_sample_step(
                x=x,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                conditions=conditions,
                conditions_mask=conditions_mask,
                model=model,
                model_kwargs=model_kwargs,
                sample_kwargs=sample_kwargs,
            )
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def euler_sample_step(
        self,
        x: th.Tensor,
        curr_noise_level: th.Tensor,
        next_noise_level: th.Tensor,
        conditions: Optional[th.Tensor],
        conditions_mask: Optional[th.Tensor] = None,
        model = None,
        model_kwargs = None,
        sample_kwargs = None,
        **kwargs,
    ):
        """
        Key idea: dk is derived from a difference in true noise levels, ultimately enforced by the scheduling matrix
        """
        sample_kwargs = sample_kwargs or {}
        show_curr_noise_level, _show_next_noise_level, true_curr_noise_level, true_next_noise_level = (
            self._resolve_sampling_noise_levels(curr_noise_level, next_noise_level, sample_kwargs)
        )
        valid_step = true_next_noise_level < true_curr_noise_level

        model_pred = self._sampling_model_predictions(
            x=x,
            k=show_curr_noise_level,
            conditions=conditions,
            conditions_mask=conditions_mask,
            model=model,
            model_kwargs=model_kwargs,
        )
        x_start = model_pred.pred_x_start
        pred_v = model_pred.pred_v

        dk = true_next_noise_level - true_curr_noise_level # negative valued
        dk = th.where(valid_step, dk, th.zeros_like(dk))
        x_pred = x - add_shape_channels(dk, pred_v.shape) * pred_v
        step_delta = x_pred - x
        step_delta_scale = th.where(valid_step, th.abs(dk), th.ones_like(dk))

        output = StepOutput(
            x = x_pred, # may need to add mask for "only where noise decreased like in ddpm"
            pred_xstart = x_start,
            pred_v = pred_v,
            step_delta = step_delta,
            step_delta_scale = step_delta_scale,
            next_momentum = th.zeros_like(x_pred),
            additional_output = model_pred.additional_output,
        )

        return output

    def heun_sample_step(
        self,
        x: th.Tensor,
        curr_noise_level: th.Tensor,
        next_noise_level: th.Tensor,
        conditions: Optional[th.Tensor],
        conditions_mask: Optional[th.Tensor] = None,
        model = None,
        model_kwargs = None,
        sample_kwargs = None,
        **kwargs,
    ):
        """
        Heun (predictor-corrector) step for flow-matching ODE.

        Key idea: dk is derived from a difference in true noise levels, ultimately enforced by the scheduling matrix
        """
        sample_kwargs = sample_kwargs or {}
        show_curr_noise_level, show_next_noise_level, true_curr_noise_level, true_next_noise_level = (
            self._resolve_sampling_noise_levels(curr_noise_level, next_noise_level, sample_kwargs)
        )
        valid_step = true_next_noise_level < true_curr_noise_level
        dk = true_next_noise_level - true_curr_noise_level # negative valued
        dk = th.where(valid_step, dk, th.zeros_like(dk))
        model_pred = self._sampling_model_predictions(
            x=x,
            k=show_curr_noise_level,
            conditions=conditions,
            conditions_mask=conditions_mask,
            model=model,
            model_kwargs=model_kwargs,
        )
        pred_v = model_pred.pred_v
        x_euler = x - add_shape_channels(dk, pred_v.shape) * pred_v
        model_pred_next = self._sampling_model_predictions(
            x=x_euler,
            k=show_next_noise_level,
            conditions=conditions,
            conditions_mask=conditions_mask,
            model=model,
            model_kwargs=model_kwargs,
        )
        pred_v_next = model_pred_next.pred_v

        x_pred = x - add_shape_channels(dk, pred_v.shape) * 0.5 * (pred_v + pred_v_next)
        step_delta = x_pred - x

        step_delta_scale = th.where(valid_step, th.abs(dk), th.ones_like(dk))
        return StepOutput(
            x=x_pred,
            pred_xstart=model_pred_next.pred_x_start,
            pred_v=0.5 * (pred_v + pred_v_next),
            step_delta=step_delta,
            step_delta_scale=step_delta_scale,
            next_momentum=th.zeros_like(x_pred),
            additional_output=model_pred.additional_output,
        )

    def _predict_v_from_xstart(self, x_k, k, x_start):
        r"""
        Predict velocity from a noisy image x_k and clean image x_0.
        
        This function implements the deterministic reverse process formula that 
        computes the predicted velocity given a clean image at noise level k
        and the noisy image.
        
        Math: (x_0 - x_k)/k = v
        """
        return (
            (x_start - x_k)/add_shape_channels(k, x_k.shape).clamp(min=1e-4)
        )
    
    def _predict_xstart_from_v(self, x_k, k, v):
        r"""
        Predict the clean image x_0 from a noisy image x_k and velocity.
        
        This function implements the deterministic reverse process formula that 
        computes the predicted clean image (x_0) given a noisy image at noise level k
        and the velocity.
        
        Math: x_0 = x_k + k v
        """
        return (
            x_k + add_shape_channels(k, x_k.shape) * v
        )

    def _predict_v(self, x_start, k, noise):
        # Math: v = \bar{\alpha} \epsilon - \sqrt{1-\bar{\alpha}} x_0
        r"""
        Predict the velocity from the clean image x_0 and noise.

        Math: v = \frac{x_0 - x_k}{k} =  \frac{x_0 - (1-k) x_0 - k noise)}{k} = x_0 - noise
        """
        return (
            x_start - noise
        )

    def compute_loss_weights(
        self,
        k: th.Tensor,
        strategy: Literal["uniform"],
    ) -> th.Tensor:
        """
        TODO: clean up strategy, epsilon
        Compute the loss weights for the given timesteps.
        Used for training, reweighting the loss for different timesteps.
        :param k: the timesteps to compute the loss weights for.
        :param strategy: the strategy to use for computing the loss weights.
        :return: a tensor of shape [N, **] containing the loss weights for each timestep.
        """
        if strategy == "uniform":
            return th.ones_like(k)
        
        match self.loss_target_type:
            case ModelMeanType.START_X:
                return 1/(k.clamp(min=1e-4)**2)
            case ModelMeanType.VELOCITY:
                return th.ones_like(k)
            case _:
                raise ValueError(f"unknown objective {self.model_mean_type}")

    ### Training losses ###
    def training_loss(
        self,
        model,
        x_start,
        k,
        masks,
        conditions,
        model_kwargs,
        noise=None,
        show_k: Optional[th.Tensor] = None,
    ):
        """
        Compute training losses for a single timestep.
        x_start: (bs, f, c, h, w)
        conds: (bs, f, d)
        k: (bs, f) diffusion/flow noise levels
        """
        
        model_kwargs['masks'] = masks

        if noise is None:
            noise = th.randn_like(x_start)
            noise_abs_max = model_kwargs.get("noise_abs_max", None)
            if noise_abs_max is not None:
                noise = th.clamp(noise, -noise_abs_max, noise_abs_max)
        x_k = self.q_sample(x_start, k, noise=noise)
        x_k = x_k.to(x_start.dtype)

        terms = {}

        if self.loss_type not in (LossType.MSE, LossType.RESCALED_MSE):
            raise NotImplementedError(f"Loss type {self.loss_type} is not supported.")
        
        model_pred = self.model_predictions(
            x=x_k,
            k=k,
            conditions=conditions,
            model=model,
            model_kwargs=model_kwargs,
            show_k=show_k,
        )

        additional_loss = None
        if isinstance(model_pred.additional_output, dict):
            terms.update(model_pred.additional_output)
        elif th.is_tensor(model_pred.additional_output):
            additional_loss = model_pred.additional_output
        elif model_pred.additional_output is not None:
            raise TypeError(
                f"Unexpected additional_output type: {type(model_pred.additional_output)}"
            )

        if self.loss_target_type == ModelMeanType.START_X:
            target = x_start
            pred = model_pred.pred_x_start
        elif self.loss_target_type == ModelMeanType.VELOCITY:
            target = self._predict_v(x_start, k, noise)
            pred = model_pred.pred_v
        else:
            raise NotImplementedError(self.loss_target_type)

        assert pred.shape == target.shape == x_start.shape

        terms["mse"] = masked_mean_flat((target - pred) ** 2, masks, dims_to_exclude=[0, 1]) 

        terms["loss"] = terms["mse"]
        if additional_loss is not None:
            terms["loss"] += additional_loss
        
        if th.isnan(terms["loss"]).any():
            raise RuntimeError(bold_red("loss is nan"))
            
        # Convert model output back to x0 for logging/visualization.
        predicted_x_start = model_pred.pred_x_start
            
        terms['original_x'] = x_start # (bs, t, c, h, w)
        terms['predicted_x_start'] = predicted_x_start

        # Loss reweighting by timestep; for flow matching this is usually uniform unless
        # using x0/eps parameterizations that imply 1/k^2 or 1/(1-k)^2 scaling.
        loss_weight = self.compute_loss_weights(k, self.loss_weighting.strategy)
        terms["loss"] = terms["loss"] * loss_weight

        # mse has not been touched
        if self.loss_target_type == ModelMeanType.START_X:
            terms["x_mse"] = terms["mse"]
        else:
            terms["x_mse"] = masked_mean_flat(
                (terms["original_x"] - terms["predicted_x_start"]) ** 2,
                masks,
                dims_to_exclude=[0, 1],
            )
        
        return terms
