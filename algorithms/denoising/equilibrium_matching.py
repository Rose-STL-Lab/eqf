# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import torch as th
from .diffusion_utils import add_shape_channels
from utils.print_utils import bold_red
from typing import Literal, Optional, Any
from logging import getLogger
from .interface import ModelMeanType, LossType, StepOutput
from .gaussian_diffusion import masked_mean_flat
from .flow_matching import FlowMatching
from .equilibrium_c import EquilibriumC
logger = getLogger(__name__)

from collections import namedtuple
ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_w", "pred_v", "pred_x_start", "model_out", "additional_output"]
)


def resolve_learning_rate_eta_from_config(
    cfg, num_sampling_steps
):
    """
    Resolve the EQF gradient-sampling step size `eta`.

    This is the trajectory-step parameter used by gradient-style samplers:
        x_next = x - eta * pred_w   (or momentum variants built from this term)

    Where
      - `default_eta_multiple` is the user-facing knob (1.0 means "default budget").
      - `lambda_divisor` is:
          * `equilibrium_lambda` if `use_lambda_divisor=true`
          * `1.0` otherwise

    Rationale:
      1) Divide by `num_sampling_steps` to keep the total update budget roughly stable
         when changing expected sampling depth D.
      2) Optionally divide by `equilibrium_lambda` because EQF drift magnitude scales
         with lambda; this keeps comparable effective step budgets across lambda values
         when `use_lambda_divisor=true`.

    Note:
      This function only sets the integration step size (trajectory dynamics). Any
      additional normalization used for convergence metrics/epsilon thresholds is a
      separate concern handled downstream.
    """
    if not getattr(cfg, "use_lambda_divisor", False):
        raise ValueError("Must specify lambda divisor, and make sure that lambda is set correctly for your model!")
    eta_multiple = getattr(cfg, "default_eta_multiple", 1.0)
    lambda_divisor = 1.0 if not getattr(cfg, "use_lambda_divisor", False) else cfg.equilibrium_lambda # separates out effect from lambda
    return (eta_multiple / lambda_divisor) * 1 / num_sampling_steps

class EquilibriumMatching(FlowMatching):
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
            w = v c(k)
        
        Given x_k, \hat{w}, and k
            \hat{x}_0 = x_k + k \hat{w} / c(k)
        
        Given x_k, \hat{x}_0 and k
            \hat{w} = (\hat{x}_0 - x_k) c(k) /k
        
        Given \hat{w} and k
            v = \hat{w} / c(k)

        Sampling note
            We integrate in a reparameterized time s with ds = c(k) dk, so the drift uses w directly.
            With the noise schedule here, dk = next_noise - curr_noise <= 0, which accounts for the sign.
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
        super().__init__(
            betas=betas,
            num_sampling_steps=num_sampling_steps,
            model_mean_type=model_mean_type,
            loss_type=loss_type,
            cfg=cfg,
            logger=logger,
            **kwargs,
        )
        # Training c(k): includes equilibrium_lambda.
        self.c_helper = EquilibriumC(cfg, apply_lambda=True)
        # Inference c(k): shape-only schedule (no equilibrium_lambda factor).
        self.c_helper_inference = EquilibriumC(cfg, apply_lambda=False)
        self.equilibrium_lambda = cfg.equilibrium_lambda or 4.0
        self.equilibrium_schedule = cfg.equilibrium_schedule
        self.determinstic_c_fn = self.c_helper.deterministic
        self.determinstic_c_fn_inference = self.c_helper_inference.deterministic

        if self.model_mean_type == ModelMeanType.EPSILON:
            raise NotImplementedError("Have not implemented epsilon for EQF")

    def _select_k_model(
        self,
        k: th.Tensor,
        show_k: Optional[th.Tensor],
    ) -> th.Tensor:
        # In deciding what noise level to show the model, we use the following hierarchy:
        # 1. If we hard-passed in a specific noise level using "show_k", use it.
        # 2. Otherwise use the configured force-show policy. Options are random,
        # fixed, or the true noise level.
        # 3. Original default behavior show the model 0.0
        fsk_type = getattr(self.cfg, "force_show_k_type", "fixed")
        if show_k is not None:
            return show_k * (self.num_noise_levels - 1)

        if fsk_type == "random":
            return th.rand_like(k) * (self.num_noise_levels - 1)
        if fsk_type == "fixed":
            fsk_value = float(getattr(self.cfg, "force_show_k", 0.0))
            if not (0.0 <= fsk_value <= 1.0):
                raise ValueError(
                    f"`force_show_k` must be in [0, 1], got {fsk_value}."
                )
            return th.full_like(k, fsk_value) * (self.num_noise_levels - 1)
        if fsk_type == "true_noise_level":
            return k * (self.num_noise_levels - 1)
        raise ValueError(
            f"Invalid `force_show_k_type={fsk_type}`. Expected one of: None, 'random', 'fixed', 'true_noise_level'."
        )

    def model_predictions(
        self,
        x,
        k,
        c,
        conditions=None,
        conditions_mask=None,
        model=None,
        model_kwargs=None,
        show_k=None,
    ) -> Any:
        r"""
        Don't unscale \lambda here so that the model can be correctly supervised in training_loss
        """
        # k is continuous in [0, 1]; scale only for model input, keep continuous for conversions
        k_cont = k

        if model_kwargs is None:
            model_kwargs = {}

        k_model = self._select_k_model(
            k=k_cont,
            show_k=show_k,
        )

        if model is None:
            raise ValueError("`model` must not be None")
        model_output = model(x, k_model, conditions, **model_kwargs)
        additional_output = None
        if isinstance(model_output, tuple):
            model_output, additional_output = model_output

        k_tensor = add_shape_channels(k_cont, x.shape)
        # Use a safe c for divisions; keep unclamped c available for the generic w formula.
        c_safe = c.clamp(min=1e-4)
        k_safe = k_tensor.clamp(min=1e-4)

        if self.model_mean_type == ModelMeanType.START_X:
            pred_x_start = model_output
            pred_w = self._predict_w_from_xstart(x_k=x, x_start=pred_x_start, c=c, k=k_safe)
            pred_v = pred_w / c_safe
        elif self.model_mean_type == ModelMeanType.VELOCITY:
            # importantly: in this case the pred_w does not depend on c (so it does not depend on k)
            pred_w = model_output
            pred_v = pred_w / c_safe
            pred_x_start = self._predict_xstart_from_w(x_k=x, w=pred_w, c=c_safe, k=k_tensor)
        else:
            raise NotImplementedError

        model_pred = ModelPrediction(
            pred_w, pred_v, pred_x_start, model_output, additional_output
        )

        return model_pred

    def guided_model_predictions(
        self,
        x,
        k,
        c,
        conditions=None,
        conditions_mask=None,
        model=None,
        model_kwargs=None,
        show_k=None,
        history_mask=None,
        history_noise=None,
        text_on: bool = False,
    ) -> Any:
        r"""
        Guidance for EqF predictions: composes text classifier-free guidance and
        history guidance.

        The text conditional/unconditional branch is selected through model_kwargs
        (`prompt_embed_key`) so callers can keep `conditions` reserved for
        repo-side external conditioning. History guidance instead replaces the
        clean context (history) tokens with pure noise to form the "null history"
        branch.
        """
        k_cont = k
        model_kwargs = model_kwargs or {}

        k_model = self._select_k_model(
            k=k_cont,
            show_k=show_k,
        )

        if model is None:
            raise ValueError("`model` must not be None")

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
        model_output_cc, additional_output_cc = _run(
            x, k_model, prompt_key="positive" if text_on else None, want_hidden=True
        )

        model_output = model_output_cc

        # Text CFG direction: amplify positive-vs-null text.
        if text_on:
            model_output_cn, _ = _run(x, k_model, prompt_key="null", want_hidden=False)
            model_output = model_output + (self.guidance_scale - 1.0) * (
                model_output_cc - model_output_cn
            )

        # History guidance direction: amplify clean-vs-null history.
        if hist_on:
            x_null, k_null_cont = self._build_null_history_inputs(
                x, k_cont, history_mask, history_noise
            )
            k_null_model = self._select_k_model(k=k_null_cont, show_k=show_k)
            model_output_nc, _ = _run(
                x_null,
                k_null_model,
                prompt_key="positive" if text_on else None,
                want_hidden=False,
            )
            model_output = model_output + (self.history_guidance_scale - 1.0) * (
                model_output_cc - model_output_nc
            )

        k_tensor = add_shape_channels(k_cont, x.shape)
        c_safe = c.clamp(min=1e-4)

        if self.model_mean_type == ModelMeanType.START_X:
            raise NotImplementedError
        elif self.model_mean_type == ModelMeanType.VELOCITY:
            pred_w = model_output
            pred_v = pred_w / c_safe
            pred_x_start = self._predict_xstart_from_w(
                x_k=x,
                w=pred_w,
                c=c_safe,
                k=k_tensor,
            )
        else:
            raise NotImplementedError

        model_pred = ModelPrediction(
            pred_w,
            pred_v,
            pred_x_start,
            model_output,
            additional_output_cc,
        )

        return model_pred

    def _sampling_model_predictions(
        self,
        *,
        x,
        k,
        c,
        conditions=None,
        conditions_mask=None,
        model=None,
        model_kwargs=None,
        show_k=None,
    ) -> Any:
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
                c=c,
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
            c=c,
            conditions=conditions,
            conditions_mask=conditions_mask,
            model=model,
            model_kwargs=model_kwargs,
            show_k=show_k,
        )

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
        scheduler_type = sample_kwargs.get("scheduler_type", "gd")
        if conditions is not None and conditions.dtype != x.dtype:
            x = x.to(conditions.dtype) # x dtype is likely to be affected as we need to apply noise to x, while conditions are just loaded from dataloader
        if scheduler_type == "gd":
            return self.gradient_sample_step(
                x=x,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                conditions=conditions,
                conditions_mask=conditions_mask,
                model=model,
                model_kwargs=model_kwargs,
                sample_kwargs=sample_kwargs,
            )
        if scheduler_type == "ngd":
            return self.nesterov_gradient_sample_step(
                x=x,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                conditions=conditions,
                conditions_mask=conditions_mask,
                model=model,
                model_kwargs=model_kwargs,
                sample_kwargs=sample_kwargs,
            )
        if scheduler_type == "lgd":
            return self.lookahead_gradient_sample_step(
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

    @staticmethod
    def _resolve_eta_tensor(eta, ref: th.Tensor) -> th.Tensor:
        if th.is_tensor(eta):
            return eta.to(device=ref.device, dtype=ref.dtype)
        return th.full_like(ref, float(eta))

    def _maybe_attach_readout_prediction(
        self,
        *,
        model_pred: Any,
        true_curr_noise_level: th.Tensor,
        sample_kwargs: dict,
    ) -> Optional[th.Tensor]:
        if not bool(sample_kwargs.get("needs_readout_prediction", False)):
            return None
        predict_fn = sample_kwargs.get("readout_predict_fn", None)
        if predict_fn is None:
            raise RuntimeError(
                "needs_readout_prediction=true requires readout_predict_fn in sample_kwargs."
            )
        extra = model_pred.additional_output
        if not isinstance(extra, dict) or "hidden_states" not in extra:
            raise RuntimeError(
                "needs_readout_prediction=true requires hidden_states in additional_output."
            )
        if "readout_t_hat" in extra:
            t_hat = extra["readout_t_hat"]
        else:
            with th.no_grad():
                t_hat = predict_fn(extra["hidden_states"])
            t_hat = t_hat.detach().to(
                device=true_curr_noise_level.device,
                dtype=true_curr_noise_level.dtype,
            ).clamp(min=0.0, max=1.0)
            extra["readout_t_hat"] = t_hat
        return t_hat

    def _resolve_sampling_step_eta(
        self,
        *,
        eta_tensor: th.Tensor,
        valid_step: th.Tensor,
        true_curr_noise_level: th.Tensor,
        sample_kwargs: dict,
        readout_t_hat: Optional[th.Tensor],
        additional_output: Optional[dict] = None,
    ) -> th.Tensor:
        """
        Single overridable seam for resolving the per-step `eta` after the
        forward pass (so the freshest `readout_t_hat` is available).

        Default behavior is unchanged: it is exactly the schedule/readout eta
        resolution used by standard inference. A caller may inject a custom
        per-frame policy via `sample_kwargs["step_eta_resolver"]` (e.g. the
        streaming unified-cleanup resolver). The injected resolver receives the
        current `eta_tensor`, `valid_step`, `true_curr_noise_level`, and
        `readout_t_hat`, and returns `(eta_tensor, extra)` where `extra` is an
        optional dict of bookkeeping tensors merged into `additional_output`
        for the caller to consume (e.g. updated tail mask / counter / done).
        """
        resolver = sample_kwargs.get("step_eta_resolver", None)
        if resolver is not None:
            eta_tensor, extra = resolver(
                eta_tensor=eta_tensor,
                valid_step=valid_step,
                true_curr_noise_level=true_curr_noise_level,
                readout_t_hat=readout_t_hat,
            )
            if extra and isinstance(additional_output, dict):
                additional_output.update(extra)
            return eta_tensor
        eta_tensor, _ = self._resolve_eta_with_current_readout(
            eta_tensor=eta_tensor,
            valid_step=valid_step,
            true_curr_noise_level=true_curr_noise_level,
            sample_kwargs=sample_kwargs,
            readout_t_hat=readout_t_hat,
        )
        return eta_tensor

    def _resolve_eta_with_current_readout(
        self,
        *,
        eta_tensor: th.Tensor,
        valid_step: th.Tensor,
        true_curr_noise_level: th.Tensor,
        sample_kwargs: dict,
        readout_t_hat: Optional[th.Tensor],
    ) -> tuple[th.Tensor, Optional[th.Tensor]]:
        source = str(sample_kwargs.get("inference_solver_index_source", "schedule")).lower()
        if source != "readout_predicted":
            return eta_tensor, None
        if readout_t_hat is None:
            raise RuntimeError(
                "inference_solver_index_source=readout_predicted requires current-step readout_t_hat."
            )
        t_hat = readout_t_hat

        # Family-agnostic reindexing: any inference schedule that exposes an
        # eta(k) function can be driven by the readout-predicted noise level.
        eta_at_k_fn = sample_kwargs.get("eta_at_k_fn", None)
        if eta_at_k_fn is None:
            return eta_tensor, t_hat

        eta_multiplier = eta_at_k_fn(t_hat.to(dtype=th.float32)).to(
            device=eta_tensor.device,
            dtype=eta_tensor.dtype,
        ).clamp_min(1e-6)
        eta_base = float(sample_kwargs.get("eta_base", 1.0))
        eta_base_tensor = self._resolve_eta_tensor(eta_base, true_curr_noise_level)
        eta_tensor_current = eta_base_tensor * eta_multiplier
        eta_tensor_current = th.where(valid_step, eta_tensor_current, eta_tensor)
        return eta_tensor_current, t_hat

    def _deterministic_c_for_sampling(self, k: th.Tensor) -> th.Tensor:
        # Sampling runs in eval mode in this codepath; keep a training-mode guard.
        if bool(self.training):
            return self.determinstic_c_fn(k=k)
        return self.determinstic_c_fn_inference(k=k)

    def gradient_sample_step(
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
        Key idea: dk is derived from a learning rate. We enforce stopping with the "true noise levels"
        which are just based on the current scheduling matrix values (when 0). Not actually noise levels, but a state tracker.

        x_next = x - eta f(x)
        """
        sample_kwargs = sample_kwargs or {}
        show_curr_noise_level, _show_next_noise_level, true_curr_noise_level, true_next_noise_level = (
            self._resolve_sampling_noise_levels(curr_noise_level, next_noise_level, sample_kwargs)
        )
        valid_step = true_next_noise_level < true_curr_noise_level

        # unpack gd params, now dk is a learning rate
        eta = sample_kwargs["eta"]
        eta_tensor = self._resolve_eta_tensor(eta, true_curr_noise_level)

       # create the step size. important: how we scale with learning rate; keep negative valued.
        active_frames_and_sign = th.where(valid_step, -th.ones_like(true_curr_noise_level), th.zeros_like(true_curr_noise_level))
        
        c_time = sample_kwargs.get("c_current_noise_levels", true_curr_noise_level)
        c = self._deterministic_c_for_sampling(c_time) # only needed for getting x pred from w pred, so we don't actually pass this into the model under "velocity sampling!"
        c = add_shape_channels(c, x.shape)
        model_pred = self._sampling_model_predictions(
            x=x,
            k=show_curr_noise_level,
            c=c,
            conditions=conditions,
            conditions_mask=conditions_mask,
            model=model,
            model_kwargs=model_kwargs,
        )
        x_start = model_pred.pred_x_start
        pred_w = model_pred.pred_w
        readout_t_hat = self._maybe_attach_readout_prediction(
            model_pred=model_pred,
            true_curr_noise_level=true_curr_noise_level,
            sample_kwargs=sample_kwargs,
        )
        eta_tensor = self._resolve_sampling_step_eta(
            eta_tensor=eta_tensor,
            valid_step=valid_step,
            true_curr_noise_level=true_curr_noise_level,
            sample_kwargs=sample_kwargs,
            readout_t_hat=readout_t_hat,
            additional_output=model_pred.additional_output,
        )

        x_pred = x - add_shape_channels(active_frames_and_sign, pred_w.shape) * pred_w * add_shape_channels(eta_tensor, pred_w.shape)
        step_delta = x_pred - x

        # To keep epsilon and eta_default_multiple as invariants will unscale later by this resolved eta
        # The comparable EQF quantity is pred_w / lambda and step_delta = -eta * pred_w
        # So |step_delta|| = eta * ||pred_w|| = eta * lambda * ||g||
        # If you normalize by eta * lambda: ||step_delta|| / (eta * lambda) = ||g|| becomes comparable
        step_delta_scale = th.abs(eta_tensor) * self.equilibrium_lambda # since this will divide step delta, it will undo the resolution of eta_base to include lambda, so we have to make sure to re-include it here.
        step_delta_scale = th.where(valid_step, step_delta_scale, th.ones_like(step_delta_scale))
        
        output = StepOutput(
            x = x_pred, # may need to add mask for "only where noise decreased like in ddpm"
            pred_xstart = x_start,
            step_delta = step_delta,
            step_delta_scale = step_delta_scale,
            next_momentum = th.zeros_like(x_pred),
            additional_output = model_pred.additional_output,
        )
        return output

    def lookahead_gradient_sample_step(
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
        Key idea: dk is derived from a learning rate. We enforce stopping with the "true noise levels"
        which are just based on the current scheduling matrix values (when 0). Not actually noise levels, but a state tracker.

        x_spec = x + mu momentum
        x_next = x - eta f(x_spec)
        momentum_next = eta f(x_spec)
        """
        sample_kwargs = sample_kwargs or {}
        show_curr_noise_level, _show_next_noise_level, true_curr_noise_level, true_next_noise_level = (
            self._resolve_sampling_noise_levels(curr_noise_level, next_noise_level, sample_kwargs)
        )
        valid_step = true_next_noise_level < true_curr_noise_level

        # unpack gd params, now dk is a learning rate
        eta = sample_kwargs["eta"]
        eta_tensor = self._resolve_eta_tensor(eta, true_curr_noise_level)
        mu = sample_kwargs["mu"]
        momentum = sample_kwargs.get("momentum", th.zeros_like(x))

        # create the step size. important: how we scale with learning rate; keep negative valued.
        active_frames_and_sign = th.where(valid_step, -th.ones_like(true_curr_noise_level), th.zeros_like(true_curr_noise_level))

        # take a lookahead step
        x_speculative = x +  mu * momentum
        
        c_time = sample_kwargs.get("c_current_noise_levels", true_curr_noise_level)
        c = self._deterministic_c_for_sampling(c_time) # only needed for getting x pred from w pred, so we don't actually pass this into the model under "velocity sampling!"
        c = add_shape_channels(c, x.shape)
        model_pred = self._sampling_model_predictions(
            x=x_speculative,
            k=show_curr_noise_level,
            c=c,
            conditions=conditions,
            conditions_mask=conditions_mask,
            model=model,
            model_kwargs=model_kwargs,
        )
        x_start = model_pred.pred_x_start
        pred_w = model_pred.pred_w
        readout_t_hat = self._maybe_attach_readout_prediction(
            model_pred=model_pred,
            true_curr_noise_level=true_curr_noise_level,
            sample_kwargs=sample_kwargs,
        )
        eta_tensor = self._resolve_sampling_step_eta(
            eta_tensor=eta_tensor,
            valid_step=valid_step,
            true_curr_noise_level=true_curr_noise_level,
            sample_kwargs=sample_kwargs,
            readout_t_hat=readout_t_hat,
            additional_output=model_pred.additional_output,
        )

        lookahead_velocity_next = add_shape_channels(eta_tensor, pred_w.shape) * pred_w
        valid_step_exp = add_shape_channels(valid_step, lookahead_velocity_next.shape)
        lookahead_velocity_next = th.where(valid_step_exp, lookahead_velocity_next, momentum)

        # Momentum is in delta_x units, so apply directly (unit step size).
        x_pred = x - add_shape_channels(active_frames_and_sign, lookahead_velocity_next.shape) * lookahead_velocity_next
        step_delta = x_pred - x

        # To keep epsilon and eta_default_multiple as invariants will unscale later by this resolved eta
        # The comparable EQF quantity is pred_w / lambda and step_delta = -eta * pred_w
        # So |step_delta|| = eta * ||pred_w|| = eta * lambda * ||g||
        # If you normalize by eta * lambda: ||step_delta|| / (eta * lambda) = ||g|| becomes comparable
        step_delta_scale = th.abs(eta_tensor) * self.equilibrium_lambda # since this will divide step delta, it will undo the resolution of eta_base to include lambda, so we have to make sure to re-include it here.
        step_delta_scale = th.where(valid_step, step_delta_scale, th.ones_like(step_delta_scale))
        output = StepOutput(
            x = x_pred, # may need to add mask for "only where noise decreased like in ddpm"
            pred_xstart = x_start,
            step_delta = step_delta,
            step_delta_scale = step_delta_scale,
            next_momentum = lookahead_velocity_next,
            additional_output = model_pred.additional_output,
        )
        return output

    def nesterov_gradient_sample_step(
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
        Key idea: dk is derived from a learning rate. We enforce stopping with the "true noise levels"
        which are just based on the current scheduling matrix values (when 0). Not actually noise levels, but a state tracker.

        x_spec = x + mu momentum
        momentum_next = mu momentum + eta f(x_spec)
        x_next = x - momentum_next
        """
        sample_kwargs = sample_kwargs or {}
        show_curr_noise_level, _show_next_noise_level, true_curr_noise_level, true_next_noise_level = (
            self._resolve_sampling_noise_levels(curr_noise_level, next_noise_level, sample_kwargs)
        )
        valid_step = true_next_noise_level < true_curr_noise_level

        # unpack gd params, now dk is a learning rate
        eta = sample_kwargs["eta"]
        eta_tensor = self._resolve_eta_tensor(eta, true_curr_noise_level)
        mu = sample_kwargs["mu"]
        momentum = sample_kwargs.get("momentum", th.zeros_like(x))

        # create the step size. important: how we scale with learning rate; keep negative valued.
        active_frames_and_sign = th.where(valid_step, -th.ones_like(true_curr_noise_level), th.zeros_like(true_curr_noise_level))

        # take a lookahead step
        x_speculative = x + mu * momentum
        
        c_time = sample_kwargs.get("c_current_noise_levels", true_curr_noise_level)
        c = self._deterministic_c_for_sampling(c_time) # only needed for getting x pred from w pred, so we don't actually pass this into the model under "velocity sampling!"
        c = add_shape_channels(c, x.shape)
        model_pred = self._sampling_model_predictions(
            x=x_speculative,
            k=show_curr_noise_level,
            c=c,
            conditions=conditions,
            conditions_mask=conditions_mask,
            model=model,
            model_kwargs=model_kwargs,
        )
        x_start = model_pred.pred_x_start
        pred_w = model_pred.pred_w
        readout_t_hat = self._maybe_attach_readout_prediction(
            model_pred=model_pred,
            true_curr_noise_level=true_curr_noise_level,
            sample_kwargs=sample_kwargs,
        )
        eta_tensor = self._resolve_sampling_step_eta(
            eta_tensor=eta_tensor,
            valid_step=valid_step,
            true_curr_noise_level=true_curr_noise_level,
            sample_kwargs=sample_kwargs,
            readout_t_hat=readout_t_hat,
            additional_output=model_pred.additional_output,
        )

        momentum_next = mu * momentum + add_shape_channels(eta_tensor, pred_w.shape) * pred_w
        valid_step_exp = add_shape_channels(valid_step, momentum_next.shape)
        momentum_next = th.where(valid_step_exp, momentum_next, momentum)

        # Momentum is in delta_x units, so apply directly (unit step size).
        x_pred = x - add_shape_channels(active_frames_and_sign, momentum_next.shape) * momentum_next
        step_delta = x_pred - x

        # To keep epsilon and eta_default_multiple as invariants will unscale later by this resolved eta
        # The comparable EQF quantity is pred_w / lambda and step_delta = -eta * pred_w
        # So |step_delta|| = eta * ||pred_w|| = eta * lambda * ||g||
        # If you normalize by eta * lambda: ||step_delta|| / (eta * lambda) = ||g|| becomes comparable
        step_delta_scale = th.abs(eta_tensor) * self.equilibrium_lambda # since this will divide step delta, it will undo the resolution of eta_base to include lambda, so we have to make sure to re-include it here.
        step_delta_scale = th.where(valid_step, step_delta_scale, th.ones_like(step_delta_scale))
        output = StepOutput(
            x=x_pred,
            pred_xstart=x_start,
            step_delta=step_delta,
            step_delta_scale=step_delta_scale,
            next_momentum=momentum_next,
            additional_output=model_pred.additional_output,
        )
        return output
    
    def _predict_xstart_from_w(self, x_k, w, c, k):
        r"""
        Predict x_0 from x_k and modulated velocity w = c(k) * v.

        Math:
            v = w / c(k)
            x_0 = x_k + k * v = x_k + w * k / c(k)
        
        For c(k) = \lambda_ k just divided by lambda_.
        """
        return x_k + k * w / c
    
    def _predict_w_from_xstart(self, x_k, x_start, c, k):
        r"""
        Predict w from x_k and denoised x_start

        Math:
            v = w / c(k)
            w = (x_0 - x_k) c(k)/k
        
        For c(k) = \lambda_ k just multiply by lambda_.
        """
        return (x_start - x_k) * c / k

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
        if  self.equilibrium_schedule == "linear": # linear saves us for now in x/v prediction
            return th.ones_like(k) * self.equilibrium_lambda
        
        match self.loss_target_type:
            case ModelMeanType.START_X:
                # w = c(k) * (x_0 - x_k) / k
                # so ||x0 - x0_hat||^2 maps to ||w - w_hat||^2 with weight c(k)^2 / k^2
                c = self.c_fn(k)
                return (c ** 2) / k.clamp(min=1e-4) ** 2
            case ModelMeanType.VELOCITY:
                # Directly predicting w (modulated velocity) needs no reweighting
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
        generator=None,
        show_k=None,
    ):
        """
        Compute training losses for a single timestep.
        x_start: (bs, f, c, h, w)
        conds: (bs, f, d)
        k: (bs, f)
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

        c = self.c_helper.resolve_c_tensor(k, x_k.shape)

        if self.loss_type not in (LossType.MSE, LossType.RESCALED_MSE):
            raise NotImplementedError(f"Loss type {self.loss_type} is not supported.")

        model_pred = self.model_predictions(
            x=x_k,
            k=k,
            c=c,
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
            target = self._predict_v(x_start, k, noise) * c
            pred = model_pred.pred_w
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

        # Loss reweighting by diffusion noise level; for eq matching this is usually uniform unless
        # using x0/eps parameterizations that imply c(k)^2/k^2 or c(k)^2/(1-k)^2 scaling.
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
