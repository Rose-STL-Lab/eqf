# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import torch as th
import torch.nn as nn
from .diffusion_utils import (
    discretized_gaussian_log_likelihood,
    normal_kl,
    _extract_into_tensor,
    add_shape_channels,
    resolve_loss_target_type,
)
from utils.print_utils import bold_red
from typing import Literal, Optional
import torch.nn.functional as F
from logging import getLogger
from .interface import ModelMeanType, ModelVarType, LossType, StepOutput
from .noise_schedule import get_named_beta_schedule
logger = getLogger(__name__)

from collections import namedtuple
ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "model_out", "additional_output"]
)


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def masked_mean_flat(tensor, mask=None, dims_to_exclude=None):
    """
    Take the mean over all dimensions not specified in dims_to_exclude with an optional mask.
        only calculate the mean where the mask is True
    Args:
        tensor: Tensor of any shape (bs, ...)
        mask: Optional mask of shape (bs, t) for tensors of shape (bs, t, c, h, w)
            or (bs,) for tensors of shape (bs, c, h, w). 
        dims_to_exclude: Optional list of dimensions to exclude from mean calculation.
            Default is [0], which excludes only the batch dimension.
    
    Returns:
        Mean tensor with shape determined by the excluded dimensions,
        averaging only over unmasked elements

    Potential TODO that only affects non-Diffusion Forcing baselines is divide by the sum of masks at the end.
    """
    if dims_to_exclude is None:
        dims_to_exclude = [0]  # Default: exclude only batch dimension
    
    # Determine which dimensions to average over
    dims_to_avg = [d for d in range(tensor.dim()) if d not in dims_to_exclude]
    
    if mask is None:
        return tensor.mean(dim=dims_to_avg)
    
    # For tensors with mask
    # Expand mask to match tensor dimensions for proper broadcasting
    expanded_mask = mask
    
    # Determine how the mask should be expanded based on the tensor and dims_to_exclude
    for dim in range(mask.dim(), tensor.dim()):
        if dim not in dims_to_exclude:
            expanded_mask = expanded_mask.unsqueeze(-1)
    
    # Apply mask
    masked_tensor = tensor * expanded_mask
    
    # Count valid elements per kept dimension
    mean = masked_tensor.mean(dim=dims_to_avg)
    
    return mean


class GaussianDiffusion(nn.Module):
    """
    Utilities for training and sampling diffusion models.
    Original ported from this codebase:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    """

    def __init__(
        self,
        *,
        betas,
        num_sampling_steps,
        model_mean_type,
        model_var_type,
        loss_type,
        cfg,
        logger, 
        **kwargs,
    ):
        super().__init__()
        self.model_mean_type = model_mean_type
        self.loss_target_type = resolve_loss_target_type(cfg, self.model_mean_type)
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        self.num_noise_levels = int(betas.shape[0])
        self.clip_noise = cfg.clip_noise # only active for sampling code
        self.loss_weighting = cfg.loss_weighting
        # Sampling convention:
        # - num_sampling_steps: number of denoising steps
        # - num_sampling_noise_levels: number of endpoints = num_sampling_steps + 1
        self.num_sampling_steps = int(num_sampling_steps)
        self.num_sampling_noise_levels = int(self.num_sampling_steps) + 1
        self.register_buffer("betas", betas)
        
        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", th.cumprod(alphas, dim=0))
        self.register_buffer("alphas_cumprod_prev", 
                            th.cat([th.tensor([1.0], device=alphas.device), 
                                self.alphas_cumprod[:-1]]))
        self.register_buffer("alphas_cumprod_next", 
                            th.cat([self.alphas_cumprod[1:], 
                                th.tensor([0.0], device=alphas.device)]))
        assert self.alphas_cumprod_prev.shape == (self.num_noise_levels,)

        # calculations for diffusion q(x_k | x_{k-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", th.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", th.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", th.log(1.0 - self.alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", th.sqrt(1.0 / self.alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", th.sqrt(1.0 / self.alphas_cumprod - 1))

        # calculations for posterior q(x_{k-1} | x_k, x_0)
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        # calculations for posterior q(x_{k-1} | x_k, x_0)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        posterior_log_variance_clipped = th.log(
            th.cat([self.posterior_variance[:1], self.posterior_variance[1:]])
        ) if len(self.posterior_variance) > 1 else th.tensor([])

        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)

        self.register_buffer("posterior_mean_coef1", 
                            self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", 
                            (1.0 - self.alphas_cumprod_prev) * th.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        self.use_causal_mask = cfg.use_causal_mask
        
        # snr: signal noise ratio
        snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        self.register_buffer("snr", snr)
        if self.loss_weighting.strategy in {"min_snr", "fused_min_snr"}:
            clipped_snr = snr.clone()
            clipped_snr = th.clamp(clipped_snr, min=None, max=self.loss_weighting.snr_clip)
            # self.clipped_snr = clipped_snr.float()
            self.register_buffer("clipped_snr", clipped_snr)
        elif self.loss_weighting.strategy == "sigmoid":
            # logsnr = th.log(snr).float()
            logsnr = th.log(snr) 
            self.register_buffer("logsnr", logsnr)

        self.logger = logger
            
    def q_mean_variance(self, x_start, k):
        """
        Get the distribution q(x_k | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param k: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, k, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, k, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, k, noise=None) -> th.Tensor:
        r"""
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_k | x_0).
        :param x_start: the initial data batch.
        :param k: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        
        Math: x_k = \sqrt{\bar{\alpha_k}} x_0 + \sqrt{1-\bar{\alpha_k}} \epsilon
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        if x_start.ndim in [4, 5]:
            # 4: images: (bs, c, h, w)
            # 5: video: (bs, f, c, h, w)
            return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * noise
            )
        else:
            raise ValueError(f"x_start.ndim must be 4 or 5, but got {x_start.ndim}")

    def q_posterior_mean_variance(self, x_start, x_k, k):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{k-1} | x_k, x_0)
        """
        assert x_start.shape == x_k.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, k, x_k.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, k, x_k.shape) * x_k
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, k, x_k.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, k, x_k.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, k, conditions, clip_denoised=True, denoised_fn=None, model_kwargs=None, **kwargs):
        """
        Apply the model to get p(x_{k-1} | x_k), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at diffusion level k.
        :param k: a 1-D Tensor of diffusion timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B = x.shape[0]
        C = x.shape[-3]
        # assert k.shape == (B,)
        strategy = kwargs.get("strategy", "next-frame")
        model_output = model(x, k, conditions, strategy = strategy, **model_kwargs) # fixed for diffusion-forcing, using x_k_cur
        if model_output.shape[0] != B:
            model_output = model_output[:B, ...]

        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None
        
        if model_output.shape[0] == 1:
            x_k = x[:1]
            k = k[:1]
        else:
            x_k = x

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            """
            https://arxiv.org/abs/2102.09672
            """
            assert False, "Not implemented"
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    th.cat([self.posterior_variance[1].unsqueeze(0), self.betas[1:]], dim=0),
                    th.log(th.cat([self.posterior_variance[1].unsqueeze(0), self.betas[1:]], dim=0)),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, k, model_output.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, k, model_output.shape)

        def process_xstart(x):
            if denoised_fn is not None: # NOTE: denoised_fn is None when sampling
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x_k=x_k, k=k, eps=model_output)
        elif self.model_mean_type == ModelMeanType.VELOCITY:
            pred_xstart = self._predict_xstart_from_v(x_k=x_k, k=k, v=model_output)
        else:
            raise NotImplementedError(f"{self.model_mean_type} is not supported yet")
        
        pred_xstart = process_xstart(pred_xstart)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_k=x_k, k=k)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_k.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }

    def _predict_xstart_from_eps(self, x_k, k, eps):
        r"""
        Predict the clean image x_0 from a noisy image x_k and predicted noise.
        
        This function implements the deterministic reverse process formula that 
        computes the predicted clean image (x_0) given a noisy image at timestep t 
        and the predicted noise.
        
        Args:
            x_k: A tensor of shape [N x C x ...] or [N x T x C x ...] representing
                the noisy image at diffusion level k.
            k: A 1-D tensor of diffusion timesteps (one per batch element or per frame).
            eps: A tensor of the same shape as x_k containing the predicted noise.
        
        Returns:
            A tensor of the same shape as x_k containing the predicted clean image x_0.
        
        Math:
            x_0 = \frac{x_k - \sqrt{1-\bar{\alpha_k}} \epsilon}{\sqrt{\bar{\alpha_k}}}
            
        where:
            - \bar{\alpha_k} is the cumulative product of (1 - \beta_i) for i=1...k
            - \beta_i is the noise schedule
        """
        assert x_k.shape == eps.shape

        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, k, x_k.shape) * x_k
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, k, x_k.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_k, k, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, k, x_k.shape) * x_k - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, k, x_k.shape).clamp(min=1e-4)

    def _predict_xstart_from_v(self, x_k, k, v):
        r"""
        Predict the clean image x_0 from a noisy image x_k and velocity.
        
        This function implements the deterministic reverse process formula that 
        computes the predicted clean image (x_0) given a noisy image at diffusion level k
        and the velocity.
        
        Math: x_0 = \sqrt{\bar{\alpha_k}} x_k - \sqrt{1-\bar{\alpha_k}} v
        """
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, k, x_k.shape) * x_k
            - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * v
        )
        
    def _predict_v(self, x_start, k, noise):
        # Math: v = \bar{\alpha} \epsilon - \sqrt{1-\bar{\alpha}} x_0
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, k, x_start.shape) * noise
            - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * x_start
        )

    def _predict_eps_from_v(self, x_k, k, v):
        """
        Get predicted epsilon from velocity
        """
        alpha_cum_k = _extract_into_tensor(self.alphas_cumprod, k, v.shape)
        beta_cum_k = _extract_into_tensor(1 - self.alphas_cumprod, k, v.shape)

        epsilon = (alpha_cum_k**0.5) * v + (beta_cum_k**0.5) * x_k
        return epsilon
    
    def condition_mean(self, cond_fn, p_mean_var, x, k, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, k, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, k, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, k, x.shape)

        eps = self._predict_eps_from_xstart(x, k, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, k, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, k, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_k=x, k=k)
        return out

    def _model_forward(
        self,
        model,
        x,
        k,
        conditions=None,
        model_kwargs=None,
    ):
        model_kwargs = model_kwargs or {}
        model_output = model(x, k.to(dtype=x.dtype), conditions, **model_kwargs)
        additional_output = None
        if isinstance(model_output, tuple):
            model_output, additional_output = model_output
        return model_output, additional_output

    def _model_predictions_from_output(
        self,
        x,
        k,
        model_output,
        additional_output=None,
        clip_pred_noise: bool = False,
    ):
        if self.model_mean_type == ModelMeanType.EPSILON:
            pred_noise = (
                th.clamp(model_output, -self.clip_noise, self.clip_noise)
                if clip_pred_noise
                else model_output
            )
            x_start = self._predict_xstart_from_eps(x, k, pred_noise)
        elif self.model_mean_type == ModelMeanType.START_X:
            x_start = model_output
            pred_noise = self._predict_eps_from_xstart(x, k, x_start)
        elif self.model_mean_type == ModelMeanType.VELOCITY:
            v = model_output
            x_start = self._predict_xstart_from_v(x, k, v)
            pred_noise = self._predict_eps_from_v(x, k, v)
        else:
            raise NotImplementedError(
                f"{self.model_mean_type} is not supported in model_predictions"
            )

        return ModelPrediction(pred_noise, x_start, model_output, additional_output)

    def model_predictions(
        self,
        x,
        k,
        conditions=None,
        conditions_mask=None,
        model=None,
        model_kwargs=None,
    ):
        model_output, additional_output = self._model_forward(
            model=model,
            x=x,
            k=k,
            conditions=conditions,
            model_kwargs=model_kwargs,
        )
        return self._model_predictions_from_output(
            x=x,
            k=k,
            model_output=model_output,
            additional_output=additional_output,
            clip_pred_noise=True,
        )
    
    def ddim_idx_to_noise_level(self, indices: th.Tensor):
        shape = indices.shape
        real_steps = th.linspace(-1, self.num_noise_levels - 1, self.num_sampling_noise_levels)
        real_steps = real_steps.long().to(indices.device)
        k = real_steps[indices.flatten()]
        return k.view(shape)

    @staticmethod
    def _resolve_sampling_noise_levels(
        curr_noise_level: th.Tensor,
        next_noise_level: th.Tensor,
        sample_kwargs: Optional[dict] = None,
    ):
        """
        Resolve decoupled noise levels for sampling:
        - model_*: levels presented to the denoiser.
        - sampler_*: levels used for integration / update masking.
        """
        sample_kwargs = sample_kwargs or {}
        show_curr = sample_kwargs.get(
            "model_current_noise_levels",
            sample_kwargs.get("show_model_current_noise_levels", curr_noise_level),
        )
        show_next = sample_kwargs.get(
            "model_next_noise_levels",
            sample_kwargs.get("show_model_next_noise_levels", next_noise_level),
        )
        true_curr = sample_kwargs.get(
            "sampler_current_noise_levels",
            sample_kwargs.get("true_current_noise_levels", curr_noise_level),
        )
        true_next = sample_kwargs.get(
            "sampler_next_noise_levels",
            sample_kwargs.get("true_next_noise_levels", next_noise_level),
        )
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
        scheduler_type = sample_kwargs.get("scheduler_type", "ddim")
        if conditions is not None and conditions.dtype != x.dtype:
            x = x.to(conditions.dtype) # x dtype is likely to be affected as we need to apply noise to x, while conditions are just loaded from dataloader
        if scheduler_type == "ddim":
            return self.ddim_sample_step(
                x=x,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                conditions=conditions,
                conditions_mask=conditions_mask,
                model=model,
                model_kwargs=model_kwargs,
                sample_kwargs=sample_kwargs,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def ddim_sample_step(
        self,
        x: th.Tensor,
        curr_noise_level: th.Tensor,
        next_noise_level: th.Tensor,
        conditions: Optional[th.Tensor],
        conditions_mask: Optional[th.Tensor] = None,
        ddim_sampling_eta: float = 0.0,
        model = None,
        model_kwargs = None,
        sample_kwargs = None,
    ):
        sample_kwargs = sample_kwargs or {}
        ddim_sampling_eta = sample_kwargs.get("ddim_sampling_eta", ddim_sampling_eta)
        show_curr_noise_level, show_next_noise_level, true_curr_noise_level, true_next_noise_level = (
            self._resolve_sampling_noise_levels(curr_noise_level, next_noise_level, sample_kwargs)
        )
        # In discrete time diffusion, `true_*` may use -1 as a sentinel for clean/frozen
        # tokens. Preserve that sentinel for gating/final-step semantics, but clamp
        # the model-visible levels before indexing alphas or passing them to the model.
        # Behavior can be driven off of show_noise_levels because by assumption in diffusion
        # they are identical, with the exception of clipping
        clipped_show_curr_noise_level = th.clamp(show_curr_noise_level, min=0)
        clipped_show_next_noise_level = th.clamp(show_next_noise_level, min=0)
        valid_step = true_next_noise_level < true_curr_noise_level

        alpha = self.alphas_cumprod[clipped_show_curr_noise_level]
        alpha_next = th.where(
            show_next_noise_level < 0,
            th.ones_like(alpha),
            self.alphas_cumprod[clipped_show_next_noise_level],
        )
        safe_alpha_next = th.where(valid_step, alpha_next, alpha)
        sigma = th.where(
            show_next_noise_level < 0,
            th.zeros_like(alpha),
            ddim_sampling_eta
            * (
                (1 - alpha / safe_alpha_next)
                * (1 - safe_alpha_next)
                / (1 - alpha)
            ).clamp(min=0).sqrt(),
        )
        sigma = th.where(valid_step, sigma, th.zeros_like(sigma))
        c = th.where(
            valid_step,
            (1 - safe_alpha_next - sigma**2).clamp(min=0).sqrt(),
            th.zeros_like(alpha),
        )

        alpha = add_shape_channels(alpha, x.shape)
        alpha_next = add_shape_channels(safe_alpha_next, x.shape)
        c = add_shape_channels(c, x.shape)
        sigma = add_shape_channels(sigma, x.shape)

        model_pred = self.model_predictions(
            x=x,
            k=clipped_show_curr_noise_level,
            conditions=conditions,
            conditions_mask=conditions_mask,
            model=model,
            model_kwargs=model_kwargs,
        )
        pred_noise = model_pred.pred_noise
        x_start = model_pred.pred_x_start

        noise = th.randn_like(x)
        noise = th.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        # only update frames where true integration levels decrease
        mask = true_next_noise_level >= true_curr_noise_level
        x_pred = th.where(
            add_shape_channels(mask, x.shape),
            x,
            x_pred,
        )
        step_delta = x_pred - x
        step_delta_scale = th.ones_like(true_curr_noise_level) # since discrete valued, we want a no-op
        pred_v = self._predict_v(x_start, clipped_show_curr_noise_level, pred_noise)
        output = StepOutput(
            x = x_pred,
            pred_xstart = x_start,
            pred_v = pred_v,
            step_delta = step_delta,
            step_delta_scale = step_delta_scale,
            next_momentum = th.zeros_like(x_pred),
            additional_output = model_pred.additional_output,
        )

        return output
    
    def _vb_terms_bpd(
            self, model, x_start, x_k, k, conditions, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_k=x_k, k=k
        )
        out = self.p_mean_variance(
            model, x_k, k, conditions, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )

        masks = model_kwargs.get("masks", None)
        
        dims_to_exclude = [i for i in range(0, x_start.dim() - 3)]
        kl = masked_mean_flat(kl, masks, dims_to_exclude) / th.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = masked_mean_flat(decoder_nll, masks, dims_to_exclude) / th.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{k-1}|x_k,x_0) || p(x_{k-1}|x_k))
        output = th.where((k == 0), decoder_nll, kl)
        if output.dim() == 2:
            # (bs, t) -> (bs) 
            output = masked_mean_flat(output)        
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def compute_loss_weights(
        self,
        k: th.Tensor,
        strategy: Literal["min_snr", "fused_min_snr", "uniform", "sigmoid"],
    ) -> th.Tensor:
        """
        Compute the loss weights for the given timesteps.
        Used for training, reweighting the loss for different timesteps.
        :param k: the timesteps to compute the loss weights for.
        :param strategy: the strategy to use for computing the loss weights.
        :return: a tensor of shape [N, **] containing the loss weights for each timestep.
        """
        if strategy == "uniform":
            return th.ones_like(k)
        
        self.snr = self.snr.to(k.device)
        snr = self.snr[k]
        epsilon_weighting = None
        match strategy:
            case "sigmoid":
                self.logsnr = self.logsnr.to(k.device)
                logsnr = self.logsnr[k]
                # sigmoid reweighting proposed by https://arxiv.org/abs/2303.00848
                # and adopted by https://arxiv.org/abs/2410.19324
                epsilon_weighting = th.sigmoid(
                    self.loss_weighting.sigmoid_bias - logsnr
                )
            case "snr+1":
                assert False, "Not used anymore"
                assert self.model_mean_type == ModelMeanType.VELOCITY, "snr+1 is used for Matrix, and only support velocity model"
                weightning =  1 / (1 - self.alphas_cumprod[k]).clamp(min=1e-8)
                return weightning
            case "min_snr":
                # min-SNR reweighting proposed by https://arxiv.org/abs/2303.09556
                self.clipped_snr = self.clipped_snr.to(k.device)
                clipped_snr = self.clipped_snr[k]
                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)  # avoid NaN
            case "fused_min_snr":
                # fused min-SNR reweighting proposed by Diffusion Forcing v1
                # with an additional support for bi-directional Fused min-SNR for non-causal models
                snr_clip, cum_snr_decay = (
                    self.loss_weighting.snr_clip,
                    self.loss_weighting.cum_snr_decay,
                )
                clipped_snr = self.clipped_snr[k]
                normalized_clipped_snr = clipped_snr / snr_clip
                normalized_snr = snr / snr_clip

                def compute_cum_snr(reverse: bool = False):
                    new_normalized_clipped_snr = (
                        normalized_clipped_snr.flip(1)
                        if reverse
                        else normalized_clipped_snr
                    )
                    cum_snr = th.zeros_like(new_normalized_clipped_snr)
                    for t in range(0, k.shape[1]):
                        if t == 0:
                            cum_snr[:, t] = new_normalized_clipped_snr[:, t]
                        else:
                            cum_snr[:, t] = (
                                cum_snr_decay * cum_snr[:, t - 1]
                                + (1 - cum_snr_decay) * new_normalized_clipped_snr[:, t]
                            )
                    cum_snr = F.pad(cum_snr[:, :-1], (1, 0, 0, 0), value=0.0)
                    return cum_snr.flip(1) if reverse else cum_snr

                if self.use_causal_mask:
                    cum_snr = compute_cum_snr()
                else:
                    # bi-directional cum_snr when not using causal mask
                    cum_snr = compute_cum_snr(reverse=True) + compute_cum_snr()
                    cum_snr *= 0.5
                clipped_fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (
                    1 - normalized_clipped_snr
                )
                fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_snr)
                clipped_snr = clipped_fused_snr * snr_clip
                snr = fused_snr * snr_clip
                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)  # avoid NaN
            case _:
                raise ValueError(f"unknown loss weighting strategy {strategy}")

        match self.loss_target_type:
            case ModelMeanType.EPSILON:
                return epsilon_weighting
            case ModelMeanType.START_X:
                return epsilon_weighting * snr
            case ModelMeanType.VELOCITY:
                return epsilon_weighting * snr / (snr + 1)
            case _:
                raise ValueError(f"unknown objective {self.model_mean_type}")

    def _prior_bpd(self, x_start, model_kwargs=None):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        # 
        """
        batch_size = x_start.shape[0]
        k = th.tensor([self.num_noise_levels - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, k)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return masked_mean_flat(kl_prior, model_kwargs.get("masks", None)) / th.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        # 
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        masks = model_kwargs.get("masks", None)
        for k_step in list(range(self.num_noise_levels))[::-1]:
            k_batch = th.tensor([k_step] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_k = self.q_sample(x_start=x_start, k=k_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_k=x_k,
                    k=k_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(masked_mean_flat((out["pred_xstart"] - x_start) ** 2, masks))
            eps = self._predict_eps_from_xstart(x_k, k_batch, out["pred_xstart"])
            mse.append(masked_mean_flat((eps - noise) ** 2, masks))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start, model_kwargs)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

    ### Training losses ###
    def training_loss(self, model, x_start, k, masks, conditions, model_kwargs, noise=None, noise_abs_max = None, ):
        """
        Compute training losses for a single timestep.
        x_start: (bs, f, c, h, w)
        conds: (bs, f, d)
        k: (bs, f) diffusion noise levels
        """

        model_kwargs = dict(model_kwargs or {})
        model_kwargs["masks"] = masks

        if noise is None:
            noise = th.randn_like(x_start)
            noise_abs_max = model_kwargs.get("noise_abs_max", None)
            if noise_abs_max is not None:
                noise = th.clamp(noise, -noise_abs_max, noise_abs_max)
        x_k = self.q_sample(x_start, k, noise=noise)
        x_k = x_k.to(x_start.dtype)
        terms = {}

        model_output, additional_output = self._model_forward(
            model=model,
            x=x_k,
            k=k,
            conditions=conditions,
            model_kwargs=model_kwargs,
        )
        model_pred = self._model_predictions_from_output(
            x=x_k,
            k=k,
            model_output=model_output,
            additional_output=additional_output,
            clip_pred_noise=False,
        )
        pred_noise = model_pred.pred_noise
        pred_x_start = model_pred.pred_x_start
        pred_v = self._predict_v(pred_x_start, k, pred_noise)

        if self.loss_target_type == ModelMeanType.START_X:
            target = x_start
            pred = pred_x_start
        elif self.loss_target_type == ModelMeanType.EPSILON:
            target = noise
            pred = pred_noise
        elif self.loss_target_type == ModelMeanType.VELOCITY:
            target = self._predict_v(x_start, k, noise)
            pred = pred_v
        elif self.loss_target_type == ModelMeanType.PREVIOUS_X:
            target = self.q_posterior_mean_variance(
                x_start=x_start, x_k=x_k, k=k
            )[0]
            pred = self.q_posterior_mean_variance(
                x_start=pred_x_start, x_k=x_k, k=k
            )[0]
        else:
            raise NotImplementedError(self.loss_target_type)

        assert pred.shape == target.shape == x_start.shape

        terms["mse"] = masked_mean_flat((target - pred) ** 2, masks, dims_to_exclude=[0, 1])
        terms["loss"] = terms["mse"]

        additional_loss = None
        if isinstance(model_pred.additional_output, dict):
            terms.update(model_pred.additional_output)
        elif th.is_tensor(model_pred.additional_output):
            additional_loss = model_pred.additional_output
        elif model_pred.additional_output is not None:
            raise TypeError(
                f"Unexpected additional_output type: {type(model_pred.additional_output)}"
            )
        if additional_loss is not None:
            terms["loss"] += additional_loss

        if th.isnan(terms["loss"]).any():
            raise RuntimeError(bold_red("loss is nan"))

        terms['original_x'] = x_start # (bs, t, c, h, w)
        # terms['noised_x'] = x_k
        terms['predicted_x_start'] = pred_x_start

        loss_weight = self.compute_loss_weights(k, self.loss_weighting.strategy)
        terms["loss"] = terms["loss"] * loss_weight

        return terms
