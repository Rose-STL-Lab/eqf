# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import torch as th
import numpy as np
from .interface import ModelMeanType

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if isinstance(arr, np.ndarray):
        res = th.from_numpy(arr).to(device=timesteps.device)[timesteps]
    else:
        res = arr[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=arr.device, dtype=arr.dtype)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def continuous_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a continuous Gaussian distribution.
    :param x: the targets
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    normalized_x = centered_x * inv_stdv
    log_probs = th.distributions.Normal(th.zeros_like(x), th.ones_like(x)).log_prob(normalized_x)
    return log_probs


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def add_shape_channels(x, target_shape):
    """
    Reshape tensor x to have the right dimensions for broadcasting with a tensor of shape target_shape.
    For example, if x has shape [B] and target_shape is [C, H, W], this will return a tensor of shape [B, 1, 1, 1].
    If x has shape [B, T] and target_shape is [B, T, C, H, W], this will return a tensor of shape [B, T, 1, 1, 1].
    """
    if isinstance(target_shape, int):
        # If target_shape is just a single int, treat as a 1D shape
        target_shape = [target_shape]
    
    # If target_shape is a tensor, get its shape
    if hasattr(target_shape, 'shape'):
        target_shape = target_shape.shape
    
    # Determine how many dimensions to add
    dims_to_add = len(target_shape) - x.dim()
    if dims_to_add <= 0:
        # x already has enough dimensions
        return x
    
    # Add the necessary 1 dimensions
    return x.view(*x.shape, *([1] * dims_to_add))


def resolve_loss_target_type(cfg, default):
    if cfg is None:
        return default
    loss_type = getattr(cfg, "loss_type", None)
    if loss_type is None:
        return default
    if isinstance(loss_type, ModelMeanType):
        return loss_type
    loss_key = str(loss_type).lower()
    if loss_key in {"x", "xstart", "x_start", "start_x"}:
        loss_key = "start_x"
    if loss_key in {"v", "velocity", "w"}:
        loss_key = "velocity"
    if loss_key in {"eps", "epsilon"}:
        loss_key = "epsilon"
    return ModelMeanType[loss_key]
