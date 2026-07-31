# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from . import flow_matching as fm
from typing import Literal, Optional
import os
from omegaconf import DictConfig, OmegaConf
from .gaussian_diffusion import GaussianDiffusion
from .flow_matching import FlowMatching
from .equilibrium_matching import EquilibriumMatching, resolve_learning_rate_eta_from_config
from typing import Union
from logging import Logger


def create_diffusion_algo(
    num_sampling_steps: int = 50,
    noise_schedule: Literal["linear", "cosine", "sigmoid", "squaredcos_cap_v2", "cosine_simple"] = "linear", 
    use_kl=False,
    mean_type: Literal["xstart", "epsilon", "velocity"] = "epsilon",
    var_type: Literal["fixed_large", "fixed_small"] = "fixed_large",
    rescale_learned_sigmas=False,
    num_noise_levels=1000,
    shift = 0.125,
    cfg: Optional[DictConfig] = None,
    logger: Optional[Logger] = None
) -> GaussianDiffusion:
    betas = gd.get_named_beta_schedule(noise_schedule, num_noise_levels, shift = shift)

    if use_kl:
        raise NotImplementedError("use_kl is deprecated; GaussianDiffusion is MSE-only.")
    if rescale_learned_sigmas:
        raise NotImplementedError(
            "rescale_learned_sigmas is deprecated; GaussianDiffusion is MSE-only"
        )
    if var_type not in {"fixed_large", "fixed_small"}:
        raise NotImplementedError(
            "GaussianDiffusion only supports fixed variance on this branch."
        )
    
    loss_type = gd.LossType.MSE

    return GaussianDiffusion(
        betas=betas,
        num_sampling_steps=num_sampling_steps,
        model_mean_type=gd.ModelMeanType[mean_type],
        model_var_type=gd.ModelVarType[var_type],
        loss_type=loss_type, # Default to MSE
        cfg = cfg,
        logger = logger
    )

def create_flow_algo(
    num_sampling_steps: int = 50,
    noise_schedule: Literal["linear"] = "linear", 
    mean_type: Literal["xstart", "epsilon", "velocity"] = "epsilon",
    num_noise_levels=1000,
    cfg: Optional[DictConfig] = None,
    logger: Optional[Logger] = None
) -> FlowMatching:

    """
    IN PROGRESS: define flow matching algorithm with a lot of simplifications
    """

    betas = gd.get_named_beta_schedule(noise_schedule, num_noise_levels) # just for scaling

    loss_type = fm.LossType.MSE

    return FlowMatching(
        betas=betas,
        num_sampling_steps=num_sampling_steps,
        model_mean_type=fm.ModelMeanType[mean_type],
        loss_type=loss_type, # Default to MSE
        cfg = cfg,
        logger = logger
    )

def create_eq_algo(
    num_sampling_steps: int = 50,
    noise_schedule: Literal["linear"] = "linear", 
    mean_type: Literal["xstart", "epsilon", "velocity"] = "epsilon",
    num_noise_levels=1000,
    cfg: Optional[DictConfig] = None,
    logger: Optional[Logger] = None
) -> EquilibriumMatching:

    """
    IN PROGRESS: define eq matching algorithm with a lot of simplifications
    """

    betas = gd.get_named_beta_schedule(noise_schedule, num_noise_levels) # just for scaling

    loss_type = fm.LossType.MSE

    return EquilibriumMatching(
        betas=betas,
        num_sampling_steps=num_sampling_steps,
        model_mean_type=fm.ModelMeanType[mean_type],
        loss_type=loss_type, # Default to MSE
        cfg = cfg,
        logger = logger
    )
