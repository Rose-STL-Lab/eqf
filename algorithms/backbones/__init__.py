from typing import Optional, List, Literal
from .cogv import create_cogv_like_model
from .wan import create_wan_model
from omegaconf import DictConfig

__all__ = ["create_main_model"]

def create_main_model(
        backbone_cfg: DictConfig,
    ):
    AVAILABLE_MODELS = ["cogv", "wan"]
    if backbone_cfg.model_type in ["cogv"]:
        return create_cogv_like_model(**backbone_cfg)
    if backbone_cfg.model_type in ["wan"]:
        return create_wan_model(backbone_cfg)
    else:
        raise ValueError(f"Invalid model type: {backbone_cfg.model_type}. Available models: {AVAILABLE_MODELS}")


