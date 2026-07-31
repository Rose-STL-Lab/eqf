from __future__ import annotations
from typing import Type, Dict, Any

def get_vae_cls_dict() -> Dict[str, Type[Any]]:
    # local imports to avoid import-time cycles
    from .image_vae import ImageVAE
    from .wan_vae import WanVideoVAE
    from .wan22_vae import Wan22VideoVAE

    return {
        "image_vae": ImageVAE,
        "wan_vae": WanVideoVAE,
        "wan22_vae": Wan22VideoVAE,
    }
