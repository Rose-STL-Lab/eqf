from __future__ import annotations

import torch
from torch import Tensor
from typing import Any, Literal, Optional
from algorithms.vae.registry import get_vae_cls_dict
from utils.torch_utils import freeze_model

class VAEMixin:
    # These attributes are provided by the main LightningModule/Algo class mixing this in.
    cfg: Any
    device: Any
    vae: Any

    def _resolve_vae_max_batch(self) -> int:
        """
        Maximum number of items to process in a single VAE encode/decode call.
        Interpretation is VAE-specific:
        - Image VAE wrappers typically treat this as a max for flattened (B*T) frames.
        - Video VAE wrappers may treat this as a conservative (B*T) budget but will
          avoid arbitrary time-chunking (block temporal models).
        If unset/None/<=0, no chunking is applied by the VAE wrapper.
        """
        v = getattr(self.cfg.vae, "max_batch_size", None)
        return 0 if v is None else int(v)

    def _load_vae(self) -> None:
        """
        Load the pretrained VAE model.
        """
        vae_cls = get_vae_cls_dict()[self.cfg.vae.cls]
        self.vae = vae_cls.from_pretrained(
            path=self.cfg.vae.pretrained_path,
            torch_dtype=(
                torch.float16 if self.cfg.vae.use_fp16 else torch.float32
            ),  # only for Diffuser's ImageVAE
            **self.cfg.vae.pretrained_kwargs,
        ).to(self.device)
        
        # dirty hack to load the model from the .pt ckpt
        # if self.cfg.vae.load_ckpt:
        #     self.vae.load_ckpt(self.cfg.vae.ckpt_path)
        freeze_model(self.vae)

    def _encode(self, frames: Tensor, data_type: Literal["rgb", "depth"] = "rgb") -> Tensor:
        """
        args:
            frames: (bs, t, c, h, w)
        """
        max_bt = self._resolve_vae_max_batch()
        # Delegate batching/chunking semantics to the VAE wrapper itself:
        # - ImageVAEs typically chunk along flattened (B*T)
        # - VideoVAEs may require block-aware temporal handling
        return self.vae.vae_encode(
            frames,
            output_shape=self.cfg.vae.pretrained_kwargs.output_shape,
            image_height=self.cfg.x_shape[1],
            image_width=self.cfg.x_shape[2],
            data_type=data_type,
            max_batch_size=max_bt,
        )

    def _decode(
        self,
        xs: Tensor,
        data_type: Literal["rgb", "depth"] = "rgb",
        *,
        desired_length: Optional[int] = None,
    ) -> Tensor:
        """
        Decode the latent codes to the original frames.
        """
        max_bt = self._resolve_vae_max_batch()
        return self.vae.vae_decode(
            xs,
            input_channels=self.cfg.latent.num_channels,
            data_type=data_type,
            desired_length=desired_length,
            max_batch_size=max_bt,
        )
