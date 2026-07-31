from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from einops import rearrange

from algorithms.vae.wan22_vae_model import WanVideoVAE38
from algorithms.wan.modules.model_wan22 import load_state_dict_path


class Wan22VideoVAE(torch.nn.Module):
    def __init__(
        self,
        pretrained_path: str,
        *,
        z_dim: int = 48,
        dim: int = 160,
        tiled: bool = True,
        tile_size: tuple[int, int] | list[int] = (30, 52),
        tile_stride: tuple[int, int] | list[int] = (15, 26),
    ) -> None:
        super().__init__()

        self.model = WanVideoVAE38(z_dim=z_dim, dim=dim).eval().requires_grad_(False)
        state_dict = load_state_dict_path(str(Path(pretrained_path)))
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                normalized[key] = value
            else:
                normalized[f"model.{key}"] = value
        missing, unexpected = self.model.load_state_dict(normalized, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Failed to load Wan2.2 VAE checkpoint strictly enough: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )

        self.z_dim = int(z_dim)
        self.is_causal = True
        self.temporal_pixel_length = 1
        self.tiled = bool(tiled)
        self.tile_size = tuple(int(v) for v in tile_size)
        self.tile_stride = tuple(int(v) for v in tile_stride)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "Wan22VideoVAE":
        kwargs.pop("torch_dtype", None)
        kwargs.pop("output_shape", None)
        return cls(pretrained_path=path, **kwargs)

    @torch.no_grad()
    def vae_encode(
        self,
        x: torch.Tensor,
        output_shape: Any = None,
        image_height: int | None = None,
        image_width: int | None = None,
        data_type: str = "rgb",
        max_batch_size: int | None = None,
    ) -> torch.Tensor:
        del output_shape, image_height, image_width, max_batch_size
        if data_type != "rgb":
            raise ValueError("Wan22VideoVAE currently supports rgb only.")
        if x.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            x = x.float()
        if x.max() > 2.0:
            x = x / 255.0
        video = rearrange(2.0 * x - 1.0, "b t c h w -> b c t h w")
        z = self.model.encode(
            video,
            device=video.device,
            tiled=self.tiled,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )
        return rearrange(z, "b c t h w -> b t c h w")

    @torch.no_grad()
    def vae_decode(
        self,
        z: torch.Tensor,
        input_channels: int | None = None,
        data_type: str = "rgb",
        desired_length: Optional[int] = None,
        max_batch_size: int | None = None,
    ) -> torch.Tensor:
        del input_channels, max_batch_size
        if data_type != "rgb":
            raise ValueError("Wan22VideoVAE currently supports rgb only.")
        hidden = rearrange(z, "b t c h w -> b c t h w")
        model_device = next(self.model.parameters()).device
        video = self.model.decode(
            hidden,
            device=model_device,
            tiled=self.tiled,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )
        video = rearrange((video.clamp_(-1, 1) + 1.0) / 2.0, "b c t h w -> b t c h w")
        if desired_length is not None:
            video = video[:, : int(desired_length)]
        # Tiled decode accumulates its output on CPU (see WanVideoVAE.tiled_decode).
        # Return on the input latent's device so downstream GPU consumers (metric
        # models, distributed metric all_gather) don't see CPU tensors, which under
        # an NCCL-only process group raises "No backend type associated with device
        # type cpu" during torchmetrics' compute()-time sync.
        return video.to(z.device)


__all__ = ["Wan22VideoVAE"]
