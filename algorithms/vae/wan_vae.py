from __future__ import annotations

from typing import Any, Optional
from typing import cast

import torch
from einops import rearrange

from algorithms.wan.modules.vae import video_vae_factory


class WanVideoVAE(torch.nn.Module):
    def __init__(
        self,
        pretrained_path: str,
        *,
        z_dim: int,
        mean: list[float],
        std: list[float],
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.is_causal = True
        self.temporal_pixel_length = 1
        self.model = video_vae_factory(pretrained_path=pretrained_path, z_dim=z_dim)
        self.register_buffer("latent_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("latent_std", torch.tensor(std, dtype=torch.float32))

    def _get_scale(self) -> list[torch.Tensor]:
        latent_mean = cast(torch.Tensor, self.latent_mean)
        latent_std = cast(torch.Tensor, self.latent_std)
        return [latent_mean, torch.reciprocal(latent_std)]

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "WanVideoVAE":
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
            raise ValueError("WanVideoVAE currently supports rgb only.")
        if x.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            x = x.float()
        if x.max() > 2.0:
            x = x / 255.0
        x = rearrange(2.0 * x - 1.0, "b t c h w -> b c t h w")
        z = self.model.encode(x, self._get_scale())
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
            raise ValueError("WanVideoVAE currently supports rgb only.")
        z = rearrange(z, "b t c h w -> b c t h w")
        x = self.model.decode(z, self._get_scale()).clamp_(-1, 1)
        x = rearrange((x + 1.0) / 2.0, "b c t h w -> b t c h w")
        if desired_length is not None:
            x = x[:, : int(desired_length)]
        return x
