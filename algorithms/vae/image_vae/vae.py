from __future__ import annotations

import types
from typing import Tuple, Callable, Any, cast

import torch
from omegaconf import OmegaConf, DictConfig
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL as DiffuserImageVAE
from einops import rearrange

from utils.ckpt_utils import (
    is_wandb_run_path,
    is_hf_path,
    wandb_to_local_path,
    download_pretrained as hf_to_local_path,
)
from ..common.distribution import DiagonalGaussianDistribution
from ..common.base_vae import VAE
from .model import Encoder, Decoder


class ImageVAE(VAE):
    """
    Pretrained ImageVAE model that can be used to encode and decode images.
    Ported from diffusion-forcing-transformer.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        ddconfig, embed_dim = cfg.ddconfig, cfg.embed_dim
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # Optional latent normalization stats (for matching diffusion-forcing-transformer `dataset.data_mean/std`)
        self.register_buffer("latent_mean", torch.zeros(1, 1, embed_dim, 1, 1), persistent=False)
        self.register_buffer("latent_std", torch.ones(1, 1, embed_dim, 1, 1), persistent=False)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> VAE:
        # supports diffuser:..., wandb run path, and hf `pretrained:` paths.
        latent_mean = kwargs.pop("latent_mean", None)
        latent_std = kwargs.pop("latent_std", None)

        if path.startswith("diffuser:"):
            path = path.replace("diffuser:", "")
            model = cls._from_pretrained_diffuser(path, **kwargs)
        else:
            model = cls._from_pretrained_custom(path)

        # attach latent stats if provided
        if isinstance(model, ImageVAE):
            model._set_latent_stats(latent_mean=latent_mean, latent_std=latent_std)
        return model

    @classmethod
    def _from_pretrained_custom(cls, path: str) -> "ImageVAE":
        if is_wandb_run_path(path):
            path = str(wandb_to_local_path(path))
        elif is_hf_path(path):
            path = hf_to_local_path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Temporary fix for vaes trained with older versions of the code (e.g. released Minecraft VAE)
        if "cfg" not in checkpoint:
            cfg = OmegaConf.create(
                {
                    "embed_dim": 4,
                    "ddconfig": {
                        "double_z": True,
                        "z_channels": 4,
                        "resolution": 256,
                        "in_channels": 3,
                        "out_ch": 3,
                        "ch": 128,
                        "ch_mult": [1, 2, 4, 4],
                        "num_res_blocks": 2,
                        "attn_resolutions": [],
                        "dropout": 0.0,
                    },
                }
            )
            checkpoint["cfg"] = cfg

        cfg = checkpoint["cfg"]
        model = cls(cfg)

        state_dict = checkpoint["state_dict"]
        # filter out loss / trainer-only weights if present
        for k in list(state_dict.keys()):
            if k.startswith("loss"):
                del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        return model

    @classmethod
    def _from_pretrained_diffuser(cls, path: str, **kwargs) -> VAE:
        vae = DiffuserImageVAE.from_pretrained(path, **kwargs)
        return diffuser_to_custom(vae)

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def _set_latent_stats(self, latent_mean=None, latent_std=None) -> None:
        """
        latent_mean/std: list/tuple/torch.Tensor of shape (C,) where C == embed_dim.
        Stored as buffers shaped (1, 1, C, 1, 1) for broadcasting over (B, T, C, H, W).
        """
        if latent_mean is None and latent_std is None:
            return
        if latent_mean is None or latent_std is None:
            raise ValueError("Must provide both latent_mean and latent_std if either is set.")
        mean = torch.as_tensor(latent_mean, dtype=torch.float32).flatten()
        std = torch.as_tensor(latent_std, dtype=torch.float32).flatten()
        if mean.numel() != self.latent_mean.shape[2] or std.numel() != self.latent_std.shape[2]:
            raise ValueError(
                f"latent_mean/std length must match embed_dim={self.latent_mean.shape[2]}, "
                f"got mean={mean.numel()} std={std.numel()}"
            )
        # Keep these as the same registered buffers (avoid re-binding attributes),
        # so Torch + type checkers don't treat them as defined outside __init__.
        self.latent_mean.copy_(mean.view(1, 1, -1, 1, 1).to(self.latent_mean.device))
        self.latent_std.copy_(std.view(1, 1, -1, 1, 1).to(self.latent_std.device))

    @torch.no_grad()
    def vae_encode(
        self,
        x: torch.Tensor,
        output_shape: Any = None,  # unused; kept for eqf VAEMixin compatibility
        image_height: int | None = None,  # unused
        image_width: int | None = None,  # unused
        data_type: str = "rgb",
        max_batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        eqf inference-time wrapper.

        Input:  (B, T, C, H, W) in [0, 1]
        Output: (B, T, C_latent, H_latent, W_latent)
        """
        if data_type != "rgb":
            raise ValueError("ImageVAE currently supports rgb only.")

        if x.dtype not in (torch.float16, torch.float32, torch.float64):
            x = x.float()
        if x.max() > 2.0:
            x = x / 255.0
        x = 2.0 * x - 1.0

        b, t, c, h, w = x.shape
        x2d = rearrange(x, "b t c h w -> (b t) c h w")

        bt = int(x2d.shape[0])
        max_bt = 0 if max_batch_size is None else int(max_batch_size)
        if max_bt > 0 and bt > max_bt:
            zs = []
            for i in range(0, bt, max_bt):
                posterior = self.encode(x2d[i : i + max_bt])
                zs.append(posterior.mode())
            z2d = torch.cat(zs, dim=0)
        else:
            posterior = self.encode(x2d)
            z2d = posterior.mode()

        z = rearrange(z2d, "(b t) c h w -> b t c h w", b=b, t=t)
        # Normalize latent if release checkpoint statistics are configured.
        z = (z - self.latent_mean) / self.latent_std
        return z

    @torch.no_grad()
    def vae_decode(
        self,
        z: torch.Tensor,
        input_channels: int | None = None,  # unused; kept for eqf VAEMixin compatibility
        data_type: str = "rgb",
        desired_length: int | None = None,
        max_batch_size: int | None = None,  # unused; VideoVAE may honor this
    ) -> torch.Tensor:
        """
        eqf inference-time wrapper.

        Input:  (B, T, C_latent, H_latent, W_latent)
        Output: (B, T, 3, H, W) in [0, 1]
        """
        if data_type != "rgb":
            raise ValueError("ImageVAE currently supports rgb only.")

        b, t = z.shape[:2]
        # Denormalize latent if stats are set
        z = z * self.latent_std + self.latent_mean
        z2d = rearrange(z, "b t c h w -> (b t) c h w")

        bt = int(z2d.shape[0])
        max_bt = 0 if max_batch_size is None else int(max_batch_size)
        if max_bt > 0 and bt > max_bt:
            xs = []
            for i in range(0, bt, max_bt):
                xs.append(self.decode(z2d[i : i + max_bt]))
            x2d = torch.cat(xs, dim=0)
        else:
            x2d = self.decode(z2d)

        x = x2d
        x = (x + 1.0) / 2.0
        x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)
        if desired_length is not None:
            desired_length = int(desired_length)
            if desired_length <= 0:
                raise ValueError(f"desired_length must be > 0, got {desired_length}")
            # Keep the last `desired_length` frames (back-crop), matching VideoVAE.decode semantics.
            x = x[:, -desired_length:]
        return x


def diffuser_to_custom(vae: DiffuserImageVAE) -> VAE:
    """
    Modify DiffuserImageVAE to be compatible with VAE abstract class.
    """

    def wrap_encode(encode: Callable) -> Callable:
        def wrapped_encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
            return encode(x).latent_dist

        return wrapped_encode

    def wrap_decode(decode: Callable) -> Callable:
        def wrapped_decode(self, z: torch.Tensor) -> torch.Tensor:
            return decode(z).sample

        return wrapped_decode

    def wrapped_forward(
        self, sample: torch.Tensor, sample_posterior: bool = True
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode(sample)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    vae.encode = types.MethodType(wrap_encode(vae.encode), vae)
    vae.decode = types.MethodType(wrap_decode(vae.decode), vae)
    vae.forward = types.MethodType(wrapped_forward, vae)

    return cast(VAE, vae)



