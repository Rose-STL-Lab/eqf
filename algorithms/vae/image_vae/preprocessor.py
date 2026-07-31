from pathlib import Path

import torch
import torch.distributed as dist
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from torch import Tensor

from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from utils.logging_utils import log_video
from utils.storage_utils import safe_torch_save
from utils.torch_utils import freeze_model

from .vae import ImageVAE


class ImageVAEPreprocessor(BasePytorchAlgo):
    """Precompute Minecraft ImageVAE latents from raw RGB videos."""

    def __init__(self, cfg: DictConfig):
        self.max_decode_length = cfg.logging.max_video_length
        self.log_every_n_batch = cfg.logging.every_n_batch
        self.cfg = cfg
        self.is_latent_diffusion = False
        super().__init__(cfg)

    def configure_model(self) -> None:
        self.vae = ImageVAE.from_pretrained(
            path=self.cfg.vae.pretrained_path,
            torch_dtype=(
                torch.float16 if self.cfg.vae.use_fp16 else torch.float32
            ),
            **self.cfg.vae.pretrained_kwargs,
        ).to(self.device)
        freeze_model(self.vae)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        raise NotImplementedError(
            "ImageVAE preprocessing only supports validation."
        )

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        raise NotImplementedError(
            "ImageVAE preprocessing only supports validation."
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        batch, latent_paths, _ = batch
        videos = batch["videos"]
        nonterminal = batch.get("nonterminal")
        latent_paths = [Path(path) for path in latent_paths]

        latents = self._encode(videos)

        if batch_idx % 100 == 0:
            self.log("dummy", 0.0)

        if batch_idx % self.log_every_n_batch == 0 and self.logger:
            reconstructed_videos = self._decode(
                latents[: self.max_decode_length]
            ).detach().cpu()
            log_video(
                reconstructed_videos,
                videos.detach().cpu()[: self.max_decode_length],
                step=self._manual_wandb_step(),
                namespace="reconstruction_vis",
                logger=self.logger.experiment,
                captions=[
                    f"{path.parent.parent.name}/{path.parent.name}/{path.stem}"
                    for path in latent_paths
                ],
            )

        for index, (latent, latent_path) in enumerate(
            zip(latents.detach().cpu(), latent_paths)
        ):
            if nonterminal is not None:
                mask = nonterminal[index].detach().cpu()
                if mask.numel() > 0 and mask.any():
                    last_valid = (
                        int(torch.nonzero(mask, as_tuple=False).max().item()) + 1
                    )
                    latent = latent[:last_valid]
            safe_torch_save(latent.clone(), latent_path)

        return None

    def on_validation_end(self) -> None:
        if getattr(self.trainer, "world_size", 1) > 1:
            try:
                self.trainer.strategy.barrier()
            except Exception as error:
                if dist.is_available() and dist.is_initialized():
                    try:
                        dist.barrier()
                    except Exception as fallback_error:
                        rank_zero_info(
                            "Latent preprocessing barrier failed: "
                            f"{error!r} / {fallback_error!r}"
                        )
        rank_zero_info("Finished preprocessing Minecraft latents.")

    def on_validation_epoch_end(self, namespace="validation") -> None:
        return

    def _encode(self, frames: Tensor) -> Tensor:
        return self.vae.vae_encode(
            frames,
            output_shape=self.cfg.vae.pretrained_kwargs.output_shape,
            image_height=frames.shape[3],
            image_width=frames.shape[4],
        )

    def _decode(self, latents: Tensor) -> Tensor:
        return self.vae.vae_decode(
            latents,
            input_channels=self.cfg.vae.latent_dim,
        )


__all__ = ["ImageVAEPreprocessor"]
