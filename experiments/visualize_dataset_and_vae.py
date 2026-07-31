from __future__ import annotations

from typing import Dict, Any, List

import torch

from datasets.video import (
    DroidVideoDataset,
    MinecraftVideoDataset,
    Re10KVideoDataset,
)
from omegaconf import DictConfig
from algorithms.common.vae_mixin import VAEMixin
from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print, is_rank_zero
from utils.logging_utils import log_video, get_validation_metrics_for_videos
from .base_exp import BaseExperiment
from .data_modules import BaseDataModule


class VisualizeDatasetAndVAEExperiment(BaseExperiment):
    """
    Logs dataset videos and also runs the configured VAE to reconstruct them.
    Useful to sanity-check:
      - dataloader
      - pretrained VAE checkpoint path
      - reconstruction quality / latent shapes
    """

    compatible_datasets = dict(
        droid=DroidVideoDataset,
        minecraft=MinecraftVideoDataset,
        re10k=Re10KVideoDataset,
    )

    def __init__(self, root_cfg, logger=None, ckpt_path=None) -> None:
        super().__init__(root_cfg, logger, ckpt_path)
        self.data_module = BaseDataModule(root_cfg, self.compatible_datasets)

    def _get_device(self) -> torch.device:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _build_vae_helper(self) -> VAEMixin:
        """
        Load and use the VAE the same way as `algorithms/denoising_video.py` does,
        via `VAEMixin._load_vae()` and its `_encode/_decode` wrappers.
        """

        class _VAEHelper(VAEMixin):
            def __init__(self, cfg: DictConfig, device: torch.device):
                self.cfg = cfg
                self.device = device
                self.vae = None

        algo_cfg = self.root_cfg.algorithm
        # Guard against the common Hydra override mistake where a config-group
        # selection is passed as a scalar value.
        if isinstance(getattr(algo_cfg, "vae", None), str):
            raise ValueError(
                f"`algorithm.vae` is a string ({algo_cfg.vae!r}). "
                "To select a VAE config group, use Hydra group override syntax: "
                "`algorithm/vae=<name>` (note the slash), NOT `algorithm.vae=...`."
            )
        if not isinstance(getattr(algo_cfg, "vae", None), DictConfig):
            raise ValueError("Missing `algorithm.vae` config; cannot load VAE.")

        helper = _VAEHelper(algo_cfg, self._get_device())
        helper._load_vae()
        return helper

    @torch.no_grad()
    def validation(self) -> None:
        if not is_rank_zero:
            return

        viz_cfg = getattr(self.cfg, "visualize", {})
        num_videos = int(getattr(viz_cfg, "num_videos", 8))
        max_batches = int(getattr(viz_cfg, "max_batches", 4))
        fps = int(getattr(viz_cfg, "fps", 10))
        normalize = bool(getattr(viz_cfg, "normalize", False))
        max_frames = getattr(viz_cfg, "max_frames", None)
        max_frames = int(max_frames) if max_frames is not None else None
        use_bf16 = bool(getattr(viz_cfg, "bf16", False))

        wb = getattr(self.logger, "experiment", None) if self.logger else None

        vae_helper = self._build_vae_helper()
        rank_zero_print(
            cyan(
                f"[visualize_dataset_and_vae] Loaded VAE cls={self.root_cfg.algorithm.vae.cls} "
                f"ckpt={self.root_cfg.algorithm.vae.pretrained_path}"
            )
        )

        loaders = self.data_module.val_dataloader()
        if isinstance(loaders, torch.utils.data.DataLoader):
            loaders = [loaders]

        collected_videos: List[torch.Tensor] = []
        collected_captions: List[str] = []

        for loader in loaders:
            if len(collected_videos) >= num_videos:
                break
            for batch_idx, batch in enumerate(loader):
                if batch is None:
                    continue
                if batch_idx >= max_batches:
                    break
                videos = batch.get("videos", None)
                if videos is None:
                    continue
                if max_frames is not None and videos.shape[1] > max_frames:
                    videos = videos[:, :max_frames]
                meta = batch.get("metadata", None)

                for i in range(videos.shape[0]):
                    if len(collected_videos) >= num_videos:
                        break
                    collected_videos.append(videos[i].detach().cpu())
                    if isinstance(meta, dict) and "path" in meta:
                        try:
                            caption = str(meta["path"][i])
                            if "clip" in meta:
                                caption += f" clip={meta['clip'][0][i]}:{meta['clip'][1][i]}"
                            collected_captions.append(caption)
                        except Exception:
                            pass

        if len(collected_videos) == 0:
            rank_zero_print(cyan("[visualize_dataset_and_vae] No videos found to log (dataset empty or latents-only)."))
            return

        device = self._get_device()
        videos = torch.stack(collected_videos, dim=0).to(device)  # (N,T,C,H,W) in [0,1]

        # Reconstruct
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16 and device.type == "cuda",
        ):
            latents = vae_helper._encode(videos)
            if use_bf16 and device.type == "cuda":
                # Wan2.2 generation creates BF16 latents before entering its VAE
                # decoder. Tiled encoding accumulates into FP32 CPU buffers, so
                # restore that expected decoder input dtype explicitly.
                latents = latents.to(dtype=torch.bfloat16)
            # Keep recon aligned to the input clip length. This matters e.g. for non-causal
            # VideoVAE paths that may pad internally to fixed temporal blocks.
            recons = vae_helper._decode(latents, desired_length=videos.shape[1])
        recons = recons.float()

        # Ensure recon length matches input length when possible (crop to shortest)
        min_t = min(videos.shape[1], recons.shape[1])
        videos = videos[:, :min_t]
        recons = recons[:, :min_t]

        # Log GT and recon side-by-side (log_video expects [-1,1])
        # log_video() expects [0, 1] (despite docstring); it clamps directly to [0, 1].
        videos_01 = videos.clamp(0.0, 1.0)
        recons_01 = recons.clamp(0.0, 1.0)

        log_video(
            observation_hats=recons_01.detach(),
            observation_gt=videos_01.detach(),
            namespace="visualize_dataset_and_vae",
            prefix=f"{self.root_cfg.dataset._name}_recon",
            captions=collected_captions,
            fps=fps,
            normalize=normalize,
            logger=wb,
            log_to_logger=wb is not None,
        )

        # Metrics expect [-1, 1] (data_range=2.0 in utils); compute on [-1, 1] tensors.
        videos_m11 = videos_01 * 2.0 - 1.0
        recons_m11 = recons_01 * 2.0 - 1.0
        metrics = get_validation_metrics_for_videos(
            recons_m11.detach().cpu(), videos_m11.detach().cpu()
        )
        if wb is not None:
            wb.log(
                {f"visualize_dataset_and_vae/{k}": float(v) for k, v in metrics.items()},
                commit=False,
            )
            wb.log({"visualize_dataset_and_vae/done": 1}, commit=True)

