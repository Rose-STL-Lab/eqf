from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

from hydra.core.hydra_config import HydraConfig
import torch

from datasets.video import (
    DroidVideoDataset,
    MinecraftVideoDataset,
    Re10KVideoDataset,
)
from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print, is_rank_zero
from utils.logging_utils import log_video
from .base_exp import BaseExperiment
from .data_modules import BaseDataModule


class VisualizeDatasetExperiment(BaseExperiment):
    """
    Loads a few batches from the dataloader and logs videos (+ optional conditions) to wandb.
    This is meant as a quick sanity check that:
      - dataset indexing works
      - dataloading works
      - clips/metadata look correct
    """

    compatible_datasets = dict(
        droid=DroidVideoDataset,
        minecraft=MinecraftVideoDataset,
        re10k=Re10KVideoDataset,
    )

    def __init__(self, root_cfg, logger=None, ckpt_path=None) -> None:
        super().__init__(root_cfg, logger, ckpt_path)
        self.data_module = BaseDataModule(root_cfg, self.compatible_datasets)

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

        if self.logger:
            wb = self.logger.experiment
        else:
            wb = None
        raw_dir = None
        if wb is None:
            raw_dir = Path(HydraConfig.get().runtime.output_dir) / "visualize_dataset"
            rank_zero_print(cyan(f"[visualize_dataset] WandB disabled; saving previews to {raw_dir}"))

        rank_zero_print(cyan(f"[visualize_dataset] Sampling num_videos={num_videos} max_batches={max_batches} fps={fps}"))

        loaders = self.data_module.val_dataloader()
        if isinstance(loaders, torch.utils.data.DataLoader):
            loaders = [loaders]

        collected_videos: List[torch.Tensor] = []
        collected_conds: List[torch.Tensor] = []
        collected_captions: List[str] = []

        for loader_idx, loader in enumerate(loaders):
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
                # (B, T, C, H, W) in [0, 1]
                if max_frames is not None and videos.shape[1] > max_frames:
                    videos = videos[:, :max_frames]

                conds = batch.get("conds", None)
                meta = batch.get("metadata", None)

                b = videos.shape[0]
                for i in range(b):
                    if len(collected_videos) >= num_videos:
                        break
                    collected_videos.append(videos[i].detach().cpu())
                    if conds is not None:
                        collected_conds.append(conds[i].detach().cpu())
                    # Best-effort caption from metadata
                    if isinstance(meta, dict) and "path" in meta:
                        try:
                            path_i = meta["path"][i]
                            caption = str(path_i)
                            if "clip" in meta:
                                caption += f" clip={meta['clip'][0][i]}:{meta['clip'][1][i]}"
                            collected_captions.append(caption)
                        except Exception:
                            pass

        if len(collected_videos) == 0:
            rank_zero_print(cyan("[visualize_dataset] No videos found to log (dataset empty or latents-only)."))
            return

        videos_to_log = torch.stack(collected_videos, dim=0)  # (N, T, C, H, W) in [0, 1]

        log_video(
            # log_video() expects [0, 1] (despite docstring); it clamps directly to [0, 1].
            observation_hats=videos_to_log,
            observation_gt=None,
            namespace="visualize_dataset",
            prefix=f"{self.root_cfg.dataset._name}",
            captions=collected_captions,
            fps=fps,
            normalize=normalize,
            logger=wb,
            log_to_logger=wb is not None,
            raw_dir=raw_dir,
        )

        # Log conditions as a small table (best-effort)
        if wb is not None and len(collected_conds) > 0:
            import wandb

            table = wandb.Table(columns=["idx", "conds_shape", "conds_preview"])
            for i, c in enumerate(collected_conds[:num_videos]):
                preview = c
                if c.ndim >= 1:
                    preview = c[: min(8, c.shape[0])]
                table.add_data(i, str(tuple(c.shape)), str(preview))
            wb.log({"visualize_dataset/conds": table}, commit=False)

        # Force a commit so videos/tables flush even though we logged with commit=False above.
        if wb is not None:
            wb.log({"visualize_dataset/done": 1}, commit=True)
