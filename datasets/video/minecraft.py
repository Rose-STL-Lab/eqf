from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from typing_extensions import override

from .utils import collate_fn_skip_none
from .base_video import BaseAdvancedVideoDataset, SPLIT


class _VideoTimestampsDataset:
    """
    Dataset used to parallelize reading timestamps/fps from a list of video paths.
    Must be top-level to be picklable for multiprocessing DataLoader workers.
    """

    def __init__(self, video_paths: List[Path]) -> None:
        self.video_paths = video_paths

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int):
        return validate_item_and_get_info(self.video_paths[idx])


def validate_item_and_get_info(video_path: Path):
    """
    Returns (timestamps, fps, video_path) or None on failure.
    timestamps is a list of frame indices [0..num_frames-1].
    """
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(str(video_path), ctx=cpu())
        num_frames = len(vr)
        fps = vr.get_avg_fps()
        timestamps = list(range(num_frames))
        return timestamps, fps, video_path
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return None


class MinecraftVideoDataset(BaseAdvancedVideoDataset):
    """
    Minecraft dataset (actions-conditioned).

    Expected folder structure under cfg.save_dir (default: data/minecraft):
      - training/**/*.mp4
      - validation/**/*.mp4
      - (optional) test/**/*.mp4

    For each video file, expects a sidecar NPZ at the same path with `.npz`
    suffix containing an `actions` array of shape (T,) with integer action ids.
      - e.g. `clip_000001.mp4` -> `clip_000001.npz` with key `actions`
    """

    _ALL_SPLITS = ["training", "validation", "test"]

    def __init__(self, cfg: DictConfig, split: str = "training", purpose: str = "training"):
        super().__init__(cfg, split=split, purpose=purpose)

    def build_transform(self) -> Callable[[torch.Tensor], torch.Tensor]:
        # Use nearest to match many Minecraft datasets' sharp pixel edges.
        return transforms.Resize(
            self.resolution,
            interpolation=InterpolationMode.NEAREST_EXACT,
            antialias=True,
        )


    @override
    def download_dataset(self):
        raise FileNotFoundError(
            f"Minecraft data is missing under {self.save_dir}. "
            "Download it explicitly with `bash scripts/download_minecraft.sh` "
            f"(or `bash scripts/download_minecraft.sh --dest {self.save_dir}`)."
        )


    @override
    def build_metadata(self, split: SPLIT) -> None:
        if (self.metadata_dir / f"{split}.pt").exists():
            return

        split_root = self.save_dir / split
        video_paths = sorted(list(split_root.glob("**/*.mp4")), key=str)

        if len(video_paths) == 0:
            torch.save(
                {"video_paths": [], "video_pts": [], "video_fps": []},
                self.metadata_dir / f"{split}.pt",
            )
            return

        dl: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(video_paths),
            batch_size=max(1, min(16, len(video_paths))),
            num_workers=4,
            collate_fn=collate_fn_skip_none,
            pin_memory=True,
            persistent_workers=True,
        )

        video_pts: List[torch.Tensor] = []
        video_fps: List[float] = []
        valid_video_paths: List[Path] = []

        with tqdm(total=len(dl), desc=f"Building metadata for {split}", position=0) as pbar:
            for batch in dl:
                pbar.update(1)
                batch_pts, batch_fps, batch_valid_video_path = list(zip(*batch))
                batch_pts = [torch.as_tensor(pts, dtype=torch.long).cpu() for pts in batch_pts]
                video_pts.extend(batch_pts)
                video_fps.extend([float(fps) if fps is not None else 0.0 for fps in batch_fps])
                valid_video_paths.extend(batch_valid_video_path)

        metadata = {
            "video_paths": valid_video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
        }
        torch.save(metadata, self.metadata_dir / f"{split}.pt")

    def _npz_path_from_video(self, video_path: Path) -> Path:
        return video_path.with_suffix(".npz")

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: Optional[int] = None
    ) -> torch.Tensor:
        """
        Load actions as external condition.
        Returns Tensor of shape:
          - (T, external_cond_dim) for one_hot
          - (T,) for action_int
        """
        if end_frame is None:
            end_frame = self.video_length(video_metadata)

        video_path: Path = video_metadata["video_paths"]
        npz_path = self._npz_path_from_video(video_path)
        actions = np.load(npz_path)["actions"][start_frame:end_frame]

        if self.cond_loading_style == "one_hot":
            # actions are assumed to be in [0, external_cond_dim)
            actions = torch.from_numpy(actions).long()
            return torch.eye(self.external_cond_dim, dtype=torch.float32)[actions]
        elif self.cond_loading_style == "action_int":
            return torch.from_numpy(actions).long()
        else:
            raise RuntimeError(
                f"Action cond loading style {self.cond_loading_style} not recognized in minecraft dataset."
            )

    @override
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        output = super().__getitem__(idx)

        # Attach metadata for logging/debugging consistency with other datasets.
        video_idx, clip_idx = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)
        start_frame, end_frame = clip_idx, min(clip_idx + self.n_frames, video_length)
        output["metadata"] = {
            "path": str(video_metadata["video_paths"]),
            "clip": [start_frame, end_frame],
        }

        if self.is_latent_preprocessing_expt:
            return (
                output,
                self.video_metadata_to_latent_path(video_metadata).as_posix(),
                self.video_metadata_to_latent_depth_path(video_metadata).as_posix(),
            )
        return output


