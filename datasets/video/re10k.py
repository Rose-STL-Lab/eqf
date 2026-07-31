from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from torchvision import transforms
from typing_extensions import override

from decord import VideoReader, cpu

from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan

from .base_video import BaseAdvancedVideoDataset, SPLIT


class Re10KVideoDataset(BaseAdvancedVideoDataset):
    """
    RealEstate10K dataset wired into eqf's BaseAdvancedVideoDataset.

    Temporal downsampling is intentionally left out here. We assume RE10K is
    already normalized to 10 fps and treat cached `video_pts` as the
    authoritative sampling timeline for clip extraction.
    """

    _ALL_SPLITS = ["training", "validation"]
    _SPLIT_TO_VIDEO_DIR = {
        "training": "training_256",
        "validation": "test_256",
    }
    _SPLIT_TO_POSE_DIR = {
        "training": "training_poses",
        "validation": "test_poses",
    }
    _SPLIT_TO_BOOTSTRAP_METADATA = {
        "training": "training.pt",
        "validation": "test.pt",
    }
    _NORMALIZED_METADATA_KEYS = {
        "video_paths",
        "video_pts",
        "video_fps",
        "pose_path",
        "caption",
    }

    def __init__(
        self,
        cfg: DictConfig,
        split: str = "training",
        purpose: str = "training",
    ):
        self.poses_enabled = bool(getattr(cfg, "poses", True))
        super().__init__(cfg, split=split, purpose=purpose)
        if not self.poses_enabled:
            self.external_cond_dim = 0

    @override
    def _should_download(self) -> bool:
        source_dir = self.save_dir / self._SPLIT_TO_VIDEO_DIR[self.split]
        return not source_dir.exists()

    def build_transform(self) -> Callable[[torch.Tensor], torch.Tensor]:
        # dataset.resolution is stored as [width, height], while Resize expects [height, width].
        target_hw = (self.resolution[1], self.resolution[0])
        return transforms.Resize(target_hw, antialias=True)

    def download_dataset(self) -> None:
        raise ValueError(
            "RE10K is expected to already exist on disk under `data/re10k/` "
            "with `training_256/`, `test_256/`, `training_poses/`, and `test_poses/`."
        )

    @staticmethod
    def _normalize_relative_path(path: Path | str, markers: Tuple[str, ...]) -> Path:
        path = Path(path)
        parts = path.parts
        for marker in markers:
            if marker in parts:
                idx = parts.index(marker)
                return Path(*parts[idx:])
        return path

    @staticmethod
    def _read_csv_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value)

    @staticmethod
    def _read_csv_int(value: Any) -> Optional[int]:
        if pd.isna(value) or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _read_csv_float(value: Any) -> Optional[float]:
        if pd.isna(value) or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _metadata_is_normalized(self, metadata_path: Path) -> bool:
        if not metadata_path.exists():
            return False
        metadata = torch.load(metadata_path, weights_only=False)
        if not self._NORMALIZED_METADATA_KEYS.issubset(metadata.keys()):
            return False
        if len(metadata["video_paths"]) == 0:
            return True

        first_video_path = Path(metadata["video_paths"][0])
        first_pose_path = Path(metadata["pose_path"][0])
        first_video_pts = torch.as_tensor(metadata["video_pts"][0], dtype=torch.long)
        expected_video_pts = torch.arange(len(first_video_pts), dtype=torch.long)
        return (
            str(first_video_path).startswith(str(self.save_dir))
            and str(first_pose_path).startswith(str(self.save_dir))
            and torch.equal(first_video_pts, expected_video_pts)
        )

    def _probe_video(self, video_path: Path) -> Tuple[torch.Tensor, float]:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        num_frames = len(vr)
        fps = vr.get_avg_fps()
        if fps is None:
            fps = float(getattr(self.cfg, "fps", 10))
        return torch.arange(num_frames, dtype=torch.long), float(fps)

    @override
    def build_metadata(self, split: SPLIT) -> None:
        metadata_path = self.metadata_dir / f"{split}.pt"
        if self._metadata_is_normalized(metadata_path):
            return

        csv_path = self.save_dir / "metadata.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing RE10K metadata CSV at {csv_path}")

        df = pd.read_csv(csv_path, keep_default_na=False)
        split_df = df[df["split"] == split]
        if split_df.empty:
            raise ValueError(f"No RE10K CSV rows found for split `{split}`")

        rows: List[Tuple[Path, torch.Tensor, float, Path, str]] = []
        missing_pose_count = 0
        missing_video_count = 0
        probed_video_count = 0

        with tqdm(
            split_df.itertuples(index=False),
            total=len(split_df),
            desc=f"Normalizing re10k/{split} metadata",
        ) as pbar:
            for idx, row in enumerate(pbar, start=1):
                rel_video_path = Path(self._read_csv_value(row.video_path))
                video_path = self.save_dir / rel_video_path
                if not video_path.exists():
                    missing_video_count += 1
                    continue

                caption = self._read_csv_value(row.caption)
                rel_pose_path = Path(self._read_csv_value(row.pose_path))
                pose_path = self.save_dir / rel_pose_path if rel_pose_path.as_posix() != "." else Path("")

                if self.poses_enabled and (not rel_pose_path.as_posix() or not pose_path.is_file()):
                    missing_pose_count += 1
                    continue

                n_frames = self._read_csv_int(row.n_frames)
                csv_fps = self._read_csv_float(row.fps)
                if n_frames is not None and n_frames > 0:
                    video_pts = torch.arange(n_frames, dtype=torch.long)
                    video_fps = csv_fps if csv_fps is not None else float(getattr(self.cfg, "fps", 10))
                else:
                    video_pts, video_fps = self._probe_video(video_path)
                    probed_video_count += 1

                rows.append((video_path, video_pts, video_fps, pose_path, caption))

                if idx == len(split_df) or idx % 500 == 0:
                    pbar.set_postfix(
                        kept=len(rows),
                        missing_pose=missing_pose_count,
                        missing_video=missing_video_count,
                        probed=probed_video_count,
                    )

        if not rows:
            raise ValueError(f"RE10K split `{split}` is empty after metadata normalization")

        rank_zero_print(
            cyan(
                f"re10k/{split}: wrote normalized metadata for {len(rows)} videos; "
                f"dropped {missing_pose_count} rows with missing poses; "
                f"skipped {missing_video_count} rows with missing videos; "
                f"probed {probed_video_count} videos missing from timing cache"
            )
        )

        metadata = {
            "video_paths": [row[0] for row in rows],
            "video_pts": [row[1] for row in rows],
            "video_fps": [row[2] for row in rows],
            "pose_path": [row[3] for row in rows],
            "caption": [row[4] for row in rows],
        }
        torch.save(metadata, metadata_path)

    def load_cond(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load the raw camera pose and concatenate to flatten the intrinsics (4) and extrinsics (12).
        The intrinsics form 

        K  = [[W fx, 0, Wcx],
              [0, H fy, Hcy],
              [0,  0,   1]]

        The intrinsics form
        E = [r_11, ..., r_33, t_x, t_y, t_z]

        We get rid of k1, k2 which are at pose[:, 4:6] (distortion only) since they are not needed for the NeRF-like training.
        Only having retained the compact 16 values, we concat to form the pose_16 vector.
        """
        if not self.poses_enabled:
            clip_len = self.video_length(video_metadata) if end_frame is None else end_frame - start_frame
            return torch.empty((clip_len, 0), dtype=torch.float32)

        if end_frame is None:
            end_frame = self.video_length(video_metadata)

        pose_path = Path(video_metadata["pose_path"])
        pose = torch.load(pose_path, map_location="cpu", weights_only=False)
        pose = torch.as_tensor(pose)
        if pose.ndim != 2 or pose.shape[-1] < 18:
            raise ValueError(
                f"Expected RE10K pose tensor at {pose_path} to have shape (T, 18+), got {tuple(pose.shape)}"
            )

        pose_16 = torch.cat([pose[:, :4], pose[:, 6:]], dim=-1).to(torch.float32)
        frame_ids = torch.as_tensor(
            video_metadata["video_pts"][start_frame:end_frame],
            dtype=torch.long,
        )
        if frame_ids.numel() == 0:
            return pose_16.new_empty((0, pose_16.shape[-1]))
        if frame_ids.max().item() >= pose_16.shape[0]:
            raise ValueError(
                f"Pose tensor at {pose_path} is shorter than requested frame id {frame_ids.max().item()}"
            )
        return pose_16.index_select(0, frame_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        output = super().__getitem__(idx)

        if self.is_latent_preprocessing_expt:
            video_idx, clip_idx = idx, 0
        else:
            video_idx, clip_idx = self.get_clip_location(idx)

        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)
        if self.is_latent_preprocessing_expt:
            start_frame, end_frame = 0, min(self.n_frames, video_length)
        else:
            start_frame, end_frame = clip_idx, min(clip_idx + self.n_frames, video_length)

        output["metadata"] = {
            "path": str(video_metadata["video_paths"]),
            "clip": [start_frame, end_frame],
            "pose_path": str(video_metadata["pose_path"]),
            "caption": video_metadata.get("caption", ""),
        }
        if "conds" in output:
            output["pose_16"] = output["conds"]

        return output
