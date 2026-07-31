from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from typing_extensions import override

from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan

from .base_video import BaseAdvancedVideoDataset, SPLIT


class DroidVideoDataset(BaseAdvancedVideoDataset):
    """
    DROID robot video dataset. Assumes that the cleaned_metadata.csv from the LVP
    paper is downloaded along the dataset.

    This class reads cleaned Droid metadata directly from
    `save_dir / metadata_path`. The CSV is the source artifact; the current
    repo's `metadata/{split}.pt` files are derived caches containing sampled
    `video_pts` for the shared base video loader.
    """

    _ALL_SPLITS = ["training", "validation", "test"]
    _METADATA_KEYS = {
        "video_paths",
        "video_pts",
        "video_fps",
        "source_fps",
        "caption",
        "gemini_caption",
        "original_caption",
        "has_caption",
        "prompt_embed_path",
    }
    _CAPTION_NORMALIZATION_VERSION = "gemini_first_v2"
    _PATH_NORMALIZATION_VERSION = "droid_raw_cleaned_v1"

    def __init__(
        self,
        cfg: DictConfig,
        split: str = "training",
        purpose: str = "training",
    ):
        self.metadata_path = Path(str(getattr(cfg, "metadata_path", "cleaned_metadata.csv")))
        self.force_download = bool(getattr(cfg, "force_download", False))
        self.test_percentage = float(getattr(cfg, "test_percentage", 0.01))
        self.shuffle_seed = int(getattr(cfg, "shuffle_seed", 0))
        self.target_fps = float(getattr(cfg, "fps", 16))
        self.target_n_frames = int(getattr(cfg, "max_frames", getattr(cfg, "n_frames", 49)))
        self.id_token = str(getattr(cfg, "id_token", "") or "")
        self.load_prompt_embed = bool(getattr(cfg, "load_prompt_embed", False))
        self.max_text_tokens = int(getattr(cfg, "max_text_tokens", 512))
        self.trim_mode = str(getattr(cfg, "trim_mode", "speedup"))
        self.pad_mode = str(getattr(cfg, "pad_mode", "slowdown"))
        self.check_video_path = bool(getattr(cfg, "check_video_path", False))

        download_cfg = getattr(cfg, "download", {})
        self.override_fps = getattr(download_cfg, "override_fps", None)

        if self.trim_mode not in ("speedup", "random_cut"):
            raise ValueError("dataset.trim_mode must be one of ['speedup', 'random_cut']")
        if self.pad_mode not in ("slowdown", "pad_last", "discard"):
            raise ValueError(
                "dataset.pad_mode must be one of ['slowdown', 'pad_last', 'discard']"
            )

        super().__init__(cfg, split=split, purpose=purpose)

    @property
    def metadata_csv_path(self) -> Path:
        if self.metadata_path.is_absolute():
            return self.metadata_path
        return self.save_dir / self.metadata_path

    @property
    def cache_manifest_path(self) -> Path:
        return self.metadata_dir / "droid_cache_manifest.json"

    @override
    def _should_download(self) -> bool:
        """
        This assumes that we have the metadata CSV downloaded from the LVP HuggingFace,
        as described in the README.
        """
        return not self.metadata_csv_path.exists()

    @override
    def download_dataset(self) -> None:
        """
        The original Droid cleaned metadata is downloaded externally to use Gemini captions.
        """
        raise FileNotFoundError(
            f"Missing required Droid cleaned metadata CSV: {self.metadata_csv_path}"
        )

    def build_transform(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Minimal, does not have transformations from original LVP repo.
        """
        return transforms.Resize((self.resolution[1], self.resolution[0]), antialias=True)

    @override
    def build_metadata(self, split: SPLIT) -> None:
        metadata_path = self.metadata_dir / f"{split}.pt"
        if not self.force_download and self._metadata_cache_is_current(metadata_path):
            return
        self._build_all_metadata_caches()

    def _metadata_cache_is_current(self, metadata_path: Path) -> bool:
        if not metadata_path.exists():
            return False
        if not self.cache_manifest_path.exists():
            return False
        try:
            with self.cache_manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            return False
        if manifest != self._cache_manifest():
            return False
        try:
            metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
        except Exception:
            return False
        if not self._METADATA_KEYS.issubset(metadata.keys()):
            return False
        return True

    def _cache_manifest(self) -> Dict[str, Any]:
        csv_stat = self.metadata_csv_path.stat() if self.metadata_csv_path.exists() else None
        filtering = getattr(self.cfg, "filtering", None)
        filtering_cfg = (
            OmegaConf.to_container(filtering, resolve=True)
            if isinstance(filtering, DictConfig)
            else filtering
        )
        return {
            "metadata_csv_path": self.metadata_csv_path.resolve().as_posix()
            if self.metadata_csv_path.exists()
            else str(self.metadata_csv_path),
            "metadata_csv_mtime_ns": csv_stat.st_mtime_ns if csv_stat is not None else None,
            "metadata_csv_size": csv_stat.st_size if csv_stat is not None else None,
            "target_fps": self.target_fps,
            "target_n_frames": self.target_n_frames,
            "trim_mode": self.trim_mode,
            "pad_mode": self.pad_mode,
            "override_fps": self.override_fps,
            "test_percentage": self.test_percentage,
            "shuffle_seed": self.shuffle_seed,
            "filtering": filtering_cfg,
            "caption_normalization": self._CAPTION_NORMALIZATION_VERSION,
            "path_normalization": self._PATH_NORMALIZATION_VERSION,
        }

    def _build_all_metadata_caches(self) -> None:
        if not self.metadata_csv_path.exists():
            raise FileNotFoundError(
                f"Missing required Droid cleaned metadata CSV: {self.metadata_csv_path}"
            )

        records = self._load_csv_records()
        records = self._filter_records(records)
        split_to_records = self._split_records(records)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        for split, split_records in split_to_records.items():
            metadata = self._records_to_metadata(split_records)
            torch.save(metadata, self.metadata_dir / f"{split}.pt")
            rank_zero_print(cyan(f"droid/{split}: cached {len(split_records)} videos"))
        with self.cache_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self._cache_manifest(), f, indent=2, sort_keys=True)

    def _load_csv_records(self) -> List[Dict[str, Any]]:
        df = pd.read_csv(self.metadata_csv_path, keep_default_na=False)
        records = df.to_dict("records")
        if not records:
            raise ValueError(f"No Droid records found in {self.metadata_csv_path}")
        return records

    def _filter_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtering = getattr(self.cfg, "filtering", None)
        if filtering is not None and bool(getattr(filtering, "disable", False)):
            return records

        before = len(records)
        filtered = [record for record in records if self._record_passes_filter(record)]
        rank_zero_print(
            cyan(
                f"{self.metadata_csv_path}: filtered {before - len(filtered)} "
                f"records from {before} to {len(filtered)}"
            )
        )
        return filtered

    def _split_records(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        if any("split" in record and str(record["split"]) for record in records):
            return {
                split: [record for record in records if str(record.get("split", "")) == split]
                for split in self._ALL_SPLITS
            }

        shuffled = list(records)
        random.Random(self.shuffle_seed).shuffle(shuffled)
        n_validation = int(len(shuffled) * self.test_percentage)
        if n_validation <= 0:
            return {"training": shuffled, "validation": [], "test": []}
        return {
            "training": shuffled[:-n_validation],
            "validation": shuffled[-n_validation:],
            "test": [],
        }

    def _records_to_metadata(self, records: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        rows = [self._record_to_metadata_row(record) for record in records]
        return {
            "video_paths": [row["video_paths"] for row in rows],
            "video_pts": [row["video_pts"] for row in rows],
            "video_fps": [row["video_fps"] for row in rows],
            "source_fps": [row["source_fps"] for row in rows],
            "caption": [row["caption"] for row in rows],
            "gemini_caption": [row["gemini_caption"] for row in rows],
            "original_caption": [row["original_caption"] for row in rows],
            "has_caption": [row["has_caption"] for row in rows],
            "prompt_embed_path": [row["prompt_embed_path"] for row in rows],
            "width": [row["width"] for row in rows],
            "height": [row["height"] for row in rows],
            "n_frames": [row["n_frames"] for row in rows],
        }

    def _record_to_metadata_row(self, record: Dict[str, Any]) -> Dict[str, Any]:
        video_path = self._resolve_record_video_path(record)
        if self.check_video_path and not video_path.is_file():
            raise FileNotFoundError(f"Missing Droid video file: {video_path}")

        n_frames = self._raw_n_frames(record)
        if n_frames <= 0:
            n_frames = self._probe_video_length(video_path)
        source_fps = self._source_fps_from_metadata(record.get("fps", None))
        trim_start = self._optional_int(record.get("trim_start"), default=0)
        trim_end = self._optional_int(record.get("trim_end"), default=n_frames)
        trim_start = max(0, min(trim_start, n_frames))
        trim_end = max(trim_start, min(trim_end, n_frames))
        sampled = self._temporal_sample(trim_end - trim_start, source_fps) + trim_start

        caption = self._caption_from_record(record)
        return {
            "video_paths": video_path,
            "video_pts": torch.as_tensor(sampled, dtype=torch.long),
            "video_fps": self.target_fps,
            "source_fps": source_fps,
            "caption": caption,
            "gemini_caption": self.normalize_caption_value(record.get("gemini_caption", "")),
            "original_caption": str(record.get("original_caption", "")),
            "has_caption": caption != "",
            "prompt_embed_path": str(record.get("prompt_embed_path", "")),
            "width": self._optional_int(record.get("width"), default=0),
            "height": self._optional_int(record.get("height"), default=0),
            "n_frames": n_frames,
        }

    def _record_passes_filter(self, record: Dict[str, Any]) -> bool:
        if bool(getattr(self.cfg.filtering, "has_caption", False)):
            if not self._caption_from_record(record):
                return False

        effective_n_frames = self._record_n_frames(record)
        if effective_n_frames <= 0:
            return False
        height = self._optional_int(record.get("height"), default=0)
        width = self._optional_int(record.get("width"), default=0)
        source_fps = self._source_fps_from_metadata(record.get("fps", None))

        if not self._in_range(height, getattr(self.cfg.filtering, "height", None)):
            return False
        if not self._in_range(width, getattr(self.cfg.filtering, "width", None)):
            return False
        if not self._in_range(source_fps, getattr(self.cfg.filtering, "fps", None)):
            return False
        if not self._in_range(effective_n_frames, getattr(self.cfg.filtering, "n_frames", None)):
            return False

        required_source_frames = self._n_frames_in_src(source_fps)
        if effective_n_frames < required_source_frames and self.pad_mode == "discard":
            return False
        return True

    def _record_n_frames(self, record: Dict[str, Any]) -> int:
        if self._has_value(record.get("trim_start")) and self._has_value(record.get("trim_end")):
            start = self._optional_int(record.get("trim_start"), default=0)
            end = self._optional_int(record.get("trim_end"), default=start)
            return max(0, end - start)
        return self._raw_n_frames(record)

    def _raw_n_frames(self, record: Dict[str, Any]) -> int:
        return self._optional_int(record.get("n_frames"), default=0)

    def _probe_video_length(self, video_path: Path) -> int:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        return len(vr)

    def _caption_from_record(self, record: Dict[str, Any]) -> str:
        for key in ("gemini_caption", "caption", "original_caption"):
            value = record.get(key, "")
            caption = self.normalize_caption_value(value)
            if caption:
                return caption
        return ""

    @staticmethod
    def normalize_caption_value(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split())

    def _resolve_record_video_path(self, record: Dict[str, Any]) -> Path:
        raw_path = str(record["video_path"])
        path = Path(raw_path)
        if path.is_absolute():
            return path

        rel_path = self.normalize_video_path(raw_path)
        candidates = [
            self.save_dir / rel_path,
            self.save_dir / self._unsanitize_episode_timestamp(rel_path),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1]

    @classmethod
    def normalize_video_path(cls, value: Any) -> Path:
        """
        LVP processing uses droid_raw as the root for the data, while 
        we assume data lives under data/droid. This unifies child paths.
        """
        path = str(value).replace("\\", "/").strip()
        for prefix in ("droid_raw/", "data/droid/"):
            if path.startswith(prefix):
                path = path[len(prefix) :]
                break
        return Path(path)

    @staticmethod
    def _unsanitize_episode_timestamp(path: Path) -> Path:
        parts = list(path.parts)
        for idx, part in enumerate(parts):
            if "_" not in part or ":" in part:
                continue
            pieces = part.rsplit("_", 3)
            if len(pieces) != 4:
                continue
            prefix, minute, second, year = pieces
            if (
                minute.isdigit()
                and len(minute) == 2
                and second.isdigit()
                and len(second) == 2
                and year.isdigit()
                and len(year) == 4
            ):
                parts[idx] = f"{prefix}:{minute}:{second}_{year}"
        return Path(*parts)

    def _source_fps_from_metadata(self, value: Any) -> float:
        if self.override_fps is not None:
            return float(self.override_fps)
        if value is None or value == "":
            return self.target_fps
        return float(value)

    def _n_frames_in_src(self, source_fps: float) -> int:
        return round(self.target_n_frames / self.target_fps * source_fps)

    def _temporal_sample(self, n_frames: int, source_fps: float) -> np.ndarray:
        target_len = self._n_frames_in_src(source_fps)
        if n_frames <= 0:
            raise ValueError("Cannot sample from a zero-frame Droid video.")

        if n_frames < target_len:
            if self.pad_mode == "pad_last":
                indices = np.linspace(0, target_len - 1, self.target_n_frames)
                indices = np.clip(indices, 0, n_frames - 1)
            elif self.pad_mode == "slowdown":
                indices = np.linspace(0, n_frames - 1, self.target_n_frames)
            else:
                raise ValueError("Short Droid video was not filtered out.")
        elif n_frames > target_len:
            if self.trim_mode == "random_cut":
                start = np.random.randint(0, n_frames - target_len + 1)
                indices = start + np.linspace(0, target_len - 1, self.target_n_frames)
            else:
                indices = np.linspace(0, n_frames - 1, self.target_n_frames)
        else:
            indices = np.linspace(0, n_frames - 1, self.target_n_frames)
        return np.round(indices).astype(int)

    @staticmethod
    def _in_range(value: float, value_range: Any) -> bool:
        if value_range is None:
            return True
        return value_range[0] <= value <= value_range[1]

    @staticmethod
    def _optional_int(value: Any, default: int) -> int:
        if value is None or value == "":
            return default
        return int(float(value))

    @staticmethod
    def _has_value(value: Any) -> bool:
        return value is not None and value != ""

    @staticmethod
    def _optional_bool(value: Any, default: bool) -> bool:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("1", "true", "yes", "y")

    def load_cond(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        if end_frame is None:
            end_frame = self.video_length(video_metadata)
        return torch.empty((max(0, end_frame - start_frame), 0), dtype=torch.float32)

    @override
    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        if end_frame is None:
            end_frame = self.video_length(video_metadata)
        video_path = video_metadata["video_paths"]
        video_pts = video_metadata["video_pts"]

        indices = torch.as_tensor(video_pts[start_frame:end_frame], dtype=torch.long)
        vr = VideoReader(str(video_path), ctx=cpu(0))
        if indices.numel() > 0:
            indices = indices.clamp_(0, max(len(vr) - 1, 0))
        video = torch.as_tensor(vr.get_batch(indices.tolist()).asnumpy())
        return video.permute(0, 3, 1, 2) / 255.0

    def _load_prompt_embed(
        self, video_metadata: Dict[str, Any]
    ) -> Tuple[torch.Tensor, int]:
        prompt_embed_path = str(video_metadata.get("prompt_embed_path", ""))
        if not prompt_embed_path:
            raise ValueError("Droid record missing required key `prompt_embed_path`.")
        path = Path(prompt_embed_path)
        if not path.is_absolute():
            path = self.save_dir / path
        prompt_embed = self._load_single_prompt_embed(path)
        prompt_embed_len = int(prompt_embed.size(0))
        if prompt_embed_len < self.max_text_tokens:
            padding = torch.zeros(
                self.max_text_tokens - prompt_embed_len,
                prompt_embed.size(1),
                dtype=prompt_embed.dtype,
                device=prompt_embed.device,
            )
            prompt_embed = torch.cat([prompt_embed, padding], dim=0)
        elif prompt_embed_len > self.max_text_tokens:
            prompt_embed = prompt_embed[: self.max_text_tokens]
            prompt_embed_len = self.max_text_tokens
        return prompt_embed, prompt_embed_len

    @staticmethod
    def _load_single_prompt_embed(path: Path) -> torch.Tensor:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            for key in ("prompt_embeds", "prompt_embed", "context", "embedding", "embeds"):
                if key in obj:
                    obj = obj[key]
                    break
            else:
                raise ValueError(f"No prompt embedding tensor found in {path}.")
        if isinstance(obj, (list, tuple)):
            if len(obj) != 1:
                raise ValueError(f"Expected one prompt embedding in {path}, got {len(obj)}.")
            obj = obj[0]
        if not torch.is_tensor(obj):
            raise TypeError(f"Expected prompt embedding tensor in {path}, got {type(obj)}.")
        if obj.ndim == 3 and obj.shape[0] == 1:
            obj = obj[0]
        if obj.ndim != 2:
            raise ValueError(f"Expected prompt embedding shape (L, D), got {tuple(obj.shape)}.")
        return obj.detach().cpu()

    @override
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        output = super().__getitem__(idx)
        if self.is_latent_preprocessing_expt:
            video_idx, clip_idx = idx, 0
        else:
            video_idx, clip_idx = self.get_clip_location(idx)

        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)
        end_frame = min(clip_idx + self.n_frames, video_length)
        caption = str(video_metadata.get("caption", ""))
        output["metadata"] = {
            "path": str(video_metadata["video_paths"]),
            "clip": [clip_idx, end_frame],
            "caption": caption,
            "gemini_caption": str(video_metadata.get("gemini_caption", "")),
            "source_fps": float(video_metadata.get("source_fps", self.target_fps)),
        }
        output["prompts"] = self.id_token + caption

        if self.load_prompt_embed:
            prompt_embeds, prompt_embed_len = self._load_prompt_embed(video_metadata)
            output["prompt_embeds"] = prompt_embeds
            output["prompt_embed_len"] = prompt_embed_len

        return output
