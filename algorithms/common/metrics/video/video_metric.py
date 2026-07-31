from typing import Any, Dict, List, Optional, Set, Tuple, Iterable, Union, cast
import torch
from torch import Tensor
from torch import nn
from einops import rearrange, repeat
from torchmetrics import MeanSquaredError
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)
from .fid import FrechetInceptionDistance
from .fvd import FrechetVideoDistance
from .fvd_over_time import FVDOverTime
from .fvmd import FrechetVideoMotionDistance
from .inception_score import InceptionScore
from .lpips import LearnedPerceptualImagePatchSimilarity
from .masked_mse import MaskedMeanSquaredError
from .vbench import VBench
from .types import VideoMetricType, VideoMetricModelType, EXCLUDED_VIDEO_METRICS_FROM_WANDB_LOG
from .shared_registry import SharedVideoMetricModelRegistry
from .per_frame_mse import PerFrameMSE


def filter_excluded_video_metrics(
        metrics_dict: dict[str, object],
        excluded: set[str] = EXCLUDED_VIDEO_METRICS_FROM_WANDB_LOG,
    ) -> tuple[dict[str, object], dict[str, object]]:
    """
    Split metrics dict into (to_log, to_excluded) based on excluded set.
    Excluded set contains bare metric names like 'per_frame_mse'.
    """
    to_log, to_excluded = {}, {}
    for k, v in metrics_dict.items():
        metric_name = k.split("/", 1)[1] if "/" in k else k
        if metric_name in excluded:
            to_excluded[k] = v
        else:
            to_log[k] = v
    return to_log, to_excluded


class VideoMetric(nn.Module):
    """
    A class that wraps all video metrics.

    Args:
        registry: A registry of models used for computing video metrics. When multiple `VideoMetric` instances are created to evaluate multiple models or tasks,
            the same registry shall be passed to all instances to avoid redundant model loading and save GPU memory.
        metric_types: List of video metric types. For supported video metric types, see `VideoMetricType`.
    """

    # Evaluated for the entire video
    VIDEO_WISE_METRICS = {
        VideoMetricType.FVD,
        VideoMetricType.FVD_OVER_TIME,
        VideoMetricType.IS,
        VideoMetricType.REAL_IS,
        VideoMetricType.FVMD,
        VideoMetricType.VBENCH,
        VideoMetricType.REAL_VBENCH,
        VideoMetricType.PER_FRAME_MSE,
    }

    # Evaluated for the entire video, evaluated using "shared" I3D features
    I3D_DEPENDENT_METRICS = {
        VideoMetricType.FVD,
        VideoMetricType.IS,
        VideoMetricType.REAL_IS,
    }

    # Returns a dict from compute() (curve + scalar summaries), handled
    # specially in `log()` like VBench but also exposes a curve to the caller.
    CURVE_METRICS = {
        VideoMetricType.FVD_OVER_TIME,
    }

    # VBench metrics, compute() returns a dictionary of dimension scores and a final VBench score
    VBENCH_METRICS = {VideoMetricType.VBENCH, VideoMetricType.REAL_VBENCH}

    # Evaluated for each "non-context" frame of the video
    FRAME_WISE_METRICS = {
        VideoMetricType.LPIPS,
        VideoMetricType.FID,
        VideoMetricType.MSE,
        VideoMetricType.MASKED_MSE,
        VideoMetricType.SSIM,
        VideoMetricType.PSNR,
    }

    def __init__(
        self,
        registry: SharedVideoMetricModelRegistry,
        metric_types: List[str] | List[VideoMetricType],
        split_batch_size: int = 16,
        frame_split_batch_size: Optional[int] = None,
        vbench_frame_split_batch_size: Optional[int] = None,
        resolution: Union[int, Tuple[int, int]] = (224, 224), # (h, w)
        fvd_over_time_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        modules = {}
        metric_types = [VideoMetricType(metric_type) for metric_type in metric_types]
        fvd_over_time_cfg = dict(fvd_over_time_cfg or {})
        if vbench_frame_split_batch_size is not None:
            vbench_frame_split_batch_size = int(vbench_frame_split_batch_size)
            if vbench_frame_split_batch_size <= 0:
                raise ValueError(
                    "vbench_frame_split_batch_size must be positive or None, "
                    f"got {vbench_frame_split_batch_size}"
                )
        for metric_type in metric_types:
            match metric_type:
                case VideoMetricType.LPIPS:
                    module = LearnedPerceptualImagePatchSimilarity(
                        registry=registry, normalize=True
                    )
                case VideoMetricType.FID:
                    module = FrechetInceptionDistance(registry=registry, normalize=True)
                case VideoMetricType.FVMD:
                    module = FrechetVideoMotionDistance(registry=registry)
                case VideoMetricType.VBENCH | VideoMetricType.REAL_VBENCH:
                    module = VBench(
                        registry=registry,
                        resolution=resolution,
                        vbench_metrics_frame_batch_size=vbench_frame_split_batch_size,
                    )
                case VideoMetricType.FVD:
                    module = FrechetVideoDistance()
                case VideoMetricType.FVD_OVER_TIME:
                    module = FVDOverTime(**fvd_over_time_cfg)
                case VideoMetricType.IS | VideoMetricType.REAL_IS:
                    module = InceptionScore()
                case VideoMetricType.MSE:
                    module = MeanSquaredError()
                case VideoMetricType.MASKED_MSE:
                    module = MaskedMeanSquaredError()
                case VideoMetricType.PER_FRAME_MSE:
                    module = PerFrameMSE()
                case VideoMetricType.SSIM:
                    module = StructuralSimilarityIndexMeasure(data_range=1.0)
                case VideoMetricType.PSNR:
                    module = PeakSignalNoiseRatio(data_range=1.0)
                case _:
                    raise ValueError(f"Unknown video metric type: {metric_type}")
            registry.register_for_metric(metric_type)
            modules[metric_type] = module

        self.metrics = nn.ModuleDict(modules)

        self.registry = registry
        self.split_batch_size = int(split_batch_size)
        if frame_split_batch_size is None:
            self.frame_split_batch_size = self.split_batch_size
        else:
            self.frame_split_batch_size = int(frame_split_batch_size)

    def keys(self) -> Iterable[VideoMetricType]:
        return self.metrics.keys()

    def items(self) -> Iterable[Tuple[VideoMetricType, nn.Module]]:
        return self.metrics.items()

    def values(self) -> Iterable[nn.Module]:
        return self.metrics.values()

    def _filtered_items(
        self, metric_types: Set[VideoMetricType], not_in: bool = False
    ) -> Iterable[Tuple[VideoMetricType, nn.Module]]:
        if not_in:
            return filter(lambda x: x[0] not in metric_types, self.items())
        return filter(lambda x: x[0] in metric_types, self.items())

    def _extract_i3d_features(self, x: Tensor) -> Tensor:
        """
        Extract I3D features. Requires a batch of videos of shape (B, T, C, H, W) and range [0, 1].
        """
        # temporally pad both ends to be at least 9 frames
        if x.shape[1] < 9:
            pad = (10 - x.shape[1]) // 2
            x = torch.cat(
                [
                    repeat(x[:, 0:1], "b 1 c h w -> b t c h w", t=pad).clone(),
                    x,
                    repeat(x[:, -1:], "b 1 c h w -> b t c h w", t=pad),
                ],
                dim=1,
            )
        x = 2.0 * x - 1.0
        x = rearrange(torch.clamp(x, -1.0, 1.0), "b t c h w -> b c t h w").contiguous()
        # Run I3D outside autocast with explicit float32 inputs. Lightning wraps
        # validation_step in autocast under precision=bf16/16, which would otherwise
        # cause the I3D forward to emit bf16/fp16 features. That both (a) slightly
        # corrupts the feature statistics (sum/cov accumulated in float64 from
        # low-precision sources) and (b) trips the inherited torchmetrics FID
        # compute() that casts the final FVD scalar back to orig_dtype, producing
        # quantized outputs (e.g. multiples of 1/16 for bf16 values in [8, 16)).
        i3d = cast(nn.Module, self.registry[VideoMetricModelType.I3D]).to(
            device=x.device,
            dtype=torch.float32,
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            return i3d(
                x.float(),
                rescale=False,
                resize=True,
                return_features=True,
            )

    def forward(
        self,
        preds: Tensor,
        target: Tensor,
        context_mask: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        """
        Update video metrics with the given predictions and targets.
        Args:
            preds: Predictions of shape (B, T, C, H, W), [0, 1]
            target: Targets of shape (B, T, C, H, W), [0, 1]
            context_mask: A boolean tensor of shape (T,) indicating whether each frame is a context frame.
                1) Frame-wise metrics are computed only for non-context frames.
                2) Context frames of generated videos will be overwritten with the corresponding frames of the real videos.
            mask: Optional[Tensor]: A mask to calculate Masked MSE. If None, all values are considered. (B, T, H, W)
        """
        # Metrics can be created lazily during validation after Lightning has
        # already moved the parent module. Keep metric states and shared models
        # anchored to the incoming tensors before any feature extraction.
        self.to(device=preds.device)

        # NOTE: Due to the large memory consumption of video metrics (especially VBench and LPIPS),
        # we split the batch into smaller chunks (max size = split_batch_size) and update per chunk.
        preds_split = preds.split(self.split_batch_size, dim=0)
        target_split = target.split(self.split_batch_size, dim=0)
        if mask is not None:
            mask = mask.unsqueeze(2).repeat(1, 1, preds.shape[2], 1, 1)
            assert mask.shape == preds.shape, (
                "Shape of mask must be the same as preds. "
                f"Got mask shape: {mask.shape}, preds shape: {preds.shape}"
            )
            mask_split = mask.split(self.split_batch_size, dim=0)
        else:
            mask_split = None

        assert len(preds_split) == len(
            target_split
        ), "Batch size of preds and target must be the same."
        if mask_split is not None:
            assert len(preds_split) == len(
                mask_split
            ), "Batch size of preds and mask must be the same."
            assert preds_split.shape == mask_split.shape, (
                "Shape of preds and mask must be the same. "
                f"Got preds shape: {preds_split.shape}, mask shape: {mask_split.shape}"
            )
        
        if mask_split is not None:
            for preds_chunk, target_chunk, mask_chunk in zip(preds_split, target_split, mask_split):
                self._update(preds_chunk, target_chunk, context_mask, mask_chunk)
        else:
            for preds_chunk, target_chunk in zip(preds_split, target_split):
                self._update(preds_chunk, target_chunk, context_mask, None)

    def _update(
        self, preds: Tensor, target: Tensor, context_mask: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ):
        """
        Note:
            FVD, IS, REAL_IS: (B, C) (I3D features)
            FVMD: (B, T, C, H, W), [0, 1]
            FID, LPIPS, MSE, MASKED_MSE, SSIM, PSNR: (B, C, H, W), [0, 1]
        """
        if context_mask is None:
            context_mask = torch.zeros(
                preds.shape[1], device=preds.device, dtype=torch.bool
            )
        else:
            context_mask = context_mask.to(device=preds.device, dtype=torch.bool)
        target = target.to(device=preds.device)
        if mask is not None:
            mask = mask.to(device=preds.device)

        # replace all NaNs with 0 / clamp to [0, 1] / convert to float32
        preds, target = map(
            lambda x: torch.clamp(torch.nan_to_num(x, nan=0.0), 0.0, 1.0).to(
                torch.float32
            ),
            (preds, target),
        )
        # overwrite context frames of generated videos with the corresponding frames of the real videos
        preds = torch.where(
            rearrange(context_mask, "t -> 1 t 1 1 1"),
            target,
            preds,
        )
        # update I3D-dependent video-wise metrics
        i3d_dependent_metrics = self.I3D_DEPENDENT_METRICS.intersection(self.keys())
        if i3d_dependent_metrics:
            fake_features, real_features = None, None
            if {VideoMetricType.FVD, VideoMetricType.IS}.intersection(self.keys()):
                fake_features = self._extract_i3d_features(preds)
            if {VideoMetricType.FVD, VideoMetricType.REAL_IS}.intersection(self.keys()):
                real_features = self._extract_i3d_features(target)
            for metric_type, module in self._filtered_items(self.I3D_DEPENDENT_METRICS):
                if metric_type == VideoMetricType.FVD:
                    module.update(fake_features, real_features)
                else:
                    module.update(
                        fake_features
                        if metric_type == VideoMetricType.IS
                        else real_features
                    )

        # update FVD-over-time (per-window I3D features). Each window is
        # extracted from the full clip and passed through the same
        # _extract_i3d_features helper as FVD/IS, so preprocessing matches.
        if VideoMetricType.FVD_OVER_TIME in self.keys():
            fvdt: FVDOverTime = self.metrics[VideoMetricType.FVD_OVER_TIME]
            T_pred, T_targ = preds.shape[1], target.shape[1]
            if fvdt.reference_mode == "time_aligned":
                max_start = min(T_pred, T_targ) - fvdt.clip_len
            else:
                max_start = T_pred - fvdt.clip_len
            if max_start >= 0:
                gen_starts = list(range(0, max_start + 1, fvdt.gen_stride))
                if fvdt.reference_mode == "global_pooled":
                    ref_max_start = T_targ - fvdt.clip_len
                    ref_starts = (
                        list(range(0, ref_max_start + 1, fvdt.ref_stride))
                        if ref_max_start >= 0
                        else []
                    )
                else:  # time_aligned
                    ref_starts = gen_starts

                fake_feats: Dict[int, Tensor] = {}
                for t in gen_starts:
                    fake_feats[t] = self._extract_i3d_features(
                        preds[:, t : t + fvdt.clip_len]
                    )

                real_feats: Dict[Any, Tensor] = {}
                for s in ref_starts:
                    real_feats[s] = self._extract_i3d_features(
                        target[:, s : s + fvdt.clip_len]
                    )
                fvdt.update(fake_feats, real_feats)

        # update VBench metrics
        for metric_type, module in self._filtered_items(self.VBENCH_METRICS):
            module.update(preds if metric_type == VideoMetricType.VBENCH else target)

        # update other video-wise metrics (handled bespoke above:
        # I3D_DEPENDENT_METRICS, VBENCH_METRICS, FVD-over-time)
        fvd_over_time_set = {VideoMetricType.FVD_OVER_TIME}
        for metric_type, module in self._filtered_items(
            self.VIDEO_WISE_METRICS
            - self.I3D_DEPENDENT_METRICS
            - self.VBENCH_METRICS
            - fvd_over_time_set
        ):
            module.update(preds, target)

        # reshape a batch of videos to a batch of image frames
        preds_frames, target_frames = map(
            lambda x: rearrange(x[:, ~context_mask], "b t c h w -> (b t) c h w"),
            (preds, target),
        )
        mask_frames: Optional[Tensor] = None
        if mask is not None:
            # Mask is expected to align with (B, T, C, H, W) after the expansion in forward().
            mask_frames = rearrange(mask[:, ~context_mask], "b t c h w -> (b t) c h w")

        # update frame-wise metrics (cap frame-batch size to avoid OOM on long videos)
        pred_splits = preds_frames.split(self.frame_split_batch_size, dim=0)
        targ_splits = target_frames.split(self.frame_split_batch_size, dim=0)
        if mask_frames is not None:
            mask_splits = mask_frames.split(self.frame_split_batch_size, dim=0)
        else:
            mask_splits = None

        for metric_type, module in self._filtered_items(self.FRAME_WISE_METRICS):
            if metric_type == VideoMetricType.MASKED_MSE:
                for i, (p, t) in enumerate(zip(pred_splits, targ_splits)):
                    m = None if mask_splits is None else mask_splits[i]
                    module.update(p, t, mask=m)
            else:
                for p, t in zip(pred_splits, targ_splits):
                    module.update(p, t)

    def log(self, prefix: str):
        dict_metrics = {}
        # call compute() for VBench metrics and reorganize the results
        for metric_type, module in self._filtered_items(self.VBENCH_METRICS):
            output = module.compute()
            dict_metrics.update(
                {
                    f"{prefix}/{metric_type.value}/{key}": value
                    for key, value in output.items()
                }
            )
            # NOTE: manually calling compute() requires manual call to reset() afterwards
            # FIXME:if we need a functionality to log VBench several times within a single validation epoch,
            # we should move reset() to on_validation_epoch_end() in the Lightning module that uses this VideoMetric
            module.reset()

        # call compute() for curve metrics (e.g. FVD-over-time): expand scalar
        # summaries into log keys, and stash the full curve under a special key
        # so the caller can persist/plot it. The curve key matches the
        # excluded-from-wandb-log set, so it goes through the manual side path
        # (raw_dir JSON + wandb image plot) instead of log_dict.
        for metric_type, module in self._filtered_items(self.CURVE_METRICS):
            output = module.compute()
            scalars = output.get("scalars", {}) if isinstance(output, dict) else {}
            for key, value in scalars.items():
                dict_metrics[f"{prefix}/{metric_type.value}/{key}"] = value
            dict_metrics[f"{prefix}/{metric_type.value}"] = output
            module.reset()

        # other metrics (no need to call compute())
        dict_metrics.update(
            {
                f"{prefix}/{metric_type.value}": module
                for metric_type, module in self._filtered_items(
                    self.VBENCH_METRICS | self.CURVE_METRICS, not_in=True
                )
                if not (hasattr(module, "is_empty") and module.is_empty)
            }
        )

        return dict_metrics

    def reset(self):
        for module in self.values():
            module.reset()

if __name__ == "__main__":
    pass
