from typing import Literal, Optional
from typing import Dict
from torch import Tensor
from algorithms.common.metrics.video import VideoMetric, SharedVideoMetricModelRegistry
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan, bold_red
from utils.torch_utils import freeze_model
import torch

class MetricMixin:
    def _make_video_metric(
        self,
        registry: SharedVideoMetricModelRegistry,
        metric_types: Optional[list] = None,
    ) -> VideoMetric:
        fvd_over_time_cfg = self.logging_cfg.get("fvd_over_time", None)
        if fvd_over_time_cfg is not None:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(fvd_over_time_cfg):
                fvd_over_time_cfg = OmegaConf.to_container(fvd_over_time_cfg, resolve=True)
        return VideoMetric(
            registry,
            metric_types if metric_types is not None else self._metric_types,
            resolution=(self.cfg.x_shape[1], self.cfg.x_shape[2]),
            split_batch_size=self.logging_cfg.metrics_batch_size,
            frame_split_batch_size=(
                self.logging_cfg.metrics_frame_batch_size
                if self.logging_cfg.metrics_frame_batch_size is not None
                else self.logging_cfg.metrics_batch_size
            ),
            vbench_frame_split_batch_size=self.logging_cfg.get(
                "vbench_metrics_frame_batch_size", None
            ),
            fvd_over_time_cfg=fvd_over_time_cfg,
        )

    def _has_multiple_validation_dataloaders(self) -> bool:
        trainer = getattr(self, "trainer", None)
        val_dataloaders = getattr(trainer, "val_dataloaders", None)
        return val_dataloaders is not None and len(val_dataloaders) > 1

    def _get_or_create_loader_metric(
        self,
        task: Literal["prediction", "reconstruction"],
        dataloader_idx: int,
    ) -> Optional[VideoMetric]:
        if not self._has_multiple_validation_dataloaders():
            return getattr(self, f"metrics_{task}", None)

        attr_name = f"metrics_{task}_by_loader"
        if not hasattr(self, attr_name):
            setattr(self, attr_name, torch.nn.ModuleDict())
        metric_store = getattr(self, attr_name)
        key = str(dataloader_idx)
        if key not in metric_store:
            # FVD-over-time is only meaningful for prediction.
            metric_types = (
                self._metric_types
                if task == "prediction"
                else [m for m in self._metric_types if m != "fvd_over_time"]
            )
            metric_store[key] = self._make_video_metric(
                self._metric_registry, metric_types=metric_types
            )
            metric_store[key] = metric_store[key].to(self.device)
            freeze_model(metric_store[key])
        return metric_store[key]

    def _build_metrics(self) -> None:
        """
        Build the metrics.
        """
        
        # Metrics 
        if len(self.tasks) == 0:
            return
        registry = SharedVideoMetricModelRegistry()
        metric_types = list(self.logging_cfg.metrics)
        if self.cfg.training.strategy == "fsdp":
            rank_zero_print(bold_red("Using FSDP, metrics like FVD, IS, REAL IS, FVD_OVER_TIME is not compatible, because they use torch_script."))
            for fsdp_skip in ("fvd", "is", "real_is", "fvd_over_time"):
                if fsdp_skip in metric_types:
                    metric_types.remove(fsdp_skip)

        self._metric_registry = registry
        self._metric_types = metric_types

        # FVD-over-time is only meaningful for prediction/rollouts; strip it
        # from other tasks (e.g. reconstruction) where it would be redundant.
        recon_metric_types = [m for m in metric_types if m != "fvd_over_time"]

        for task in self.tasks:
            match task:
                case "prediction":
                    self.metrics_prediction = self._make_video_metric(registry)
                    freeze_model(self.metrics_prediction)
                case "reconstruction":
                    if not self.training:
                        self.tasks.remove("reconstruction")
                        rank_zero_print(cyan("Reconstruction is not supported during training, removing it from the tasks."))
                    else:
                        self.metrics_reconstruction = self._make_video_metric(
                            registry, metric_types=recon_metric_types
                        )
                        freeze_model(self.metrics_reconstruction)
    

    def _metrics(
        self,
        task: Literal["prediction", "reconstruction"],
        dataloader_idx: Optional[int] = None,
    ) -> Optional[VideoMetric]:
        """
        Get the appropriate metrics object for the given task.
        """
        if dataloader_idx is not None:
            return self._get_or_create_loader_metric(task, dataloader_idx)
        return getattr(self, f"metrics_{task}", None)


    def _update_metrics(self, all_videos: Dict[str, Tensor], dataloader_idx: int = 0, verbose: bool = False) -> None:
        """Update metrics for the specific dataloader during validation/test step."""
        if (
            self.logging_cfg.n_metrics_frames is not None
        ):  # only consider the first n_metrics_frames for evaluation
            all_videos = {
                k: v[:, : self.logging_cfg.n_metrics_frames] for k, v in all_videos.items()
            }

        gt_videos = all_videos["gt"]
        for task in self.tasks:
            if task in all_videos:
                metric = self._metrics(task, dataloader_idx=dataloader_idx)
                videos = all_videos[task]
            else:
                if verbose:
                    rank_zero_print(cyan(f"{task} is not carried in this run, check whether this is expected behavior!"))
                continue
            context_mask = torch.zeros(videos.shape[1], dtype=torch.bool, device=videos.device)
            match task:
                case "prediction":
                    context_mask = self.get_prediction_context_mask(
                        videos.shape[1],
                        device=videos.device,
                    )
                case "reconstruction":
                    context_mask = context_mask
            if self.logging_cfg.n_metrics_frames is not None:
                context_mask = context_mask[: self.logging_cfg.n_metrics_frames]
            metric(videos, gt_videos, context_mask=context_mask)
  
