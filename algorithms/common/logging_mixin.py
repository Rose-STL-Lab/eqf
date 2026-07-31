from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from einops import reduce
from utils.distributed_utils import is_rank_zero, get_rank
from utils.logging_utils import log_video, log_video_as_images
from utils.video_utils import write_video_to_file


class LoggingMixin:
    def _get_task_logged_video_count(self, dataloader_idx: int, task: str) -> int:
        if not hasattr(self, "_num_logged_videos_by_task"):
            self._num_logged_videos_by_task = {}
        return int(self._num_logged_videos_by_task.get((dataloader_idx, task), 0))

    def _increment_task_logged_video_count(
        self, dataloader_idx: int, task: str, amount: int
    ) -> None:
        if not hasattr(self, "_num_logged_videos_by_task"):
            self._num_logged_videos_by_task = {}
        key = (dataloader_idx, task)
        self._num_logged_videos_by_task[key] = (
            self._num_logged_videos_by_task.get(key, 0) + int(amount)
        )

    def _bucketed_loss(
        self,
        k: Tensor,
        loss_per_token: Tensor,
        masks: Optional[Tensor],
        num_buckets: int,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[list[str]]]:
        """
        Bucket per-token loss by continuous noise level k in [0, 1]
        Returns (bucket_means, bucket_counts), each shape [num_buckets].
        """
        if num_buckets <= 0:
            return None, None, None
        if k is None or loss_per_token is None:
            return None, None, None

        k = k.clamp(0, 1)
        if masks is None:
            valid = torch.ones_like(loss_per_token, dtype=torch.bool)
        else:
            valid = reduce(masks.bool(), "b t ... -> b t", torch.any)

        bucket_idx = torch.floor(k * num_buckets).long().clamp(0, num_buckets - 1)
        flat_idx = bucket_idx[valid]
        flat_loss = loss_per_token[valid]

        bucket_sum = torch.zeros(
            num_buckets, device=loss_per_token.device, dtype=loss_per_token.dtype
        )
        bucket_count = torch.zeros(
            num_buckets, device=loss_per_token.device, dtype=loss_per_token.dtype
        )
        bucket_sum.scatter_add_(0, flat_idx, flat_loss)
        bucket_count.scatter_add_(
            0, flat_idx, torch.ones_like(flat_loss, dtype=loss_per_token.dtype)
        )
        bucket_means = bucket_sum / bucket_count.clamp_min(1)
        bucket_labels = [
            f"[{i / num_buckets:.2f}, {(i + 1) / num_buckets:.2f}]"
            for i in range(num_buckets)
        ]
        return bucket_means, bucket_count, bucket_labels

    def _update_bucketed_loss_metrics(
        self,
        metrics_dict: Dict[str, Tensor],
        namespace: str,
        k: Optional[Tensor],
        loss_per_token: Optional[Tensor],
        masks: Optional[Tensor],
        metric_name: str = "x_loss_bucket",
        continuous_only: bool = True,
    ) -> None:
        """
        Assumes we have precaculated the loss (designed to be used immendiately after the training forward pass)
        """

        num_buckets = getattr(self.logging_cfg, "loss_bucket_count", 0)
        if num_buckets <= 0 or k is None or loss_per_token is None:
            return
        if continuous_only and not getattr(self, "use_continuous_timesteps", False):
            return

        bucket_means, _, bucket_labels = self._bucketed_loss(
            k=k,
            loss_per_token=loss_per_token,
            masks=masks,
            num_buckets=num_buckets,
        )
        if bucket_means is None or bucket_labels is None:
            return

        metrics_dict.update(
            {
                f"{namespace}/{metric_name}_{bucket_labels[i]}": bucket_means[i]
                for i in range(bucket_means.shape[0])
            }
        )

    def _log_videos(self, all_videos: Dict[str, Tensor], namespace: str, dataloader_idx: int = 0, video_metadata: Optional[Dict[str, Tensor]] = None) -> None:
        """Log videos during validation/test step."""
        rank = get_rank()

        if self.trainer.sanity_checking and (not self.logging_cfg.sanity_generation):
            return

        should_log_to_logger = bool(self.logger) and is_rank_zero
        should_save_raw = self.logging_cfg.raw_dir is not None
        wandb_step = self._manual_wandb_step() if should_log_to_logger else None

        # Nothing to log on this rank
        if not should_log_to_logger and not should_save_raw:
            return

        batch_size, n_frames = all_videos["gt"].shape[:2]
        
        # Also handle auxiliary videos if present
        extra_video_keys = []
        if "latent_map_mag" in all_videos:
            extra_video_keys.append("latent_map_mag")

        for task in list(self.tasks) + extra_video_keys:
            task_logged_videos = self._get_task_logged_video_count(dataloader_idx, task)
            if task_logged_videos >= self.logging_cfg.max_num_videos:
                continue
            num_videos_to_log = min(
                self.logging_cfg.max_num_videos - task_logged_videos,
                batch_size,
            )
            cut_videos = lambda x: x[:num_videos_to_log]
            if task not in all_videos:
                # rank_zero_print(cyan(f"{task} is not carried in this run, check whether this is expected behavior!"))
                continue
            if task == "prediction":
                context_frames = self.get_prediction_context_indices(
                    n_frames,
                    device=all_videos[task].device,
                )
            elif task == "reconstruction":
                context_frames = 0 # reconstruction in diffusion won't have context frames
            elif task == "latent_map_mag":
                # already aligned with prediction concat; no special context overlay in visualization
                context_frames = 0
            else:
                raise ValueError(f"Invalid task: {task}")

            raw_dir = self.logging_cfg.raw_dir if task == "prediction" else None # Only log raw if doing prediction, not reconstruction
            log_images_this_task = (
                self.logging_cfg.log_video_as_images.enable
                and (should_log_to_logger or raw_dir is not None)
            )

            # Skip if this rank has nothing to do for this task
            if not should_log_to_logger and raw_dir is None and not log_images_this_task:
                continue

            log_video(
                cut_videos(all_videos[task].clone()), # to avoid modifying the original tensor
                cut_videos(all_videos["gt"].clone()) if task != "latent_map_mag" else None,
                step=wandb_step,
                namespace=f"{task}_vis_{namespace}_loader_id={dataloader_idx}",
                logger=self.logger.experiment if should_log_to_logger else None,
                indent=task_logged_videos,
                raw_dir=raw_dir,
                context_frames=context_frames,
                captions=f"{task} | gt" if task != "latent_map_mag" else f"{task}",
                video_metadata=video_metadata,
                rank=rank,
                log_to_logger=should_log_to_logger,
            )

            if log_images_this_task:
                video_format = self.logging_cfg.log_video_as_images.format # png, pdf
                raw_indices = self.logging_cfg.log_video_as_images.indices
                if raw_indices is None:
                    raise ValueError("indices is required when log_video_as_images is enabled")
                indices = list(raw_indices)
                log_video_as_images(
                    cut_videos(all_videos[task].clone()), 
                    cut_videos(all_videos["gt"].clone()) if task != "latent_map_mag" else None,
                    step=wandb_step,
                    namespace=f"{task}_vis_as_images_{namespace}_loader_id={dataloader_idx}",
                    logger=self.logger.experiment if should_log_to_logger else None,
                    indent=task_logged_videos,
                    indices=indices, 
                    format=video_format,
                    raw_dir=raw_dir,
                    video_metadata=video_metadata,
                    rank=rank,
                    log_to_logger=should_log_to_logger,
                )
            self._increment_task_logged_video_count(dataloader_idx, task, batch_size)

        self.num_logged_videos[dataloader_idx] += batch_size
    

    def _log_videos_for_validation_only(self, all_videos: Dict[str, Tensor], namespace: str, dataloader_idx: int = 0) -> None:
        """
        We want to use multiple devices to run extremely long videos, so it's not good practice to use gather_data here, in this case. Therefore, we want each device to log their own videos.
        """
        rank = get_rank()
        should_log_to_logger = bool(self.logger) and is_rank_zero
        wandb_step = self._manual_wandb_step() if should_log_to_logger else None

        too_large_to_gather = False
        try:
            gathered_videos = self.gather_data(all_videos)
            all_videos = gathered_videos
        except:
            print("gather_data failed")
            too_large_to_gather = True
            torch.cuda.empty_cache()

        batch_size, n_frames = all_videos["gt"].shape[:2]

        for task in self.tasks:
            task_logged_videos = self._get_task_logged_video_count(dataloader_idx, task)
            if task_logged_videos >= self.logging_cfg.max_num_videos:
                continue
            if task not in all_videos:
                # rank_zero_print(cyan(f"{task} is not carried in this run, check whether this is expected behavior!"))
                continue
            if task == "prediction":
                context_frames = self.get_prediction_context_indices(
                    n_frames,
                    device=all_videos[task].device,
                )
            else:
                context_frames = torch.tensor(
                    [0, n_frames - 1], device=self.device, dtype=torch.long
                )
            
            num_videos_to_log = min(
                self.logging_cfg.max_num_videos - task_logged_videos,
                batch_size,
            )

            verbose_namespace = f"{task}_vis_{namespace}_loader_id={dataloader_idx}"
            cut_videos = lambda x: x[:num_videos_to_log]

            raw_dir = self.logging_cfg.raw_dir if not too_large_to_gather else f"{self.logger.save_dir}/{task}_vis_{namespace}_loader_id={dataloader_idx}/device_id={self.trainer.local_rank}"
            if too_large_to_gather:
                log_video(
                    cut_videos(all_videos[task].clone()),
                    cut_videos(all_videos["gt"].clone()),
                    step=wandb_step,
                    namespace=verbose_namespace,
                    logger=self.logger.experiment if should_log_to_logger else None,
                    indent=task_logged_videos,
                    raw_dir=raw_dir,
                    context_frames=context_frames,
                    captions=f"{task} | gt",
                    rank=rank,
                    log_to_logger=should_log_to_logger,
                )
            else:
                if should_log_to_logger:
                    log_video(
                        cut_videos(all_videos[task].clone()),
                        cut_videos(all_videos["gt"].clone()),
                        step=wandb_step,
                        namespace=verbose_namespace,
                        logger=self.logger.experiment,
                        indent=task_logged_videos,
                        raw_dir=raw_dir,
                        context_frames=context_frames,
                        captions=f"{task} | gt",
                        rank=rank,
                        log_to_logger=should_log_to_logger,
                    )
            self._increment_task_logged_video_count(dataloader_idx, task, batch_size)


    def _temp_save_videos(self, latents: Tensor, path: str):
        """
        Temp save the videos to the given path.
        """
        videos = self._decode(latents)
        write_video_to_file(videos[0], path)
