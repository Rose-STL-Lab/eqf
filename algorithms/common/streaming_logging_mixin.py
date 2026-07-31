from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import wandb

from algorithms.streaming_state import resolve_streaming_controls
from utils.distributed_utils import is_rank_zero
from utils.velocity_utils import (
    log_global_context_readout_noise_level_schedules,
    log_global_denoising_schedules,
    log_global_frame_nfe_bars,
    log_global_readout_noise_level_raw_schedules,
    log_global_readout_noise_level_schedules,
    log_global_velocity_schedules,
)


def _unpack_other_results_dict(other_results_by_task, name):
    """Helper to unpack other_results dict for the tasks."""
    global_values = {
        task: (results or {}).get(name)
        for task, results in (other_results_by_task or {}).items()
    }
    return {k: v for k, v in global_values.items() if v is not None}


def _unpack_other_results_scalar(other_results_by_task, name):
    """Helper to unpack scalar-like per-task values from other_results."""
    values = {}
    for task, results in (other_results_by_task or {}).items():
        if not isinstance(results, dict):
            continue
        value = results.get(name)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                continue
            value = value.detach().cpu().reshape(-1)[0].item()
        values[task] = int(value)
    return values


class StreamingLoggingMixin:
    _GLOBAL_GRID_KEYS = (
        "global_denoising_schedule",
        "global_readout_noise_level_schedule",
        "global_readout_noise_level_raw_schedule",
        "global_velocity_schedule",
        "global_context_readout_noise_level_schedule",
    )
    _GLOBAL_SCALAR_KEYS = ("bake_in_inner_steps", "denoising_depth", "total_nfes_allowed")

    def _calculate_total_nfes(self) -> Tuple[float, int]:
        """
        Compute canonical total allowed NFEs for logging, along with denoising depth.
            In fixed inference, this will output the deterministic number of NFEs
            In adaptive inference, this will output the maximum number of NFEs based on self.num_sampling_steps

        Returns:
            total_nfes_allowed: canonical inference budget including optional bake-in.
            num_sampling_steps: denoising depth D used to define the budget.
        """
        controls = resolve_streaming_controls(
            self.cfg.tasks.prediction.streaming,
            self.forward_window_size_in_tokens,
            self.num_sampling_steps,
            self.validation_n_sliding_context_tokens,
            self.validation_n_initial_context_tokens,
            self._n_frames_to_n_tokens(self.cfg.tasks.prediction.streaming.stride_in_frames),
        )

        num_sampling_steps = int(controls.num_sampling_steps)
        horizon_tokens = int(controls.validation_horizon_tokens)
        stride_tokens = int(controls.stride_in_tokens)
        total_generated_tokens = max(
            int(self._n_frames_to_n_tokens(int(self.cfg.n_frames)))
            - int(controls.initial_context_tokens),
            0,
        )

        total_nfes_allowed = float(num_sampling_steps * total_generated_tokens / horizon_tokens)

        if controls.should_bake_in:
            inner_steps = int(controls.resolved_inner_steps_per_emit)
            bake_in_rows = inner_steps * max((horizon_tokens // stride_tokens) - 1, 0)
            total_nfes_allowed += float(bake_in_rows)

        return float(total_nfes_allowed), num_sampling_steps

    def _init_global_grid_epoch_accumulator(self) -> None:
        self._global_grid_epoch_accumulator = {}

    def _max_grid_samples_to_keep(self) -> int:
        return max(1, int(self.logging_cfg.max_num_videos))

    def _accumulate_global_grids(
        self,
        *,
        other_results_by_task: Dict[str, Dict],
        dataloader_idx: int,
    ) -> None:
        """
        Accumulate self._global_grid_epoch_accumulator, an epoch-level nested dict keyed by dataloader idx and task.
        """
        if not hasattr(self, "_global_grid_epoch_accumulator"):
            self._init_global_grid_epoch_accumulator()

        max_keep = self._max_grid_samples_to_keep()
        dl_store = self._global_grid_epoch_accumulator.setdefault(dataloader_idx, {})

        for task, results in (other_results_by_task or {}).items():
            if not isinstance(results, dict):
                continue
            task_store = dl_store.setdefault(task, {})
            for key in self._GLOBAL_GRID_KEYS:
                value = results.get(key)
                if value is None or not isinstance(value, torch.Tensor) or value.ndim != 3:
                    continue
                value_cpu = value.detach().cpu()
                prev = task_store.get(key)
                if prev is None:
                    merged = value_cpu
                else:
                    merged = torch.cat([prev, value_cpu], dim=0)
                if int(merged.shape[0]) > max_keep:
                    merged = merged[:max_keep]
                task_store[key] = merged
            for key in self._GLOBAL_SCALAR_KEYS:
                value = results.get(key)
                if value is None:
                    continue
                if isinstance(value, torch.Tensor):
                    if value.numel() == 0:
                        continue
                    value = value.detach().cpu().reshape(-1)[0].item()
                task_store[key] = int(value)

    def _flush_global_grid_epoch_accumulator(self, *, namespace: str) -> None:
        accumulator = getattr(self, "_global_grid_epoch_accumulator", None)
        if not accumulator:
            return

        for dataloader_idx, other_results_by_task in accumulator.items():
            if not other_results_by_task:
                continue

            self._log_velocity_grids_for_wandb(
                other_results_by_task=other_results_by_task,
                namespace=namespace,
                dataloader_idx=int(dataloader_idx),
                num_logged_videos_override=0,
            )
            self._log_global_denoising_schedules(
                _unpack_other_results_dict(other_results_by_task, "global_denoising_schedule"),
                _unpack_other_results_scalar(other_results_by_task, "bake_in_inner_steps"),
                _unpack_other_results_scalar(other_results_by_task, "denoising_depth"),
                namespace=namespace,
                dataloader_idx=int(dataloader_idx),
                num_logged_videos_override=0,
            )
            self._log_global_readout_noise_level_schedules(
                _unpack_other_results_dict(
                    other_results_by_task, "global_readout_noise_level_schedule"
                ),
                namespace=namespace,
                dataloader_idx=int(dataloader_idx),
                num_logged_videos_override=0,
            )
            self._log_global_readout_noise_level_raw_schedules(
                _unpack_other_results_dict(
                    other_results_by_task, "global_readout_noise_level_raw_schedule"
                ),
                _unpack_other_results_dict(other_results_by_task, "global_denoising_schedule"),
                namespace=namespace,
                dataloader_idx=int(dataloader_idx),
                num_logged_videos_override=0,
            )
            self._log_global_velocity_schedules(
                _unpack_other_results_dict(other_results_by_task, "global_velocity_schedule"),
                namespace=namespace,
                dataloader_idx=int(dataloader_idx),
                num_logged_videos_override=0,
            )
            self._log_global_context_readout_noise_level_schedules(
                _unpack_other_results_dict(
                    other_results_by_task, "global_context_readout_noise_level_schedule"
                ),
                namespace=namespace,
                dataloader_idx=int(dataloader_idx),
                num_logged_videos_override=0,
            )

        self._init_global_grid_epoch_accumulator()

    def _should_log_velocity_grids_for_wandb(
        self,
        *,
        dataloader_idx: int,
        controls,
        num_logged_videos_override: Optional[int] = None,
    ) -> bool:
        """
        Only log on wandb if the S=H and fixed.
        """
        if controls is None:
            return False
        if not bool(getattr(self.logging_cfg, "log_global_velocity_schedule", False)):
            return False
        if str(getattr(controls, "mode", "")).lower() != "fixed":
            return False
        if int(getattr(controls, "stride_in_tokens", -1)) != int(
            getattr(controls, "validation_horizon_tokens", -2)
        ):
            return False
        if not self.logger or (not is_rank_zero):
            return False
        if self.trainer.sanity_checking and (not self.logging_cfg.sanity_generation):
            return False
        num_logged_videos = (
            int(self.num_logged_videos[dataloader_idx])
            if num_logged_videos_override is None
            else int(num_logged_videos_override)
        )
        if num_logged_videos >= int(self.logging_cfg.max_num_videos):
            return False
        return True

    def _extract_velocity_chunk_curves(
        self,
        *,
        vel_grid: Tensor,
        depth: int,
        horizon: int,
        frame_width: Optional[int] = None,
    ):
        """
        Take average over batch and chunks of frames.
        """
        vel = vel_grid.detach().cpu().float()
        if vel.numel() == 0:
            return None
        if frame_width is None:
            frame_width = horizon

        if vel.shape[1] > 1 and (int(vel.shape[1]) - 1) % depth == 0:
            vel = vel[:, 1:, :]
        if vel.shape[1] < depth:
            return None

        total_touches = int(vel.shape[1])
        generated_frames = int(vel.shape[2])
        num_chunks_touches = total_touches // depth
        num_chunks_frames = (generated_frames + horizon - 1) // horizon
        num_chunks = int(min(num_chunks_touches, num_chunks_frames))
        if num_chunks <= 0:
            return None

        vel = vel[:, : num_chunks * depth, :]
        chunk_curves = []
        chunk_payload = {}
        for ci in range(num_chunks):
            ts = ci * depth
            te = (ci + 1) * depth
            fs = ci * horizon
            fe = min(fs + int(frame_width), generated_frames)
            if fs >= fe:
                continue
            chunk_block = vel[:, ts:te, fs:fe]
            chunk_curve = torch.nanmean(chunk_block, dim=(0, 2))
            chunk_curves.append(chunk_curve)
            chunk_payload[f"chunk_{ci}"] = chunk_curve.tolist()

        if not chunk_curves:
            return None

        chunk_stack = torch.stack(chunk_curves, dim=0)
        chunk_mean = torch.nanmean(chunk_stack, dim=0)
        return chunk_stack, chunk_mean, chunk_payload

    def _log_velocity_grids_for_wandb(
        self,
        *,
        other_results_by_task: Dict[str, Dict],
        namespace: str,
        dataloader_idx: int,
        num_logged_videos_override: Optional[int] = None,
    ) -> None:
        controls = resolve_streaming_controls(
            self.cfg.tasks.prediction.streaming,
            self.forward_window_size_in_tokens,
            self.num_sampling_steps,
            self.validation_n_sliding_context_tokens,
            self.validation_n_initial_context_tokens,
            self._n_frames_to_n_tokens(
                self.cfg.tasks.prediction.streaming.stride_in_frames
            ),
        )

        if not self._should_log_velocity_grids_for_wandb(
            dataloader_idx=dataloader_idx,
            controls=controls,
            num_logged_videos_override=num_logged_videos_override,
        ):
            return

        if not hasattr(self, "velocity_grids_for_wandb"):
            self.velocity_grids_for_wandb = {}

        depth = int(max(int(getattr(controls, "num_sampling_steps", 1)), 1))
        horizon = int(max(int(getattr(controls, "validation_horizon_tokens", 1)), 1))
        use_log_y = bool(getattr(self.logging_cfg, "velocity_log_yscale", False))

        def _build_chunk_series(
            *,
            vel_grid: Optional[Tensor],
            frame_width: int,
            title: str,
            chart_key: str,
            payload_key: str,
        ):
            if vel_grid is None or not isinstance(vel_grid, torch.Tensor) or vel_grid.ndim != 3:
                return None

            curves = self._extract_velocity_chunk_curves(
                vel_grid=vel_grid,
                depth=depth,
                horizon=horizon,
                frame_width=frame_width,
            )
            if curves is None:
                return None
            chunk_stack, chunk_mean, chunk_payload = curves

            plot_chunk_stack = chunk_stack
            plot_chunk_mean = chunk_mean
            title_suffix = ""
            if use_log_y:
                min_positive = 1e-12
                plot_chunk_stack = torch.where(
                    torch.isfinite(plot_chunk_stack),
                    torch.clamp(plot_chunk_stack, min=min_positive),
                    plot_chunk_stack,
                )
                plot_chunk_mean = torch.where(
                    torch.isfinite(plot_chunk_mean),
                    torch.clamp(plot_chunk_mean, min=min_positive),
                    plot_chunk_mean,
                )
                plot_chunk_stack = torch.log10(plot_chunk_stack)
                plot_chunk_mean = torch.log10(plot_chunk_mean)
                title_suffix = " [log10]"

            xs = list(range(1, depth + 1))
            keys = [f"chunk_{i}" for i in range(int(chunk_stack.shape[0]))] + [
                "chunk_mean"
            ]
            ys = [plot_chunk_stack[i].tolist() for i in range(int(chunk_stack.shape[0]))] + [
                plot_chunk_mean.tolist()
            ]
            chart = wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=keys,
                title=f"{task} {title}{title_suffix}",
                xname="denoising_depth",
            )
            self._wandb_log(
                {
                    f"{payload_key}/{chart_key}": chart,
                },
                commit=False,
            )
            return {
                "denoising_depth": depth,
                "num_chunks": int(chunk_stack.shape[0]),
                "chunk_curves": chunk_payload,
                "chunk_mean": chunk_mean.tolist(),
                "use_log_y": use_log_y,
            }

        for task, results in (other_results_by_task or {}).items():
            if not isinstance(results, dict):
                continue

            payload_key = f"{task}_velocity_{namespace}_loader_id={dataloader_idx}"
            future_payload = _build_chunk_series(
                vel_grid=results.get("global_velocity_schedule", None),
                frame_width=horizon,
                title="chunkwise active horizon velocity vs denoising depth",
                chart_key="velocity_grids_for_wandb",
                payload_key=payload_key,
            )
            if future_payload is not None:
                self.velocity_grids_for_wandb[payload_key] = future_payload

    def _log_global_grids(
        self,
        other_results_by_task: Dict[str, Dict],
        namespace: str,
        dataloader_idx: int = 0,
        video_metadata: Optional[Dict[str, Tensor]] = None,
    ) -> None:
        _ = namespace
        _ = video_metadata
        self._accumulate_global_grids(
            other_results_by_task=other_results_by_task,
            dataloader_idx=dataloader_idx,
        )
        self._log_eta_multiplier_comparison(
            other_results_by_task=other_results_by_task,
            namespace=namespace,
            dataloader_idx=dataloader_idx,
        )

    def _log_eta_multiplier_comparison(
        self,
        *,
        other_results_by_task: Dict[str, Dict],
        namespace: str,
        dataloader_idx: int,
    ) -> None:
        if not self.logger or (not is_rank_zero):
            return
        if self.trainer.sanity_checking and (not self.logging_cfg.sanity_generation):
            return
        if int(self.num_logged_videos[dataloader_idx]) >= int(self.logging_cfg.max_num_videos):
            return
        import numpy as np
        import matplotlib.pyplot as plt

        for task, results in (other_results_by_task or {}).items():
            if not isinstance(results, dict):
                continue
            local_sum_actual = results.get("eta_multiplier_local_sum_actual", None)
            local_sum_expected = results.get("eta_multiplier_local_sum_expected", None)
            local_count = results.get("eta_multiplier_local_count", None)
            if local_sum_actual is None or local_sum_expected is None or local_count is None:
                continue
            if isinstance(local_sum_actual, torch.Tensor):
                local_sum_actual_np = local_sum_actual.detach().cpu().float().numpy()
            else:
                local_sum_actual_np = np.asarray(local_sum_actual, dtype=np.float32)
            if isinstance(local_sum_expected, torch.Tensor):
                local_sum_expected_np = local_sum_expected.detach().cpu().float().numpy()
            else:
                local_sum_expected_np = np.asarray(local_sum_expected, dtype=np.float32)
            if isinstance(local_count, torch.Tensor):
                local_count_np = local_count.detach().cpu().float().numpy()
            else:
                local_count_np = np.asarray(local_count, dtype=np.float32)
            n = int(
                min(
                    local_sum_actual_np.shape[0],
                    local_sum_expected_np.shape[0],
                    local_count_np.shape[0],
                )
            )
            if n <= 1:
                continue
            denom = np.maximum(local_count_np[:n], 1e-12)
            valid = local_count_np[:n] > 0
            actual_np = np.full((n,), np.nan, dtype=np.float32)
            expected_np = np.full((n,), np.nan, dtype=np.float32)
            actual_np[valid] = local_sum_actual_np[:n][valid] / denom[valid]
            expected_np[valid] = local_sum_expected_np[:n][valid] / denom[valid]
            x = np.arange(n, dtype=np.int32)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8.5, 3.8), constrained_layout=True)
            ax.plot(x, expected_np[:n], color="#1f77b4", linewidth=1.5, label="scheduled eta multiplier")
            ax.plot(x, actual_np[:n], color="#d62728", linewidth=1.5, label="readout-driven eta multiplier")
            ax.set_title(f"{task} | eta multiplier vs local denoising step ({namespace})", fontsize=11)
            ax.set_xlabel("local denoising step (frame-aligned)", fontsize=9)
            ax.set_ylabel("multiplier (active-token mean)", fontsize=9)
            ax.grid(True, alpha=0.20, linewidth=0.8)
            ax.legend(fontsize=8, loc="best")
            ax.tick_params(labelsize=8)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            self.log_image(
                key=f"{task}_eta_multiplier_{namespace}_loader_id={dataloader_idx}/eta_multiplier_comparison",
                image=img,
            )
            plt.close(fig)

    def _log_global_denoising_schedules(
        self,
        global_schedules: Optional[Dict[str, Optional[Tensor]]],
        bake_in_inner_steps_by_task: Optional[Dict[str, int]],
        denoising_depth_by_task: Optional[Dict[str, int]],
        namespace: str,
        dataloader_idx: int = 0,
        video_metadata: Optional[Dict[str, Tensor]] = None,
        num_logged_videos_override: Optional[int] = None,
    ) -> None:
        _ = video_metadata
        if not bool(getattr(self.logging_cfg, "log_global_denoising_schedule", False)):
            return
        num_logged_videos = (
            int(self.num_logged_videos[dataloader_idx])
            if num_logged_videos_override is None
            else int(num_logged_videos_override)
        )
        log_global_denoising_schedules(
            global_schedules,
            denoising_depth_by_task=denoising_depth_by_task,
            namespace=namespace,
            dataloader_idx=dataloader_idx,
            sanity_checking=bool(self.trainer.sanity_checking),
            sanity_generation=bool(self.logging_cfg.sanity_generation),
            max_num_videos=int(self.logging_cfg.max_num_videos),
            num_logged_videos=num_logged_videos,
            has_logger=self.logger is not None,
            is_rank_zero=bool(is_rank_zero),
            log_image=lambda k, img: self.log_image(key=k, image=img),
        )
        if bool(getattr(self.logging_cfg, "log_frame_nfe", False)):
            log_global_frame_nfe_bars(
                global_schedules,
                bake_in_inner_steps_by_task=bake_in_inner_steps_by_task,
                namespace=namespace,
                dataloader_idx=dataloader_idx,
                sanity_checking=bool(self.trainer.sanity_checking),
                sanity_generation=bool(self.logging_cfg.sanity_generation),
                max_num_videos=int(self.logging_cfg.max_num_videos),
                num_logged_videos=num_logged_videos,
                has_logger=self.logger is not None,
                is_rank_zero=bool(is_rank_zero),
                log_image=lambda k, img: self.log_image(key=k, image=img),
            )

    def _log_global_velocity_schedules(
        self,
        global_velocity_schedules: Optional[Dict[str, Optional[Tensor]]],
        namespace: str,
        dataloader_idx: int = 0,
        video_metadata: Optional[Dict[str, Tensor]] = None,
        num_logged_videos_override: Optional[int] = None,
    ) -> None:
        _ = video_metadata
        if not bool(getattr(self.logging_cfg, "log_global_velocity_schedule", False)):
            return
        num_logged_videos = (
            int(self.num_logged_videos[dataloader_idx])
            if num_logged_videos_override is None
            else int(num_logged_videos_override)
        )
        log_global_velocity_schedules(
            global_velocity_schedules,
            namespace=namespace,
            dataloader_idx=dataloader_idx,
            sanity_checking=bool(self.trainer.sanity_checking),
            sanity_generation=bool(self.logging_cfg.sanity_generation),
            max_num_videos=int(self.logging_cfg.max_num_videos),
            num_logged_videos=num_logged_videos,
            has_logger=self.logger is not None,
            is_rank_zero=bool(is_rank_zero),
            log_image=lambda k, img: self.log_image(key=k, image=img),
            velocity_log_yscale=bool(getattr(self.logging_cfg, "velocity_log_yscale", False)),
        )

    def _log_global_readout_noise_level_schedules(
        self,
        global_readout_noise_level_schedules: Optional[Dict[str, Optional[Tensor]]],
        namespace: str,
        dataloader_idx: int = 0,
        video_metadata: Optional[Dict[str, Tensor]] = None,
        num_logged_videos_override: Optional[int] = None,
    ) -> None:
        _ = video_metadata
        if not bool(getattr(self.logging_cfg, "log_global_readout_noise_level_schedule", False)):
            return
        num_logged_videos = (
            int(self.num_logged_videos[dataloader_idx])
            if num_logged_videos_override is None
            else int(num_logged_videos_override)
        )
        log_global_readout_noise_level_schedules(
            global_readout_noise_level_schedules,
            namespace=f"{namespace}_readout",
            dataloader_idx=dataloader_idx,
            sanity_checking=bool(self.trainer.sanity_checking),
            sanity_generation=bool(self.logging_cfg.sanity_generation),
            max_num_videos=int(self.logging_cfg.max_num_videos),
            num_logged_videos=num_logged_videos,
            has_logger=self.logger is not None,
            is_rank_zero=bool(is_rank_zero),
            log_image=lambda k, img: self.log_image(key=k, image=img),
        )

    def _log_global_readout_noise_level_raw_schedules(
        self,
        global_readout_noise_level_raw_schedules: Optional[Dict[str, Optional[Tensor]]],
        global_denoising_schedules: Optional[Dict[str, Optional[Tensor]]],
        namespace: str,
        dataloader_idx: int = 0,
        video_metadata: Optional[Dict[str, Tensor]] = None,
        num_logged_videos_override: Optional[int] = None,
    ) -> None:
        _ = video_metadata
        if not bool(getattr(self.logging_cfg, "log_global_readout_noise_level_schedule", False)):
            return
        num_logged_videos = (
            int(self.num_logged_videos[dataloader_idx])
            if num_logged_videos_override is None
            else int(num_logged_videos_override)
        )
        summaries = log_global_readout_noise_level_raw_schedules(
            global_readout_noise_level_raw_schedules,
            global_denoising_schedules,
            emit_noise_level=float(self.cfg.tasks.prediction.streaming.emit_noise_level),
            namespace=f"{namespace}_readout",
            dataloader_idx=dataloader_idx,
            sanity_checking=bool(self.trainer.sanity_checking),
            sanity_generation=bool(self.logging_cfg.sanity_generation),
            max_num_videos=int(self.logging_cfg.max_num_videos),
            num_logged_videos=num_logged_videos,
            has_logger=self.logger is not None,
            is_rank_zero=bool(is_rank_zero),
            log_image=lambda k, img: self.log_image(key=k, image=img),
        )
        for key, value in summaries.items():
            self.log(
                f"inference/{key}",
                torch.as_tensor(float(value), device=self.device, dtype=torch.float32),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                # `summaries` are produced only on rank zero because the image
                # helper returns early on nonzero ranks. Do not enter a
                # distributed reduction here or rank zero can hang waiting for
                # ranks that never call this log statement.
                sync_dist=False,
            )

    def _log_global_context_readout_noise_level_schedules(
        self,
        global_context_readout_noise_level_schedules: Optional[Dict[str, Optional[Tensor]]],
        namespace: str,
        dataloader_idx: int = 0,
        video_metadata: Optional[Dict[str, Tensor]] = None,
        num_logged_videos_override: Optional[int] = None,
    ) -> None:
        _ = video_metadata
        if not bool(getattr(self.logging_cfg, "log_global_readout_noise_level_schedule", False)):
            return
        num_logged_videos = (
            int(self.num_logged_videos[dataloader_idx])
            if num_logged_videos_override is None
            else int(num_logged_videos_override)
        )
        log_global_context_readout_noise_level_schedules(
            global_context_readout_noise_level_schedules,
            namespace=f"{namespace}_context_readout",
            dataloader_idx=dataloader_idx,
            sanity_checking=bool(self.trainer.sanity_checking),
            sanity_generation=bool(self.logging_cfg.sanity_generation),
            max_num_videos=int(self.logging_cfg.max_num_videos),
            num_logged_videos=num_logged_videos,
            has_logger=self.logger is not None,
            is_rank_zero=bool(is_rank_zero),
            log_image=lambda k, img: self.log_image(key=k, image=img),
        )

    def _log_inference_step_counts(
        self,
        *,
        other_results_by_task: Dict[str, Dict],
        dataloader_idx: int,
    ) -> None:
        """
        Log actual work counters from streaming inference.
        """
        _ = dataloader_idx
        for task, results in (other_results_by_task or {}).items():
            if not isinstance(results, dict):
                continue
            raw_nfe = results.get("raw_nfe", None)
            raw_token_nfe = results.get("raw_token_nfe", None)
            nfe_per_step = results.get("nfe_per_step", 1.0)
            forward_evals = results.get("forward_evals", None)
            inner_steps = results.get("inner_steps", None)
            if raw_nfe is None:
                continue

            raw_nfe_t = (
                raw_nfe
                if isinstance(raw_nfe, torch.Tensor)
                else torch.as_tensor(raw_nfe, device=self.device)
            ).to(device=self.device, dtype=torch.float32)
            raw_token_nfe_t = None
            if raw_token_nfe is not None:
                raw_token_nfe_t = (
                    raw_token_nfe
                    if isinstance(raw_token_nfe, torch.Tensor)
                    else torch.as_tensor(raw_token_nfe, device=self.device)
                ).to(device=self.device, dtype=torch.float32)

            nfe_per_step_f = float(nfe_per_step) if nfe_per_step is not None else 1.0
            denom = max(nfe_per_step_f, 1e-8)
            touches_per_video_mean = (raw_nfe_t / denom).mean()

            self.log(
                f"inference/{task}/raw_nfe_mean",
                raw_nfe_t.mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            if forward_evals is not None:
                self.log(
                    f"inference/{task}/forward_evals",
                    torch.as_tensor(float(forward_evals), device=self.device),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
            if inner_steps is not None:
                self.log(
                    f"inference/{task}/inner_steps",
                    torch.as_tensor(float(inner_steps), device=self.device),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
            tail_finish_summary = results.get("tail_finish", None)
            if isinstance(tail_finish_summary, dict) and tail_finish_summary.get("enabled", False):
                self.log(
                    f"inference/{task}/tail_finish/total_tail_steps",
                    torch.as_tensor(float(tail_finish_summary.get("total_tail_steps", 0)), device=self.device),
                    on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,
                )
                self.log(
                    f"inference/{task}/tail_finish/n_invocations",
                    torch.as_tensor(float(tail_finish_summary.get("n_invocations", 0)), device=self.device),
                    on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,
                )
                self.log(
                    f"inference/{task}/tail_finish/total_tail_tokens",
                    torch.as_tensor(float(tail_finish_summary.get("total_tail_tokens", 0)), device=self.device),
                    on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,
                )
                avg_k = tail_finish_summary.get("avg_trigger_k", None)
                if avg_k is not None:
                    self.log(
                        f"inference/{task}/tail_finish/avg_trigger_k",
                        torch.as_tensor(float(avg_k), device=self.device),
                        on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,
                    )
            self.log(
                f"inference/{task}/touches_per_video_mean",
                touches_per_video_mean,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            total_nfes_allowed = results.get("total_nfes_allowed", None)
            if total_nfes_allowed is not None:
                max_nfes = float(total_nfes_allowed)
                if max_nfes > 0:
                    self.log(
                        f"inference/{task}/max_possible_nfes",
                        torch.as_tensor(max_nfes, device=self.device),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
                    self.log(
                        f"inference/{task}/max_possible_forward_evals",
                        torch.as_tensor(max_nfes * float(denom), device=self.device),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
                    self.log(
                        f"inference/{task}/raw_nfe_over_max_possible_mean",
                        (raw_nfe_t / max_nfes).mean(),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )

            if raw_token_nfe_t is not None:
                total_generated_tokens = results.get("total_generated_tokens", None)
                if total_generated_tokens is not None:
                    gen_tokens = int(total_generated_tokens)
                    if gen_tokens > 0:
                        self.log(
                            f"inference/{task}/token_touches_per_generated_token_mean",
                            (raw_token_nfe_t / denom / float(gen_tokens)).mean(),
                            on_step=False,
                            on_epoch=True,
                            prog_bar=False,
                            sync_dist=True,
                        )
                self.log(
                    f"inference/{task}/raw_token_nfe_mean",
                    raw_token_nfe_t.mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    f"inference/{task}/token_touches_per_video_mean",
                    (raw_token_nfe_t / denom).mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

    def _log_inference_step_to_noise_level_thresholds(
        self,
        *,
        other_results_by_task: Dict[str, Dict],
        dataloader_idx: int,
    ) -> None:
        """
        For each user-specified noise-level threshold T, log the empirical first step
        index (0..num_sampling_steps) at which the noise-level readout predicts
        k_hat <= T, averaged across (sample, generated frame) pairs. If a frame
        never reaches T, its value defaults to num_sampling_steps. Also log the
        static scheduled step index for comparison.

        Requires `algorithm.readout.enabled=true` and a populated
        `global_readout_noise_level_schedule` (which the matching gate in
        `_streaming_inference` ensures when thresholds are configured).

        Note on step accounting:
          `step_index = num_sampling_steps - n_act` counts steps applied to a
          frame. For pyramid initial noise profiles a frame may skip early
          indices; the value remains an honest count of applied steps.
        """
        _ = dataloader_idx
        thresholds_cfg = getattr(self.logging_cfg, "inference_step_to_noise_level_thresholds", None)
        if not thresholds_cfg:
            return
        # Silently no-op when the current model has no noise-level readout: this
        # metric is meaningless without k_hat, and the sampler gate already
        # avoids collecting readouts in that case.
        if not bool(getattr(self, "readout_enabled", False)):
            return
        try:
            thresholds = [float(t) for t in thresholds_cfg]
        except (TypeError, ValueError):
            return
        if len(thresholds) == 0:
            return

        scheduled_steps_by_T = self._compute_scheduled_step_to_threshold(thresholds)

        for task, results in (other_results_by_task or {}).items():
            if not isinstance(results, dict):
                continue
            sched = results.get("global_denoising_schedule", None)
            t_hat = results.get("global_readout_noise_level_schedule", None)
            if not isinstance(sched, torch.Tensor) or sched.ndim != 3:
                continue
            if not isinstance(t_hat, torch.Tensor) or t_hat.ndim != 3:
                continue

            depth = results.get("denoising_depth", None)
            if depth is None:
                depth = int(getattr(self, "num_sampling_steps", 0))
            if isinstance(depth, torch.Tensor):
                depth = int(depth.detach().cpu().reshape(-1)[0].item())
            depth = int(depth)
            if depth <= 0:
                continue

            sched_f = sched.detach().to(dtype=torch.float32).cpu()
            t_hat_f = t_hat.detach().to(dtype=torch.float32).cpu()

            if sched_f.shape != t_hat_f.shape:
                continue

            local_step = (float(depth) - sched_f).clamp(min=0.0, max=float(depth))

            # Active frame mask: any touch has a finite readout for that frame.
            t_hat_finite = torch.isfinite(t_hat_f)
            active_frame_mask = t_hat_finite.any(dim=1)  # (B, F)
            if not bool(active_frame_mask.any()):
                continue

            for T in thresholds:
                hit = t_hat_finite & (t_hat_f <= float(T))  # (B, touches, F)
                any_hit = hit.any(dim=1)  # (B, F)
                # argmax on int returns the first index of the max value; when
                # any_hit is True the first True wins; when False we overwrite
                # the gathered value with the default below.
                first_idx = torch.argmax(hit.to(dtype=torch.int8), dim=1)  # (B, F)
                gathered = local_step.gather(1, first_idx.unsqueeze(1)).squeeze(1)  # (B, F)
                default_val = torch.full_like(gathered, float(depth))
                per_frame = torch.where(any_hit, gathered, default_val)

                active = active_frame_mask
                if not bool(active.any()):
                    continue
                mean_val = per_frame[active].mean()

                key = f"inference/{task}/step_to_noise_level_{float(T):.4f}"
                self.log(
                    key,
                    mean_val.to(device=self.device, dtype=torch.float32),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

                scheduled_step = scheduled_steps_by_T.get(float(T))
                if scheduled_step is not None:
                    self.log(
                        f"inference/{task}/scheduled_step_to_noise_level_{float(T):.4f}",
                        torch.as_tensor(float(scheduled_step), device=self.device, dtype=torch.float32),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )

    def _compute_scheduled_step_to_threshold(
        self,
        thresholds: list,
    ) -> Dict[float, float]:
        """
        Compute the first step index at which the scheduled noise level is <= T
        for each threshold in the list. Uses the inference schedule lookup when
        available; returns an empty dict otherwise (callers should skip logging
        the scheduled baseline in that case).
        """
        lookup_getter = getattr(self, "_get_inference_schedule_lookup", None)
        if lookup_getter is None:
            return {}
        try:
            lookup = lookup_getter()
        except Exception:
            return {}
        if lookup is None or getattr(lookup, "levels", None) is None:
            return {}

        depth = int(getattr(self, "num_sampling_steps", 0))
        if depth <= 0:
            return {}

        levels = lookup.levels.detach().to(dtype=torch.float32).cpu()
        # levels[idx] is the scheduled noise level for n_act=idx; higher idx -> noisier.
        # step_idx = depth - n_act, so for step_idx s the scheduled noise level is levels[depth - s].
        out: Dict[float, float] = {}
        N = int(levels.shape[0]) - 1
        for T in thresholds:
            T_f = float(T)
            found: Optional[int] = None
            for s in range(0, depth + 1):
                idx = depth - s
                if idx < 0 or idx > N:
                    continue
                if float(levels[idx].item()) <= T_f:
                    found = s
                    break
            out[T_f] = float(found) if found is not None else float(depth)
        return out

