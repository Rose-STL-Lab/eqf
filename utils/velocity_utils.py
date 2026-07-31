from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

import matplotlib.pyplot as plt


class DenoisingStateRows:
    """
    Tracks global denoising, readout-time, and active-velocity rows during streaming inference.

    This encapsulates the mutable row buffers that were previously tracked as
    local globals in the sampler loop.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        total_length: int,
        total_generated_frames: int,
        initial_context_tokens: int,
        num_sampling_steps: int,
        max_touch_rows: int,
        device: torch.device,
    ) -> None:
        self.batch_size = int(batch_size)
        self.total_length = int(total_length)
        self.total_generated_frames = int(total_generated_frames)
        self.initial_context_tokens = int(initial_context_tokens)
        self.num_sampling_steps = int(num_sampling_steps)
        self.max_touch_rows = int(max_touch_rows)
        self.device = device

        self.global_schedule_rows = torch.full(
            (self.batch_size, self.max_touch_rows, self.total_generated_frames),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self.global_readout_noise_level_rows = torch.full(
            (self.batch_size, self.max_touch_rows, self.total_generated_frames),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self.global_readout_noise_level_raw_rows = torch.full(
            (self.batch_size, self.max_touch_rows, self.total_generated_frames),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self.global_velocity_rows = torch.full(
            (self.batch_size, self.max_touch_rows, self.total_generated_frames),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self.global_context_readout_noise_level_rows = torch.full(
            (self.batch_size, self.max_touch_rows, self.total_length),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self.locked_global_velocity = torch.full(
            (self.batch_size, self.total_generated_frames),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self._next_row_idx = 0
        self._last_row_idx: Optional[int] = None
        self._overflow_count = 0

    def _reserve_row_index(self) -> int:
        if self._next_row_idx < self.max_touch_rows:
            idx = int(self._next_row_idx)
            self._next_row_idx += 1
        else:
            # Keep shape invariant; overwrite the last row if we exceed the budget.
            idx = int(self.max_touch_rows - 1)
            self._overflow_count += 1
        self._last_row_idx = idx
        return idx

    def _valid_generated_mask(self, abs_idx: Tensor) -> Tensor:
        return torch.logical_and(
            abs_idx >= self.initial_context_tokens,
            abs_idx < self.total_length,
        )

    def _append_global_schedule_row(
        self,
        *,
        row_idx: int,
        active_abs_frame_idx: Tensor,
        state_n_act: Tensor,
        init: bool = False,
    ) -> None:
        if self.total_generated_frames == 0:
            return

        row = self.global_schedule_rows[:, row_idx]
        if init:
            row.fill_(float(self.num_sampling_steps))
        elif row_idx > 0:
            row.copy_(self.global_schedule_rows[:, row_idx - 1])
        else:
            row.fill_(float(self.num_sampling_steps))

        valid_active = self._valid_generated_mask(active_abs_frame_idx)
        if bool(valid_active.any()) and (not init):
            active_gen_idx = active_abs_frame_idx[valid_active] - self.initial_context_tokens
            row[:, active_gen_idx] = state_n_act[:, valid_active].to(dtype=torch.float32)

    def _append_global_readout_noise_level_row(
        self,
        *,
        row_idx: int,
        active_abs_frame_idx: Tensor,
        readout_frame_noise_level: Optional[Tensor] = None,
        init: bool = False,
    ) -> None:
        if self.total_generated_frames == 0:
            return

        row = self.global_readout_noise_level_rows[:, row_idx]
        if init:
            row.fill_(float("nan"))
        elif row_idx > 0:
            row.copy_(self.global_readout_noise_level_rows[:, row_idx - 1])
        else:
            row.fill_(float("nan"))

        if readout_frame_noise_level is not None:
            valid_active = self._valid_generated_mask(active_abs_frame_idx)
            if bool(valid_active.any()):
                active_gen_idx = active_abs_frame_idx[valid_active] - self.initial_context_tokens
                row[:, active_gen_idx] = readout_frame_noise_level[:, valid_active].to(dtype=torch.float32)

    def _append_global_readout_noise_level_raw_row(
        self,
        *,
        row_idx: int,
        active_abs_frame_idx: Tensor,
        readout_frame_noise_level: Optional[Tensor] = None,
    ) -> None:
        if self.total_generated_frames == 0:
            return

        row = self.global_readout_noise_level_raw_rows[:, row_idx]
        row.fill_(float("nan"))
        if readout_frame_noise_level is not None:
            valid_active = self._valid_generated_mask(active_abs_frame_idx)
            if bool(valid_active.any()):
                active_gen_idx = active_abs_frame_idx[valid_active] - self.initial_context_tokens
                row[:, active_gen_idx] = readout_frame_noise_level[:, valid_active].to(dtype=torch.float32)

    def _append_global_velocity_row(
        self,
        *,
        row_idx: int,
        active_abs_frame_idx: Tensor,
        frame_res: Optional[Tensor] = None,
    ) -> None:
        if self.total_generated_frames == 0:
            return

        row = self.global_velocity_rows[:, row_idx]
        row.copy_(self.locked_global_velocity)
        if frame_res is not None:
            valid_active = self._valid_generated_mask(active_abs_frame_idx)
            if bool(valid_active.any()):
                active_gen_idx = active_abs_frame_idx[valid_active] - self.initial_context_tokens
                row[:, active_gen_idx] = frame_res[:, valid_active].to(dtype=torch.float32)

    def _append_global_context_readout_noise_level_row(
        self,
        *,
        row_idx: int,
        total_committed: int,
        sliding_context_tokens: int,
        context_readout_noise_level: Optional[Tensor] = None,
        context_abs_frame_idx: Optional[Tensor] = None,
    ) -> None:
        row = self.global_context_readout_noise_level_rows[:, row_idx]
        row.fill_(float("nan"))
        if context_readout_noise_level is None:
            return

        if context_abs_frame_idx is not None:
            if context_abs_frame_idx.ndim != 1:
                raise ValueError(
                    "context_abs_frame_idx must be 1D when provided, got "
                    f"shape={tuple(context_abs_frame_idx.shape)}."
                )
            if int(context_readout_noise_level.shape[1]) != int(context_abs_frame_idx.numel()):
                raise ValueError(
                    "context_readout_noise_level width must match context_abs_frame_idx length, got "
                    f"{context_readout_noise_level.shape[1]} and {int(context_abs_frame_idx.numel())}."
                )
            ctx_abs = context_abs_frame_idx.to(device=self.device, dtype=torch.long)
            ctx_valid = torch.logical_and(ctx_abs >= 0, ctx_abs < self.total_length)
            if bool(ctx_valid.any()):
                ctx_global_idx = ctx_abs[ctx_valid]
                ctx_local_idx = torch.nonzero(ctx_valid, as_tuple=False).squeeze(-1)
                row[:, ctx_global_idx] = context_readout_noise_level[:, ctx_local_idx].to(
                    dtype=torch.float32
                )
            return

        if sliding_context_tokens > 0:
            ctx_abs = torch.arange(
                int(total_committed) - int(sliding_context_tokens),
                int(total_committed),
                device=self.device,
                dtype=torch.long,
            )
            ctx_valid = torch.logical_and(ctx_abs >= 0, ctx_abs < self.total_length)
            if bool(ctx_valid.any()):
                ctx_global_idx = ctx_abs[ctx_valid]
                ctx_local_idx = torch.nonzero(ctx_valid, as_tuple=False).squeeze(-1)
                row[:, ctx_global_idx] = context_readout_noise_level[:, ctx_local_idx].to(
                    dtype=torch.float32
                )

    def append_rows(
        self,
        *,
        active_abs_frame_idx: Tensor,
        state_n_act: Tensor,
        total_committed: int,
        sliding_context_tokens: int,
        frame_res: Optional[Tensor] = None,
        context_readout_noise_level: Optional[Tensor] = None,
        readout_frame_noise_level: Optional[Tensor] = None,
        context_abs_frame_idx: Optional[Tensor] = None,
        init: bool = False,
    ) -> None:
        row_idx = self._reserve_row_index()
        self._append_global_schedule_row(
            row_idx=row_idx,
            active_abs_frame_idx=active_abs_frame_idx,
            state_n_act=state_n_act,
            init=init,
        )
        self._append_global_readout_noise_level_row(
            row_idx=row_idx,
            active_abs_frame_idx=active_abs_frame_idx,
            readout_frame_noise_level=readout_frame_noise_level,
            init=init,
        )
        self._append_global_readout_noise_level_raw_row(
            row_idx=row_idx,
            active_abs_frame_idx=active_abs_frame_idx,
            readout_frame_noise_level=None if init else readout_frame_noise_level,
        )
        self._append_global_velocity_row(
            row_idx=row_idx,
            active_abs_frame_idx=active_abs_frame_idx,
            frame_res=frame_res,
        )
        self._append_global_context_readout_noise_level_row(
            row_idx=row_idx,
            total_committed=total_committed,
            sliding_context_tokens=sliding_context_tokens,
            context_readout_noise_level=context_readout_noise_level,
            context_abs_frame_idx=context_abs_frame_idx,
        )

    def _snap_emit_prefix_to_zero(
        self,
        *,
        emit_size: int,
        active_abs_frame_idx: Tensor,
    ) -> None:
        if (
            emit_size <= 0
            or self._last_row_idx is None
            or self.total_generated_frames == 0
        ):
            return
        emit_abs = active_abs_frame_idx[:emit_size]
        emit_valid = self._valid_generated_mask(emit_abs)
        if not bool(emit_valid.any()):
            return
        emit_gen_idx = emit_abs[emit_valid] - self.initial_context_tokens
        self.global_schedule_rows[:, self._last_row_idx, emit_gen_idx] = 0.0

    def _snap_emit_prefix_readout_noise_level(
        self,
        *,
        emit_size: int,
        active_abs_frame_idx: Tensor,
        readout_frame_noise_level: Optional[Tensor],
    ) -> None:
        if (
            emit_size <= 0
            or self._last_row_idx is None
            or self.total_generated_frames == 0
            or readout_frame_noise_level is None
        ):
            return
        emit_abs = active_abs_frame_idx[:emit_size]
        emit_valid = self._valid_generated_mask(emit_abs)
        if not bool(emit_valid.any()):
            return
        emit_gen_idx = emit_abs[emit_valid] - self.initial_context_tokens
        local_emit_idx = torch.nonzero(emit_valid, as_tuple=False).squeeze(-1)
        self.global_readout_noise_level_rows[:, self._last_row_idx, emit_gen_idx] = (
            readout_frame_noise_level[:, local_emit_idx].to(dtype=torch.float32)
        )

    def _snap_emit_prefix_velocity(
        self,
        *,
        emit_size: int,
        active_abs_frame_idx: Tensor,
        frame_res: Optional[Tensor],
    ) -> None:
        if emit_size <= 0 or frame_res is None or self.total_generated_frames == 0:
            return
        emit_abs = active_abs_frame_idx[:emit_size]
        emit_valid = self._valid_generated_mask(emit_abs)
        if not bool(emit_valid.any()):
            return
        emit_gen_idx = emit_abs[emit_valid] - self.initial_context_tokens
        local_emit_idx = torch.nonzero(emit_valid, as_tuple=False).squeeze(-1)
        self.locked_global_velocity[:, emit_gen_idx] = frame_res[:, local_emit_idx].to(
            dtype=torch.float32
        )

    def snap_rows(
        self,
        *,
        emit_size: int,
        active_abs_frame_idx: Tensor,
        frame_res: Optional[Tensor],
        readout_frame_noise_level: Optional[Tensor],
    ) -> None:
        self._snap_emit_prefix_to_zero(
            emit_size=emit_size,
            active_abs_frame_idx=active_abs_frame_idx,
        )
        self._snap_emit_prefix_readout_noise_level(
            emit_size=emit_size,
            active_abs_frame_idx=active_abs_frame_idx,
            readout_frame_noise_level=readout_frame_noise_level,
        )
        self._snap_emit_prefix_velocity(
            emit_size=emit_size,
            active_abs_frame_idx=active_abs_frame_idx,
            frame_res=frame_res,
        )

    def to_other_results(self) -> Dict[str, Optional[Tensor]]:
        return {
            "global_denoising_schedule": self.global_schedule_rows.detach().cpu(),
            "global_readout_noise_level_schedule": self.global_readout_noise_level_rows.detach().cpu(),
            "global_readout_noise_level_raw_schedule": self.global_readout_noise_level_raw_rows.detach().cpu(),
            "global_velocity_schedule": self.global_velocity_rows.detach().cpu(),
            "global_context_readout_noise_level_schedule": self.global_context_readout_noise_level_rows.detach().cpu(),
            "global_touch_rows_written": int(min(self._next_row_idx, self.max_touch_rows)),
            "global_touch_overflow_count": int(self._overflow_count),
        }

def _fig_to_image(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = np.asarray(buf)[..., :3]
    plt.close(fig)
    return img


def _nanmean_keep_nan(x: np.ndarray, axis: int) -> np.ndarray:
    """
    NaN-aware mean that keeps NaN when all values along `axis` are NaN.
    Avoids RuntimeWarning: Mean of empty slice.
    """
    valid = np.isfinite(x)
    count = valid.sum(axis=axis)
    summed = np.nansum(x, axis=axis)
    mean = summed / np.maximum(count, 1)
    mean = mean.astype(np.float32, copy=False)
    mean[count == 0] = np.nan
    return mean


def _touch_support_fraction_any_frame(x: np.ndarray) -> np.ndarray:
    """
    Fraction of samples that have at least one finite frame at each touch.

    x: (B, touches, frames)
    returns: (touches,)
    """
    finite_any_frame = np.isfinite(x).any(axis=2)  # (B, touches)
    return finite_any_frame.mean(axis=0).astype(np.float32, copy=False)


def _mask_sparse_touch_rows_for_aggregate(
    x: np.ndarray,
    *,
    touch_support_fraction: np.ndarray,
    min_support_frac: float = 0.10,
) -> np.ndarray:
    """
    Mask aggregate rows where support is too sparse.
    """
    out = np.asarray(x, dtype=np.float32).copy()
    low_support = touch_support_fraction < float(min_support_frac)
    if out.ndim == 2:
        out[low_support, :] = np.nan
    elif out.ndim == 1:
        out[low_support] = np.nan
    else:
        raise ValueError(f"Expected 1D/2D aggregate array, got shape={out.shape}")
    return out


def _stabilize_generated_series_for_aggregate(x: np.ndarray) -> np.ndarray:
    """
    Plotting-only stabilization for generated-frame schedules (B, touches, frames).

    For each sample/frame series, carry the last finite value forward until that
    sample's last touch where any generated frame is defined, then keep NaN after.
    This reduces sparse-tail aggregate artifacts without changing stored tensors.
    """
    out = np.asarray(x, dtype=np.float32).copy()
    if out.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={out.shape}")

    bsz, touches, frames = out.shape
    if bsz == 0 or touches == 0 or frames == 0:
        return out

    finite = np.isfinite(out)
    touch_idx = np.arange(touches, dtype=np.int64)

    # Last touch where this sample has any finite generated-frame value.
    sample_any_finite_by_touch = finite.any(axis=2)  # (B, touches)
    sample_last_touch = np.where(
        sample_any_finite_by_touch,
        touch_idx[None, :],
        -1,
    ).max(axis=1)  # (B,)

    for bi in range(bsz):
        cutoff = int(sample_last_touch[bi])
        if cutoff < 0:
            continue

        # No aggregate contribution past the sample's effective horizon.
        if cutoff + 1 < touches:
            out[bi, cutoff + 1 :, :] = np.nan

        for fi in range(frames):
            line = out[bi, :, fi]
            finite_idx = np.flatnonzero(np.isfinite(line))
            if finite_idx.size == 0:
                continue
            last_idx = int(finite_idx[-1])
            if last_idx < cutoff:
                line[last_idx + 1 : cutoff + 1] = line[last_idx]

    return out


def _prepare_line_values_for_plot(
    values: np.ndarray,
    *,
    use_log_y: bool,
    min_positive: float = 1e-12,
) -> np.ndarray:
    """
    Prepare line values for plotting.

    When log y-scale is requested, finite values are floored to `min_positive`
    so zeros remain visible instead of breaking log scaling.
    """
    out = np.asarray(values, dtype=np.float32)
    if not use_log_y:
        return out
    out = out.copy()
    valid = np.isfinite(out)
    out[valid] = np.maximum(out[valid], float(min_positive))
    return out


def _maybe_set_log_yscale(
    ax: plt.Axes,
    *,
    use_log_y: bool,
    candidates: Tuple[np.ndarray, ...],
) -> bool:
    """Enable log y-scale only when at least one positive finite value exists."""
    if not use_log_y:
        return False
    for arr in candidates:
        values = np.asarray(arr)
        if np.any(np.isfinite(values) & (values > 0.0)):
            ax.set_yscale("log")
            return True
    return False


def log_global_denoising_schedules(
    schedules: Optional[Dict[str, Optional[Tensor]]],
    *,
    denoising_depth_by_task: Optional[Dict[str, int]] = None,
    namespace: str,
    dataloader_idx: int,
    sanity_checking: bool,
    sanity_generation: bool,
    max_num_videos: int,
    num_logged_videos: int,
    has_logger: bool,
    is_rank_zero: bool,
    log_image: Callable[[str, np.ndarray], None],
) -> None:
    """
    Log global denoising schedule visualizations.

    schedules:
      task -> schedule_grid, where schedule_grid is (B, steps, generated_frames)
      with integer n_act values.
    """
    if not schedules:
        return
    if sanity_checking and (not sanity_generation):
        return
    if num_logged_videos >= max_num_videos:
        return
    if not has_logger or not is_rank_zero:
        return

    remaining_examples = int(max_num_videos - num_logged_videos)
    if remaining_examples <= 0:
        return

    max_examples_preview = min(3, remaining_examples)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    def _compute_panel_size(
        *,
        num_touches: int,
        num_generated_frames: int,
        denoising_depth: int,
    ) -> Tuple[float, float]:
        # Width should follow generated horizon directly.
        frames_per_inch = 22.0
        panel_w = float(
            np.clip(
                float(max(int(num_generated_frames), 1)) / frames_per_inch,
                6.0,
                24.0,
            )
        )

        # Height should primarily follow denoising depth.
        # Extra rollout passes over the same depth grow height only logarithmically.
        depth = max(int(denoising_depth), 1)
        touches = max(int(num_touches), 1)
        passes_over_depth = max(float(touches) / float(depth), 1.0)
        effective_depth = float(depth) * (1.0 + np.log2(passes_over_depth))
        depth_rows_per_inch = 18.0
        panel_h = float(np.clip(effective_depth / depth_rows_per_inch, 3.0, 10.0))
        return panel_w, panel_h

    for task, schedule_grid in schedules.items():
        if schedule_grid is None or schedule_grid.ndim != 3:
            continue

        sched = schedule_grid.detach().cpu().float()
        valid_items = ~torch.isnan(sched).all(dim=(1, 2))
        if not torch.any(valid_items):
            continue
        sched = sched[valid_items]
        if sched.numel() == 0:
            continue

        bsz, steps, frames = map(int, sched.shape)
        if bsz <= 0 or steps <= 0 or frames <= 0:
            continue

        num_examples_preview = min(max_examples_preview, min(remaining_examples, bsz))
        if num_examples_preview <= 0:
            continue

        sched_full = sched.numpy()  # (B, S, F), no downsampling
        depth_hint = None
        if denoising_depth_by_task is not None:
            depth_hint = denoising_depth_by_task.get(task)
        if depth_hint is None or int(depth_hint) <= 0:
            inferred_depth = int(np.nanmax(sched_full))
            denoising_depth = max(inferred_depth, 1)
        else:
            denoising_depth = max(int(depth_hint), 1)

        key_prefix = f"{task}_denoising_{namespace}_loader_id={dataloader_idx}"

        def _log_schedule_view(
            sched_view: np.ndarray,
            key_suffix: str,
            title_suffix: str,
            num_examples_to_plot: int,
        ) -> None:
            if not np.any(np.isfinite(sched_view)):
                return
            sched_for_avg = _stabilize_generated_series_for_aggregate(sched_view)
            avg_sched = _nanmean_keep_nan(sched_for_avg, axis=0)
            support_frac = _touch_support_fraction_any_frame(sched_view)
            avg_sched = _mask_sparse_touch_rows_for_aggregate(
                avg_sched,
                touch_support_fraction=support_frac,
            )
            vmax = float(np.nanmax(sched_view))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            panel_w, panel_h = _compute_panel_size(
                num_touches=int(avg_sched.shape[0]),
                num_generated_frames=int(avg_sched.shape[1]),
                denoising_depth=denoising_depth,
            )

            fig_avg, ax_avg = plt.subplots(
                nrows=1,
                ncols=1,
                figsize=(panel_w, panel_h),
                constrained_layout=True,
            )
            im_avg = ax_avg.imshow(
                avg_sched,
                aspect="auto",
                origin="lower",
                vmin=0.0,
                vmax=float(vmax),
                cmap=cmap,
                interpolation="nearest",
            )
            ax_avg.set_title(
                f"{task} | global denoising schedule (batch mean){title_suffix}",
                fontsize=11,
            )
            ax_avg.set_xlabel("generated frame index", fontsize=9)
            ax_avg.set_ylabel("global touch", fontsize=9)
            ax_avg.tick_params(labelsize=7)
            cbar_avg = fig_avg.colorbar(im_avg, ax=ax_avg, fraction=0.020, pad=0.01)
            cbar_avg.set_label("n_act", rotation=90)
            log_image(
                f"{key_prefix}/global_denoising_schedule_average{key_suffix}",
                _fig_to_image(fig_avg),
            )

            ex_cols = max(1, int(num_examples_to_plot))
            fig_ex, axes_ex = plt.subplots(
                nrows=1,
                ncols=ex_cols,
                figsize=(panel_w * ex_cols, panel_h),
                squeeze=False,
                constrained_layout=True,
            )
            fig_ex.suptitle(
                f"{task} | global denoising schedule (examples){title_suffix}",
                fontsize=11,
            )
            for j in range(ex_cols):
                ax = axes_ex[0, j]
                im = ax.imshow(
                    sched_view[j],
                    aspect="auto",
                    origin="lower",
                    vmin=0.0,
                    vmax=float(vmax),
                    cmap=cmap,
                    interpolation="nearest",
                )
                ax.set_title(f"ex {j}", fontsize=9)
                ax.set_xlabel("generated frame index", fontsize=8)
                if j == 0:
                    ax.set_ylabel("global touch", fontsize=8)
                ax.tick_params(labelsize=7)
            cbar_ex = fig_ex.colorbar(im, ax=axes_ex.ravel().tolist(), fraction=0.020, pad=0.01)
            cbar_ex.set_label("n_act", rotation=90)
            log_image(
                f"{key_prefix}/global_denoising_schedule_examples{key_suffix}",
                _fig_to_image(fig_ex),
            )

        _log_schedule_view(
            sched_full,
            key_suffix="",
            title_suffix=" (full resolution)",
            num_examples_to_plot=num_examples_preview,
        )


def log_global_frame_nfe_bars(
    schedules: Optional[Dict[str, Optional[Tensor]]],
    *,
    bake_in_inner_steps_by_task: Optional[Dict[str, int]],
    namespace: str,
    dataloader_idx: int,
    sanity_checking: bool,
    sanity_generation: bool,
    max_num_videos: int,
    num_logged_videos: int,
    has_logger: bool,
    is_rank_zero: bool,
    log_image: Callable[[str, np.ndarray], None],
) -> None:
    """
    Log per-frame NFE bar plots derived from global denoising schedules.

    For each generated frame, NFE is counted as the number of touches where the
    denoising sentinel strictly decreases. Bake-in transitions are included.
    """
    if not schedules:
        return
    if sanity_checking and (not sanity_generation):
        return
    if num_logged_videos >= max_num_videos:
        return
    if not has_logger or not is_rank_zero:
        return

    remaining_examples = int(max_num_videos - num_logged_videos)
    if remaining_examples <= 0:
        return

    for task, schedule_grid in schedules.items():
        if schedule_grid is None or schedule_grid.ndim != 3:
            continue

        sched = schedule_grid.detach().cpu().float()
        valid_items = ~torch.isnan(sched).all(dim=(1, 2))
        if not torch.any(valid_items):
            continue
        sched = sched[valid_items]
        if sched.numel() == 0:
            continue
        bsz, steps, frames = map(int, sched.shape)
        if bsz <= 0 or steps <= 1 or frames <= 0:
            continue

        # Touch transitions are row-to-row decreases in the denoising sentinel.
        deltas = sched[:, 1:, :] - sched[:, :-1, :]  # (B, steps-1, frames)
        key_prefix = f"{task}_denoising_{namespace}_loader_id={dataloader_idx}"

        # Per-video total touches: at each transition, count 1 touch for a video
        # if any generated frame decreased.
        video_touch_counts = (deltas < 0).any(dim=2).sum(dim=1).to(torch.long)  # (B,)
        if video_touch_counts.numel() > 0:
            hist_vid = torch.bincount(video_touch_counts)
            nonzero_vid = hist_vid > 0
            xs_vid = torch.nonzero(nonzero_vid, as_tuple=False).squeeze(-1).cpu().numpy()
            ys_vid = hist_vid[nonzero_vid].cpu().numpy()

            fig_vid_hist, ax_vid_hist = plt.subplots(
                nrows=1,
                ncols=1,
                figsize=(8.5, 3.8),
                constrained_layout=True,
            )
            ax_vid_hist.bar(xs_vid, ys_vid, width=0.9, color="#1f77b4", alpha=0.85)
            ax_vid_hist.set_title(
                f"{task} | total touches per video distribution (batch)",
                fontsize=11,
            )
            ax_vid_hist.set_xlabel("total touches per video", fontsize=9)
            ax_vid_hist.set_ylabel("video count", fontsize=9)
            ax_vid_hist.grid(True, axis="y", alpha=0.20, linewidth=0.8)
            ax_vid_hist.tick_params(labelsize=8)
            log_image(
                f"{key_prefix}/global_video_touch_count_distribution_batch",
                _fig_to_image(fig_vid_hist),
            )


def log_global_velocity_schedules(
    schedules: Optional[Dict[str, Optional[Tensor]]],
    *,
    namespace: str,
    dataloader_idx: int,
    sanity_checking: bool,
    sanity_generation: bool,
    max_num_videos: int,
    num_logged_videos: int,
    has_logger: bool,
    is_rank_zero: bool,
    log_image: Callable[[str, np.ndarray], None],
    velocity_log_yscale: bool = False,
) -> None:
    """
    Log full-resolution global velocity schedules as line plots (no downsampling).

    schedules:
      task -> vel_grid, where vel_grid is (B, touches, generated_frames)
      with NaN indicating "no gradient defined for this frame at this touch".
    """
    if not schedules:
        return
    if sanity_checking and (not sanity_generation):
        return
    if num_logged_videos >= max_num_videos:
        return
    if not has_logger or not is_rank_zero:
        return

    remaining_examples = int(max_num_videos - num_logged_videos)
    if remaining_examples <= 0:
        return

    max_examples = min(3, remaining_examples)
    cmap = plt.cm.turbo

    try:
        import matplotlib as mpl
    except Exception:
        mpl = None

    for task, vel_grid in schedules.items():
        if vel_grid is None or vel_grid.ndim != 3:
            continue

        vel = vel_grid.detach().cpu().float()
        if vel.numel() == 0:
            continue

        valid_items = ~torch.isnan(vel).all(dim=(1, 2))
        if not torch.any(valid_items):
            continue
        vel = vel[valid_items]

        bsz, touches, frames = map(int, vel.shape)
        if bsz <= 0 or touches <= 0 or frames <= 0:
            continue

        num_examples = min(max_examples, bsz)
        if num_examples <= 0:
            continue

        x = np.arange(touches)
        key_prefix = f"{task}_velocity_{namespace}_loader_id={dataloader_idx}"

        vel_np = vel.numpy()
        vel_for_avg = _stabilize_generated_series_for_aggregate(vel_np)
        mean_by_frame = _nanmean_keep_nan(vel_for_avg, axis=0)  # (touches, frames)
        support_frac = _touch_support_fraction_any_frame(vel_np)
        mean_by_frame = _mask_sparse_touch_rows_for_aggregate(
            mean_by_frame,
            touch_support_fraction=support_frac,
        )

        fig_avg, ax_avg = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(10.5, 3.8),
            constrained_layout=True,
        )
        norm = (
            mpl.colors.Normalize(vmin=0.0, vmax=float(max(1, frames - 1)))
            if mpl is not None
            else None
        )
        for fi in range(frames):
            color = (
                cmap(norm(fi))
                if norm is not None
                else cmap(float(fi) / max(1.0, float(max(1, frames - 1))))
            )
            ax_avg.plot(
                x,
                _prepare_line_values_for_plot(
                    mean_by_frame[:, fi], use_log_y=velocity_log_yscale
                ),
                color=color,
                linewidth=0.9,
                alpha=0.85,
            )
        avg_uses_log_y = _maybe_set_log_yscale(
            ax_avg,
            use_log_y=velocity_log_yscale,
            candidates=(mean_by_frame,),
        )
        ax_avg.set_title(f"{task} | global velocity vs touch (batch mean, full resolution)", fontsize=11)
        ax_avg.set_xlabel("global touch / NFE", fontsize=9)
        ax_avg.set_ylabel("|v| (log scale)" if avg_uses_log_y else "|v|", fontsize=9)
        ax_avg.grid(True, alpha=0.20, linewidth=0.8)
        ax_avg.tick_params(labelsize=7)
        if mpl is not None:
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig_avg.colorbar(sm, ax=ax_avg, fraction=0.020, pad=0.01)
            cbar.set_label("generated frame index", rotation=90)
        log_image(
            f"{key_prefix}/global_velocity_schedule_lines_fullres",
            _fig_to_image(fig_avg),
        )

        # Example-wise lines by frame (full resolution, no downsampling).
        fig_ex, axes_ex = plt.subplots(
            nrows=1,
            ncols=num_examples,
            figsize=(max(9.6, 3.6 * num_examples), 3.8),
            squeeze=False,
            constrained_layout=True,
        )
        fig_ex.suptitle(
            f"{task} | global velocity vs touch (examples, full resolution)",
            fontsize=11,
        )
        for j in range(num_examples):
            ax = axes_ex[0, j]
            ex = vel[j].numpy()  # (touches, frames)
            for fi in range(frames):
                color = (
                    cmap(norm(fi))
                    if norm is not None
                    else cmap(float(fi) / max(1.0, float(max(1, frames - 1))))
                )
                ax.plot(
                    x,
                    _prepare_line_values_for_plot(
                        ex[:, fi], use_log_y=velocity_log_yscale
                    ),
                    color=color,
                    linewidth=0.85,
                    alpha=0.80,
                )
            ex_uses_log_y = _maybe_set_log_yscale(
                ax,
                use_log_y=velocity_log_yscale,
                candidates=(ex,),
            )
            ax.set_title(f"ex {j}", fontsize=9)
            ax.set_xlabel("global touch / NFE", fontsize=8)
            if j == 0:
                ax.set_ylabel("|v| (log scale)" if ex_uses_log_y else "|v|", fontsize=8)
            ax.grid(True, alpha=0.20, linewidth=0.8)
            ax.tick_params(labelsize=7)
        if mpl is not None:
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig_ex.colorbar(sm, ax=axes_ex.ravel().tolist(), fraction=0.020, pad=0.01)
            cbar.set_label("generated frame index", rotation=90)
        log_image(
            f"{key_prefix}/global_velocity_schedule_examples_fullres",
            _fig_to_image(fig_ex),
        )

def log_global_readout_noise_level_schedules(
    schedules: Optional[Dict[str, Optional[Tensor]]],
    *,
    namespace: str,
    dataloader_idx: int,
    sanity_checking: bool,
    sanity_generation: bool,
    max_num_videos: int,
    num_logged_videos: int,
    has_logger: bool,
    is_rank_zero: bool,
    log_image: Callable[[str, np.ndarray], None],
) -> None:
    """
    Log full-resolution global readout-time schedules as line plots.

    schedules:
      task -> time_grid, where time_grid is (B, touches, generated_frames)
      with NaN indicating "no readout noise level defined for this frame at this touch".
    """
    if not schedules:
        return
    if sanity_checking and (not sanity_generation):
        return
    if num_logged_videos >= max_num_videos:
        return
    if not has_logger or not is_rank_zero:
        return

    remaining_examples = int(max_num_videos - num_logged_videos)
    if remaining_examples <= 0:
        return

    max_examples = min(3, remaining_examples)
    cmap = plt.cm.turbo

    try:
        import matplotlib as mpl
    except Exception:
        mpl = None

    for task, time_grid in schedules.items():
        if time_grid is None or time_grid.ndim != 3:
            continue

        time = time_grid.detach().cpu().float()
        if time.numel() == 0:
            continue

        valid_items = ~torch.isnan(time).all(dim=(1, 2))
        if not torch.any(valid_items):
            continue
        time = time[valid_items]

        bsz, touches, frames = map(int, time.shape)
        if bsz <= 0 or touches <= 0 or frames <= 0:
            continue

        num_examples = min(max_examples, bsz)
        if num_examples <= 0:
            continue

        x = np.arange(touches)
        key_prefix = f"{task}_time_{namespace}_loader_id={dataloader_idx}"

        time_np = time.numpy()
        time_for_avg = _stabilize_generated_series_for_aggregate(time_np)
        mean_by_frame = _nanmean_keep_nan(time_for_avg, axis=0)  # (touches, frames)
        support_frac = _touch_support_fraction_any_frame(time_np)
        mean_by_frame = _mask_sparse_touch_rows_for_aggregate(
            mean_by_frame,
            touch_support_fraction=support_frac,
        )

        fig_avg, ax_avg = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(10.5, 3.8),
            constrained_layout=True,
        )
        norm = (
            mpl.colors.Normalize(vmin=0.0, vmax=float(max(1, frames - 1)))
            if mpl is not None
            else None
        )
        for fi in range(frames):
            color = (
                cmap(norm(fi))
                if norm is not None
                else cmap(float(fi) / max(1.0, float(max(1, frames - 1))))
            )
            ax_avg.plot(
                x,
                mean_by_frame[:, fi],
                color=color,
                linewidth=0.9,
                alpha=0.85,
            )
        ax_avg.set_title(
            f"{task} | global readout noise level vs touch (batch mean, full resolution)",
            fontsize=11,
        )
        ax_avg.set_xlabel("global touch / NFE", fontsize=9)
        ax_avg.set_ylabel("readout noise level", fontsize=9)
        ax_avg.grid(True, alpha=0.20, linewidth=0.8)
        ax_avg.tick_params(labelsize=7)
        if mpl is not None:
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig_avg.colorbar(sm, ax=ax_avg, fraction=0.020, pad=0.01)
            cbar.set_label("generated frame index", rotation=90)
        log_image(
            f"{key_prefix}/global_readout_noise_level_schedule_lines_fullres",
            _fig_to_image(fig_avg),
        )

        fig_ex, axes_ex = plt.subplots(
            nrows=1,
            ncols=num_examples,
            figsize=(max(9.6, 3.6 * num_examples), 3.8),
            squeeze=False,
            constrained_layout=True,
        )
        fig_ex.suptitle(
            f"{task} | global readout noise level vs touch (examples, full resolution)",
            fontsize=11,
        )
        for j in range(num_examples):
            ax = axes_ex[0, j]
            ex = time[j].numpy()  # (touches, frames)
            for fi in range(frames):
                color = (
                    cmap(norm(fi))
                    if norm is not None
                    else cmap(float(fi) / max(1.0, float(max(1, frames - 1))))
                )
                ax.plot(
                    x,
                    ex[:, fi],
                    color=color,
                    linewidth=0.85,
                    alpha=0.80,
                )
            ax.set_title(f"ex {j}", fontsize=9)
            ax.set_xlabel("global touch / NFE", fontsize=8)
            if j == 0:
                ax.set_ylabel("readout noise level", fontsize=8)
            ax.grid(True, alpha=0.20, linewidth=0.8)
            ax.tick_params(labelsize=7)
        if mpl is not None:
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig_ex.colorbar(sm, ax=axes_ex.ravel().tolist(), fraction=0.020, pad=0.01)
            cbar.set_label("generated frame index", rotation=90)
        log_image(
            f"{key_prefix}/global_readout_noise_level_schedule_examples_fullres",
            _fig_to_image(fig_ex),
        )


def log_global_readout_noise_level_raw_schedules(
    raw_schedules: Optional[Dict[str, Optional[Tensor]]],
    denoising_schedules: Optional[Dict[str, Optional[Tensor]]],
    *,
    emit_noise_level: float,
    namespace: str,
    dataloader_idx: int,
    sanity_checking: bool,
    sanity_generation: bool,
    max_num_videos: int,
    num_logged_videos: int,
    has_logger: bool,
    is_rank_zero: bool,
    log_image: Callable[[str, np.ndarray], None],
) -> Dict[str, int]:
    """
    Log raw per-touch readout heatmaps.

    Unlike `global_readout_noise_level_schedule`, this grid is sparse: a frame
    is finite only at touches where it was in the active window and read out.
    """
    summaries: Dict[str, int] = {}
    if not raw_schedules:
        return summaries
    if sanity_checking and (not sanity_generation):
        return summaries
    if num_logged_videos >= max_num_videos:
        return summaries
    if not has_logger or not is_rank_zero:
        return summaries

    remaining_examples = int(max_num_videos - num_logged_videos)
    if remaining_examples <= 0:
        return summaries

    max_examples = min(3, remaining_examples)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")
    discrepancy_cmap = plt.cm.Reds.copy()
    discrepancy_cmap.set_bad(color="white")

    for task, raw_grid in raw_schedules.items():
        if raw_grid is None or raw_grid.ndim != 3:
            continue

        raw = raw_grid.detach().cpu().float()
        if raw.numel() == 0:
            continue

        valid_items = ~torch.isnan(raw).all(dim=(1, 2))
        if not torch.any(valid_items):
            continue
        raw = raw[valid_items]
        bsz, touches, frames = map(int, raw.shape)
        if bsz <= 0 or touches <= 0 or frames <= 0:
            continue

        sched_grid = None
        if denoising_schedules:
            sched_grid = denoising_schedules.get(task)
        sched = None
        if isinstance(sched_grid, torch.Tensor) and sched_grid.ndim == 3:
            sched = sched_grid.detach().cpu().float()
            if sched.shape[0] == valid_items.shape[0]:
                sched = sched[valid_items]
            if sched.shape != raw.shape:
                sched = None

        raw_np = raw.numpy()
        finite = np.isfinite(raw_np)
        mean_heatmap = _nanmean_keep_nan(raw_np, axis=0)
        num_examples = min(max_examples, bsz)
        key_prefix = f"{task}_raw_readout_{namespace}_loader_id={dataloader_idx}"

        fig_mean, ax_mean = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(10.5, 4.6),
            constrained_layout=True,
        )
        im_mean = ax_mean.imshow(
            mean_heatmap,
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            interpolation="nearest",
        )
        ax_mean.set_title(
            f"{task} | raw readout k_hat by touch/frame (batch mean)",
            fontsize=11,
        )
        ax_mean.set_xlabel("generated frame index", fontsize=9)
        ax_mean.set_ylabel("global touch / NFE", fontsize=9)
        ax_mean.tick_params(labelsize=7)
        cbar_mean = fig_mean.colorbar(im_mean, ax=ax_mean, fraction=0.020, pad=0.01)
        cbar_mean.set_label("raw readout k_hat", rotation=90)
        log_image(
            f"{key_prefix}/global_readout_noise_level_raw_heatmap_mean",
            _fig_to_image(fig_mean),
        )

        fig_ex, axes_ex = plt.subplots(
            nrows=1,
            ncols=num_examples,
            figsize=(max(9.6, 3.8 * num_examples), 4.6),
            squeeze=False,
            constrained_layout=True,
        )
        fig_ex.suptitle(
            f"{task} | raw readout k_hat by touch/frame (examples)",
            fontsize=11,
        )
        for j in range(num_examples):
            ax = axes_ex[0, j]
            im = ax.imshow(
                raw_np[j],
                aspect="auto",
                origin="lower",
                vmin=0.0,
                vmax=1.0,
                cmap=cmap,
                interpolation="nearest",
            )
            ax.set_title(f"ex {j}", fontsize=9)
            ax.set_xlabel("generated frame index", fontsize=8)
            if j == 0:
                ax.set_ylabel("global touch / NFE", fontsize=8)
            ax.tick_params(labelsize=7)
        cbar_ex = fig_ex.colorbar(im, ax=axes_ex.ravel().tolist(), fraction=0.020, pad=0.01)
        cbar_ex.set_label("raw readout k_hat", rotation=90)
        log_image(
            f"{key_prefix}/global_readout_noise_level_raw_heatmap_examples",
            _fig_to_image(fig_ex),
        )

        # Discrepancy heuristic: all frames read out at a touch are under the
        # emit threshold, but at least one of those same frames is read out
        # again later, implying it was not emitted immediately.
        finite_any = finite.any(axis=2)
        active_max = np.full((bsz, touches), np.nan, dtype=np.float32)
        for bi in range(bsz):
            for ti in range(touches):
                vals = raw_np[bi, ti][finite[bi, ti]]
                if vals.size > 0:
                    active_max[bi, ti] = np.max(vals).astype(np.float32)
        expected_done = finite_any & (active_max <= float(emit_noise_level))
        read_again = np.zeros((bsz, touches), dtype=bool)
        for ti in range(touches - 1):
            future_finite = finite[:, ti + 1 :, :].any(axis=1)
            read_again[:, ti] = (finite[:, ti, :] & future_finite).any(axis=1)
        discrepancy = expected_done & read_again

        if sched is not None:
            # Suppress rows that were already snapped to clean at emit time.
            sched_np = sched.numpy()
            active_sched = np.isfinite(sched_np) & (sched_np > 0)
            active_now = finite & active_sched
            expected_done = active_now.any(axis=2) & (active_max <= float(emit_noise_level))
            read_again = np.zeros((bsz, touches), dtype=bool)
            for ti in range(touches - 1):
                future_active = active_now[:, ti + 1 :, :].any(axis=1)
                read_again[:, ti] = (active_now[:, ti, :] & future_active).any(axis=1)
            discrepancy = expected_done & read_again

        summaries[f"{task}/raw_readout_expected_done_but_still_active_touches"] = int(
            discrepancy.sum()
        )

        fig_disc, axes_disc = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(10.5, 5.2),
            sharex=True,
            constrained_layout=True,
        )
        mean_active_max = _nanmean_keep_nan(active_max, axis=0)
        axes_disc[0].plot(
            np.arange(touches),
            mean_active_max,
            color="#1f77b4",
            linewidth=1.4,
            label="batch mean active-window max readout",
        )
        axes_disc[0].axhline(
            float(emit_noise_level),
            color="#d62728",
            linestyle="--",
            linewidth=1.1,
            label="emit_noise_level",
        )
        axes_disc[0].set_ylabel("max k_hat", fontsize=9)
        axes_disc[0].set_title(
            f"{task} | raw readout discrepancy heuristic",
            fontsize=11,
        )
        axes_disc[0].grid(True, alpha=0.20, linewidth=0.8)
        axes_disc[0].legend(fontsize=8, loc="best")
        disc_img = discrepancy[:num_examples].astype(np.float32)
        disc_img[~finite_any[:num_examples]] = np.nan
        im_disc = axes_disc[1].imshow(
            disc_img,
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap=discrepancy_cmap,
            interpolation="nearest",
        )
        axes_disc[1].set_xlabel("global touch / NFE", fontsize=9)
        axes_disc[1].set_ylabel("example", fontsize=9)
        axes_disc[1].tick_params(labelsize=7)
        cbar_disc = fig_disc.colorbar(im_disc, ax=axes_disc.ravel().tolist(), fraction=0.020, pad=0.01)
        cbar_disc.set_label("expected done but read again", rotation=90)
        log_image(
            f"{key_prefix}/global_readout_noise_level_raw_discrepancy",
            _fig_to_image(fig_disc),
        )

    return summaries


def log_global_context_readout_noise_level_schedules(
    schedules: Optional[Dict[str, Optional[Tensor]]],
    *,
    namespace: str,
    dataloader_idx: int,
    sanity_checking: bool,
    sanity_generation: bool,
    max_num_videos: int,
    num_logged_videos: int,
    has_logger: bool,
    is_rank_zero: bool,
    log_image: Callable[[str, np.ndarray], None],
) -> None:
    """
    Log full-resolution global context-readout schedules as line plots.

    schedules:
      task -> time_grid, where time_grid is (B, touches, total_frames)
      with NaN indicating "no context readout noise level defined for this frame at this touch".
    """
    if not schedules:
        return
    if sanity_checking and (not sanity_generation):
        return
    if num_logged_videos >= max_num_videos:
        return
    if not has_logger or not is_rank_zero:
        return

    remaining_examples = int(max_num_videos - num_logged_videos)
    if remaining_examples <= 0:
        return

    max_examples = min(3, remaining_examples)
    cmap = plt.cm.turbo

    try:
        import matplotlib as mpl
    except Exception:
        mpl = None

    for task, time_grid in schedules.items():
        if time_grid is None or time_grid.ndim != 3:
            continue

        time = time_grid.detach().cpu().float()
        if time.numel() == 0:
            continue

        valid_items = ~torch.isnan(time).all(dim=(1, 2))
        if not torch.any(valid_items):
            continue
        time = time[valid_items]

        bsz, touches, frames = map(int, time.shape)
        if bsz <= 0 or touches <= 0 or frames <= 0:
            continue

        num_examples = min(max_examples, bsz)
        if num_examples <= 0:
            continue

        x = np.arange(touches)
        key_prefix = f"{task}_time_{namespace}_loader_id={dataloader_idx}"

        time_np = time.numpy()
        mean_by_frame = _nanmean_keep_nan(time_np, axis=0)  # (touches, frames)
        support_frac = _touch_support_fraction_any_frame(time_np)
        mean_by_frame = _mask_sparse_touch_rows_for_aggregate(
            mean_by_frame,
            touch_support_fraction=support_frac,
        )

        fig_avg, ax_avg = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(10.5, 3.8),
            constrained_layout=True,
        )
        norm = (
            mpl.colors.Normalize(vmin=0.0, vmax=float(max(1, frames - 1)))
            if mpl is not None
            else None
        )
        for fi in range(frames):
            color = (
                cmap(norm(fi))
                if norm is not None
                else cmap(float(fi) / max(1.0, float(max(1, frames - 1))))
            )
            ax_avg.plot(
                x,
                mean_by_frame[:, fi],
                color=color,
                linewidth=0.9,
                alpha=0.85,
            )
        ax_avg.set_title(
            f"{task} | global context readout noise level vs touch (batch mean, full resolution)",
            fontsize=11,
        )
        ax_avg.set_xlabel("global touch / NFE", fontsize=9)
        ax_avg.set_ylabel("context readout noise level", fontsize=9)
        ax_avg.grid(True, alpha=0.20, linewidth=0.8)
        ax_avg.tick_params(labelsize=7)
        if mpl is not None:
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig_avg.colorbar(sm, ax=ax_avg, fraction=0.020, pad=0.01)
            cbar.set_label("global frame index", rotation=90)
        log_image(
            f"{key_prefix}/global_context_readout_noise_level_schedule_lines_fullres",
            _fig_to_image(fig_avg),
        )

        fig_ex, axes_ex = plt.subplots(
            nrows=1,
            ncols=num_examples,
            figsize=(max(9.6, 3.6 * num_examples), 3.8),
            squeeze=False,
            constrained_layout=True,
        )
        fig_ex.suptitle(
            f"{task} | global context readout noise level vs touch (examples, full resolution)",
            fontsize=11,
        )
        for j in range(num_examples):
            ax = axes_ex[0, j]
            ex = time[j].numpy()  # (touches, frames)
            for fi in range(frames):
                color = (
                    cmap(norm(fi))
                    if norm is not None
                    else cmap(float(fi) / max(1.0, float(max(1, frames - 1))))
                )
                ax.plot(
                    x,
                    ex[:, fi],
                    color=color,
                    linewidth=0.85,
                    alpha=0.80,
                )
            ax.set_title(f"ex {j}", fontsize=9)
            ax.set_xlabel("global touch / NFE", fontsize=8)
            if j == 0:
                ax.set_ylabel("context readout noise level", fontsize=8)
            ax.grid(True, alpha=0.20, linewidth=0.8)
            ax.tick_params(labelsize=7)
        if mpl is not None:
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig_ex.colorbar(sm, ax=axes_ex.ravel().tolist(), fraction=0.020, pad=0.01)
            cbar.set_label("global frame index", rotation=90)
        log_image(
            f"{key_prefix}/global_context_readout_noise_level_schedule_examples_fullres",
            _fig_to_image(fig_ex),
        )
