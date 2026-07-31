"""Plotting and persistence helpers for FVD-over-time curves.

Mirrors the matplotlib -> uint8 RGB array convention used in
``utils/noise_level_eval_utils.py`` so the result can be passed straight to
``BasePytorchAlgo.log_image``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch

from utils.distributed_utils import is_rank_zero, rank_zero_print
from utils.noise_level_eval_utils import _fig_to_image  # noqa: F401
from utils.print_utils import cyan


def make_fvd_over_time_curve(
    *,
    start_indices: Sequence[int],
    fvd_values: Sequence[float],
    title: str,
    xlabel: str = "generated start frame",
    ylabel: str = "FVD",
) -> np.ndarray:
    import matplotlib.pyplot as plt  # local import: matplotlib is heavy

    x = np.asarray(start_indices, dtype=np.float32)
    y = np.asarray(fvd_values, dtype=np.float32)
    valid = np.isfinite(y)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if valid.any():
        ax.plot(x[valid], y[valid], marker="o", linewidth=2.0)
        # Use log-y when the dynamic range warrants it, since FVD spans orders
        # of magnitude in long rollouts.
        positive = valid & (y > 0)
        if positive.sum() >= 2:
            y_pos = y[positive]
            if y_pos.max() / max(y_pos.min(), 1e-6) > 10.0:
                ax.set_yscale("log")
    else:
        ax.text(
            0.5,
            0.5,
            "No valid data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _fig_to_image(fig)


def jsonify_curve_payload(payload: Any) -> Any:
    """Recursively convert tensors / numpy arrays in a curve payload to
    JSON-serializable Python primitives."""
    if isinstance(payload, torch.Tensor):
        return payload.detach().cpu().tolist()
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, dict):
        return {str(k): jsonify_curve_payload(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [jsonify_curve_payload(v) for v in payload]
    if isinstance(payload, np.floating):
        return float(payload)
    if isinstance(payload, np.integer):
        return int(payload)
    return payload


def _log_fvd_over_time_image(
    log_image: Callable[..., None],
    base_key: str,
    payload: Dict[str, Any],
    task: str,
    loader_idx: Optional[int],
) -> None:
    start_indices = payload.get("start_indices") or []
    fvd_values = payload.get("fvd") or []
    if not start_indices or not fvd_values:
        return
    suffix = f"/dataloader_idx_{loader_idx}" if loader_idx is not None else ""
    title = f"FVD over time ({task})"
    if loader_idx is not None:
        title += f" loader {loader_idx}"
    config = payload.get("config", {})
    if config:
        title += (
            f" | mode={config.get('reference_mode')} "
            f"clip_len={config.get('clip_len')} "
            f"gen_stride={config.get('gen_stride')}"
        )
    image = make_fvd_over_time_curve(
        start_indices=start_indices,
        fvd_values=fvd_values,
        title=title,
    )
    image_key = f"{base_key}/curve{suffix}"
    log_image(image_key, image)


def _log_fvd_over_time_wandb_chart(
    wandb_experiment: Any,
    base_key: str,
    payload: Dict[str, Any],
    task: str,
    loader_idx: Optional[int],
    step: Optional[int],
) -> None:
    start_indices = payload.get("start_indices") or []
    fvd_values = payload.get("fvd") or []
    if not start_indices or not fvd_values:
        return

    import wandb

    num_gen_clips = payload.get("num_gen_clips") or [None] * len(start_indices)
    num_ref_clips = payload.get("num_ref_clips") or [None] * len(start_indices)
    rows = list(zip(start_indices, fvd_values, num_gen_clips, num_ref_clips))
    table = wandb.Table(
        columns=["start_idx", "fvd", "num_gen_clips", "num_ref_clips"],
        data=rows,
    )
    suffix = f"/dataloader_idx_{loader_idx}" if loader_idx is not None else ""
    title = f"FVD over time ({task})"
    if loader_idx is not None:
        title += f" loader {loader_idx}"
    table_key = f"{base_key}/table{suffix}"
    line_key = f"{base_key}/line{suffix}"
    payload_to_log = {
        table_key: table,
        line_key: wandb.plot.line(table, "start_idx", "fvd", title=title),
    }
    log_kwargs: Dict[str, Any] = {"commit": False}
    if step is not None:
        log_kwargs["step"] = int(step)
    wandb_experiment.log(payload_to_log, **log_kwargs)


# Plot dispatchers per metric name. Add new entries here for additional
# curve-shaped metrics.
_CURVE_IMAGE_LOGGERS: Dict[
    str,
    Callable[
        [Callable[..., None], str, Dict[str, Any], str, Optional[int]],
        None,
    ],
] = {
    "fvd_over_time": _log_fvd_over_time_image,
}


def handle_curve_metrics(
    curve_metrics: Dict[str, Any],
    *,
    task: str,
    loader_idx: Optional[int],
    raw_dir: Optional[str],
    step: Optional[int],
    log_image: Optional[Callable[..., None]],
    wandb_experiment: Optional[Any] = None,
) -> None:
    """Persist curve-shaped metrics (e.g. fvd_over_time) that don't fit the
    scalar Lightning log path:
      - rank-zero JSON dump under ``raw_dir`` (if provided), and
      - W&B image/table/line artifacts when a logger is available.

    Args:
        curve_metrics: dict from `VideoMetric.log()` filtered to the
            excluded-from-wandb keys; values are payload dicts produced by the
            curve metric's ``compute()``.
        task: e.g. "prediction".
        loader_idx: optional dataloader index for multi-loader runs.
        raw_dir: directory to write JSON artifacts to; if None, JSON dump is
            skipped.
        step: step suffix for filenames (e.g. global step). Defaults to 0.
        log_image: callable matching ``BasePytorchAlgo.log_image`` signature
            ``(key: str, image: np.ndarray) -> None``.
        wandb_experiment: optional W&B experiment object. If provided, FVD
            curves are also logged as a table and a W&B line plot.
    """
    if not is_rank_zero:
        return

    for k, payload in curve_metrics.items():
        if not isinstance(payload, dict):
            continue
        metric_name = k.split("/", 1)[1] if "/" in k else k

        # Wandb image plot.
        plot_fn = _CURVE_IMAGE_LOGGERS.get(metric_name)
        if plot_fn is not None:
            if (
                payload.get("start_indices")
                and payload.get("fvd")
                and log_image is not None
            ):
                try:
                    plot_fn(log_image, k, payload, task, loader_idx)
                except Exception as e:
                    rank_zero_print(
                        cyan(f"[curve metrics] failed to log image for {k}: {e}")
                    )

        # W&B table + line chart for FVD-over-time. This complements the image
        # because media images can be easy to miss in the W&B UI.
        if metric_name == "fvd_over_time":
            if wandb_experiment is not None:
                try:
                    _log_fvd_over_time_wandb_chart(
                        wandb_experiment, k, payload, task, loader_idx, step
                    )
                except Exception as e:
                    rank_zero_print(
                        cyan(f"[curve metrics] failed to log wandb chart for {k}: {e}")
                    )

        # Local JSON dump.
        if raw_dir:
            try:
                out_dir = os.path.join(raw_dir, metric_name)
                os.makedirs(out_dir, exist_ok=True)
                loader_tag = f"_loader{loader_idx}" if loader_idx is not None else ""
                step_int = int(step) if step is not None else 0
                fname = f"{task}{loader_tag}_step{step_int:08d}.json"
                out_path = os.path.join(out_dir, fname)
                with open(out_path, "w") as f:
                    json.dump(jsonify_curve_payload(payload), f, indent=2)
            except Exception as e:
                rank_zero_print(
                    cyan(f"[curve metrics] failed to dump JSON for {k}: {e}")
                )
