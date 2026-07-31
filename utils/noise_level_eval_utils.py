from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _fig_to_image(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = buf.reshape(height, width, 4)[..., :3].copy()
    plt.close(fig)
    return image


def make_noise_level_abs_delta_error_curve(
    *,
    x_values: Sequence[float],
    mean_error: Sequence[float],
    counts: Sequence[float],
    title: str,
    xlabel: str = "|fake time - true time|",
    ylabel: str = "time prediction error",
) -> np.ndarray:
    x = np.asarray(x_values, dtype=np.float32)
    y = np.asarray(mean_error, dtype=np.float32)
    c = np.asarray(counts, dtype=np.float32)
    valid = np.isfinite(y) & (c > 0)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if valid.any():
        ax.plot(x[valid], y[valid], marker="o", linewidth=2.0)
    else:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _fig_to_image(fig)


def make_noise_level_grid_error_matrix(
    *,
    grid_values: Sequence[float],
    mean_error_matrix: np.ndarray,
    title: str,
    xlabel: str = "fake time",
    ylabel: str = "true time",
    colorbar_label: str = "time prediction error",
) -> np.ndarray:
    values = [f"{float(v):.2f}" for v in grid_values]
    n_values = len(values)
    fig_width = max(6.5, min(12.0, 0.55 * max(n_values, 1) + 1.5))
    fig_height = max(5.5, min(11.0, 0.50 * max(n_values, 1) + 1.5))
    font_size = max(4.5, min(10.0, 180.0 / max(n_values, 1) / 2.0))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(mean_error_matrix, interpolation="nearest", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(values)))
    ax.set_yticks(np.arange(len(values)))
    ax.set_xticklabels(values, rotation=45, ha="right")
    ax.set_yticklabels(values)
    ax.set_xticks(np.arange(-0.5, len(values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(values), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(mean_error_matrix.shape[0]):
        for col_idx in range(mean_error_matrix.shape[1]):
            cell_value = float(mean_error_matrix[row_idx, col_idx])
            if np.isfinite(cell_value):
                label = f"{cell_value:.3f}"
            else:
                label = "-"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                color="red",
                fontsize=font_size,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.55,
                    "edgecolor": "none",
                    "pad": 0.2,
                },
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    return _fig_to_image(fig)


def reduce_grid_abs_delta_curve(
    *,
    grid_values: Sequence[float],
    sum_error_matrix: np.ndarray,
    count_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(grid_values, dtype=np.float32)
    delta_to_sum: dict[float, float] = {}
    delta_to_count: dict[float, float] = {}
    for true_idx, true_value in enumerate(values):
        for fake_idx, fake_value in enumerate(values):
            count = float(count_matrix[true_idx, fake_idx])
            if count <= 0:
                continue
            delta = float(abs(fake_value - true_value))
            delta_to_sum[delta] = delta_to_sum.get(delta, 0.0) + float(
                sum_error_matrix[true_idx, fake_idx]
            )
            delta_to_count[delta] = delta_to_count.get(delta, 0.0) + count

    if not delta_to_sum:
        return (
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
        )

    deltas = np.asarray(sorted(delta_to_sum.keys()), dtype=np.float32)
    counts = np.asarray([delta_to_count[d] for d in deltas], dtype=np.float32)
    mean_errors = np.asarray(
        [delta_to_sum[d] / max(delta_to_count[d], 1.0) for d in deltas],
        dtype=np.float32,
    )
    return deltas, mean_errors, counts
