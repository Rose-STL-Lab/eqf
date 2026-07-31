from __future__ import annotations

import random

import torch

POSE_CONDITIONING_DIMS = {
    "ray": 6,
    "plucker": 6,
    # 3 coords * 15 frequencies * 2 (sin/cos) for origin, plus the same for
    # direction. These channels are produced on the WAN latent spatial grid.
    "ray_encoding": 180,
}


def resolve_pose_conditioning_dim(conditioning_type: str) -> int:
    try:
        return POSE_CONDITIONING_DIMS[str(conditioning_type).lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported WAN pose conditioning type: {conditioning_type}"
        ) from exc


def select_latent_pose_indices(
    num_frames: int,
    stride: int,
    downsample_technique: str,
) -> torch.Tensor:
    """
    Select indices of frames to keep as pose conditions after temporal downsampling in the VAE.
    With a causal VAE, we always keep the first frame. Beyond that
    """
    if num_frames <= 0:
        return torch.empty(0, dtype=torch.long)
    if stride <= 0:
        raise ValueError(f"Temporal stride must be positive, got {stride}.")

    method = str(downsample_technique).lower()
    indices = [0]
    start = 1
    while start < num_frames:
        end = min(start + stride, num_frames)
        if method == "first":
            chosen = start
        elif method == "last":
            chosen = end - 1
        elif method == "random":
            chosen = random.randint(start, end - 1)
        else:
            raise ValueError(
                f"Unsupported pose downsample technique: {downsample_technique}"
            )
        indices.append(chosen)
        start += stride
    return torch.tensor(indices, dtype=torch.long)
