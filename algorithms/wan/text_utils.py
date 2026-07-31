from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_single_prompt_embed(path: str | Path) -> torch.Tensor:
    obj: Any = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("prompt_embeds", "prompt_embed", "context", "embedding", "embeds"):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise ValueError(
                f"Could not find a prompt embedding tensor in {path}; "
                f"available keys are {list(obj.keys())}."
            )
    if isinstance(obj, (list, tuple)):
        if len(obj) != 1:
            raise ValueError(
                f"Expected exactly one prompt embedding in {path}, got {len(obj)}."
            )
        obj = obj[0]
    if not torch.is_tensor(obj):
        raise TypeError(f"Expected {path} to contain a tensor, got {type(obj)}.")
    if obj.ndim == 3 and obj.shape[0] == 1:
        obj = obj[0]
    if obj.ndim != 2:
        raise ValueError(
            f"Expected a single prompt embedding with shape (L, D), got {tuple(obj.shape)}."
        )
    return obj.detach().cpu()


def repeat_prompt_embed(
    prompt_embed: torch.Tensor,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    prompt_embed = prompt_embed.to(device=device, dtype=dtype)
    return [prompt_embed for _ in range(batch_size)]
