from typing import Callable, Any, Optional
import torch
from torch import Tensor
from einops import rearrange


def chunked_forward(
    func: Callable[..., Tensor],
    *tensor_args: Tensor,
    chunk_size: Optional[int] = None,
    **kwargs: Any,
) -> Tensor:
    """
    Run a tensor batch through func in chunks along dim 0.
    """
    if chunk_size is None:
        return func(*tensor_args, **kwargs)
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive or None, got {chunk_size}")
    if not tensor_args:
        return func(**kwargs)

    batch_size = tensor_args[0].shape[0]
    for tensor in tensor_args[1:]:
        if tensor.shape[0] != batch_size:
            raise ValueError(
                "All chunked tensor arguments must have the same dim-0 size. "
                f"Got {batch_size} and {tensor.shape[0]}."
            )
    if batch_size <= chunk_size:
        return func(*tensor_args, **kwargs)

    outputs = []
    tensor_splits = [tensor.split(chunk_size, dim=0) for tensor in tensor_args]
    for chunk_args in zip(*tensor_splits):
        outputs.append(func(*chunk_args, **kwargs))
    return torch.cat(outputs, dim=0)


def videos_as_images(
    func: Callable[..., Tensor], num_video_args: int = 1
) -> Callable[..., Tensor]:
    """
    Wrapper that enables a function that operates on a batch of images to operate on a batch of videos.
    Can also be used as a decorator.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Tensor:
        # check if the first argument is a tensor or not
        # if not, assume it is "self"
        new_args = list(args)
        videos_idx = 1 if not isinstance(new_args[0], Tensor) else 0
        b = new_args[videos_idx].shape[0]
        for idx in range(videos_idx, videos_idx + num_video_args):
            new_args[idx] = rearrange(new_args[idx], "b t c h w -> (b t) c h w")
        return rearrange(
            func(*new_args, **kwargs),
            "(b t) ... -> b t ...",
            b=b,
        )

    return wrapper
