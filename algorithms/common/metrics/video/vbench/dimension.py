from abc import ABC, abstractmethod
from typing import Any, Optional
from torch import Tensor
from torch import nn
from algorithms.common.metrics.video.shared_registry import (
    SharedVideoMetricModelRegistry,
)


class Dimension(nn.Module, ABC):
    """
    Base class for evaluation dimensions in VBench.
    """

    def __init__(
        self,
        registry: SharedVideoMetricModelRegistry,
        vbench_metrics_frame_batch_size: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.registry = registry
        if vbench_metrics_frame_batch_size is not None:
            vbench_metrics_frame_batch_size = int(vbench_metrics_frame_batch_size)
            if vbench_metrics_frame_batch_size <= 0:
                raise ValueError(
                    "vbench_metrics_frame_batch_size must be positive or None, "
                    f"got {vbench_metrics_frame_batch_size}"
                )
        self.vbench_metrics_frame_batch_size = vbench_metrics_frame_batch_size

    @abstractmethod
    def forward(self, videos: Tensor) -> Tensor:
        """
        Compute the dimension score.
        Args:
            videos: Videos of shape (B, T, C, H, W), uint8, range [0, 255].
        Returns:
            The computed dimension score of shape (B,) of range [0, 1].
        """
        raise NotImplementedError
