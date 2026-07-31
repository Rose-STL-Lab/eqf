from typing import Optional
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.image import FrechetInceptionDistance as _FrechetInceptionDistance
from .shared_registry import SharedVideoMetricModelRegistry


class BaseFrechetDistance(_FrechetInceptionDistance, ABC):
    """
    Base class for Fréchet distance metrics (e.g. FID, FVD).
    AAdapted from `torchmetrics.image.FrechetInceptionDistance` to work with shared model registry and support different feature extractors and modalities (e.g. images, videos).
    """

    orig_dtype: torch.dtype

    def __init__(
        self,
        registry: Optional[SharedVideoMetricModelRegistry],
        features: int,
        reset_real_features=True,
        **kwargs,
    ):
        # pylint: disable=non-parent-init-called
        Metric.__init__(self, **kwargs)

        self.registry = registry
        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        num_features = features
        mx_nb_feets = (num_features, num_features)
        self.add_state(
            "real_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            torch.zeros(mx_nb_feets).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            torch.zeros(mx_nb_feets).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

    @property
    def is_empty(self) -> bool:
        # pylint: disable=no-member
        # Frechet distances estimate a sample covariance with an `n - 1` denominator,
        # so they require at least 2 samples on each side. Treat <2-sample states as
        # "empty" so VideoMetric.log() silently skips them instead of producing NaN/Inf
        # (e.g. when overfitting on a single clip with num_validation_clips=1).
        #
        # Count samples GLOBALLY (summed across distributed ranks) rather than
        # per-rank: torchmetrics' compute() all-gathers/sums these states across
        # ranks before computing, so the relevant population is the global one
        # (e.g. 2 GPUs x 1 clip each = 2 samples -> computable). Reducing here also
        # guarantees every rank reaches the same include/exclude decision, avoiding
        # a collective desync where some ranks call compute() and others skip it.
        real_n = self.real_features_num_samples
        fake_n = self.fake_features_num_samples
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            counts = torch.stack(
                [real_n.detach().reshape(()), fake_n.detach().reshape(())]
            )
            # Run the collective on CUDA when the process group is NCCL-only;
            # never-updated states may still live on CPU, which NCCL can't reduce.
            if counts.device.type == "cpu" and torch.cuda.is_available():
                counts = counts.to(torch.device("cuda", torch.cuda.current_device()))
            else:
                counts = counts.clone()
            torch.distributed.all_reduce(
                counts, op=torch.distributed.ReduceOp.SUM
            )
            real_n, fake_n = counts[0], counts[1]
        return bool(real_n < 2) or bool(fake_n < 2)

    @abstractmethod
    def extract_features(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _check_input(fake: Tensor, real: Tensor) -> bool:
        return True

    def _update(self, x: Tensor, real: bool) -> None:
        # pylint: disable=no-member
        features = self.extract_features(x)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += features.size(0)
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += features.size(0)

    def update(self, fake: Tensor, real: Tensor) -> None:
        if not self._check_input(fake, real):
            return
        self._update(fake, real=False)
        self._update(real, real=True)

    def compute(self) -> Tensor:
        # NOTE: torchmetrics' parent compute() ends with `.to(self.orig_dtype)`, where
        # `orig_dtype` is captured from the feature extractor output. Under Lightning
        # mixed-precision (bf16/fp16), the feature extractor runs inside autocast and
        # returns low-precision tensors, so the final FVD/FID scalar gets quantized
        # to bf16/fp16 spacing (e.g. 1/16 for bf16 values in [8, 16)). Force the
        # returned scalar to float32 so global statistics aren't lost at the final cast.
        return super().compute().float()
