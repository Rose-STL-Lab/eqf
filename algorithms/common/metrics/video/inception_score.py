from typing import Any
import torch
from torch import Tensor
from torchmetrics import Metric


class InceptionScore(Metric):
    """
    Calculates Inception Score (IS) to evaluate how realistic the videos are.
    Requires a batch of videos of shape (B, num_classes).
    Adapted from `torchmetrics.image.InceptionScore` to work with videos.
    """

    higher_is_better: bool = True
    is_differentiable: bool = False
    full_state_update: bool = False
    orig_dtype: torch.dtype

    def __init__(self, num_classes: int = 400, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_state(
            "prob_sum", torch.zeros(num_classes).double(), dist_reduce_fx="sum"
        )
        self.add_state("num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")
        self.add_state(
            "neg_entropy_sum", torch.tensor(0.0).double(), dist_reduce_fx="sum"
        )

    @property
    def is_empty(self) -> bool:
        # pylint: disable=no-member
        # Count samples GLOBALLY (summed across distributed ranks): compute() sums
        # the state across ranks, and reducing here keeps the include/exclude
        # decision identical on every rank, preventing a collective desync.
        num = self.num_samples
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            num = num.detach().reshape(())
            # Run the collective on CUDA when the process group is NCCL-only;
            # never-updated states may still live on CPU, which NCCL can't reduce.
            if num.device.type == "cpu" and torch.cuda.is_available():
                num = num.to(torch.device("cuda", torch.cuda.current_device()))
            else:
                num = num.clone()
            torch.distributed.all_reduce(num, op=torch.distributed.ReduceOp.SUM)
        return bool(num == 0)

    def update(self, features: Tensor) -> None:
        """
        Update the state with extracted features.
        Args:
            features: Features of shape (B, num_classes). The features are logits extracted from a video classifier (e.g. I3D, C3D).
        """
        # pylint: disable=no-member
        self.orig_dtype = features.dtype
        features = features.double()
        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)
        self.num_samples += features.size(0)
        self.prob_sum += prob.sum(dim=0)
        self.neg_entropy_sum += (prob * log_prob).sum()

    def compute(self) -> Tensor:
        """
        Compute the Inception Score (IS).
        Returns:
            The computed IS.
        """
        # pylint: disable=no-member
        mean_prob = self.prob_sum / self.num_samples
        # calculate KL divergence
        kl = (
            self.neg_entropy_sum / self.num_samples
            - (mean_prob * mean_prob.log()).sum()
        )
        # NOTE: avoid casting to self.orig_dtype here. Under Lightning mixed-precision
        # the feature extractor runs inside autocast and emits bf16/fp16, which would
        # quantize the final IS scalar to that dtype's spacing (e.g. multiples of
        # 1/16 for bf16 values in [8, 16)). Return float32 unconditionally.
        return kl.exp().float()
