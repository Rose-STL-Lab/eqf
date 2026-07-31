from functools import partial
from typing import Optional
import torch
from torch import nn, Tensor
from types import SimpleNamespace
from algorithms.common.metrics.video.utils import chunked_forward, videos_as_images
from utils.huggingface_utils import download_from_hf
from .raft import RAFT as _RAFT
from .utils import InputPadder


class RAFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _RAFT(
            SimpleNamespace(
                small=False, mixed_precision=False, alternate_corr=False
            )
        )

        state_dict = torch.load(
            download_from_hf("metrics_models/raft-things.pth"), map_location="cpu"
        )  # comes from https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
        self.model.load_state_dict(
            {k.replace("module.", ""): v for k, v in state_dict.items()}
        )

    def train(self, mode: bool) -> "RAFT":
        super().train(False)

    @partial(videos_as_images, num_video_args=2)
    def forward(
        self,
        images1: Tensor,
        images2: Tensor,
        iters: int,
        frame_batch_size: Optional[int] = None,
    ) -> Tensor:
        return chunked_forward(
            lambda chunk1, chunk2: self.model(
                chunk1, chunk2, iters=iters, test_mode=True
            )[1],
            images1,
            images2,
            chunk_size=frame_batch_size,
        )
