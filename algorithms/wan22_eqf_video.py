import torch

from algorithms.denoising import create_eq_algo
from algorithms.denoising.equilibrium_matching import EquilibriumMatching
from .wan22_forcing_video import Wan22ForcingVideo


class Wan22EqFVideo(Wan22ForcingVideo):
    """
    EqF variant of Wan22ForcingVideo.
    """

    denoising_algo: EquilibriumMatching

    def create_denoising_algo(self) -> torch.nn.Module:
        return create_eq_algo(
            num_sampling_steps=self.num_sampling_steps,
            num_noise_levels=self.num_noise_levels,
            mean_type=self.denoising_cfg.mean_type,
            cfg=self.denoising_cfg,
            logger=self.logger,
        )


__all__ = ["Wan22EqFVideo"]
