import torch
from omegaconf import DictConfig

from algorithms.denoising import create_eq_algo
from algorithms.denoising.equilibrium_matching import EquilibriumMatching
from algorithms.noiselevelpred_video import NoiseLevelPredVideo


class NoiseLevelPredEqFVideo(NoiseLevelPredVideo):
    """
    Timestep-prediction variant of EqForcingVideo.
    """

    denoising_algo: EquilibriumMatching

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def create_denoising_algo(self) -> torch.nn.Module:
        return create_eq_algo(
            num_sampling_steps=self.num_sampling_steps,
            num_noise_levels=self.num_noise_levels,
            mean_type=self.denoising_cfg.mean_type,
            cfg=self.denoising_cfg,
            logger=self.logger,
        )
