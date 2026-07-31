from omegaconf import DictConfig
import torch
from algorithms.flowf_video import FlowForcingVideo
from algorithms.denoising import create_eq_algo
from algorithms.denoising.equilibrium_matching import EquilibriumMatching

class EqForcingVideo(FlowForcingVideo):

    denoising_algo: EquilibriumMatching

    """
    An algorithm for training and evaluating equilibrium forcing for video generation.
    """
    def __init__(self, cfg: DictConfig) -> None:

        super().__init__(cfg)


    def create_denoising_algo(self) -> torch.nn.Module:
        """
        Create denoising algorithm, abstract method to be overridden by downstream classes
        """
        return create_eq_algo(
                num_sampling_steps=self.num_sampling_steps,
                num_noise_levels=self.num_noise_levels,
                mean_type = self.denoising_cfg.mean_type,
                cfg = self.denoising_cfg,
                logger = self.logger
            )  # default (handled internally for scaling): 1000 steps, linear noise schedule
