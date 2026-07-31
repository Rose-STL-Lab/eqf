from omegaconf import DictConfig
import torch
# Diffusion Models
from algorithms.denoising_video import DenoisingVideo
from algorithms.denoising import create_flow_algo
from algorithms.denoising.flow_matching import FlowMatching


class FlowForcingVideo(DenoisingVideo):
    """
    An algorithm for training and evaluating flow forcing for video generation.
    """
    
    denoising_algo: FlowMatching
    
    def __init__(self, cfg: DictConfig) -> None:
        
        super().__init__(cfg)
    
    
    def create_denoising_algo(self) -> torch.nn.Module:
        """
        Create denoising algorithm, abstract method to be overridden by downstream classes
        """
        return create_flow_algo(
                num_sampling_steps=self.num_sampling_steps,
                num_noise_levels=self.num_noise_levels,
                mean_type = self.denoising_cfg.mean_type,
                cfg = self.denoising_cfg,
                logger = self.logger
            ) # default (handled internally for scaling): 1000 steps, linear noise schedule


