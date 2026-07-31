from omegaconf import DictConfig
import torch
# Diffusion Models
from algorithms.denoising_video import DenoisingVideo
from algorithms.denoising import create_diffusion_algo
from algorithms.denoising.gaussian_diffusion import GaussianDiffusion

class DFoTVideo(DenoisingVideo):
    """
    An algorithm for training and evaluating
    World Models' memory ability on video datasets on different tasks
    """
    
    denoising_algo: GaussianDiffusion
    
    def __init__(self, cfg: DictConfig) -> None:
        
        super().__init__(cfg)
    
    
    def create_denoising_algo(self) -> torch.nn.Module:
        """
        Create denoising algorithm, abstract method to be overridden by downstream classes
        """
        return create_diffusion_algo(
                num_sampling_steps=self.num_sampling_steps,
                num_noise_levels=self.num_noise_levels,
                noise_schedule = self.denoising_cfg.noise_schedule,
                mean_type = self.denoising_cfg.mean_type,
                var_type = self.denoising_cfg.var_type,
                cfg = self.denoising_cfg,
                logger = self.logger
            )
