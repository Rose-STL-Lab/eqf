from typing import Optional

import torch
from torch import Tensor

from omegaconf import DictConfig

class PropertyMixin:

    # ---------------------------------------------------------------------
    # NOTE: n_{frames, tokens} indicates the number of frames/tokens
    # that the model actually processes during training/validation.
    # During validation, it may be different from max_{frames, tokens},
    # ---------------------------------------------------------------------
    @property
    def num_sampling_steps(self) -> int:
        """
        Number of *denoising steps* during sampling.
        """
        return int(self.cfg.denoising.num_sampling_steps)

    @property
    def num_sampling_noise_levels(self) -> int:
        """
        Number of noise levels (including both endpoints) for sampling schedules.
        By convention: `num_sampling_noise_levels = num_sampling_steps + 1`.
        """
        return int(self.num_sampling_steps) + 1

    @property
    def sampling_scheduler_type(self) -> str:
        return str(self.cfg.denoising.sampling_algorithm).strip().lower()

    @property
    def training_n_context_frames(self) -> int:
        if self.cfg.fixed_context.enabled or self.cfg.uniform_future.enabled:
            if self.cfg.training_context_frames is None:
                raise ValueError(
                    "training_context_frames must be set when "
                    "fixed_context or uniform_future is enabled."
                )
            return int(self.cfg.training_context_frames)
        elif self.cfg.variable_context.enabled:
            return 0
        else: # standard no history diffusion forcing
            if int(self.cfg.training_context_frames) != 0:
                raise ValueError("training_context_frames must be set to 0 if using standard no history diffusion forcing during training")
            return 0

    @property
    def validation_n_initial_context_frames(self) -> int:
        if self.cfg.tasks.prediction.streaming.initial_context_frames is None:
            return self.validation_n_sliding_context_frames  # no initial context defaults to sliding
        if self.cfg.fixed_context.enabled:
            assert self.cfg.tasks.prediction.streaming.initial_context_frames == self.training_n_context_frames, "if fixed context then must have same train/val context"
        return int(self.cfg.tasks.prediction.streaming.initial_context_frames)

    @property
    def validation_n_sliding_context_frames(self) -> int:
        if self.cfg.tasks.prediction.streaming.sliding_context_frames is None:
            raise ValueError("sliding_context_frames must be set in prediction.streaming")
        if self.cfg.fixed_context.enabled:
            assert self.cfg.tasks.prediction.streaming.sliding_context_frames == self.training_n_context_frames, "if fixed context then must have same train/val context"
        return int(self.cfg.tasks.prediction.streaming.sliding_context_frames)

    @property
    def training_n_context_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.training_n_context_frames)

    @property
    def validation_n_initial_context_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.validation_n_initial_context_frames)

    @property
    def validation_n_sliding_context_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.validation_n_sliding_context_frames)

    @property
    def prediction_sampling_strategy(self) -> str:
        prediction_cfg = self.cfg.tasks.prediction
        if isinstance(prediction_cfg, DictConfig):
            strategy = prediction_cfg.get("sampling_strategy", "streaming")
        else:
            strategy = getattr(prediction_cfg, "sampling_strategy", "streaming")
        return str(strategy).strip().lower()

    def get_prediction_context_mask(
        self,
        n_frames: int,
        *,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        This is used for metrics, which are in frame space.
        """
        n_frames = int(n_frames)
        device = device if device is not None else self.device
        context_mask = torch.zeros(n_frames, dtype=torch.bool, device=device)

        match self.prediction_sampling_strategy:
            case "streaming":
                context_mask[: min(self.validation_n_initial_context_frames, n_frames)] = True
            case strategy:
                raise NotImplementedError(
                    f"Prediction context helper does not support sampling strategy `{strategy}`."
                )

        return context_mask

    def get_prediction_context_indices(
        self,
        n_frames: int,
        *,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        return torch.nonzero(
            self.get_prediction_context_mask(n_frames, device=device),
            as_tuple=False,
        ).flatten()
    
    # ---------------------------------------------------------------------
    # NOTE: max_{frames, tokens} indicates the maximum number of frames/tokens
    # that the model can process within a single forward pass.
    # ---------------------------------------------------------------------
    
    @property
    def forward_window_size_in_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.forward_window_size_in_frames)
    
    @property
    def forward_window_size_in_frames(self):
        return self.backbone_cfg.forward_window_size
    

    def _n_frames_to_n_tokens(self, n_frames: int) -> int:
        """
        Converts the number of frames to the number of tokens.
        - Chunk-wise VideoVAE: 1st frame -> 1st token, then every self.temporal_downsampling_factor frames -> next token.
        - ImageVAE or Non-latent Diffusion: 1 token per frame.
        """
        if self.temporal_downsampling_factor == 1:
            return n_frames
        else: # temporal video vae
            if self.vae.is_causal:
                return (n_frames - 1) // self.temporal_downsampling_factor + 1
            else: # non-causal video VAE
                # Non-causal VideoVAEs typically pad time to a multiple of some temporal block size
                # (often equal to the temporal downsampling factor), then encode. To keep all
                # downstream shapes consistent, we mirror that padding behavior here.
                temporal_pixel_length = getattr(self.vae, "temporal_pixel_length", None)
                if temporal_pixel_length is not None and int(temporal_pixel_length) > 0:
                    t_pix = int(temporal_pixel_length)
                    padded_frames = ((int(n_frames) + t_pix - 1) // t_pix) * t_pix
                    return padded_frames // self.temporal_downsampling_factor
                # Fallback: pad to the temporal downsampling factor.
                return (int(n_frames) + self.temporal_downsampling_factor - 1) // self.temporal_downsampling_factor
        
