from __future__ import annotations

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from algorithms.noiselevelpred_video import NoiseLevelPredVideo
from algorithms.readout import NoiseLevelReadout

from .wan_t2v_pose_forcing_video import WanT2VPoseForcingVideo


class WanPoseNoiseLevelPredVideo(NoiseLevelPredVideo, WanT2VPoseForcingVideo):
    """
    Readout-only noise-level predictor on top of the WAN pose forcing stack.
    """

    def _configure_readout_head(self) -> None:
        tap_every = max(1, int(self.readout_cfg.tap_every))
        num_layers = int(self.backbone_cfg.num_layers)
        num_taps = (num_layers + tap_every - 1) // tap_every
        feature_dim = int(self.backbone_cfg.dim)

        self.readout_head = NoiseLevelReadout(
            feature_dim=feature_dim,
            num_layers=num_taps,
            adapter_dim=int(self.readout_cfg.adapter_dim),
            num_mlp_layers=int(self.readout_cfg.num_mlp_layers),
            pooling=self.readout_cfg.pooling,
            per_layer_adapter=bool(self.readout_cfg.per_layer_adapter),
            mix_layers=bool(self.readout_cfg.mix_layers),
            mix_method=self.readout_cfg.mix_method,
            target_type=self.readout_cfg.target,
            loss_type=self.readout_cfg.loss,
            eps=float(self.readout_cfg.eps),
        )
        self.readout_enabled = True

    def training_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
        namespace="training",
        noise_level_override: torch.Tensor | None = None,
        show_noise_level_override: torch.Tensor | None = None,
        ablation_eval_mode: bool = False,
    ) -> STEP_OUTPUT:
        *_, metadata = batch
        with self._prompt_context(metadata):
            return super().training_step(
                batch,
                batch_idx,
                dataloader_idx=dataloader_idx,
                namespace=namespace,
                noise_level_override=noise_level_override,
                show_noise_level_override=show_noise_level_override,
                ablation_eval_mode=ablation_eval_mode,
            )


__all__ = ["WanPoseNoiseLevelPredVideo"]
