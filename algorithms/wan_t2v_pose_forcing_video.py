from __future__ import annotations

from typing import Any

import torch
from einops import rearrange

from .wan_forcing_video import WanDenoisingModel, WanForcingVideo
from .wan_pose_mixin import WanPoseConditioningMixin


class WanPoseDenoisingModel(WanDenoisingModel):
    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        conditions: torch.Tensor | None = None,
        *,
        prompt_embed_key: str = "positive", # use positive prompt by default, usually in training
        return_hidden_states: bool = False,
        hidden_states_tap_every: int = 1,
        **_: Any,
    ) -> torch.Tensor:
        if conditions is None:
            raise ValueError("WanPoseDenoisingModel requires processed pose conditions.")

        context = self._select_prompt_embeds(prompt_embed_key)
        x_wan = rearrange(x, "b t c h w -> b c t h w").to(
            device=self.device, dtype=self.dtype
        )
        context = [u.to(device=self.device, dtype=self.dtype) for u in context]
        _, _, n_frames, height, width = x_wan.shape
        patch_t, patch_h, patch_w = tuple(self.backbone.patch_size)
        seq_len = (
            ((n_frames + patch_t - 1) // patch_t)
            * ((height + patch_h - 1) // patch_h)
            * ((width + patch_w - 1) // patch_w)
        )
        # `conditions` already lives on the WAN latent grid:
        #   - global: (B, T_lat, 12, 1, 1)
        #   - ray/plucker: (B, T_lat, 6, H_lat, W_lat)
        #   - ray_encoding: (B, T_lat, 180, H_lat, W_lat)
        pose = conditions.to(device=self.device, dtype=self.dtype)

        backbone_output = self.backbone(
            x_wan,
            t=k,
            context=context,
            seq_len=seq_len,
            clip_fea=None,
            y=None,
            pose=pose,
            return_hidden_states=return_hidden_states,
            hidden_states_tap_every=hidden_states_tap_every,
        )
        additional_output = None
        if isinstance(backbone_output, tuple):
            pred, additional_output = backbone_output
        else:
            pred = backbone_output
        pred = -1 * pred
        pred = rearrange(pred, "b c t h w -> b t c h w")
        if additional_output is not None:
            return pred, additional_output
        return pred


class WanT2VPoseForcingVideo(WanPoseConditioningMixin, WanForcingVideo):
    """
    Pose-conditioned WAN on the shared FlowF stack.
    """

    def _load_pretrained_backbone(self, backbone: torch.nn.Module, path: str):
        # WanModelPose adds pose-specific parameters that do not exist in the
        # original Wan checkpoint family. Use its custom loader so we can copy
        # base Wan weights while leaving pose-only parameters freshly initialized.
        if hasattr(backbone, "load_pretrained_base_weights"):
            backbone.load_pretrained_base_weights(path)
            return backbone
        return backbone.from_pretrained(path)

    def _make_denoising_model(self, backbone: torch.nn.Module) -> WanPoseDenoisingModel:
        return WanPoseDenoisingModel(backbone)


__all__ = ["WanT2VPoseForcingVideo"]
