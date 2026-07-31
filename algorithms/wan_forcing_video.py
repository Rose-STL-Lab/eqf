from __future__ import annotations

import math
from typing import Any

import torch
from einops import rearrange
from omegaconf import DictConfig

from algorithms.wan.text_utils import repeat_prompt_embed
from .wan_forcing_base import BaseWanForcingVideo, PromptConditionedDenoiser


class WanDenoisingModel(PromptConditionedDenoiser):
    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        conditions: torch.Tensor | None = None,
        *,
        prompt_embed_key: str = "positive",
        **_: Any,
    ) -> torch.Tensor:
        """
        Repo-facing denoiser interface.

        Inputs are repo-layout latents `(B, T, C, H, W)`. Wan's raw backbone
        predicts `noise - data`; repo FlowMatching consumes `data - noise`, so
        this adapter returns the sign-corrected repo velocity.
        """

        del conditions
        context = self._select_prompt_embeds(prompt_embed_key)

        x_wan = rearrange(x, "b t c h w -> b c t h w").to(
            device=self.device, dtype=self.dtype
        )
        context = [u.to(device=self.device, dtype=self.dtype) for u in context]
        _, _, n_frames, height, width = x_wan.shape
        patch_t, patch_h, patch_w = tuple(self.backbone.patch_size)
        seq_len = (
            math.ceil(n_frames / patch_t)
            * math.ceil(height / patch_h)
            * math.ceil(width / patch_w)
        )

        # WanModel is trained to predict (noise-data), while our repo is built upon (data-noise).
        # However both use the convention that x_k = (1 - k) x + k \epsilon
        # This is why we don't need to flip the k convention when passing k into the model (for flow matching)
        pred = -1 * self.backbone(
            x_wan,
            t=k,
            context=context,
            seq_len=seq_len,
            clip_fea=None,
            y=None,
        )
        return rearrange(pred, "b c t h w -> b t c h w")


class WanForcingVideo(BaseWanForcingVideo):
    """
    Wan2.1 training/inference on the shared eqf diffusion-forcing stack.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _make_denoising_model(self, backbone: torch.nn.Module) -> WanDenoisingModel:
        return WanDenoisingModel(backbone)

    def _prepare_cached_prompt_embeds(
        self,
        prompt_embeds: Any,
        prompt_embed_lens: Any | None = None,
    ) -> list[torch.Tensor]:
        if not torch.is_tensor(prompt_embeds):
            raise TypeError(
                "Wan cached prompt embeddings must be a tensor, "
                f"got {type(prompt_embeds)}."
            )
        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if prompt_embeds.ndim != 3:
            raise ValueError(
                "Wan cached prompt embeddings must have shape (B, L, D), "
                f"got {tuple(prompt_embeds.shape)}."
            )

        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.model_dtype)
        if prompt_embed_lens is None:
            prompt_embed_lens = torch.full(
                (prompt_embeds.shape[0],),
                prompt_embeds.shape[1],
                device=prompt_embeds.device,
                dtype=torch.long,
            )
        else:
            prompt_embed_lens = torch.as_tensor(
                prompt_embed_lens,
                device=prompt_embeds.device,
                dtype=torch.long,
            ).flatten()

        if prompt_embed_lens.numel() != prompt_embeds.shape[0]:
            raise ValueError(
                "Wan cached prompt_embed_len must have one value per prompt "
                f"embedding, got {prompt_embed_lens.numel()} for batch "
                f"{prompt_embeds.shape[0]}."
            )

        max_len = prompt_embeds.shape[1]
        return [
            embed[: int(seq_len)]
            for embed, seq_len in zip(
                prompt_embeds,
                prompt_embed_lens.clamp(min=0, max=max_len),
            )
        ]

    def encode_text(self, texts: list[str]) -> list[torch.Tensor]:
        if self._force_null_prompt:
            return repeat_prompt_embed(
                self._require_null_prompt_embed("`force_null_prompt=true`"),
                len(texts),
                device=self.device,
                dtype=self.model_dtype,
            )
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        seq_lens = mask.gt(0).sum(dim=1).long()
        text_encoder = self.__dict__["text_encoder"]

        if self.text_encoder_device.type == "cpu":
            text_encoder.to(self.text_encoder_device)
            ids = ids.to(self.text_encoder_device)
            mask = mask.to(self.text_encoder_device)
            context = text_encoder(ids, mask)
            return [u[:v].to(self.device) for u, v in zip(context, seq_lens)]

        text_encoder.to(self.device)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        context = text_encoder(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]


__all__ = ["WanDenoisingModel", "WanForcingVideo"]
