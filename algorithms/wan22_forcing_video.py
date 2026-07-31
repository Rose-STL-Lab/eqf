from __future__ import annotations

from typing import Any

import torch
from einops import rearrange
from omegaconf import DictConfig

from .wan_forcing_base import BaseWanForcingVideo, PromptConditionedDenoiser


class Wan22DenoisingModel(PromptConditionedDenoiser):
    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        conditions: torch.Tensor | None = None,
        *,
        prompt_embed_key: str = "positive",
        fuse_vae_embedding_in_latents: bool = False,
        return_hidden_states: bool = False,
        hidden_states_tap_every: int = 1,
        **_: Any,
    ) -> torch.Tensor:
        """
        Repo-facing Wan2.2 denoiser interface.

        The repo trains latents as `(B, T, C, H, W)`, while Wan2.2 expects
        `(B, C, T, H, W)` and stacked text context. The backbone predicts
        `noise - data`, so this returns the repo velocity `data - noise`.

        When `return_hidden_states` is set (noise-level readout training), the
        backbone returns tapped hidden states alongside the prediction; we
        forward them as `(pred, additional_output)` for the denoising algo.
        """

        del conditions
        context = self._select_prompt_embeds(prompt_embed_key)
        x_wan = rearrange(x, "b t c h w -> b c t h w").to(
            device=self.device, dtype=self.dtype
        )
        context = context.to(device=self.device, dtype=self.dtype)
        timestep = k.to(device=self.device, dtype=self.dtype)
        backbone_output = self.backbone(
            x_wan,
            timestep=timestep,
            context=context,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
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


class Wan22ForcingVideo(BaseWanForcingVideo):
    """
    Wan2.2-TI2V-5B training/inference on the shared diffusion-forcing stack.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _make_denoising_model(self, backbone: torch.nn.Module) -> Wan22DenoisingModel:
        return Wan22DenoisingModel(backbone)

    def _prepare_cached_prompt_embeds(
        self,
        prompt_embeds: Any,
        prompt_embed_lens: Any | None = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(prompt_embeds):
            raise TypeError(
                "Wan22 cached prompt embeddings must be a tensor, "
                f"got {type(prompt_embeds)}."
            )
        if prompt_embeds.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if prompt_embeds.ndim != 3:
            raise ValueError(
                "Wan22 cached prompt embeddings must have shape (B, L, D), "
                f"got {tuple(prompt_embeds.shape)}."
            )

        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.model_dtype)
        if prompt_embed_lens is None:
            return prompt_embeds

        prompt_embed_lens = torch.as_tensor(
            prompt_embed_lens,
            device=prompt_embeds.device,
            dtype=torch.long,
        ).flatten()
        if prompt_embed_lens.numel() != prompt_embeds.shape[0]:
            raise ValueError(
                "Wan22 cached prompt_embed_len must have one value per prompt "
                f"embedding, got {prompt_embed_lens.numel()} for batch "
                f"{prompt_embeds.shape[0]}."
            )

        prompt_embeds = prompt_embeds.clone()
        max_len = prompt_embeds.shape[1]
        for i, seq_len in enumerate(prompt_embed_lens.clamp(min=0, max=max_len)):
            prompt_embeds[i, int(seq_len) :] = 0
        return prompt_embeds

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        if self._force_null_prompt:
            return (
                self._require_null_prompt_embed("`force_null_prompt=true`")
                .to(device=self.device, dtype=self.model_dtype)
                .unsqueeze(0)
                .expand(len(texts), -1, -1)
            )

        if self.tokenizer is None:
            raise ValueError(
                "Wan22ForcingVideo.encode_text requires a live tokenizer; disable "
                "`algorithm.load_prompt_embed` or provide cached prompt embeddings."
            )
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        text_encoder = self.__dict__["text_encoder"]
        if text_encoder is None:
            raise ValueError(
                "Wan22ForcingVideo.encode_text requires a live text encoder; disable "
                "`algorithm.load_prompt_embed` or provide cached prompt embeddings."
            )

        if self.text_encoder_device.type == "cpu":
            text_encoder.to(self.text_encoder_device)
            ids = ids.to(self.text_encoder_device)
            mask = mask.to(self.text_encoder_device)
            context = text_encoder(ids, mask)
        else:
            text_encoder.to(self.device)
            ids = ids.to(self.device)
            mask = mask.to(self.device)
            context = text_encoder(ids, mask)

        for i, seq_len in enumerate(mask.gt(0).sum(dim=1).long()):
            context[i, seq_len:] = 0
        return context.to(device=self.device, dtype=self.model_dtype)


__all__ = ["Wan22DenoisingModel", "Wan22ForcingVideo"]
