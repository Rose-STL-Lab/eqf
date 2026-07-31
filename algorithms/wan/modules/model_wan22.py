from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import torch
from einops import rearrange


def _load_state_dict_file(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_state_dict_path(path_or_pattern: str) -> dict[str, torch.Tensor]:
    path = Path(path_or_pattern)
    if path.is_dir():
        candidates = sorted(path.glob("diffusion_pytorch_model*.safetensors"))
        if not candidates:
            candidates = sorted(path.glob("*.safetensors"))
        if not candidates:
            candidates = sorted(path.glob("*.pth"))
    else:
        candidates = [Path(p) for p in sorted(glob.glob(str(path_or_pattern)))]
        if not candidates and path.exists():
            candidates = [path]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found for {path_or_pattern}")

    merged: dict[str, torch.Tensor] = {}
    for candidate in candidates:
        state = _load_state_dict_file(str(candidate))
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        merged.update(state)
    return merged


class Wan22DiffSynthModel(torch.nn.Module):
    def __init__(
        self,
        *,
        has_image_input: bool = False,
        patch_size: list[int] | tuple[int, int, int] = (1, 2, 2),
        in_dim: int = 48,
        dim: int = 3072,
        ffn_dim: int = 14336,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 48,
        num_heads: int = 24,
        num_layers: int = 30,
        eps: float = 1e-6,
        seperated_timestep: bool = True,
        require_clip_embedding: bool = False,
        require_vae_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
    ) -> None:
        super().__init__()
        from . import wan22_dit as wan_video_dit
        from .wan22_dit import WanModel

        if not torch.cuda.is_available():
            wan_video_dit.FLASH_ATTN_3_AVAILABLE = False
            wan_video_dit.FLASH_ATTN_2_AVAILABLE = False
            wan_video_dit.SAGE_ATTN_AVAILABLE = False

        patch_size_tuple = tuple(patch_size)
        if len(patch_size_tuple) != 3:
            raise ValueError(f"Wan2.2 patch_size must have length 3, got {patch_size}.")
        patch_size_tuple = (
            int(patch_size_tuple[0]),
            int(patch_size_tuple[1]),
            int(patch_size_tuple[2]),
        )

        self.model = WanModel(
            has_image_input=has_image_input,
            patch_size=patch_size_tuple,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            eps=eps,
            seperated_timestep=seperated_timestep,
            require_clip_embedding=require_clip_embedding,
            require_vae_embedding=require_vae_embedding,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        self.patch_size = patch_size_tuple
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.dim = int(dim)
        self.freq_dim = int(freq_dim)
        self.seperated_timestep = bool(seperated_timestep)
        self.fuse_vae_embedding_in_latents = bool(fuse_vae_embedding_in_latents)
        self._use_gradient_checkpointing = False

    def gradient_checkpointing_enable(self, p: float = 1.0) -> None:
        del p
        self._use_gradient_checkpointing = True

    def from_pretrained(self, path: str) -> "Wan22DiffSynthModel":
        state_dict = load_state_dict_path(path)
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                key = key[len("model.") :]
            normalized[key] = value
        missing, unexpected = self.model.load_state_dict(normalized, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Failed to load Wan2.2 DiT checkpoint strictly enough: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        return self

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        fuse_vae_embedding_in_latents: bool = False,
        return_hidden_states: bool = False,
        hidden_states_tap_every: int = 1,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(dtype=model_dtype)
        context = context.to(dtype=model_dtype)
        timestep = timestep.to(dtype=model_dtype)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward_text_only(
                x=x,
                timestep=timestep,
                context=context,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                return_hidden_states=return_hidden_states,
                hidden_states_tap_every=hidden_states_tap_every,
            )

    def _forward_text_only(
        self,
        *,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        return_hidden_states: bool = False,
        hidden_states_tap_every: int = 1,
    ) -> torch.Tensor:
        from .gradient_checkpoint import gradient_checkpoint_forward
        from .wan22_dit import sinusoidal_embedding_1d

        dit = self.model
        timestep = timestep.to(device=x.device, dtype=x.dtype)
        context = context.to(device=x.device, dtype=x.dtype)

        patched = dit.patchify(x)
        f, h, w = patched.shape[2:]
        hidden = rearrange(patched, "b c f h w -> b (f h w) c").contiguous()
        freqs = torch.cat(
            [
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(hidden.device)

        if timestep.ndim == 2:
            if timestep.shape[1] != f:
                raise ValueError(
                    "Wan2.2 per-frame timestep shape must match latent frames: "
                    f"got {tuple(timestep.shape)} for latent frames={f}."
                )
            timestep_tokens = (
                timestep[:, :, None]
                .expand(timestep.shape[0], f, h * w)
                .reshape(timestep.shape[0], f * h * w)
            )
            time_emb = dit.time_embedding(
                sinusoidal_embedding_1d(dit.freq_dim, timestep_tokens.flatten())
                .to(x.dtype)
            ).view(timestep_tokens.shape[0], timestep_tokens.shape[1], dit.dim)
            t_mod = dit.time_projection(time_emb).unflatten(2, (6, dit.dim))
            head_time = time_emb
        elif dit.seperated_timestep and fuse_vae_embedding_in_latents:
            timestep_tokens = torch.concat(
                [
                    torch.zeros(
                        (1, h * w),
                        dtype=x.dtype,
                        device=x.device,
                    ),
                    torch.ones(
                        (f - 1, h * w),
                        dtype=x.dtype,
                        device=x.device,
                    )
                    * timestep.reshape(-1)[0],
                ]
            ).flatten()
            time_emb = dit.time_embedding(
                sinusoidal_embedding_1d(dit.freq_dim, timestep_tokens).unsqueeze(0)
            )
            t_mod = dit.time_projection(time_emb).unflatten(2, (6, dit.dim))
            head_time = time_emb
        else:
            timestep_flat = timestep.flatten()
            time_emb = dit.time_embedding(
                sinusoidal_embedding_1d(dit.freq_dim, timestep_flat).to(x.dtype)
            )
            t_mod = dit.time_projection(time_emb).unflatten(1, (6, dit.dim))
            head_time = time_emb

        context = dit.text_embedding(context)
        if hidden.shape[0] != context.shape[0]:
            hidden = torch.concat([hidden] * context.shape[0], dim=0)

        # Optional hidden-state tapping for the noise-level readout head. We tap
        # the same block indices the readout head expects (`i % tap_every == 0`,
        # matching algorithms/wan/modules/model_pose.py) and reshape the flat
        # `(b, f*h*w, c)` token sequence into the `(b, t, h, w, c)` layout the
        # readout consumes. Taps are detached so no gradient flows into the
        # frozen denoiser.
        hidden_states_list = [] if return_hidden_states else None
        hidden_states_tap_every = max(1, int(hidden_states_tap_every))
        for i, block in enumerate(dit.blocks):
            if dit.training:
                hidden = gradient_checkpoint_forward(
                    block,
                    self._use_gradient_checkpointing,
                    False,
                    hidden,
                    context,
                    t_mod,
                    freqs,
                )
            else:
                hidden = block(hidden, context, t_mod, freqs)

            if return_hidden_states and (i % hidden_states_tap_every == 0):
                seq_len_expected = f * h * w
                if hidden.shape[1] < seq_len_expected:
                    raise ValueError(
                        "Wan2.2 hidden-state tap reshape expected at least "
                        f"{seq_len_expected} tokens, got {hidden.shape[1]}."
                    )
                hidden_states_5d = hidden[:, :seq_len_expected].reshape(
                    hidden.shape[0], f, h, w, hidden.shape[-1]
                )
                hidden_states_list.append(hidden_states_5d.detach())

        hidden = dit.head(hidden, head_time)
        output = dit.unpatchify(hidden, (f, h, w))  # pyright: ignore[reportArgumentType]
        if return_hidden_states:
            return output, {"hidden_states": hidden_states_list}
        return output


__all__ = ["Wan22DiffSynthModel", "load_state_dict_path"]
