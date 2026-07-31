from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint

from .model import WanModel, sinusoidal_embedding_1d


class RandomDropoutPatchEmbed(nn.Module):
    def __init__(
        self,
        pose_dim: int,
        embed_dim: int,
        patch_size: tuple[int, int],
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels=pose_dim,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        nn.init.zeros_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim * 6),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        self.dropout = RandomEmbeddingDropout(p=dropout_prob)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, f, _, _, _ = x.shape
        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.patch_embed(x)
        x = rearrange(x, "(b f) d hp wp -> b (f hp wp) d", b=b, f=f)
        x = self.proj(x)
        return self.dropout(x, mask)


class RandomEmbeddingDropout(nn.Module):
    """
    Randomly nullify the input embeddings with a given probability.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, emb: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Randomly nullify the input embeddings with a probability p during training. For inference, the embeddings are nullified only if mask is provided.
        Args:
            emb: input embeddings of shape (B, ...)
            mask: mask tensor of shape (B, ). Only allowed during inference. If provided, embeddings for masked batches will be zeroed.
        """
        if mask is not None:
            assert not self.training, "embedding mask is only allowed during inference"
            assert mask.ndim == 1, "embedding mask should be of shape (B,)"

        if self.training and self.p > 0:
            mask = torch.rand(emb.shape[:1], device=emb.device) < self.p
        if mask is not None:
            mask = rearrange(mask, "... -> ..." + " 1" * (emb.ndim - 1))
            emb = torch.where(mask, torch.zeros_like(emb), emb)
        return emb # returns the same shape that emb was at the start



class WanModelPose(WanModel):
    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        pose_conditioning_type="ray_encoding",
        pose_dim=180,
        pose_dropout_prob=0.1,
    ):
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )
        self.pose_conditioning_type = str(pose_conditioning_type).lower()
        self.pose_dim = int(pose_dim)

        self.pose_embedding = RandomDropoutPatchEmbed(
            pose_dim=self.pose_dim,
            embed_dim=dim,
            patch_size=(int(self.patch_size[1]), int(self.patch_size[2])),
            dropout_prob=pose_dropout_prob,
        )
                

    def load_pretrained_base_weights(self, path: str) -> None:
        # Base Wan checkpoints do not contain the pose-conditioning parameters
        # introduced by WanModelPose. Load the original backbone weights with
        # strict=False so only the shared Wan weights are restored.
        base_model = WanModel.from_pretrained(path)
        self.load_state_dict(base_model.state_dict(), strict=False)

    def _pose_patch_tokens(
        self, pose: torch.Tensor, n_frames: int, seq_len: int
    ) -> torch.Tensor:
        if pose.shape[1] != n_frames:
            raise ValueError(
                f"Pose latent time dimension {pose.shape[1]} must match WAN latent "
                f"time dimension {n_frames}."
            )
        pose = pose.to(
            device=self.patch_embedding.weight.device,
            dtype=self.patch_embedding.weight.dtype,
        )
        pose_tokens = self.pose_embedding(pose)
        if pose_tokens.shape[1] > seq_len:
            raise ValueError(
                f"Pose token length {pose_tokens.shape[1]} exceeds seq_len {seq_len}."
            )
        if pose_tokens.shape[1] < seq_len:
            pose_tokens = torch.cat(
                [
                    pose_tokens,
                    pose_tokens.new_zeros(
                        pose_tokens.shape[0],
                        seq_len - pose_tokens.shape[1],
                        pose_tokens.shape[2],
                    ),
                ],
                dim=1,
            )
        return pose_tokens

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        pose=None,
        return_hidden_states: bool = False,
        hidden_states_tap_every: int = 1,
    ):
        n_frames = x.shape[2]
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x] # (B, C, T, H, W) --> (B, dim, T/p_T, H/p_H, W/p_W)
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x] # (B, dim, T/p_T, H/p_H, W/p_W) --> (B, P, dim) for P = (T/p_T * H/p_H * W/p_W)
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        t_shape = tuple(t.shape)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        ) # (B, ) or (BT, ) --> (B, freq_dim) or (BT, freq_dim) --> (B, dim) or (BT, dim)
        if t.ndim == 2:
            e = e.unflatten(dim=0, sizes=t_shape) # (BT, dim) --> (B, T, dim) -- unflatten different representation for each frame
        else:
            e = repeat(e, "b c -> b f c", f=n_frames) # (B, dim) --> (B, T, dim) -- same representation for each frame

        pose_tokens = None
        if pose is not None:
            pose_tokens = self._pose_patch_tokens(
                pose, n_frames=n_frames, seq_len=seq_len
            ).type_as(x)
            pose_tokens = pose_tokens.unflatten(-1, (6, self.dim))

        e0 = self.time_projection(e).unflatten(-1, (6, self.dim)) # (B, T, dim) --> (B, T, 6*dim) --> (B, T, 6, dim)

        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            pose_tokens=pose_tokens,
        )

        hidden_states_list = [] if return_hidden_states else None
        hidden_states_tap_every = max(1, int(hidden_states_tap_every))
        for i, block in enumerate(self.blocks):
            block = partial(block, **kwargs)
            if i in self.gradient_checkpointing_indices:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

            if return_hidden_states and (i % hidden_states_tap_every == 0):
                if not torch.equal(grid_sizes, grid_sizes[:1].expand_as(grid_sizes)):
                    raise ValueError(
                        "WanModelPose hidden-state tapping requires uniform latent patch grids "
                        f"within a batch. Got grid_sizes={grid_sizes.tolist()}."
                    )
                t_p, h_p, w_p = [int(v) for v in grid_sizes[0].tolist()]
                seq_len_expected = t_p * h_p * w_p
                if seq_len_expected > x.shape[1]:
                    raise ValueError(
                        "WanModelPose hidden-state tap reshape expected at least "
                        f"{seq_len_expected} tokens, got {x.shape[1]}."
                    )
                hidden_states_5d = x[:, :seq_len_expected].reshape(
                    x.shape[0], t_p, h_p, w_p, x.shape[-1]
                )
                hidden_states_list.append(hidden_states_5d.detach())

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        output = torch.stack(x)
        if return_hidden_states:
            return output, {"hidden_states": hidden_states_list}
        return output
