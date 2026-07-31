from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F


class NoiseLevelReadout(nn.Module):
    """
    Predicts per-frame noise levels from a list of tapped hidden states.

    Inputs are a list of num_layers tensors, each of shape (B, T, Hp, Wp, D),
    extracted from intermediate residual-stream activations of a video diffusion
    model.  The module produces predictions of shape (B, T) — one scalar per
    frame — which are trained to recover the per-frame noise level k ∈ [0, 1].

    Pipeline
    --------
    1. Spatial pooling  (B, T, Hp, Wp, D) -> (B, T, D)
    2. Per-layer adapter  (B, T, D) -> (B, T, adapter_dim)
    3. Layer mixing  (num_layers, B, T, adapter_dim) -> (B, T, adapter_dim)
    4. Per-frame MLP  (B, T, adapter_dim) -> (B, T)

    Design choices
    --------------
    pooling : "mean" | "attention"
        How to collapse the spatial patch grid (Hp, Wp) into a single vector
        per frame.
        - "mean": simple average; no extra parameters, fastest.
        - "attention": learned linear D->1 scoring over the Hp*Wp spatial
          tokens, then a softmax-weighted sum.  Lets the model focus on the
          most informative patches (e.g. high-motion regions).

    per_layer_adapter : bool
        Whether each tapped layer gets its own LayerNorm+Linear projection
        from feature_dim to adapter_dim.
        - True: more expressive; different layers may encode noise level
          information in different subspaces, so separate projections can
          align them before mixing.
        - False: a single shared adapter; fewer parameters, acts as a
          regulariser and assumes the representation across layers is
          roughly aligned.

    mix_layers / mix_method : "last" | "mean" | "softmax"
        How to combine the num_layers adapted representations.
        - "last": only the final tapped layer is used; cheapest, but discards
          earlier layers that may carry stronger noise level signal.
        - "mean": unweighted average across all layers.
        - "softmax": a learned num_layers-dimensional weight vector (softmax
          normalised) produces a weighted sum; the model can discover which
          layer depths are most informative for noise-level prediction.

    target_type : "log_t" | "logit_t" | "sigmoid_t"
        The transform applied to both the raw prediction and the ground-truth
        noise level before computing the loss.  Choosing a transform that spreads
        the target distribution more uniformly can make regression easier.
        - "log_t": predict log(t); emphasises precision at small t (high noise).
        - "logit_t": predict logit(t) = log(t / (1-t)); symmetric around t=0.5,
          natural for Flow Matching where t is bounded in (0, 1).
        - "sigmoid_t": apply sigmoid to the raw prediction and compare to t
          directly; keeps targets in (0, 1) without unbounded outputs.
    """

    def __init__(
        self,
        feature_dim: int,
        num_layers: int,
        adapter_dim: int,
        num_mlp_layers: int,
        pooling: str,
        per_layer_adapter: bool,
        mix_layers: bool,
        mix_method: str,
        target_type: str,
        loss_type: str,
        eps: float,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.pooling = pooling
        self.per_layer_adapter = per_layer_adapter
        self.mix_layers = mix_layers
        self.mix_method = mix_method
        self.target_type = target_type
        self.loss_type = loss_type
        self.eps = eps
        self.num_mlp_layers = num_mlp_layers

        # Spatial attention pooling operates on feature_dim (before adapter).
        if pooling == "attention":
            self.pool_attn = nn.Linear(feature_dim, 1, bias=False)

        if per_layer_adapter:
            self.adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(feature_dim),
                        nn.Linear(feature_dim, adapter_dim),
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.adapter = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, adapter_dim),
            )

        if mix_layers and mix_method == "softmax":
            self.mix_logits = nn.Parameter(torch.zeros(num_layers))
        else:
            self.mix_logits = None

        # Per-frame MLP: (B, T, adapter_dim) -> (B, T, 1).
        mlp_layers = [nn.LayerNorm(adapter_dim)]
        if num_mlp_layers == 1:
            mlp_layers.append(nn.Linear(adapter_dim, 1))
        else:
            mlp_layers.append(nn.Linear(adapter_dim, adapter_dim))
            mlp_layers.append(nn.SiLU())
            for _ in range(num_mlp_layers - 2):
                mlp_layers.append(nn.Linear(adapter_dim, adapter_dim))
                mlp_layers.append(nn.SiLU())
            mlp_layers.append(nn.Linear(adapter_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def predict(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Run forward pass and apply the inverse of the training target transform to
        return per-frame noise level estimates in [0, 1].

        Args:
            hidden_states: list of num_layers tensors, each (B, T, Hp, Wp, D).
        Returns:
            noise_level_hat: (B, T) estimated noise levels in [0, 1].
        """
        pred_raw = self.forward(hidden_states)
        return self.predict_from_raw(pred_raw)

    def predict_from_raw(self, pred_raw: torch.Tensor) -> torch.Tensor:
        if self.target_type == "logit_t":
            return torch.sigmoid(pred_raw)
        if self.target_type == "log_t":
            return torch.exp(pred_raw)
        if self.target_type == "sigmoid_t":
            return torch.sigmoid(pred_raw)
        raise ValueError(f"Unknown target_type: {self.target_type}")

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hidden_states: list of num_layers tensors, each (B, T, Hp, Wp, D).
        Returns:
            pred: (B, T) raw predictions (before any target transform).
        """
        if len(hidden_states) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} hidden states, got {len(hidden_states)}"
            )

        per_layer = []
        for idx, h in enumerate(hidden_states):
            h = h.detach()
            if h.ndim != 5:
                raise ValueError(
                    f"Expected 5D input (B, T, Hp, Wp, D), got {h.ndim}D at layer {idx}"
                )
            b, t, hp, wp, d = h.shape

            # 1. Spatial pool: (B, T, Hp, Wp, D) -> (B, T, D).
            if self.pooling == "mean":
                h = h.mean(dim=(2, 3))
            elif self.pooling == "attention":
                h_flat = h.reshape(b * t, hp * wp, d)
                scores = self.pool_attn(h_flat).squeeze(-1)      # (B*T, Hp*Wp)
                weights = torch.softmax(scores, dim=1)
                h = (weights.unsqueeze(-1) * h_flat).sum(dim=1)  # (B*T, D)
                h = h.view(b, t, d)
            else:
                raise ValueError(f"Unknown pooling type: {self.pooling}")

            # 2. Adapter: (B, T, D) -> (B, T, adapter_dim).
            if self.per_layer_adapter:
                h = self.adapters[idx](h)
            else:
                h = self.adapter(h)

            per_layer.append(h)  # (B, T, adapter_dim)

        # 3. Mix layers: (num_layers, B, T, adapter_dim) -> (B, T, adapter_dim).
        stacked = torch.stack(per_layer, dim=0)
        mixed = self._mix_layers(stacked)

        # 4. Per-frame prediction: (B, T, adapter_dim) -> (B, T).
        return self.mlp(mixed).squeeze(-1)

    def compute_loss(
        self,
        hidden_states: List[torch.Tensor],
        noise_levels: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        """
        Args:
            hidden_states: list of (B, T, Hp, Wp, D) tensors.
            noise_levels: (B, T) per-frame noise levels in [0, 1].
            masks: optional (B, T) or (B, T, ...) boolean mask for valid frames.
            return_details: if True, also return per-element loss (B, T) and target (B, T).
        """
        if noise_levels.ndim != 2:
            raise ValueError(
                f"noise_levels must be 2D (B, T), got shape {tuple(noise_levels.shape)}. "
                "Per-frame noise levels are required."
            )

        pred_raw = self.forward(hidden_states)              # (B, T)
        loss, per_element, noise_level_target = self.compute_error_from_raw(
            pred_raw,
            noise_levels,
            masks=masks,
            return_details=True,
        )

        if return_details:
            return loss, per_element, noise_level_target
        return loss

    def compute_error_from_raw(
        self,
        pred_raw: torch.Tensor,
        noise_levels: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        loss_type: Optional[str] = None,
        return_details: bool = False,
    ):
        noise_level_target = noise_levels.float()
        pred, target = self._transform_pred_target(pred_raw, noise_level_target)
        per_element = self._loss_fn(pred, target, reduction="none", loss_type=loss_type)

        if masks is not None:
            if masks.ndim > 2:
                valid = masks.bool().flatten(2).any(dim=2)  # (B, T)
            else:
                valid = masks.bool()
            valid = valid.to(dtype=per_element.dtype, device=per_element.device)
            denom = valid.sum().clamp_min(1)
            loss = (per_element * valid).sum() / denom
        else:
            loss = per_element.mean()

        if return_details:
            return loss, per_element, noise_level_target
        return loss

    def _mix_layers(self, stacked: torch.Tensor) -> torch.Tensor:
        # stacked: (num_layers, B, T, adapter_dim)
        if not self.mix_layers or self.mix_method == "last":
            return stacked[-1]
        if self.mix_method == "mean":
            return stacked.mean(dim=0)
        if self.mix_method == "softmax":
            weights = torch.softmax(self.mix_logits, dim=0).view(-1, 1, 1, 1)
            return (weights * stacked).sum(dim=0)
        raise ValueError(f"Unknown mix method: {self.mix_method}")

    def _transform_pred_target(
        self, pred_raw: torch.Tensor, t_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_clamped = t_target.clamp(min=self.eps, max=1.0 - self.eps)
        if self.target_type == "log_t":
            return pred_raw, torch.log(t_clamped)
        if self.target_type == "logit_t":
            return pred_raw, torch.logit(t_clamped)
        if self.target_type == "sigmoid_t":
            return torch.sigmoid(pred_raw), t_clamped
        raise ValueError(f"Unknown target_type: {self.target_type}")

    def _loss_fn(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
        loss_type: Optional[str] = None,
    ) -> torch.Tensor:
        resolved_loss_type = self.loss_type if loss_type is None else loss_type
        if resolved_loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target, reduction=reduction)
        if resolved_loss_type in {"mse", "l2"}:
            return F.mse_loss(pred, target, reduction=reduction)
        raise ValueError(f"Unknown loss_type: {resolved_loss_type}")
