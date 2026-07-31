"""
Unified single-loop adaptive cleanup ("tail finish").

There is no separate finishing loop and no frame freezing: every active frame
is updated on every inner step, and the *only* thing that changes when a frame
enters its cleanup phase is its per-step `eta`.

Using the freshest predicted noise level `sigma_hat = readout_t_hat`:
- A frame enters cleanup once `sigma_hat <= sigma_thresh` (configured via
  `streaming.emit_noise_level`). It then keeps a
  per-frame counter `j` of how many cleanup steps it has taken.
- cleanup frames: `delta_k = sigma_hat * m / (S - j)` (final step lands at 0)
  and `eta = delta_k`, where `S = n_cleanup_steps` and `m = step_multiplier`.
  This assumes EqF's constant equilibrium schedule (`c(k) == 1`): a GD step of
  size `eta` along the modulated velocity `w = c(k) * v` then moves the noise
  level by exactly `eta`. A non-constant schedule (EqM) raises
  NotImplementedError (it would need `eta = delta_k / c(sigma_hat)`).
- other active frames: `eta = eta_base * eta_at_k(sigma_hat)` (the standard
  readout-predicted step size); if no `eta_at_k` schedule is configured we keep
  the incoming schedule eta.
- a frame is "done" once `j >= S`; the caller forces its denoising counter to 0
  so the ordinary counter-based emit / stopping logic picks it up.

This policy is injected into the EqF eta seam
(`EquilibriumMatching._resolve_sampling_step_eta`) via
`sample_kwargs["step_eta_resolver"]` so it runs right after the forward pass
with the most up-to-date `sigma_hat`. Streaming state carries the per-frame
`tail_mask` / `tail_j`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from algorithms.denoising import EquilibriumMatching


class TailFinishMixin:
    """
    Provides the unified single-loop cleanup eta policy. Consumed by
    `StreamingInferenceMixin._inner_refine_step`, which injects
    `_compute_cleanup_step_eta` into the EqF eta seam.
    """

    # Provided by the composed algorithm class (e.g. the denoising video algo).
    denoising_algo: "EquilibriumMatching"

    def _compute_cleanup_step_eta(
        self,
        *,
        # Context captured per inner step by `_inner_refine_step`.
        active_start: int,
        active_end: int,
        active_mask: Tensor,
        tail_mask: Tensor,
        tail_j: Tensor,
        n_cleanup_steps: int,
        step_multiplier: float,
        sigma_thresh: float,
        eta_base: float,
        eta_at_k_fn,
        # Provided by the EqF eta seam after the forward pass.
        eta_tensor: Tensor,
        valid_step: Tensor,
        true_curr_noise_level: Tensor,
        readout_t_hat: Optional[Tensor],
    ) -> Tuple[Tensor, dict]:
        """
        Unified per-frame eta policy for the single-loop adaptive cleanup.

        Using the freshest `sigma_hat = readout_t_hat`:
          - A frame enters cleanup once `sigma_hat <= sigma_thresh`. It stays
            active and keeps being updated; only its `eta` changes.
          - cleanup frames: `delta_k = sigma_hat * m / (S - j)` (the final
            step, `S - j == 1`, lands exactly at zero) and `eta = delta_k`
            (EqF constant `c`; a non-constant schedule raises
            NotImplementedError).
          - standard frames: `eta = eta_base * eta_at_k(sigma_hat)` (your
            `c(sigma_hat)`); if no `eta_at_k` schedule is configured we keep the
            incoming schedule eta.

        Returns the full-window `eta_tensor` (only the active slice is changed)
        plus an `extra` dict with the updated tail mask / counter / done flags
        (active-slice shaped) that the caller persists into the sampling state.
        """
        if readout_t_hat is None:
            raise RuntimeError(
                "Unified adaptive cleanup requires readout_t_hat (sigma_hat); "
                "ensure readout collection is enabled."
            )
        device = eta_tensor.device
        S = int(n_cleanup_steps)
        m = float(step_multiplier)

        sigma_hat = (
            readout_t_hat[:, active_start:active_end]
            .to(device=device, dtype=torch.float32)
            .clamp(0.0, 1.0)
        )
        active = active_mask.to(device=device, dtype=torch.bool)
        tail = tail_mask.to(device=device, dtype=torch.bool)
        j = tail_j.to(device=device, dtype=torch.long)

        newly_entering = active & (~tail) & (sigma_hat <= float(sigma_thresh))
        tail_next = tail | newly_entering

        remaining = (S - j).clamp(min=1).to(dtype=torch.float32)
        delta_k = sigma_hat * m / remaining
        # Final cleanup step (remaining == 1) drives the noise exactly to zero.
        delta_k = torch.where(remaining <= 1.0, sigma_hat, delta_k).clamp(min=0.0)

        # EqF assumption: the equilibrium schedule is constant (c(k) == 1). Under
        # constant c, a GD step of size `eta` along the modulated velocity
        # `w = c(k) * v` moves the predicted noise level by exactly `eta`, so the
        # cleanup step size is simply the desired decrement: eta_cleanup = delta_k.
        # For EqM (non-constant c) the step moves x by `eta * c(k) * v`, so you
        # would instead need `eta = delta_k / c(sigma_hat)`; that is not supported.
        schedule = str(getattr(self.denoising_algo, "equilibrium_schedule", "")).lower()
        if schedule != "constant":
            raise NotImplementedError(
                f"Unified tail-finish cleanup assumes a constant equilibrium "
                f"schedule (EqF, c(k) == 1), but got '{schedule}'. For EqM "
                f"(non-constant c) convert delta_k to an eta via "
                f"eta = delta_k / c(sigma_hat). Cleanup is not yet implemented for EqM"
            )
        eta_cleanup = delta_k

        incoming_active = eta_tensor[:, active_start:active_end]
        if eta_at_k_fn is not None:
            eta_mult = (
                eta_at_k_fn(sigma_hat)
                .to(device=device, dtype=eta_cleanup.dtype)
                .clamp_min(1e-6)
            )
            eta_standard = float(eta_base) * eta_mult
        else:
            eta_standard = incoming_active.to(dtype=eta_cleanup.dtype)

        eta_active = torch.where(tail_next, eta_cleanup, eta_standard)
        # Leave inactive (frozen / finished) tokens at their incoming eta; they
        # are masked out of the x-update by valid_step anyway.
        eta_active = torch.where(
            active, eta_active, incoming_active.to(dtype=eta_active.dtype)
        )

        eta_tensor = eta_tensor.clone()
        eta_tensor[:, active_start:active_end] = eta_active.to(dtype=eta_tensor.dtype)

        j_next = torch.where(tail_next & active, j + 1, j)
        done = tail_next & active & (j_next >= S)

        extra = {
            "cleanup_tail_mask_next": tail_next,
            "cleanup_tail_j_next": j_next,
            "cleanup_done": done,
        }
        return eta_tensor, extra
