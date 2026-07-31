"""
Equilibrium c(k) schedule utilities for Equilibrium Matching (EqF).

This module centralizes deterministic equilibrium schedules.

Conventions (matches existing codebase):
- k in [0, 1]
- k = 0 is data
- k = 1 is noise
"""

from __future__ import annotations

from typing import Callable, Literal

import torch as th

from .diffusion_utils import add_shape_channels
from .noise_schedule import (
    constant_noise_schedule,
    linear_noise_schedule,
    truncated_noise_schedule,
)

class EquilibriumC:
    """
    Helper for deterministic c(k).
    """

    def __init__(self, cfg, *, apply_lambda: bool = True):
        self.schedule_name = str(cfg.equilibrium_schedule).lower()
        raw_lambda = float(cfg.equilibrium_lambda or 4.0)
        self.lambda_ = raw_lambda if bool(apply_lambda) else 1.0
        self.truncated_a = float((cfg.truncated_a or 0.8) if self.schedule_name == "truncated" else 0.8)

        self._c_det_fn: Callable[[th.Tensor], th.Tensor]
        if self.schedule_name == "linear":
            self._c_det_fn = lambda k: linear_noise_schedule(k, self.lambda_)
        elif self.schedule_name == "truncated":
            self._c_det_fn = lambda k: truncated_noise_schedule(k, self.lambda_, self.truncated_a)
        elif self.schedule_name == "constant":
            # Note: existing codebase semantics use c(k)=1 (not scaled by lambda) for the constant ablation.
            self._c_det_fn = constant_noise_schedule
        else:
            raise NotImplementedError(self.schedule_name)

    def deterministic(self, k: th.Tensor) -> th.Tensor:
        return self._c_det_fn(k)

    def resolve_c_tensor(self, k: th.Tensor, target_shape) -> th.Tensor:
        """Resolve deterministic c(k) as a tensor broadcastable to target_shape."""
        return add_shape_channels(self.deterministic(k), target_shape)


