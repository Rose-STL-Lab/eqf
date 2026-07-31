from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class InferenceScheduleLookup:
    # Index by denoising counter n in [0, N], where N = num_sampling_steps.
    # levels are native-time / continuous noise levels in [t_end, 1].
    levels: torch.Tensor  # shape [N+1], monotone increasing with n
    # Relative step-size scale versus the identity schedule (1/N per step).
    # For step n->n-1, eta_scale[n] applies.
    eta_scale: torch.Tensor  # shape [N+1]
    # Discrete index projection for diffusion-time mapping.
    discrete_indices: torch.Tensor  # shape [N+1], int64 in [0, N]
    # Family-agnostic analytical eta multiplier as a function of the
    # diffusion-time / native noise level k in [0, 1]. Equivalent to
    # A * dk/ds evaluated at k, where A is the family's normalization
    # constant (trivially 1 for identity, r-dependent for sd3, the
    # integral A = \int_{t_end}^{1} 1/c(k) dk for the c_function family
    # which pairs with c(k) to give eta = A * c(k), and the closed form
    # eta(k) = (lambda_max - lambda_min)/2 * k * (1 - k) for the
    # linear_logsnr family). Used by the readout-predicted solver-index
    # path so that any schedule can reindex the step size from a
    # predicted native noise level rather than the discrete counter n.
    eta_at_k: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


def _ensure_strictly_positive_c(c: torch.Tensor) -> None:
    if not bool(torch.all(c > 0)):
        raise ValueError("c(t) must be strictly positive on [t_end, 1].")


def _interp1d_monotone(
    x_query: torch.Tensor,
    x_src: torch.Tensor,
    y_src: torch.Tensor,
) -> torch.Tensor:
    """
    Piecewise-linear interpolation for monotone x_src.
    x_src is expected ascending. x_query is clamped to [x_src[0], x_src[-1]].
    Use src (x,y) pairs, and build an approximation for "y_query" by 
    interpolating x_query into x_src.
    """
    x0 = x_src[0]
    x1 = x_src[-1]
    q = x_query.clamp(min=float(x0), max=float(x1))
    idx = torch.searchsorted(x_src, q, right=False)
    idx = idx.clamp(min=1, max=x_src.numel() - 1)
    lo = idx - 1
    hi = idx
    x_lo = x_src[lo]
    x_hi = x_src[hi]
    y_lo = y_src[lo]
    y_hi = y_src[hi]
    w = (q - x_lo) / (x_hi - x_lo).clamp_min(1e-12)
    return y_lo + w * (y_hi - y_lo)


def _build_identity_levels(num_sampling_steps: int) -> torch.Tensor:
    """
    Return evenly spaced points between [0,1], to be used to define either solver time or native time
    """
    denom = max(int(num_sampling_steps), 1)
    return torch.linspace(0.0, 1.0, steps=denom + 1, dtype=torch.float32)


def _build_sd3_levels(num_sampling_steps: int, r: float) -> torch.Tensor:
    """
    Return points between [0, 1] using the SD3 time-shift convention:

        k(s) = r * s / (1 + (r - 1) * s)
    """
    if r <= 0:
        raise ValueError("SD3 schedule requires r > 0.")
    s = _build_identity_levels(num_sampling_steps)
    r_val = float(r)
    return r_val * s / (1.0 + (r_val - 1.0) * s).clamp_min(1e-12)


def _build_linear_logsnr_levels(
    num_sampling_steps: int,
    lambda_max: float,
    lambda_min: float,
    lambda_shift: float,
) -> torch.Tensor:
    """
    Return native noise levels k in (0, 1) corresponding to a log-SNR grid that
    is linear in the denoising counter n in [0, N]:
        lambda_n = lambda_max + (n / N) * (lambda_min - lambda_max) + lambda_shift
    where the rectified-flow log-SNR convention
        lambda(k) = 2 * log((1 - k) / k)
    is used to invert lambda back to k via k = sigmoid(-lambda / 2).

    Requires lambda_max > lambda_min: lambda_max sits at the data side (n=0,
    small k) and lambda_min at the noise side (n=N, large k), so the resulting
    levels are monotone increasing with n as required by the lookup convention.
    """
    if not (float(lambda_max) > float(lambda_min)):
        raise ValueError(
            "linear_logsnr requires lambda_max > lambda_min "
            "(lambda_max corresponds to the data side, lambda_min to the noise side)."
        )
    s = _build_identity_levels(num_sampling_steps)
    lam = float(lambda_max) + s * (float(lambda_min) - float(lambda_max)) + float(lambda_shift)
    return torch.sigmoid(-0.5 * lam)


def _build_c_function_levels(
    num_sampling_steps: int,
    c_fn: Callable[[torch.Tensor], torch.Tensor],
    t_end: float,
    dense_points: int,
) -> tuple[torch.Tensor, float]:
    """
    Return points between [0,1] by calculating area under 1/c function, 
    defining a linearly spaced grid to numerically integrate (using trapezoidal method).
    That helps you get total `A.` With that, the amount we have 
    progressed so far through \\int_{} 1/c is the amount of "time" that has passed. Given
    an arbitrary point, we can get there (s_dense). 
    """
    if not (0.0 <= float(t_end) < 1.0):
        raise ValueError("t_end must satisfy 0 <= t_end < 1.")
    n_dense = max(int(dense_points), 256)
    t_dense = torch.linspace(float(t_end), 1.0, steps=n_dense, dtype=torch.float32)
    c_dense = c_fn(t_dense)
    c_dense = c_dense.to(dtype=torch.float32, device=t_dense.device)
    _ensure_strictly_positive_c(c_dense)

    inv_c = 1.0 / c_dense # c(t) --> 0 as t --> 0 means gets large (explodes beyond t_end),  
    dt = torch.diff(t_dense)
    trap = 0.5 * (inv_c[:-1] + inv_c[1:]) * dt
    # cumulative integral from t_end to t
    cum = torch.cat([torch.zeros(1, dtype=t_dense.dtype), torch.cumsum(trap, dim=0)], dim=0) # increasing wrt t
    A = float(cum[-1].item()) # defines the total amount of time
    if not (A > 0):
        raise ValueError("Invalid c(t): normalization integral A must be positive.")
    # s(t) = (A - int_{t_end}^{t} 1/c) / A
    s_dense = (A - cum) / A # normalize time passed, now decreasing wrt t
    s_dense = s_dense.clamp(0.0, 1.0)

    s_grid = _build_identity_levels(num_sampling_steps)
    # s_dense decreases with t_dense; invert via ascending x for interpolation.
    t_of_s_desc = _interp1d_monotone(
        x_query=s_grid,
        x_src=torch.flip(s_dense, dims=[0]),
        y_src=torch.flip(t_dense, dims=[0]),
    )
    # Internal convention for schedule lookup is index n from clean->noise
    # (0..N), so levels must be monotone increasing with index.
    return torch.flip(t_of_s_desc, dims=[0]), A


def build_inference_schedule_lookup(
    *,
    num_sampling_steps: int,
    family: str,
    is_continuous: bool,
    t_end: float = 1e-3,
    sd3_r: float = 6.0,
    logsnr_max: float = 15.0,
    logsnr_min: float = -15.0,
    logsnr_shift: float = 0.0,
    c_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    c_dense_points: int = 4096,
) -> InferenceScheduleLookup:
    """
    Builds an object that helps us map from a noise level to a multipier on 
    the default step size (which would be 1/D). 
    
    If solver time is s and the inference schedule is t(s),
    we take small steps where |dt(s)/ds| is small. That multipliers into
    the solver-default step size (like 1/D) to determine the actual step.

    Levels is the important grid: it takes a standard, evenly spaced grid
    in solver time s and stores the mapped t(s), so that its differences
    are step sizes in the actual schedule we care about. We can use levels
    to tell the model the noise level that can differ from proportional to 
    number of steps take so far.
    """
    family_norm = str(family).strip().lower()
    eta_at_k: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    if family_norm in {"identity", "native_grid"}:
        levels = _build_identity_levels(num_sampling_steps)
        # k(s) = s  =>  dk/ds = 1 for all k.
        def _eta_at_k_identity(k: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(k, dtype=torch.float32)
        eta_at_k = _eta_at_k_identity
    elif family_norm in {"sd3", "sd3_warp"}:
        levels = _build_sd3_levels(num_sampling_steps, r=float(sd3_r))
        # analytical solution for the sd3 schedule:
        # k(s) = r s / (1 + (r-1) s)
        #   => dk/ds = (r - (r-1) k)^2 / r.
        r_val = float(sd3_r)
        def _eta_at_k_sd3(k: torch.Tensor) -> torch.Tensor:
            k32 = k.to(dtype=torch.float32)
            return (r_val - k32 * (r_val - 1.0)).pow(2) / r_val
        eta_at_k = _eta_at_k_sd3
    elif family_norm in {"linear_logsnr", "linear_log_snr", "logsnr_linear"}:
        # Levels chosen so log-SNR is linear in solver index n / N:
        #   lambda_n = lambda_max + (n/N)(lambda_min - lambda_max) + shift,
        #   k_n      = sigmoid(-lambda_n / 2)   (rectified-flow convention).
        # Closed-form rate: with s = n / N,
        #   dlambda/ds = lambda_min - lambda_max,
        #   dk/dlambda = -k(1-k)/2,
        # so eta(k) = dk/ds = (lambda_max - lambda_min)/2 * k * (1 - k).
        # The shift only translates lambda; it cancels in dk/ds.
        levels = _build_linear_logsnr_levels(
            num_sampling_steps,
            lambda_max=float(logsnr_max),
            lambda_min=float(logsnr_min),
            lambda_shift=float(logsnr_shift),
        )
        half_range = 0.5 * (float(logsnr_max) - float(logsnr_min))
        def _eta_at_k_linear_logsnr(k: torch.Tensor) -> torch.Tensor:
            k32 = k.to(dtype=torch.float32)
            return half_range * k32 * (1.0 - k32)
        eta_at_k = _eta_at_k_linear_logsnr
    elif family_norm in {"c_function", "c_fn"}:
        if c_fn is None:
            raise ValueError("c_function family requires c_fn.")
        levels, c_norm_A = _build_c_function_levels(
            num_sampling_steps=num_sampling_steps,
            c_fn=c_fn,
            t_end=float(t_end),
            dense_points=int(c_dense_points),
        )
        # A * c(k) matches the per-step eta_scale derivation in this family.
        A_val = float(c_norm_A)
        def _eta_at_k_c_function(k: torch.Tensor) -> torch.Tensor:
            k32 = k.to(dtype=torch.float32)
            return A_val * c_fn(k32).to(dtype=torch.float32)
        eta_at_k = _eta_at_k_c_function
    else:
        raise ValueError(f"Unknown inference schedule family: {family!r}")

    # Enforce monotonic increasing levels with n (0 -> clean, N -> noisy).
    min_level = float(t_end) if family_norm in {"c_function", "c_fn"} else 0.0
    levels = levels.clamp(min=min_level, max=1.0)
    if not bool(torch.all(torch.diff(levels) >= -1e-6)):
        raise ValueError("Constructed schedule levels are not monotone increasing.")
    levels = torch.maximum(levels, torch.cummax(levels, dim=0).values)

    N = max(int(num_sampling_steps), 1)
    # Use the family's analytical eta(k) = A * dk/ds evaluated at the
    # schedule's grid points. Semantically equivalent to the readout-predicted
    # path (which evaluates eta_at_k at a predicted k), just sampled at the
    # nominal schedule level for each solver index. Preferred over a
    # backward-difference of `levels` because it is exact per-point rather
    # than an integrated finite-difference approximation and removes the
    # need to special-case index 0.
    eta_scale = eta_at_k(levels).clamp_min(1e-6)

    # Diffusion projects continuous levels onto nearest denoising counter index.
    discrete_indices = torch.round(levels * float(N)).to(dtype=torch.long).clamp(min=0, max=N)

    if not is_continuous:
        # Keep eta scale present for consistent interfaces.
        eta_scale = torch.ones_like(eta_scale)
        # Discrete-timestep sampling does not use continuous step-size
        # scaling; readout-predicted reindexing collapses to a no-op.
        def _eta_at_k_discrete(k: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(k, dtype=torch.float32)
        eta_at_k = _eta_at_k_discrete

    return InferenceScheduleLookup(
        levels=levels,
        eta_scale=eta_scale,
        discrete_indices=discrete_indices,
        eta_at_k=eta_at_k,
    )
