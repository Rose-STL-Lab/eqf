"""
FVD-over-time: computes FVD on fixed-length temporal windows extracted from
longer videos to produce a curve over generated rollout time.

Two reference modes:
  - global_pooled: every generated window is compared against one shared reference
    distribution built from windows sampled across all reference videos at
    `ref_stride`-spaced start indices. Cheaper and lower-variance, but tells you
    "do generated clips look like real clips from anywhere in the trajectory?".
  - time_aligned: each generated start time `t` is compared against the reference
    distribution at the same start time. Strictly horizon-specific, but noisier
    because each FVD has only N_ref reference clips to estimate from.

Inputs to update() are pre-extracted I3D features (one tensor per window start
index), so this metric inherits the FVD/I3D preprocessing exactly from the
existing pipeline (`VideoMetric._extract_i3d_features`).
"""

from typing import Any, Dict, List, Optional, Tuple
import time

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torchmetrics import Metric
from torchmetrics.image.fid import _compute_fid

from utils.distributed_utils import rank_zero_print_once
from utils.print_utils import cyan
from .base_fid import BaseFrechetDistance


_GLOBAL_REF_KEY = "ref_global"


class _WindowFeatureAccumulator(BaseFrechetDistance):
    """
    Single-side feature accumulator (only one of fake/real is used per instance).
    Extends BaseFrechetDistance to inherit the float64 sum/cov_sum state, but
    overrides update() to take pre-extracted features and a `real` flag so that
    each window only accumulates one side.
    """

    def __init__(self, features: int = 400) -> None:
        super().__init__(registry=None, features=features, reset_real_features=True)

    def extract_features(self, x: Tensor) -> Tensor:
        return x

    def update(self, features: Tensor, real: bool) -> None:  # type: ignore[override]
        # Routed through torchmetrics' _wrap_update so _update_count increments
        # and dist sync is handled correctly.
        self._update(features, real=real)


class FVDOverTime(Metric):
    """
    Windowed FVD over generated rollout time.

    Args:
        clip_len: number of frames per clip passed to FVD/I3D.
        gen_stride: stride between generated start times.
        ref_stride: stride for sampling reference clips in global_pooled mode.
            Ignored in time_aligned mode.
        reference_mode: "global_pooled" or "time_aligned".
        features: I3D feature dimension (400 for the StyleGAN-V I3D used here).

    Inputs to ``update``:
        fake_features_per_t: dict[int, Tensor[B, features]] keyed by generated
            window start index.
        real_features_per_t: dict[Hashable, Tensor[B, features]]. In
            ``global_pooled`` mode the keys are arbitrary; one accumulator pools
            them all. In ``time_aligned`` mode the keys must match
            ``fake_features_per_t``.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        clip_len: int = 25,
        gen_stride: int = 25,
        ref_stride: int = 25,
        reference_mode: str = "global_pooled",
        fvd_distance_backend: str = "torch_eigh",
        compare_fvd_distance_backends: bool = False,
        fvd_distance_abs_tolerance: float = 1e-3,
        fvd_distance_rel_tolerance: float = 1e-4,
        features: int = 400,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if clip_len < 9:
            raise ValueError(
                f"clip_len must be >= 9 for I3D, got clip_len={clip_len}."
            )
        if gen_stride <= 0 or ref_stride <= 0:
            raise ValueError(
                f"gen_stride and ref_stride must be positive, "
                f"got gen_stride={gen_stride}, ref_stride={ref_stride}."
            )
        if reference_mode not in ("global_pooled", "time_aligned"):
            raise ValueError(
                f"reference_mode must be 'global_pooled' or 'time_aligned', "
                f"got {reference_mode!r}."
            )
        if fvd_distance_backend not in ("torchmetrics", "torch_eigh"):
            raise ValueError(
                "fvd_distance_backend must be 'torchmetrics' or 'torch_eigh', "
                f"got {fvd_distance_backend!r}."
            )

        self.clip_len = int(clip_len)
        self.gen_stride = int(gen_stride)
        self.ref_stride = int(ref_stride)
        self.reference_mode = reference_mode
        self.fvd_distance_backend = fvd_distance_backend
        self.compare_fvd_distance_backends = bool(compare_fvd_distance_backends)
        self.fvd_distance_abs_tolerance = float(fvd_distance_abs_tolerance)
        self.fvd_distance_rel_tolerance = float(fvd_distance_rel_tolerance)
        self.features = int(features)

        # Per-window accumulators. Keys are stringified ints (or _GLOBAL_REF_KEY)
        # because nn.ModuleDict only allows string keys.
        self._fakes = nn.ModuleDict()
        self._reals = nn.ModuleDict()

    @staticmethod
    def _key(t: int) -> str:
        return f"t{int(t)}"

    def _ensure_fake(self, t: int) -> _WindowFeatureAccumulator:
        key = self._key(t)
        if key not in self._fakes:
            mod = _WindowFeatureAccumulator(features=self.features).to(self.device)
            self._fakes[key] = mod
        return self._fakes[key]

    def _ensure_real(self, key: str) -> _WindowFeatureAccumulator:
        if key not in self._reals:
            mod = _WindowFeatureAccumulator(features=self.features).to(self.device)
            self._reals[key] = mod
        return self._reals[key]

    def update(  # type: ignore[override]
        self,
        fake_features_per_t: Dict[int, Tensor],
        real_features_per_t: Dict[Any, Tensor],
    ) -> None:
        if not fake_features_per_t:
            return

        for t, feats in fake_features_per_t.items():
            self._ensure_fake(int(t)).update(feats, real=False)

        if self.reference_mode == "global_pooled":
            ref_mod = self._ensure_real(_GLOBAL_REF_KEY)
            for feats in real_features_per_t.values():
                ref_mod.update(feats, real=True)
        else:  # time_aligned
            for t, feats in real_features_per_t.items():
                self._ensure_real(self._key(int(t))).update(feats, real=True)

    @property
    def is_empty(self) -> bool:
        if not self._fakes:
            return True
        for key, fake in self._fakes.items():
            if int(fake.fake_features_num_samples.item()) < 2:
                continue
            ref = self._reference_for_fake(key)
            if ref is None:
                continue
            if int(ref.real_features_num_samples.item()) >= 2:
                return False
        return True

    def _reference_for_fake(
        self, fake_key: str
    ) -> Optional[_WindowFeatureAccumulator]:
        key = _GLOBAL_REF_KEY if self.reference_mode == "global_pooled" else fake_key
        return self._reals[key] if key in self._reals else None

    def _gen_starts_sorted(self) -> List[int]:
        starts: List[int] = []
        for key in self._fakes.keys():
            assert key.startswith("t"), key
            starts.append(int(key[1:]))
        starts.sort()
        return starts

    @staticmethod
    def _stats_from_accumulator(
        mod: _WindowFeatureAccumulator, real: bool
    ) -> Optional[Tuple[Tensor, Tensor, int]]:
        # Read state (already synced when called inside sync_context).
        if real:
            n = int(mod.real_features_num_samples.item())
            if n < 2:
                return None
            mean = (mod.real_features_sum / n).unsqueeze(0)
            cov_num = mod.real_features_cov_sum - n * mean.t().mm(mean)
        else:
            n = int(mod.fake_features_num_samples.item())
            if n < 2:
                return None
            mean = (mod.fake_features_sum / n).unsqueeze(0)
            cov_num = mod.fake_features_cov_sum - n * mean.t().mm(mean)
        cov = cov_num / (n - 1)
        return mean.squeeze(0), cov, n

    @staticmethod
    def _dist_sum(x: Tensor) -> Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def _stats_from_sum_cov_num(
        features_sum: Tensor, features_cov_sum: Tensor, num_samples: Tensor
    ) -> Optional[Tuple[Tensor, Tensor, int]]:
        n = int(num_samples.item())
        if n < 2:
            return None
        mean = (features_sum / n).unsqueeze(0)
        cov_num = features_cov_sum - n * mean.t().mm(mean)
        cov = cov_num / (n - 1)
        return mean.squeeze(0), cov, n

    @staticmethod
    def _symmetrize(x: Tensor) -> Tensor:
        return 0.5 * (x + x.transpose(-1, -2))

    @classmethod
    def _matrix_sqrt_psd(cls, x: Tensor) -> Tensor:
        x = cls._symmetrize(x)
        eigvals, eigvecs = torch.linalg.eigh(x)
        sqrt_eigvals = eigvals.clamp_min(0).sqrt()
        return (eigvecs * sqrt_eigvals.unsqueeze(0)) @ eigvecs.transpose(-1, -2)

    @classmethod
    def _compute_fid_torch_eigh(
        cls, mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor
    ) -> Tensor:
        """Compute Frechet distance with a GPU-friendly symmetric PSD eig path.

        This uses the equivalent identity:
            Tr(sqrt(sigma1 @ sigma2))
          = Tr(sqrt(sqrt(sigma1) @ sigma2 @ sqrt(sigma1)))
        where the inner matrix is symmetric positive semidefinite up to
        numerical error, so `torch.linalg.eigh` can stay on GPU.
        """
        mu1 = mu1.double()
        mu2 = mu2.double()
        sigma1 = cls._symmetrize(sigma1.double())
        sigma2 = cls._symmetrize(sigma2.double())

        diff = mu1 - mu2
        sqrt_sigma1 = cls._matrix_sqrt_psd(sigma1)
        middle = cls._symmetrize(sqrt_sigma1 @ sigma2 @ sqrt_sigma1)
        covmean_eigvals = torch.linalg.eigvalsh(middle)
        tr_covmean = covmean_eigvals.clamp_min(0).sqrt().sum()
        fid = (
            diff.dot(diff)
            + torch.trace(sigma1)
            + torch.trace(sigma2)
            - 2.0 * tr_covmean
        )
        return fid.real.float()

    def _compute_fid_selected(
        self, mu_ref: Tensor, sigma_ref: Tensor, mu_fake: Tensor, sigma_fake: Tensor
    ) -> tuple[Tensor, Dict[str, float]]:
        timings: Dict[str, float] = {}

        if self.compare_fvd_distance_backends:
            torchmetrics_start = time.perf_counter()
            fvd_torchmetrics = _compute_fid(mu_ref, sigma_ref, mu_fake, sigma_fake)
            timings["torchmetrics"] = time.perf_counter() - torchmetrics_start

            eigh_start = time.perf_counter()
            fvd_eigh = self._compute_fid_torch_eigh(
                mu_ref, sigma_ref, mu_fake, sigma_fake
            )
            timings["torch_eigh"] = time.perf_counter() - eigh_start

            selected = (
                fvd_eigh
                if self.fvd_distance_backend == "torch_eigh"
                else fvd_torchmetrics
            )
            timings["abs_diff"] = float((fvd_torchmetrics - fvd_eigh).abs().item())
            denom = max(float(fvd_torchmetrics.abs().item()), 1e-12)
            timings["rel_diff"] = timings["abs_diff"] / denom
            return selected, timings

        if self.fvd_distance_backend == "torch_eigh":
            selected = self._compute_fid_torch_eigh(
                mu_ref, sigma_ref, mu_fake, sigma_fake
            )
        else:
            selected = _compute_fid(mu_ref, sigma_ref, mu_fake, sigma_fake)
        return selected, timings

    def compute(self) -> Dict[str, Any]:
        start_indices: List[int] = []
        fvd_values: List[float] = []
        num_gen_clips: List[int] = []
        num_ref_clips: List[int] = []
        backend_timings: Dict[str, float] = {
            "torchmetrics": 0.0,
            "torch_eigh": 0.0,
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
        }

        gen_starts = self._gen_starts_sorted()
        if not gen_starts:
            return {
                "start_indices": [],
                "fvd": [],
                "num_gen_clips": [],
                "num_ref_clips": [],
                "scalars": {
                    "mean": float("nan"),
                    "first": float("nan"),
                    "final": float("nan"),
                    "max": float("nan"),
                },
                "config": {
                    "clip_len": self.clip_len,
                    "gen_stride": self.gen_stride,
                    "ref_stride": self.ref_stride,
                    "reference_mode": self.reference_mode,
                    "fvd_distance_backend": self.fvd_distance_backend,
                    "compare_fvd_distance_backends": self.compare_fvd_distance_backends,
                    "fvd_distance_abs_tolerance": self.fvd_distance_abs_tolerance,
                    "fvd_distance_rel_tolerance": self.fvd_distance_rel_tolerance,
                },
            }

        fake_keys = [self._key(t) for t in gen_starts]
        first_fake = self._fakes[fake_keys[0]]
        device = first_fake.fake_features_sum.device
        dtype = first_fake.fake_features_sum.dtype
        cov_shape = first_fake.fake_features_cov_sum.shape

        fake_sum = torch.stack(
            [self._fakes[key].fake_features_sum.detach().clone() for key in fake_keys]
        )
        fake_cov_sum = torch.stack(
            [
                self._fakes[key].fake_features_cov_sum.detach().clone()
                for key in fake_keys
            ]
        )
        fake_num = torch.stack(
            [
                self._fakes[key].fake_features_num_samples.detach().clone()
                for key in fake_keys
            ]
        )

        if self.reference_mode == "global_pooled":
            real_keys = [_GLOBAL_REF_KEY]
        else:
            real_keys = fake_keys

        zero_sum = torch.zeros(self.features, device=device, dtype=dtype)
        zero_cov = torch.zeros(cov_shape, device=device, dtype=dtype)
        zero_num = torch.zeros((), device=device, dtype=torch.long)

        real_sum = torch.stack(
            [
                (
                    self._reals[key].real_features_sum.detach().clone()
                    if key in self._reals
                    else zero_sum.clone()
                )
                for key in real_keys
            ]
        )
        real_cov_sum = torch.stack(
            [
                (
                    self._reals[key].real_features_cov_sum.detach().clone()
                    if key in self._reals
                    else zero_cov.clone()
                )
                for key in real_keys
            ]
        )
        real_num = torch.stack(
            [
                (
                    self._reals[key].real_features_num_samples.detach().clone()
                    if key in self._reals
                    else zero_num.clone()
                )
                for key in real_keys
            ]
        )

        fake_sum = self._dist_sum(fake_sum)
        fake_cov_sum = self._dist_sum(fake_cov_sum)
        fake_num = self._dist_sum(fake_num)
        real_sum = self._dist_sum(real_sum)
        real_cov_sum = self._dist_sum(real_cov_sum)
        real_num = self._dist_sum(real_num)

        ref_stats_cache: Dict[int, Optional[Tuple[Tensor, Tensor, int]]] = {}

        for i, t in enumerate(gen_starts):
            fake_stats = self._stats_from_sum_cov_num(
                fake_sum[i], fake_cov_sum[i], fake_num[i]
            )
            if fake_stats is None:
                continue

            ref_idx = 0 if self.reference_mode == "global_pooled" else i
            if ref_idx not in ref_stats_cache:
                ref_stats_cache[ref_idx] = self._stats_from_sum_cov_num(
                    real_sum[ref_idx], real_cov_sum[ref_idx], real_num[ref_idx]
                )
            ref_stats = ref_stats_cache[ref_idx]
            if ref_stats is None:
                continue

            mu_fake, sigma_fake, n_fake = fake_stats
            mu_ref, sigma_ref, n_ref = ref_stats
            fvd_t, timing = self._compute_fid_selected(
                mu_ref, sigma_ref, mu_fake, sigma_fake
            )
            backend_timings["torchmetrics"] += timing.get("torchmetrics", 0.0)
            backend_timings["torch_eigh"] += timing.get("torch_eigh", 0.0)
            backend_timings["max_abs_diff"] = max(
                backend_timings["max_abs_diff"], timing.get("abs_diff", 0.0)
            )
            backend_timings["max_rel_diff"] = max(
                backend_timings["max_rel_diff"], timing.get("rel_diff", 0.0)
            )

            start_indices.append(t)
            fvd_values.append(float(fvd_t.item()))
            num_gen_clips.append(n_fake)
            num_ref_clips.append(n_ref)

        if self.compare_fvd_distance_backends:
            within_abs = (
                backend_timings["max_abs_diff"] <= self.fvd_distance_abs_tolerance
            )
            within_rel = (
                backend_timings["max_rel_diff"] <= self.fvd_distance_rel_tolerance
            )
            status = "OK" if within_abs or within_rel else "WARNING"
            rank_zero_print_once(
                cyan(
                    "[FVDOverTime] backend comparison "
                    f"status={status} selected={self.fvd_distance_backend} "
                    f"torchmetrics_time={backend_timings['torchmetrics']:.2f}s "
                    f"torch_eigh_time={backend_timings['torch_eigh']:.2f}s "
                    f"max_abs_diff={backend_timings['max_abs_diff']:.6g} "
                    f"max_rel_diff={backend_timings['max_rel_diff']:.6g} "
                    f"abs_tol={self.fvd_distance_abs_tolerance:.6g} "
                    f"rel_tol={self.fvd_distance_rel_tolerance:.6g}"
                )
            )

        if not fvd_values:
            scalars = {
                "mean": float("nan"),
                "first": float("nan"),
                "final": float("nan"),
                "max": float("nan"),
            }
        else:
            scalars = {
                "mean": float(sum(fvd_values) / len(fvd_values)),
                "first": float(fvd_values[0]),
                "final": float(fvd_values[-1]),
                "max": float(max(fvd_values)),
            }

        return {
            "start_indices": start_indices,
            "fvd": fvd_values,
            "num_gen_clips": num_gen_clips,
            "num_ref_clips": num_ref_clips,
            "scalars": scalars,
            "config": {
                "clip_len": self.clip_len,
                "gen_stride": self.gen_stride,
                "ref_stride": self.ref_stride,
                "reference_mode": self.reference_mode,
                "fvd_distance_backend": self.fvd_distance_backend,
                "compare_fvd_distance_backends": self.compare_fvd_distance_backends,
                "fvd_distance_abs_tolerance": self.fvd_distance_abs_tolerance,
                "fvd_distance_rel_tolerance": self.fvd_distance_rel_tolerance,
            },
            "backend_comparison": backend_timings
            if self.compare_fvd_distance_backends
            else None,
        }

    def reset(self) -> None:
        super().reset()
        for mod in self._fakes.values():
            mod.reset()
        for mod in self._reals.values():
            mod.reset()
