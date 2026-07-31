from typing import Dict, Optional, Tuple, Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from algorithms.denoising.inference_schedule import (
    InferenceScheduleLookup,
    build_inference_schedule_lookup,
)
from algorithms.denoising.noise_schedule import (
    constant_noise_schedule,
    linear_noise_schedule,
    truncated_noise_schedule,
)


class InferenceMixin:
    @torch.no_grad()
    def _run_eval_denoising_once(
        self,
        batch,
        batch_idx,
        dataloader_idx,
        namespace="training",
        prepare_visuals: bool = True,
        training_step_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        xs, conditions, masks, gt_videos, video_metadata = batch

        xs = xs[:, : self.forward_window_size_in_tokens]
        if conditions is not None:
            # Just take the first forward_window_size conditions
            conditions = conditions[:, : self.forward_window_size_in_tokens]
        masks = masks[:, : self.forward_window_size_in_tokens]
        if gt_videos is not None:
            if self.is_latent_diffusion and self.is_latent_video_vae:
                gt_videos = gt_videos[:, : self.forward_window_size_in_frames]
            else:
                gt_videos = gt_videos[:, : self.forward_window_size_in_tokens]

        batch = (xs, conditions, masks, gt_videos, video_metadata)
        training_step_kwargs = training_step_kwargs or {}
        output = self.training_step(
            batch,
            batch_idx,
            dataloader_idx=dataloader_idx,
            namespace=namespace,
            **training_step_kwargs,
        )

        result = {
            "output": output,
            "gt_videos": gt_videos,
            "reconstruction": None,
            "video_metadata": video_metadata,
        }
        if not prepare_visuals:
            return result

        if self.is_latent_diffusion:
            if gt_videos is None:
                return result
            if not (hasattr(self, "vae") and self.vae is not None):
                return result

        gt_videos_vis = gt_videos if self.is_latent_diffusion else output["original_x"]
        recons = output["predicted_x_start"]
        if self.is_latent_diffusion:
            recons = self._decode(recons, desired_length=int(gt_videos_vis.shape[1]))

        if recons.shape[1] < gt_videos_vis.shape[1]:
            recons = F.pad(
                recons,
                (0, 0, 0, 0, 0, 0, 0, gt_videos_vis.shape[1] - recons.shape[1], 0, 0),
            )
        elif recons.shape[1] > gt_videos_vis.shape[1]:
            recons = recons[:, : gt_videos_vis.shape[1]]

        result["gt_videos"] = gt_videos_vis
        result["reconstruction"] = recons
        return result

    def _build_explicit_inference_c_fn(self, c_cfg: Any) -> Callable[[Tensor], Tensor]:
        schedule_name = str(getattr(c_cfg, "schedule", "linear")).lower()
        # Inference c_function uses shape-only c(t); lambda scaling is handled by eta_base.
        lambda_ = 1.0
        truncated_a = float(getattr(c_cfg, "truncated_a", 0.8))

        if schedule_name == "linear":
            return lambda k: linear_noise_schedule(k, lambda_)
        if schedule_name == "truncated":
            return lambda k: truncated_noise_schedule(k, lambda_, truncated_a)
        if schedule_name == "constant":
            return constant_noise_schedule
        raise ValueError(
            "inference_schedule.c_function.schedule must be one of: "
            "linear, truncated, constant."
        )

    def _get_inference_schedule_cfg(self) -> Optional[Any]:
        denoising_cfg = getattr(self.cfg, "denoising", None)
        if denoising_cfg is None or not hasattr(denoising_cfg, "inference_schedule"):
            return None
        sched_cfg = denoising_cfg.inference_schedule
        if not bool(getattr(sched_cfg, "enabled", False)):
            return None
        return sched_cfg

    def _get_inference_schedule_lookup(self) -> Optional[InferenceScheduleLookup]:
        """
        Cachable way to get a step size vs noise level schedule based on 
        either uniform, a c function, or classic SNR-space schedulers.
        """
        sched_cfg = self._get_inference_schedule_cfg()
        if sched_cfg is None:
            return None
        c_cfg = getattr(sched_cfg, "c_function", None)
        logsnr_cfg = getattr(sched_cfg, "linear_logsnr", None)
        cache_key = (
            int(self.num_sampling_steps),
            bool(self.use_continuous_timesteps),
            str(getattr(sched_cfg, "family", "identity")).lower(),
            float(getattr(sched_cfg, "t_end", 1e-3)),
            float(getattr(getattr(sched_cfg, "sd3", None), "r", 6.0)),
            int(getattr(sched_cfg, "dense_points", 4096)),
            str(getattr(c_cfg, "schedule", "linear")).lower() if c_cfg is not None else "linear",
            float(getattr(c_cfg, "truncated_a", 0.8)) if c_cfg is not None else 0.8,
            float(getattr(logsnr_cfg, "lambda_max", 15.0)) if logsnr_cfg is not None else 15.0,
            float(getattr(logsnr_cfg, "lambda_min", -15.0)) if logsnr_cfg is not None else -15.0,
            float(getattr(logsnr_cfg, "shift", 0.0)) if logsnr_cfg is not None else 0.0,
        )
        prev_key = getattr(self, "_inference_schedule_cache_key", None)
        if prev_key == cache_key and hasattr(self, "_inference_schedule_lookup"):
            return self._inference_schedule_lookup
        family = str(getattr(sched_cfg, "family", "identity")).lower()
        c_fn = None
        if family in {"c_function", "c_fn"}:
            if c_cfg is None:
                raise ValueError(
                    "inference_schedule.family=c_function requires "
                    "inference_schedule.c_function settings."
                )
            c_fn = self._build_explicit_inference_c_fn(c_cfg)
        lookup = build_inference_schedule_lookup(
            num_sampling_steps=int(self.num_sampling_steps),
            family=family,
            is_continuous=bool(self.use_continuous_timesteps),
            t_end=float(getattr(sched_cfg, "t_end", 1e-3)),
            sd3_r=float(getattr(getattr(sched_cfg, "sd3", None), "r", 6.0)),
            logsnr_max=float(getattr(logsnr_cfg, "lambda_max", 15.0)) if logsnr_cfg is not None else 15.0,
            logsnr_min=float(getattr(logsnr_cfg, "lambda_min", -15.0)) if logsnr_cfg is not None else -15.0,
            logsnr_shift=float(getattr(logsnr_cfg, "shift", 0.0)) if logsnr_cfg is not None else 0.0,
            c_fn=c_fn,
            c_dense_points=int(getattr(sched_cfg, "dense_points", 4096)),
        )
        self._inference_schedule_cache_key = cache_key
        self._inference_schedule_lookup = lookup
        return lookup

    def _resolve_sampling_time_payload(
        self,
        n_act: Tensor,
        num_sampling_noise_levels: int,
        n_next: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Map from current noise levels (and next noise levels) as discrete counters
        to noise levels native to a models input space (such as on [0,1] for continuous
        or a DDPM/DDIM index for discrete).
        """
        if n_next is None:
            n_next = (n_act - 1).clamp(min=0)
        lookup = self._get_inference_schedule_lookup()
        if lookup is None:
            current_level, next_level = self._n_act_to_noise_levels(
                n_act=n_act,
                num_sampling_noise_levels=num_sampling_noise_levels,
                n_next=n_next,
            )
            eta_scale = torch.ones_like(n_act, dtype=torch.float32)
            return {
                "current_level": current_level,
                "next_level": next_level,
                "eta_scale": eta_scale.to(dtype=torch.float32),
            }

        N = max(int(num_sampling_noise_levels - 1), 1)
        idx_from = n_act.clamp(min=0, max=N).long()
        idx_to = n_next.clamp(min=0, max=N).long()
        levels = lookup.levels.to(device=n_act.device) # maps from index to the noise level enforced under a schedule
        eta_table = lookup.eta_scale.to(device=n_act.device) # maps from index to step size

        if self.use_continuous_timesteps:
            # map from counter-based noise level index into a noise level
            # that may follow a warped schedule not necessarily 1/D every time
            current_level = levels[idx_from]
            next_level = levels[idx_to]
        else:
            discrete = lookup.discrete_indices.to(device=n_act.device)
            from_idx = discrete[idx_from]
            to_idx = discrete[idx_to]
            noisy_from = self.denoising_algo.ddim_idx_to_noise_level(from_idx.clamp(min=1))
            noisy_to = self.denoising_algo.ddim_idx_to_noise_level(to_idx.clamp(min=1))
            current_level = torch.where(n_act > 0, noisy_from, torch.full_like(n_act, -1))
            next_level = torch.where(n_next > 0, noisy_to, torch.full_like(n_next, -1))

        return {
            "current_level": current_level,
            "next_level": next_level,
            "eta_scale": eta_table[idx_from].to(dtype=torch.float32),
        }

    @torch.no_grad()
    def _eval_denoising(self, batch, batch_idx, dataloader_idx, namespace="training") -> None:
        """Evaluate the denoising performance during training."""
        result = self._run_eval_denoising_once(
            batch,
            batch_idx,
            dataloader_idx,
            namespace=namespace,
            prepare_visuals=True,
        )
        output = result["output"]
        gt_videos = result["gt_videos"]
        recons = result["reconstruction"]
        video_metadata = result["video_metadata"]
        if gt_videos is None or recons is None:
            return

        all_videos = {
            "gt": gt_videos,
            "reconstruction": recons,
        }

        self._log_videos(all_videos, namespace, dataloader_idx, video_metadata=video_metadata)

        if not self.trainer.sanity_checking:
            self._update_metrics(all_videos, dataloader_idx=dataloader_idx)

    def _sample_all_videos(
        self, batch, batch_idx, namespace="validation"
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Dict]]:
        """
        Sample all task-specific videos and decode latents when needed.
        """
        xs, conditions, _, gt_videos, video_metadata = batch
        if conditions is not None:
            conditions = conditions.to(device=xs.device, dtype=xs.dtype)
        all_videos: Dict[str, Tensor] = {"gt": xs}
        other_results_by_task: Dict[str, Dict] = {}
        prediction_kwargs = {}

        for task in self.tasks:
            match task:
                case "prediction":
                    sample_videos, sample_other_results = self._predict_videos(
                        xs, conditions=conditions, prediction_kwargs=prediction_kwargs
                    )
                    all_videos[task] = sample_videos
                    other_results_by_task[task] = sample_other_results
                case "reconstruction":
                    continue
                case _:
                    raise NotImplementedError

        all_videos = {k: v for k, v in all_videos.items() if v is not None}
        all_videos = {k: v.detach() for k, v in all_videos.items()}
        if self.is_latent_diffusion:
            all_videos = {
                k: self._decode(v) if k != "gt" else gt_videos
                for k, v in all_videos.items()
            }

        def cut_to_same_len(videos):
            min_len = min(v.shape[1] for v in videos.values())
            for key in videos.keys():
                videos[key] = videos[key][:, :min_len]

        cut_to_same_len(all_videos)

        return all_videos, video_metadata, other_results_by_task

    def _predict_videos(
        self,
        xs: Tensor,
        conditions: Optional[Tensor] = None,
        prediction_kwargs: Optional[Dict] = None,
    ) -> Tuple[Tensor, Dict]:
        """
        Predict videos with configurable inference strategy.
        """
        if conditions is not None:
            conditions = conditions.to(device=xs.device, dtype=xs.dtype)
        strategy = str(self.cfg.tasks.prediction.sampling_strategy).lower()
        match strategy:
            case "streaming":
                xs_pred, other_results = self._streaming_inference(
                    xs, conditions, prediction_kwargs=prediction_kwargs
                )
                return xs_pred, other_results or {}
            case _:
                raise NotImplementedError(f"Sampling strategy {strategy} is not implemented.")

    def _n_act_to_noise_levels(
        self,
        n_act: Tensor,
        num_sampling_noise_levels: int,
        n_next: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Convert per-frame denoising counter to (from_nl, to_nl).
        """
        if n_next is None:
            n_next = (n_act - 1).clamp(min=0)
        if self.use_continuous_timesteps:
            denom = max(num_sampling_noise_levels - 1, 1)
            from_nl = n_act.float() / denom
            to_nl = n_next.float() / denom
        else:
            noisy_from = self.denoising_algo.ddim_idx_to_noise_level(n_act.clamp(min=1))
            noisy_to = self.denoising_algo.ddim_idx_to_noise_level(n_next.clamp(min=1))
            from_nl = torch.where(
                n_act > 0,
                noisy_from,
                torch.full_like(n_act, -1),
            )
            to_nl = torch.where(
                n_next > 0,
                noisy_to,
                torch.full_like(n_next, -1),
            )
        return from_nl, to_nl

    def _stabilization_level_to_discrete_t(self, stabilization_level: float) -> int:
        """
        Convert stabilization level to a discrete diffusion index.
        """
        max_t = max(int(self.num_noise_levels) - 1, -1)
        t = int(float(stabilization_level) * float(self.num_noise_levels) - 1.0)
        return max(-1, min(max_t, t))

    def _apply_context_stabilization(
        self,
        x_ctx: Tensor,
        is_generated: Tensor,
        is_valid: Tensor,
        stabilization_level: float,
    ) -> Tensor:
        """
        Re-noise generated context tokens using the forward corruption kernel.
        """
        if stabilization_level <= 0 or x_ctx.shape[1] == 0:
            return x_ctx
        mask = torch.logical_and(is_generated, is_valid)
        if not bool(mask.any()):
            return x_ctx

        if self.use_continuous_timesteps:
            k = torch.full(
                mask.shape,
                float(stabilization_level),
                device=x_ctx.device,
                dtype=x_ctx.dtype,
            )
        else:
            t = self._stabilization_level_to_discrete_t(stabilization_level)
            if t < 0:
                return x_ctx
            k = torch.full(mask.shape, t, device=x_ctx.device, dtype=torch.long)

        # Use self.generator so that context stabilization noise is governed by
        # algorithm.logging.deterministic (same RNG as streaming init / training
        # noise levels). When deterministic is None, self.generator is None and
        # this is equivalent to the prior torch.randn_like behavior.
        stab_noise = torch.randn(
            x_ctx.shape,
            device=x_ctx.device,
            dtype=x_ctx.dtype,
            generator=self.generator,
        ).clamp(-self.clip_noise, self.clip_noise)
        x_noised = self.denoising_algo.q_sample(
            x_ctx,
            k,
            noise=stab_noise,
        )
        mask_expanded = mask[(...,) + (None,) * (x_ctx.ndim - 2)]
        return torch.where(mask_expanded, x_noised, x_ctx)

    def _get_ctx_noise_levels(
        self,
        act_from: Tensor,
        act_to: Tensor,
        left_context_tokens: int,
        right_context_tokens: int,
        stabilization_level: float,
        left_stabilized_ctx_mask: Optional[Tensor],
        right_stabilized_ctx_mask: Optional[Tensor],
        x_dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Return model-visible and true local-window noise levels under stabilization.
        """
        bsz = act_from.shape[0]
        device = act_from.device
        left_context_tokens = int(left_context_tokens)
        right_context_tokens = int(right_context_tokens)

        def _context_shaped_values(
            context_tokens: int,
            value: float | int,
            dtype: torch.dtype,
        ) -> Tensor:
            return torch.full(
                (bsz, context_tokens),
                value,
                device=device,
                dtype=dtype,
            )

        if self.use_continuous_timesteps:
            left_ctx_from = _context_shaped_values(left_context_tokens, 0.0, x_dtype)
            left_ctx_to = _context_shaped_values(left_context_tokens, 0.0, x_dtype)
            left_true_from = _context_shaped_values(left_context_tokens, 0.0, x_dtype)
            left_true_to = _context_shaped_values(left_context_tokens, 0.0, x_dtype)
            right_ctx_from = _context_shaped_values(right_context_tokens, 0.0, x_dtype)
            right_ctx_to = _context_shaped_values(right_context_tokens, 0.0, x_dtype)
            right_true_from = _context_shaped_values(right_context_tokens, 0.0, x_dtype)
            right_true_to = _context_shaped_values(right_context_tokens, 0.0, x_dtype)
            if (
                left_stabilized_ctx_mask is not None
                and stabilization_level > 0.0
                and stabilization_level <= 1.0
                and bool(left_stabilized_ctx_mask.any())
            ):
                left_stab = torch.full_like(left_ctx_from, stabilization_level)
                left_ctx_from = torch.where(
                    left_stabilized_ctx_mask,
                    left_stab,
                    left_ctx_from,
                )
                left_ctx_to = torch.where(
                    left_stabilized_ctx_mask,
                    left_stab,
                    left_ctx_to,
                )
            if (
                right_stabilized_ctx_mask is not None
                and stabilization_level > 0.0
                and stabilization_level <= 1.0
                and bool(right_stabilized_ctx_mask.any())
            ):
                right_stab = torch.full_like(right_ctx_from, stabilization_level)
                right_ctx_from = torch.where(
                    right_stabilized_ctx_mask,
                    right_stab,
                    right_ctx_from,
                )
                right_ctx_to = torch.where(
                    right_stabilized_ctx_mask,
                    right_stab,
                    right_ctx_to,
                )
        else:
            left_ctx_from = _context_shaped_values(left_context_tokens, -1, torch.long)
            left_ctx_to = _context_shaped_values(left_context_tokens, -1, torch.long)
            left_true_from = _context_shaped_values(left_context_tokens, -1, torch.long)
            left_true_to = _context_shaped_values(left_context_tokens, -1, torch.long)
            right_ctx_from = _context_shaped_values(right_context_tokens, -1, torch.long)
            right_ctx_to = _context_shaped_values(right_context_tokens, -1, torch.long)
            right_true_from = _context_shaped_values(right_context_tokens, -1, torch.long)
            right_true_to = _context_shaped_values(right_context_tokens, -1, torch.long)
            if (
                left_stabilized_ctx_mask is not None
                and stabilization_level > 0.0
                and bool(left_stabilized_ctx_mask.any())
            ):
                left_stab_t = self._stabilization_level_to_discrete_t(stabilization_level)
                left_stab = torch.full_like(left_ctx_from, left_stab_t)
                left_ctx_from = torch.where(
                    left_stabilized_ctx_mask,
                    left_stab,
                    left_ctx_from,
                )
                left_ctx_to = torch.where(
                    left_stabilized_ctx_mask,
                    left_stab,
                    left_ctx_to,
                )
            if (
                right_stabilized_ctx_mask is not None
                and stabilization_level > 0.0
                and bool(right_stabilized_ctx_mask.any())
            ):
                right_stab_t = self._stabilization_level_to_discrete_t(stabilization_level)
                right_stab = torch.full_like(right_ctx_from, right_stab_t)
                right_ctx_from = torch.where(
                    right_stabilized_ctx_mask,
                    right_stab,
                    right_ctx_from,
                )
                right_ctx_to = torch.where(
                    right_stabilized_ctx_mask,
                    right_stab,
                    right_ctx_to,
                )

        if int(left_context_tokens) > 0 or int(right_context_tokens) > 0:
            # sandwhich active horizon noise levels between left/right context
            from_nl = torch.cat([left_ctx_from, act_from, right_ctx_from], dim=1)
            to_nl = torch.cat([left_ctx_to, act_to, right_ctx_to], dim=1)
            true_from_nl = torch.cat([left_true_from, act_from, right_true_from], dim=1)
            true_to_nl = torch.cat([left_true_to, act_to, right_true_to], dim=1)
            return from_nl, to_nl, true_from_nl, true_to_nl

        # no context so just return
        return act_from, act_to, act_from, act_to
