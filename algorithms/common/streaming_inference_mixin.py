import functools
import math
import sys
from typing import Dict, Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor
from tqdm import tqdm

from algorithms.denoising import EquilibriumMatching, resolve_learning_rate_eta_from_config
from algorithms.denoising.interface import StepOutput
from algorithms.streaming_state import (
    StreamingControls,
    StreamingState,
    resolve_streaming_controls,
)
from algorithms.common.tail_finish_mixin import TailFinishMixin
from utils.velocity_utils import DenoisingStateRows


SamplingControls = StreamingControls
SamplingState = StreamingState


class StreamingInferenceMixin(TailFinishMixin):
    def _validate_readout_controls(self, controls: SamplingControls) -> None:
        if not controls.should_collect_readout_noise_level:
            return
        if not hasattr(self, "readout_head") or self.readout_head is None:
            raise RuntimeError(
                "Readout-enabled inference requires `self.readout_head`."
            )
        if not hasattr(self, "_readout_model_kwargs"):
            raise RuntimeError(
                "Readout-enabled inference requires `self._readout_model_kwargs`."
            )
        if getattr(self, "readout_cfg", None) is None:
            raise RuntimeError(
                "Readout-enabled inference requires `self.readout_cfg`."
            )
        if controls.uses_readout_based_stopping and not bool(getattr(self, "readout_enabled", False)):
            raise ValueError(
                "Inference `stop_based_on=readout` requires `algorithm.readout.enabled=true`."
            )

    def _get_local_window_readout_noise_levels(
        self,
        *,
        result: StepOutput,
        left_context_tokens: int,
        right_context_tokens: int,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Predict sampler-time readout timestamps for local-window context and active frames.
        """

        if not hasattr(self, "readout_head") or self.readout_head is None:
            raise RuntimeError(
                "Readout-enabled inference requires `self.readout_head`."
            )

        extra = result.additional_output
        if not isinstance(extra, dict):
            raise RuntimeError(
                "Readout-enabled inference requested hidden states, but "
                "the sampler result did not return them."
            )
        t_hat = extra.get("readout_t_hat", None)
        if t_hat is None:
            if "hidden_states" not in extra:
                raise RuntimeError(
                    "Readout-enabled inference requested hidden states, but "
                    "the sampler result did not return them."
                )
            with torch.no_grad():
                t_hat = self.readout_head.predict(extra["hidden_states"])
        if t_hat is None:
            raise RuntimeError(
                "Readout-enabled streaming inference requested hidden states, but "
                "the sampler result did not return them."
            )
        t_hat = t_hat.detach().to(dtype=torch.float32)

        active_start = int(left_context_tokens)
        active_end = int(t_hat.shape[1]) - int(right_context_tokens)

        context_readout_noise_level = None
        if left_context_tokens > 0 or right_context_tokens > 0:
            context_readout_noise_level = torch.cat(
                [
                    t_hat[:, :left_context_tokens],
                    t_hat[:, active_end:],
                ],
                dim=1,
            )
        readout_frame_noise_level = t_hat[:, active_start:active_end]
        return context_readout_noise_level, readout_frame_noise_level

    def _get_local_context_parts(
        self,
        *,
        state: SamplingState,
        controls: SamplingControls,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Return explicit left/right context tensors and masks for the local window.
        """

        left_ctx, left_is_generated, left_is_valid = state.get_context(
            controls.sliding_context_tokens
        )
        right_ctx = state.x_act.new_empty((state.B, 0, *state.x_shape))
        right_is_generated = torch.zeros(
            (state.B, 0),
            dtype=torch.bool,
            device=state.x_act.device,
        )
        right_is_valid = torch.zeros_like(right_is_generated)
        return (
            left_ctx,
            right_ctx,
            left_is_generated,
            left_is_valid,
            right_is_generated,
            right_is_valid,
        )

    def _evaluate_terminal_observational_readout(
        self,
        *,
        state: SamplingState,
        conditions: Optional[Tensor],
        controls: SamplingControls,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Run one terminal silent function evaluation to log the readout time of
        the boundary state that is about to be emitted.

        Why this exists:
        During streaming inference, the last readout prediction gathered inside
        the real denoising loop is produced by the solver evaluation that
        *created* the current active state. When we later snap emitted frames to
        their terminal logged values, that readout therefore corresponds to the
        pre-boundary solver evaluation rather than the post-update state that is
        actually being committed.

        To align logging with the emitted data, we run one final sampler
        evaluation immediately before `snap_rows`. This uses the normal sample
        step path so that scheduler-specific behavior, speculative momentum
        inputs, and hidden-state/readout plumbing remain identical to real
        inference. The evaluation is made observational only by passing an
        all-false `active_update_mask`, which guarantees that:

        - no active frame is updated,
        - no denoising counter is decremented,
        - no NFE accounting is accumulated,
        - no persistent streaming state is mutated.

        This terminal silent function evaluation is purely for logging purposes.
        Its outputs must not be fed back into stopping logic, scheduling logic,
        state updates, or any other sampling behavior.
        """
        if not controls.should_collect_readout_noise_level:
            return None, None

        cond_slice = self._slice_and_pad_window_conditions(
            conditions,
            state.global_window_start_index,
            state.global_window_end_index,
            self.forward_window_size_in_tokens,
        )

        # Force a pure boundary evaluation: the sampler still executes its usual
        # forward path, but every active token is masked out of state updates.
        no_update_mask = torch.zeros_like(state.n_act, dtype=torch.bool)
        (
            _x_act_new,
            _n_act_new,
            _momentum_new,
            _frame_res,
            _step_delta_act,
            context_readout_noise_level,
            readout_frame_noise_level,
            _eta_local_step_idx,
            _eta_multiplier_actual_active,
            _eta_multiplier_expected_active,
            _cleanup_update,
        ) = self._inner_refine_step(
            state=state,
            controls=controls,
            conditions_slice=cond_slice,
            active_update_mask=no_update_mask,
        )
        return (
            None
            if context_readout_noise_level is None
            else context_readout_noise_level.detach().clone(),
            None
            if readout_frame_noise_level is None
            else readout_frame_noise_level.detach().clone(),
        )

    def _streaming_inference(
        self,
        xs: Tensor,
        conditions: Optional[Tensor] = None,
        prediction_kwargs: Optional[Dict] = None,
    ) -> Tuple[Tensor, Dict[str, object]]:
        """
        Unified streaming inference entry point.
        """

        prediction_kwargs = prediction_kwargs or {}
        stream_cfg = self.cfg.tasks.prediction.streaming
        readout_enabled = bool(getattr(self, "readout_enabled", False))
        uses_readout_based_stopping = (
            str(getattr(stream_cfg, "mode", "fixed")).lower() == "adaptive"
            and str(getattr(stream_cfg, "stop_based_on", "gradnorm")).lower() == "readout"
        )
        sched_cfg = self._get_inference_schedule_cfg()
        solver_index_source = (
            str(getattr(sched_cfg, "inference_solver_index_source", "schedule")).lower()
            if sched_cfg is not None
            else "schedule"
        )
        # The unified single-loop cleanup always needs the per-step noise-level
        # readout (sigma_hat) to decide cleanup entry and per-frame eta.
        tail_finish_cfg = getattr(stream_cfg, "tail_finish", None)
        tail_finish_wants_readout = bool(tail_finish_cfg) and bool(
            getattr(tail_finish_cfg, "enabled", False)
        )
        should_collect_readout_noise_level = readout_enabled and (
            bool(getattr(self.logging_cfg, "log_global_readout_noise_level_schedule", False))
            or uses_readout_based_stopping
            or (solver_index_source == "readout_predicted")
            or bool(getattr(self.logging_cfg, "inference_step_to_noise_level_thresholds", None))
            or tail_finish_wants_readout
        )
        controls = resolve_streaming_controls(
            stream_cfg,
            self.forward_window_size_in_tokens,
            self.num_sampling_steps,
            self.validation_n_sliding_context_tokens,
            self.validation_n_initial_context_tokens,
            self._n_frames_to_n_tokens(stream_cfg.stride_in_frames),
            should_collect_readout_noise_level=should_collect_readout_noise_level,
        )
        self._validate_readout_controls(controls)

        context = xs[:, :controls.initial_context_tokens]
        total_length = xs.shape[1]

        xs_pred, other_results = self._run_streaming_sampler(
            context=context,
            total_length=total_length,
            controls=controls,
            conditions=conditions,
            prediction_kwargs=prediction_kwargs,
        )
        return xs_pred, other_results

    def _run_one_inner_step(
        self,
        state: SamplingState,
        *,
        conditions: Optional[Tensor],
        controls: SamplingControls,
        nfe_per_step: float,
        active_update_mask: Optional[Tensor] = None,
    ) -> Tuple[
        SamplingState,
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        """
        Run one inner denoising step and update `state` in-place.
        """
        cond_slice = self._slice_and_pad_window_conditions(
            conditions,
            state.global_window_start_index,
            state.global_window_end_index,
            self.forward_window_size_in_tokens,
        )

        nfe_active_mask = state.n_act > 0
        if active_update_mask is not None:
            nfe_active_mask = torch.logical_and(nfe_active_mask, active_update_mask)

        (
            x_act_new,
            n_act_new,
            momentum_new,
            frame_res,
            _step_delta_act,
            context_readout_noise_level,
            readout_frame_noise_level,
            eta_local_step_idx,
            eta_multiplier_actual_active,
            eta_multiplier_expected_active,
            cleanup_update,
        ) = self._inner_refine_step(
            state=state,
            controls=controls,
            conditions_slice=cond_slice,
            active_update_mask=active_update_mask,
        )

        # update state data and noise levels in place
        state.x_act = x_act_new
        state.n_act = n_act_new
        state.momentum = momentum_new
        # Persist unified-cleanup per-frame state (single-loop adaptive cleanup).
        if cleanup_update is not None:
            state.tail_mask = cleanup_update["tail_mask"]
            state.tail_j = cleanup_update["tail_j"]
        state.add_nfe(nfe_active_mask, nfe_per_step=nfe_per_step)

        frame_res_log = frame_res.detach().clone()
        frame_res_log = frame_res_log.masked_fill(~nfe_active_mask, float("nan"))

        # Legacy per-frame gradnorm/readout freezer (adaptive without unified
        # cleanup). When unified cleanup is enabled, "done" is owned entirely by
        # the cleanup completion (n_act is forced to 0 in `_inner_refine_step`),
        # so this legacy stopper is disabled to avoid emitting frames before
        # they finish their cleanup phase. This could be possibly deleted if 
        # we want to get rid of the previously implemented magnitude-based stopping
        tail_finish_cfg = getattr(controls, "tail_finish", None)
        cleanup_active = bool(
            tail_finish_cfg is not None
            and getattr(tail_finish_cfg, "enabled", False)
        )
        if controls.adaptive and not cleanup_active:
            eligible_done_mask = None
            if controls.uses_readout_based_stopping:
                if readout_frame_noise_level is None:
                    raise RuntimeError(
                        "Streaming `stop_based_on=readout` requested readout noise levels, but no readout frame times were produced."
                    )
                if controls.frame_done_noise_level > 0:
                    done_mask = readout_frame_noise_level <= controls.frame_done_noise_level
                    eligible_done_mask = torch.logical_and(done_mask, nfe_active_mask)
            elif controls.frame_done_eps > 0:
                done_mask = frame_res <= controls.frame_done_eps
                eligible_done_mask = torch.logical_and(done_mask, nfe_active_mask)

            if eligible_done_mask is not None:
                state.n_act = torch.where(
                    eligible_done_mask,
                    torch.zeros_like(state.n_act),
                    state.n_act,
                )

        return (
            state,
            frame_res,
            frame_res_log,
            context_readout_noise_level,
            readout_frame_noise_level,
            eta_local_step_idx,
            eta_multiplier_actual_active,
            eta_multiplier_expected_active,
        )

    def _inner_refine_step(
        self,
        state: SamplingState,
        controls: SamplingControls,
        conditions_slice: Optional[Tensor],
        active_update_mask: Optional[Tensor] = None,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Dict[str, Tensor]],
    ]:
        """
        One denoising step over the local window [left_ctx | x_act | right_ctx].
        Active update mask helps handle freezing some frames in horzion,
        such as during the bake-in period.
        """
        bsz = state.B
        device = state.x_act.device
        dtype = state.x_act.dtype
        (
            left_ctx,
            right_ctx,
            left_is_generated,
            left_is_valid,
            right_is_generated,
            right_is_valid,
        ) = self._get_local_context_parts(state=state, controls=controls)
        left_context_tokens = int(left_ctx.shape[1])
        right_context_tokens = int(right_ctx.shape[1])
        active_start = left_context_tokens
        active_end = left_context_tokens + state.active_horizon_tokens

        left_stabilized_ctx_mask = None
        right_stabilized_ctx_mask = None
        if controls.conditioning_mode == "stabilized_conditional":
            left_stabilized_ctx_mask = torch.logical_and(
                left_is_generated,
                left_is_valid,
            )
            right_stabilized_ctx_mask = torch.logical_and(
                right_is_generated,
                right_is_valid,
            )
            left_ctx = self._apply_context_stabilization(
                x_ctx=left_ctx,
                is_generated=left_is_generated,
                is_valid=left_is_valid,
                stabilization_level=controls.stabilization_level,
            )
            right_ctx = self._apply_context_stabilization(
                x_ctx=right_ctx,
                is_generated=right_is_generated,
                is_valid=right_is_valid,
                stabilization_level=controls.stabilization_level,
            )
        x_local = torch.cat([left_ctx, state.x_act, right_ctx], dim=1)

        if active_update_mask is not None:
            active_update_mask = active_update_mask.to(device=device, dtype=torch.bool)
        else:
            active_update_mask = torch.ones_like(state.n_act, dtype=torch.bool)
        active_mask = torch.logical_and(state.n_act > 0, active_update_mask)

        # Unified single-loop adaptive cleanup: every active frame keeps being
        # updated each step; only its eta switches when it enters the cleanup
        # phase.
        tail_finish_cfg = getattr(controls, "tail_finish", None)
        cleanup_enabled = bool(
            tail_finish_cfg is not None
            and getattr(tail_finish_cfg, "enabled", False)
            and type(self.denoising_algo) is EquilibriumMatching
        )

        # next noise level per active frame by decrementing state.n_act by 1
        n_act_next = torch.where(
            active_mask,
            (state.n_act - 1).clamp(min=0),
            state.n_act,
        )
        time_payload = self._resolve_sampling_time_payload(
            state.n_act,
            controls.num_sampling_noise_levels,
            n_next=n_act_next,
        )
        current_level = time_payload["current_level"]
        next_level = time_payload["next_level"]
        eta_scale = time_payload["eta_scale"]

        # process noise levels (single source of truth), while allowing
        # context renoising for the model pass and clean context for sampler logic.
        model_current_nl, model_next_nl, sampler_current_nl, sampler_next_nl = self._get_ctx_noise_levels(
            act_from=current_level,
            act_to=next_level,
            left_context_tokens=left_context_tokens,
            right_context_tokens=right_context_tokens,
            stabilization_level=controls.stabilization_level,
            left_stabilized_ctx_mask=left_stabilized_ctx_mask,
            right_stabilized_ctx_mask=right_stabilized_ctx_mask,
            x_dtype=dtype,
        )

        sample_kwargs = dict(
            scheduler_type=self.sampling_scheduler_type,
            model_current_noise_levels=model_current_nl,
            model_next_noise_levels=model_next_nl,
            sampler_current_noise_levels=sampler_current_nl,
            sampler_next_noise_levels=sampler_next_nl,
        )

        model_kwargs = dict(
            strategy=self.denoising_cfg.strategy,
            noise_abs_max=self.denoising_cfg.noise_abs_max,
        )
        if controls.should_collect_readout_noise_level:
            model_kwargs.update(self._readout_model_kwargs())

        # History guidance: build the "null history" branch inputs (a token mask
        # over all context/history tokens plus matching noise). The denoising
        # algo pops these out of model_kwargs and runs the extra guided pass.
        hist_guidance_scale = float(
            getattr(self.denoising_cfg, "history_guidance_scale", 0.0)
        )
        if hist_guidance_scale > 0.0 and (left_context_tokens + right_context_tokens) > 0:
            history_mask = torch.zeros(
                (bsz, x_local.shape[1]), dtype=torch.bool, device=device
            )
            if left_context_tokens > 0:
                history_mask[:, :left_context_tokens] = True
            if right_context_tokens > 0:
                history_mask[:, active_end:] = True
            history_noise = torch.randn(
                x_local.shape,
                device=device,
                dtype=dtype,
                generator=self.generator,
            ).clamp(-self.clip_noise, self.clip_noise)
            model_kwargs["history_guidance_mask"] = history_mask
            model_kwargs["history_guidance_noise"] = history_noise

        solver_index_source = "schedule" # schedule for native time, readout_predicted driven by predicted noise levels
        lookup = None
        eta_local_step_idx: Optional[Tensor] = None
        eta_multiplier_actual_active: Optional[Tensor] = None
        eta_multiplier_expected_active: Optional[Tensor] = None

        if type(self.denoising_algo) is EquilibriumMatching:
            eta_base = resolve_learning_rate_eta_from_config(
                self.cfg.denoising, self.num_sampling_steps
            ) # get eta from the config, involving base multiplier
            sched_cfg = self._get_inference_schedule_cfg()
            solver_index_source = (
                str(getattr(sched_cfg, "inference_solver_index_source", "schedule")).lower()
                if sched_cfg is not None
                else "schedule"
            )
            # c(k) for x_start reconstruction follows the current solver index.
            # Readout-predicted eta modulation is resolved inside EqF sample_step
            # from the current forward pass hidden states.
            sample_kwargs["c_current_noise_levels"] = sampler_current_nl
            sample_kwargs["inference_solver_index_source"] = solver_index_source
            sample_kwargs["needs_readout_prediction"] = bool(
                controls.should_collect_readout_noise_level
            )
            if bool(sample_kwargs["needs_readout_prediction"]):
                sample_kwargs["readout_predict_fn"] = self.readout_head.predict
            eta_scale_local = torch.cat(
                [
                    torch.ones(
                        (bsz, left_context_tokens),
                        device=eta_scale.device,
                        dtype=eta_scale.dtype,
                    ),
                    eta_scale,
                    torch.ones(
                        (bsz, right_context_tokens),
                        device=eta_scale.device,
                        dtype=eta_scale.dtype,
                    ),
                ],
                dim=1,
            )
            lookup = self._get_inference_schedule_lookup()
            if lookup is not None and lookup.eta_at_k is not None:
                # Family-agnostic: any schedule that populates eta_at_k
                # can be reindexed by a predicted native noise level k.
                sample_kwargs["eta_at_k_fn"] = lookup.eta_at_k
            # effective eta is the fully resolved per-step step size after
            # base knobs (eta_multiple, lambda divisor, 1/num_sampling_steps)
            # and schedule/readout-induced scaling.
            sample_kwargs["eta_base"] = float(eta_base)
            sample_kwargs["eta"] = eta_base * eta_scale_local

            # Inject the unified cleanup eta policy. The EqF eta seam calls this
            # after the forward pass so it can use the freshest sigma_hat
            # (readout_t_hat). It returns the per-frame eta plus updated
            # tail mask / counter / done, which we read back below.
            if cleanup_enabled:
                sample_kwargs["step_eta_resolver"] = functools.partial(
                    self._compute_cleanup_step_eta,
                    active_start=active_start,
                    active_end=active_end,
                    active_mask=active_mask,
                    tail_mask=state.tail_mask,
                    tail_j=state.tail_j,
                    n_cleanup_steps=int(controls.tail_finish.n_cleanup_steps),
                    step_multiplier=float(controls.tail_finish.step_multiplier),
                    sigma_thresh=float(controls.emit_noise_level),
                    eta_base=float(eta_base),
                    eta_at_k_fn=sample_kwargs.get("eta_at_k_fn", None),
                )

            # prepend/append no-op context momentum for shape alignment in sample_step
            sample_kwargs["mu"] = getattr(self.cfg.denoising, "mu", 0.0)
            left_ctx_mom = torch.zeros(
                bsz,
                left_context_tokens,
                *state.x_shape,
                device=device,
                dtype=dtype,
            )
            right_ctx_mom = torch.zeros(
                bsz,
                right_context_tokens,
                *state.x_shape,
                device=device,
                dtype=dtype,
            )
            active_mom = state.momentum
            sample_kwargs["momentum"] = torch.cat(
                [left_ctx_mom, active_mom, right_ctx_mom], dim=1
            )

        proc_conditions = self._mask_window_conditions(conditions_slice)

        result = self.denoising_algo.sample_step(
            x=x_local,
            curr_noise_level=model_current_nl,
            next_noise_level=model_next_nl,
            conditions=proc_conditions,
            conditions_mask=None,
            model=self.denoising_model.forward,
            model_kwargs=model_kwargs,
            sample_kwargs=sample_kwargs,
        )

        x_pred_act = result.x[:, active_start:active_end]
        active_mask_exp = rearrange(active_mask, "... -> ..." + " 1" * len(self.x_shape)) # crucial for no-op steps in things like the bake-in period, where we need no-oppability beyond just the n_act based one.
        x_act_new = torch.where(active_mask_exp, x_pred_act, state.x_act)
        n_act_new = n_act_next

        # Read back the unified-cleanup bookkeeping produced by the injected eta
        # resolver. Frames that have completed their cleanup phase (j >= S) are
        # forced to n_act=0 so the standard counter-based emit checker grabs
        # them; the tail mask / counter are persisted by `_run_one_inner_step`.
        cleanup_update: Optional[Dict[str, Tensor]] = None
        if cleanup_enabled:
            extra = result.additional_output if isinstance(result.additional_output, dict) else {}
            done = extra.get("cleanup_done", None)
            tail_mask_next = extra.get("cleanup_tail_mask_next", None)
            tail_j_next = extra.get("cleanup_tail_j_next", None)
            if done is None or tail_mask_next is None or tail_j_next is None:
                raise RuntimeError(
                    "Unified cleanup is enabled but the eta resolver did not "
                    "return tail bookkeeping (cleanup_done / cleanup_tail_mask_next "
                    "/ cleanup_tail_j_next) in additional_output."
                )
            # this an force n_act_new to 0 based on an alterantive n_act clock defined 
            # by the number of clean up steps.
            n_act_new = torch.where(done, torch.zeros_like(n_act_new), n_act_new)
            cleanup_update = {
                "tail_mask": tail_mask_next.to(device=state.tail_mask.device),
                "tail_j": tail_j_next.to(device=state.tail_j.device),
            }

        frame_res, step_delta_act, momentum_local = self._unpack_stepoutput_result(
            result=result,
            active_start=active_start,
            active_end=active_end,
            x_local=x_local,
        )
        momentum_new = momentum_local[:, active_start:active_end]

        # Optionally disable momentum for cleanup frames: once a frame is in the
        # tail-cleanup phase we zero its carried momentum, so the next step sees
        # zero input momentum and performs a pure GD update (x = x - eta * v).
        # Zeroing the carry each step propagates this for the whole phase; only
        # the entering step still carries momentum from the standard phase, since
        # cleanup entry is decided after that step's forward pass.
        if (
            cleanup_update is not None
            and cleanup_enabled
            and bool(getattr(controls.tail_finish, "disable_momentum", True))
        ):
            tail_next = cleanup_update["tail_mask"]
            tail_next_exp = rearrange(
                tail_next, "... -> ..." + " 1" * len(self.x_shape)
            ).to(device=momentum_new.device)
            momentum_new = torch.where(
                tail_next_exp,
                torch.zeros_like(momentum_new),
                momentum_new,
            )

        context_readout_noise_level: Optional[Tensor] = None
        readout_frame_noise_level: Optional[Tensor] = None
        if controls.should_collect_readout_noise_level:
            context_readout_noise_level, readout_frame_noise_level = self._get_local_window_readout_noise_levels(
                result=result,
                left_context_tokens=left_context_tokens,
                right_context_tokens=right_context_tokens,
            )
        if (
            type(self.denoising_algo) is EquilibriumMatching
            and solver_index_source == "readout_predicted"
            and bool(active_mask.any())
        ):
            eta_multiplier_expected_active = eta_scale[active_mask].detach().float()
            eta_local_step_idx = (
                controls.num_sampling_steps - state.n_act[active_mask]
            ).detach().to(dtype=torch.long)
            if (
                readout_frame_noise_level is not None
                and lookup is not None
                and lookup.eta_at_k is not None
            ):
                eta_now = lookup.eta_at_k(
                    readout_frame_noise_level.to(dtype=torch.float32)
                )
                eta_multiplier_actual_active = eta_now[active_mask].detach().float()
            else:
                eta_multiplier_actual_active = eta_multiplier_expected_active

        return (
            x_act_new,
            n_act_new,
            momentum_new,
            frame_res,
            step_delta_act,
            context_readout_noise_level,
            readout_frame_noise_level,
            eta_local_step_idx,
            eta_multiplier_actual_active,
            eta_multiplier_expected_active,
            cleanup_update,
        )

    def _unpack_stepoutput_result(
        self,
        *,
        result: StepOutput,
        active_start: int,
        active_end: int,
        x_local: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Unpack StepOutput into residual magnitudes, step deltas and momentum.
        """
        if result.step_delta is None:
            raise ValueError("Expected StepOutput.step_delta for adaptive residual logic.")

        step_delta = result.step_delta.detach()
        if result.step_delta_scale is not None:
            step_delta_scale = result.step_delta_scale.detach()
        else:
            step_delta_scale = torch.ones(
                step_delta.shape[:2],
                device=step_delta.device,
                dtype=step_delta.dtype,
            ) # compat with the rearrange and how we form step_delta_scale in denoising_algo files

        step_delta_scale = step_delta_scale.clamp_min(1e-8)
        step_delta_scale_exp = rearrange(
            step_delta_scale,
            "... -> ..." + " 1" * len(self.x_shape),
        )
        raw_velocity = bool(getattr(self.logging_cfg, "raw_velocity", True))
        if raw_velocity:
            logged_step_delta = step_delta
        else:
            logged_step_delta = step_delta / step_delta_scale_exp

        logged_act = logged_step_delta[:, active_start:active_end]
        frame_res = torch.linalg.vector_norm(
            logged_act.float(), dim=tuple(range(2, logged_act.ndim))
        )
        step_delta_act = step_delta[:, active_start:active_end]

        if result.next_momentum is not None:
            momentum_local = result.next_momentum.detach()
        else:
            momentum_local = torch.zeros_like(x_local)

        return frame_res, step_delta_act, momentum_local

    def _compute_ready_prefix_blocks(
        self,
        n_act: Tensor,
        stop_metric: Optional[Tensor],
        stride_in_tokens: int,
        stop_threshold: float,
    ) -> int:
        """
        Compute how many stride-sized prefix blocks are ready to commit.
        """
        active_horizon_tokens = n_act.shape[1]
        full_prefix_limit = active_horizon_tokens - (active_horizon_tokens % stride_in_tokens)
        if full_prefix_limit <= 0:
            return 0
        ready_prefix_blocks = 0
        for start in range(0, full_prefix_limit, stride_in_tokens):
            end = min(start + stride_in_tokens, active_horizon_tokens)
            counter_ready = torch.all(n_act[:, start:end] <= 0, dim=1)
            if stop_metric is not None and stop_threshold > 0:
                metric_ready = torch.amax(stop_metric[:, start:end], dim=1) <= stop_threshold
                block_ready = torch.logical_or(counter_ready, metric_ready)
            else:
                block_ready = counter_ready
            if bool(block_ready.all()):
                ready_prefix_blocks += 1
            else:
                break
        return ready_prefix_blocks

    def _advance_active_frame_mapping(
        self,
        active_abs_frame_idx: Tensor,
        next_abs_frame_idx: int,
        emit_size: int,
        device: torch.device,
    ) -> Tuple[Tensor, int]:
        if emit_size == 0:
            return active_abs_frame_idx, next_abs_frame_idx
        new_abs = torch.arange(
            next_abs_frame_idx,
            next_abs_frame_idx + emit_size,
            device=device,
            dtype=torch.long,
        )
        active_abs_frame_idx = torch.cat([active_abs_frame_idx[emit_size:], new_abs], dim=0)
        next_abs_frame_idx += emit_size
        return active_abs_frame_idx, next_abs_frame_idx

    def _run_streaming_sampler(
        self,
        context: Tensor,
        total_length: int,
        controls: StreamingControls,
        conditions: Optional[Tensor],
        prediction_kwargs: Optional[Dict] = None,
    ) -> Tuple[Tensor, Dict[str, object]]:
        """
        Global streaming sampler. Emits `stride_in_tokens` frames at a time until
        `total_length` frames have been produced (including initial context).
        """

        prediction_kwargs = prediction_kwargs or {}
        nfe_per_step = 2.0 if self.sampling_scheduler_type == "heun" else 1.0
        should_bake_in = controls.should_bake_in
        total_generated_frames = total_length - controls.initial_context_tokens
        total_nfes_allowed, _ = self._calculate_total_nfes()
        max_tail_rows = 0
        if controls.tail_finish.enabled:
            max_emits = int(
                math.ceil(
                    max(total_generated_frames, 0)
                    / max(int(controls.stride_in_tokens), 1)
                )
            )
            max_tail_rows = max_emits * int(controls.tail_finish.n_cleanup_steps)
        max_touch_rows = int(math.ceil(float(total_nfes_allowed))) + 1 + max_tail_rows

        batch_size = context.shape[0]

        state = StreamingState.init_from_context(
            context=context,
            active_horizon_tokens=controls.validation_horizon_tokens,
            num_sampling_steps=controls.num_sampling_steps,
            clip_noise=self.clip_noise,
            generator=self.generator,
            raw_nfe_init=0.0,
            initial_context_tokens=controls.initial_context_tokens,
        )

        active_abs_frame_idx = torch.arange(
            state.total_committed,
            state.global_window_end_index,
            device=context.device,
            dtype=torch.long,
        )
        next_abs_frame_idx = state.global_window_end_index

        row_state = DenoisingStateRows(
            batch_size=batch_size,
            total_length=total_length,
            total_generated_frames=total_generated_frames,
            initial_context_tokens=controls.initial_context_tokens,
            num_sampling_steps=controls.num_sampling_steps,
            max_touch_rows=max_touch_rows,
            device=context.device,
        )

        row_state.append_rows(
            active_abs_frame_idx=active_abs_frame_idx,
            state_n_act=state.n_act,
            total_committed=state.total_committed,
            sliding_context_tokens=controls.sliding_context_tokens,
            frame_res=None,
            context_readout_noise_level=None,
            init=True,
        )
        # These tensors track the step size used if readout is enabled
        # by scatter adding, we bucket the total step sizes used/expected across batch.
        # Exist as diagnostics to tell where readout head is speeding us up or slowing us down.
        eta_multiplier_local_sum_actual = torch.zeros(
            controls.num_sampling_steps, device=context.device, dtype=torch.float32
        ) # uses the predicted noise level to index into eta_scale
        eta_multiplier_local_sum_expected = torch.zeros(
            controls.num_sampling_steps, device=context.device, dtype=torch.float32
        ) # uses an index-driven counter to index into eta_scale, which would be the noise level for a non-blind model 
        eta_multiplier_local_count = torch.zeros(
            controls.num_sampling_steps, device=context.device, dtype=torch.float32
        ) # tracks the step like num_sampling_steps - state.n_act
        bake_in_cycles = 0
        total_inner_steps = 0
        if should_bake_in:
            bake_in_cycles = max(
                controls.validation_horizon_tokens // controls.stride_in_tokens - 1,
                0,
            )
            block_idx = torch.div(
                torch.arange(
                    controls.validation_horizon_tokens,
                    device=context.device,
                    dtype=torch.long,
                ),
                controls.stride_in_tokens,
                rounding_mode="floor",
            )

            for bake_idx in range(1, bake_in_cycles + 1):
                bake_mask = (block_idx < bake_idx).unsqueeze(0).expand(batch_size, -1)
                for _inner_idx in range(controls.resolved_inner_steps_per_emit):
                    state, frame_res, frame_res_log, context_readout_noise_level, readout_frame_noise_level, eta_local_step_idx, eta_multiplier_actual_active, eta_multiplier_expected_active = self._run_one_inner_step(
                        state=state,
                        conditions=conditions,
                        controls=controls,
                        nfe_per_step=nfe_per_step,
                        active_update_mask=bake_mask,
                    )
                    if (
                        eta_local_step_idx is not None
                        and eta_multiplier_actual_active is not None
                        and eta_multiplier_expected_active is not None
                        and eta_local_step_idx.numel() > 0
                    ):
                        idx = eta_local_step_idx.clamp(min=0, max=controls.num_sampling_steps - 1)
                        eta_multiplier_local_sum_actual.scatter_add_(
                            0, idx, eta_multiplier_actual_active
                        )
                        eta_multiplier_local_sum_expected.scatter_add_(
                            0, idx, eta_multiplier_expected_active
                        )
                        eta_multiplier_local_count.scatter_add_(
                            0, idx, torch.ones_like(eta_multiplier_actual_active)
                        )
                    total_inner_steps += 1
                    row_state.append_rows(
                        active_abs_frame_idx=active_abs_frame_idx,
                        state_n_act=state.n_act,
                        total_committed=state.total_committed,
                        sliding_context_tokens=controls.sliding_context_tokens,
                        frame_res=frame_res_log,
                        context_readout_noise_level=context_readout_noise_level,
                        readout_frame_noise_level=readout_frame_noise_level,
                    )

        total_to_emit = max(total_length - state.total_committed, 0)
        total_emitted = 0

        disable = not sys.stdout.isatty()
        pbar = tqdm(
            total=total_to_emit,
            desc="Streaming sampler",
            leave=False,
            disable=disable,
        )

        cycles_without_emit = 0
        while total_emitted < total_to_emit:
            ready_prefix_blocks = 0
            last_frame_res_log: Optional[Tensor] = None
            saw_readout_frame_noise_level = False
            for _inner_idx in range(controls.resolved_inner_steps_per_emit):
                state, frame_res, frame_res_log, context_readout_noise_level, readout_frame_noise_level, eta_local_step_idx, eta_multiplier_actual_active, eta_multiplier_expected_active = self._run_one_inner_step(
                    state=state,
                    conditions=conditions,
                    controls=controls,
                    nfe_per_step=nfe_per_step,
                )
                if (
                    eta_local_step_idx is not None
                    and eta_multiplier_actual_active is not None
                    and eta_multiplier_expected_active is not None
                    and eta_local_step_idx.numel() > 0
                ):
                    idx = eta_local_step_idx.clamp(min=0, max=controls.num_sampling_steps - 1)
                    eta_multiplier_local_sum_actual.scatter_add_(
                        0, idx, eta_multiplier_actual_active
                    )
                    eta_multiplier_local_sum_expected.scatter_add_(
                        0, idx, eta_multiplier_expected_active
                    )
                    eta_multiplier_local_count.scatter_add_(
                        0, idx, torch.ones_like(eta_multiplier_actual_active)
                    )
                total_inner_steps += 1
                last_frame_res_log = frame_res_log
                saw_readout_frame_noise_level = (
                    saw_readout_frame_noise_level
                    or readout_frame_noise_level is not None
                )
                row_state.append_rows(
                    active_abs_frame_idx=active_abs_frame_idx,
                    state_n_act=state.n_act,
                    total_committed=state.total_committed,
                    sliding_context_tokens=controls.sliding_context_tokens,
                    frame_res=frame_res_log,
                    context_readout_noise_level=context_readout_noise_level,
                    readout_frame_noise_level=readout_frame_noise_level,
                )

                stop_metric: Optional[Tensor] = None
                stop_threshold = 0.0
                # Unified cleanup mode emits purely on the counter: a frame is
                # only "done" once it has finished its cleanup phase (or run out
                # of denoising budget), at which point its n_act is forced to 0.
                # The noise-level/gradnorm emit-readiness comparison is therefore
                # disabled here (emit_noise_level now means the cleanup-entry
                # sigma_thresh, not an emit trigger).
                if controls.adaptive and not controls.tail_finish.enabled:
                    if controls.uses_readout_based_stopping:
                        if readout_frame_noise_level is None:
                            raise RuntimeError(
                                "Streaming `stop_based_on=readout` requested readout noise levels, but no readout frame times were produced."
                            )
                        stop_metric = readout_frame_noise_level
                        stop_threshold = float(controls.emit_noise_level)
                    else:
                        stop_metric = frame_res
                        stop_threshold = float(controls.emit_eps)

                # prefix-driven freezer emits frames, consecutive from the first,
                # that are ready to be emitted based on gradient/ readout noise
                # also handles case of emitting when we hit clean data, using state.n_act
                ready_prefix_blocks = self._compute_ready_prefix_blocks(
                    n_act=state.n_act,
                    stop_metric=stop_metric,
                    stride_in_tokens=controls.stride_in_tokens,
                    stop_threshold=stop_threshold,
                )

                if ready_prefix_blocks > 0:
                    break

            if ready_prefix_blocks == 0:
                cycles_without_emit += 1
                if cycles_without_emit >= controls.max_cycles_without_emit:
                    ready_prefix_blocks = 1
                else:
                    continue
            cycles_without_emit = 0

            # Tail-finish is now folded into the single inner loop above: a frame
            # in the cleanup phase keeps being updated every step with a cleanup
            # eta, and is marked done (n_act -> 0) once it completes its phase.
            # No separate freezing cleanup loop is required.

            emit_tokens = min(
                ready_prefix_blocks * controls.stride_in_tokens,
                total_to_emit - total_emitted,
            )
            if emit_tokens <= 0:
                break

            emit_readout_frame_noise_level_log: Optional[Tensor] = None
            if saw_readout_frame_noise_level:
                # Refresh the readout one final time at the emit boundary so
                # the snapped value reflects the state being committed, not the
                # solver evaluation that produced that state.
                # We intentionally do not do this for frame_res, because that quantity relies
                # on a eta-scaled forward pass (and eta can be undefined at the boundary). Namely,
                # under our current logic the frame_res is derived from a difference in x_pred and x,
                # which we wanted to set to 0, so it clashes.
                _, emit_readout_frame_noise_level_log = (
                    self._evaluate_terminal_observational_readout(
                        state=state,
                        conditions=conditions,
                        controls=controls,
                    )
                )

            row_state.snap_rows(
                emit_size=emit_tokens,
                active_abs_frame_idx=active_abs_frame_idx,
                frame_res=last_frame_res_log,
                readout_frame_noise_level=emit_readout_frame_noise_level_log,
            )

            state.advance(
                emit_tokens=emit_tokens,
                num_sampling_steps=controls.num_sampling_steps,
                clip_noise=self.clip_noise,
                generator=self.generator,
            )
            active_abs_frame_idx, next_abs_frame_idx = self._advance_active_frame_mapping(
                active_abs_frame_idx=active_abs_frame_idx,
                next_abs_frame_idx=next_abs_frame_idx,
                emit_size=emit_tokens,
                device=context.device,
            )
            total_emitted += emit_tokens
            pbar.update(emit_tokens)

        pbar.close()
        other_results = {
            **row_state.to_other_results(),
            "raw_nfe": state.raw_nfe.detach().cpu(),
            "raw_token_nfe": state.raw_token_nfe.detach().cpu(),
            "nfe_per_step": float(nfe_per_step),
            "inner_steps": int(total_inner_steps),
            "forward_evals": float(total_inner_steps) * float(nfe_per_step),
            "total_length_tokens": int(total_length),
            "initial_context_tokens": int(controls.initial_context_tokens),
            "total_generated_tokens": int(max(total_length - controls.initial_context_tokens, 0)),
            "bake_in_cycles": int(bake_in_cycles),
            "bake_in_inner_steps": int(
                bake_in_cycles * controls.resolved_inner_steps_per_emit
            ),
            "denoising_depth": int(controls.num_sampling_steps),
            "total_nfes_allowed": float(total_nfes_allowed),
        }
        if controls.tail_finish.enabled:
            # Unified single-loop cleanup: cleanup steps are regular inner steps,
            # so they are already counted in `inner_steps` / NFE accounting.
            other_results["tail_finish"] = {
                "enabled": True,
                "n_cleanup_steps": int(controls.tail_finish.n_cleanup_steps),
                "step_multiplier": float(controls.tail_finish.step_multiplier),
                "sigma_thresh": float(controls.emit_noise_level),
            }
        if controls.return_streaming_state:
            other_results["streaming_state"] = state
        if bool((eta_multiplier_local_count > 0).any()):
            other_results["eta_multiplier_local_sum_actual"] = (
                eta_multiplier_local_sum_actual.detach().cpu()
            )
            other_results["eta_multiplier_local_sum_expected"] = (
                eta_multiplier_local_sum_expected.detach().cpu()
            )
            other_results["eta_multiplier_local_count"] = eta_multiplier_local_count.detach().cpu()

        return state.committed[:, :total_length], other_results
