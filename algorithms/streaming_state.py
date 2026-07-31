from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple, Dict, Union, cast

import torch
from torch import Tensor


@dataclass
class TailFinishControls:
    """
    Resolved, validated controls for the unified single-loop adaptive cleanup.

    There is no separate finishing loop or frame freezing: every active frame is
    updated on every inner step, and only its per-step `eta` changes once it
    enters the cleanup phase. A frame enters cleanup when its predicted noise
    level `sigma_hat` drops to/below the cleanup-entry threshold (reusing
    `emit_noise_level` as `sigma_thresh`); it then takes `n_cleanup_steps` (`S`)
    finishing steps with `delta_k = sigma_hat * step_multiplier / (S - j)`
    before being marked done.

    See `algorithms/common/tail_finish_mixin.py` for usage.

    The cleanup is always closed-loop: every cleanup step uses the freshest
    `sigma_hat` to recompute `delta_k`.

    Notes:
    - `disable_momentum` (default True): when set, a frame's carried momentum is
      zeroed once it enters cleanup, so cleanup steps are pure GD updates.
    """

    enabled: bool = False
    n_cleanup_steps: int = 4
    disable_momentum: bool = True
    step_multiplier: float = 1.0


@dataclass
class StreamingControls:
    """
    Resolved, validated token-space streaming controls.
    """

    forward_window_size_in_tokens: int
    sliding_context_tokens: int
    initial_context_tokens: int
    validation_horizon_tokens: int
    # Sampling convention:
    # - num_sampling_steps: number of denoising steps
    num_sampling_steps: int
    num_sampling_noise_levels: int
    schedule: str
    mode: str
    adaptive: bool
    stride_in_tokens: int
    resolved_inner_steps_per_emit: int
    conditioning_mode: str
    stabilization_level: float
    stop_based_on: str
    emit_eps: float
    frame_done_eps: float
    emit_noise_level: float
    frame_done_noise_level: float
    should_bake_in: bool
    max_cycles_without_emit: int
    initial_noise_profile: str
    should_collect_readout_noise_level: bool
    return_streaming_state: bool
    tail_finish: TailFinishControls = field(default_factory=TailFinishControls)  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, object]:
        # `dataclasses.asdict` is not precisely typed (pyright treats it as Unknown),
        # so we cast for a stable surface type.
        return cast(Dict[str, object], asdict(self))

    def items(self):
        return self.to_dict().items()

    @property
    def uses_readout_based_stopping(self) -> bool:
        return bool(self.adaptive) and self.stop_based_on == "readout"


def resolve_streaming_controls(
    stream_cfg,
    forward_window_size_in_tokens,
    num_sampling_steps,
    validation_n_sliding_context_tokens,
    validation_n_initial_context_tokens,
    stride_in_tokens_default: Optional[int],
    *,
    should_collect_readout_noise_level: bool = False,
) -> StreamingControls:
    """
    Resolve and validate the streaming control parameters.

    Core definitions (token-space)
    --------------------------------
    `forward_window_size_in_tokens`
        Width of the local model window: how many token it sees. This is the main window size used by
        both training and streaming inference.

    `training_n_context_tokens`
        Number of leading tokens in the training window treated as context.
        During training, the remaining tokens
        (`forward_window_size_in_tokens - training_n_context_tokens`) form the
        effective training horizon.

    `validation_n_sliding_context_tokens`
        Number of committed context tokens reused on each validation streaming
        step. The validation horizon is:
        `validation_horizon_tokens = forward_window_size_in_tokens - validation_n_sliding_context_tokens`.

    `validation_n_initial_context_tokens`
        Number of initial tokens taken from ground-truth history at the start
        of streaming validation before generation begins.

    `stride_in_tokens_default`
        Number of tokens emitted/advanced per commit. If not provided, a schedule-
        dependent token-space default is chosen.

    Frame-space vs token-space
    --------------------------
    Config inputs are frame-space (`*_frames`) and are converted to token-space
    through the model temporal downsampling. This resolver works in token-space.

    Schedule and alignment rules
    -----------------------------
    - `full_sequence`: force `validation_n_sliding_context_tokens = 0`.
    - `chunk` and `full_sequence`: require `stride == validation_horizon_tokens`.
    - All schedules: require `validation_horizon_tokens % stride == 0`.
    - Fixed/adaptive inner-step configuration is constrained so that
        denoising progress aligns exactly with horizon/stride geometry.
        In short, `num_sampling_steps * stride` must be divisible by
        `validation_horizon_tokens`, and resolved `inner_steps_per_emit` must
        match the exact aligned value.
    """

    forward_window_size_in_tokens = int(forward_window_size_in_tokens)
    num_sampling_steps = int(num_sampling_steps)

    validation_sliding_context_tokens = int(validation_n_sliding_context_tokens)
    if (
        validation_sliding_context_tokens < 0
        or validation_sliding_context_tokens >= forward_window_size_in_tokens
    ):
        raise ValueError(
            f"Invalid sliding_context_tokens={validation_sliding_context_tokens}. "
            f"Expected integer in [0, {forward_window_size_in_tokens - 1}]."
        )

    schedule = str(stream_cfg.schedule).lower()
    valid_schedules = {"full_sequence", "chunk", "rolling", "overlap"}
    if schedule not in valid_schedules:
        raise ValueError(
            f"Invalid streaming schedule `{schedule}`. "
            f"Expected one of {sorted(valid_schedules)}."
        )
    if schedule == "full_sequence":
        validation_sliding_context_tokens = 0
        # Full-sequence sampling has no sliding context.

    validation_horizon_tokens = (
        forward_window_size_in_tokens - validation_sliding_context_tokens
    )

    validation_initial_context_tokens = int(validation_n_initial_context_tokens)
    if (validation_initial_context_tokens < 0 or validation_initial_context_tokens >= forward_window_size_in_tokens):
        raise ValueError(
            f"Invalid initial_context_tokens={validation_initial_context_tokens}. "
            f"Expected integer in [0, {forward_window_size_in_tokens - 1}]."
        )

    if validation_sliding_context_tokens != validation_initial_context_tokens:
        raise ValueError(
            "Validation sliding context tokens and initial context tokens expected to be the same. "
            "Prompt-only generation should override both values to 0."
            f" validation_sliding_context_tokens={validation_sliding_context_tokens}, "
            f"validation_initial_context_tokens={validation_initial_context_tokens}"
        )

    mode = str(stream_cfg.mode).lower()
    if mode not in {"fixed", "adaptive"}:
        raise ValueError(
            f"Invalid streaming mode `{mode}`. Expected one of ['fixed', 'adaptive']."
        )
    adaptive = mode == "adaptive"
    
    should_bake_in = getattr(stream_cfg, "should_bake_in", False) # specify directly for adaptive
    if schedule in {"rolling", "overlap"} and not should_bake_in:
        raise ValueError(
            f"Doing a {schedule} inference but not baking in"
        )
    if schedule in {"chunk", "full_sequence"} and should_bake_in:
        raise ValueError(
            f"Doing a {schedule} inference but baking in"
        )
    
    if stride_in_tokens_default is not None:
        stride_in_tokens = int(stride_in_tokens_default)
    else:
        if schedule in {"chunk", "full_sequence"}:
            stride_in_tokens = validation_horizon_tokens
        elif schedule == "rolling":
            stride_in_tokens = 1
        else:  # overlap
            stride_in_tokens = max(1, validation_horizon_tokens // 2)


    if stride_in_tokens < 1 or stride_in_tokens > validation_horizon_tokens:
        raise ValueError(
            f"Invalid stride_in_tokens={stride_in_tokens}. "
            f"Expected integer in [1, {validation_horizon_tokens}]."
        )
    if validation_horizon_tokens % stride_in_tokens != 0:
        raise ValueError(
            "All modes require validation_horizon_tokens divisible by stride_in_tokens "
            f"for alignment: got validation_horizon_tokens={validation_horizon_tokens}, "
            f"stride_in_tokens={stride_in_tokens}."
        )

    if schedule in {"chunk", "full_sequence"} and stride_in_tokens != validation_horizon_tokens:
        raise ValueError(
            f"Invalid stride_in_tokens={stride_in_tokens}. "
            "Expected stride_in_tokens to equal validation_horizon_tokens="
            f"{validation_horizon_tokens} for full_sequence/chunk schedule."
        )

    denoise_depth = max(num_sampling_steps, 1)
    num_sampling_noise_levels = num_sampling_steps + 1
    if schedule in {"chunk", "full_sequence"}:
        aligned_inner_default = denoise_depth
    else:
        # Derived from the fixed-alignment condition:
        #   (validation_horizon_tokens / stride_in_tokens) * k = denoise_depth.
        aligned_inner_default = max(
            1,
            int(
                round(
                    denoise_depth * float(stride_in_tokens)
                    / float(max(validation_horizon_tokens, 1))
                )
            ),
        )
        expected_touches_per_stride = denoise_depth * stride_in_tokens
        if expected_touches_per_stride % validation_horizon_tokens != 0:
            raise ValueError(
                "Fixed mode cannot satisfy exact alignment with integer inner steps: "
                "num_sampling_steps*stride_in_tokens="
                f"expected_touches_per_stride {expected_touches_per_stride} not divisible by validation_horizon_tokens="
                f"{validation_horizon_tokens}."
            )
    
    if adaptive:
        max_inner_steps = stream_cfg.max_inner_steps
        if max_inner_steps is None:
            max_inner_steps = aligned_inner_default
        max_inner_steps = int(max_inner_steps)

        expected_inner = denoise_depth * stride_in_tokens // validation_horizon_tokens
        if max_inner_steps != expected_inner:
            raise ValueError(
                "Adaptive mode requires exact alignment of the worst case "
                "(validation_horizon_tokens/stride_in_tokens)*inner_steps_per_emit "
                "= num_sampling_steps. "
                f"Got inner_steps_per_emit={max_inner_steps}, "
                f"but expected {expected_inner} for "
                f"validation_horizon_tokens={validation_horizon_tokens}, "
                f"stride_in_tokens={stride_in_tokens}, num_sampling_steps={num_sampling_steps}."
            )
        resolved_inner_steps_per_emit = max_inner_steps
    else:
        inner_steps_per_emit = stream_cfg.inner_steps_per_emit
        if inner_steps_per_emit is None:
            inner_steps_per_emit = aligned_inner_default
        inner_steps_per_emit = int(inner_steps_per_emit)
        expected_inner = denoise_depth * stride_in_tokens // validation_horizon_tokens
        if inner_steps_per_emit != expected_inner:
            raise ValueError(
                "Fixed mode requires exact alignment "
                "(validation_horizon_tokens/stride_in_tokens)*inner_steps_per_emit "
                "= num_sampling_steps. "
                f"Got inner_steps_per_emit={inner_steps_per_emit}, "
                f"but expected {expected_inner} for "
                f"validation_horizon_tokens={validation_horizon_tokens}, "
                f"stride_in_tokens={stride_in_tokens}, num_sampling_steps={num_sampling_steps}."
            )
        resolved_inner_steps_per_emit = inner_steps_per_emit

    conditioning_mode = str(stream_cfg.conditioning_mode).lower()
    if conditioning_mode not in {"conditional", "stabilized_conditional"}:
        raise ValueError(
            f"Invalid conditioning_mode `{conditioning_mode}`. "
            "Expected one of ['conditional', 'stabilized_conditional']."
        )
    stabilization_level = float(stream_cfg.stabilization_level)
    if stabilization_level > 0.0 and conditioning_mode == "conditional":
        raise ValueError("Stabilization level set to nonzero but conditioning mode set to conditional")

    stop_based_on = str(getattr(stream_cfg, "stop_based_on", "gradnorm")).lower()
    if stop_based_on not in {"gradnorm", "readout"}:
        raise ValueError(
            f"Invalid stop_based_on `{stop_based_on}`. Expected one of ['gradnorm', 'readout']."
        )

    emit_eps = float(stream_cfg.emit_eps)
    frame_done_eps = float(stream_cfg.frame_done_eps)
    emit_noise_level = float(stream_cfg.emit_noise_level)
    frame_done_noise_level = float(stream_cfg.frame_done_noise_level)

    raw_max_cycles_without_emit = stream_cfg.max_cycles_without_emit
    if isinstance(raw_max_cycles_without_emit, str):
        normalized_max_cycles_without_emit = raw_max_cycles_without_emit.strip().lower()
        if normalized_max_cycles_without_emit in {"none", "one"}:
            max_cycles_without_emit = 1 # During fixed inference emit every inner steps mandatorily
        elif normalized_max_cycles_without_emit in {"default", "h_over_s"}:
            # This default makes it so that we can take up to the full denoising depth to emit a stride
            # Given max_inner = (DS/H) and we emit H/S --> D is max possible (for the emit, not overall)
            if not adaptive and normalized_max_cycles_without_emit == "default":
                max_cycles_without_emit = 1 # During fixed inference emit every inner steps mandatorily
            else:
                max_cycles_without_emit = max(1, validation_horizon_tokens // stride_in_tokens)
        else:
            try:
                max_cycles_without_emit = int(normalized_max_cycles_without_emit)
            except ValueError:
                raise ValueError(
                    f"Invalid max_cycles_without_emit='{raw_max_cycles_without_emit}'. "
                    "Expected integer >= 1 or one of ['none', 'one', 'default', 'h_over_s']."
                )
    else:
        max_cycles_without_emit = int(raw_max_cycles_without_emit)

    if max_cycles_without_emit < 1:
        raise ValueError(
            f"Invalid max_cycles_without_emit={max_cycles_without_emit}. "
            "Expected integer >= 1."
        )

    profile_cfg = stream_cfg.initial_noise_profile
    if profile_cfg is None:
        initial_noise_profile = "flat"
    else:
        initial_noise_profile = str(profile_cfg).lower()
        if initial_noise_profile not in {"flat", "pyramid"}:
            raise ValueError(
                f"Invalid initial_noise_profile `{initial_noise_profile}`. "
                "Expected one of {'flat', 'pyramid'}."
            )

    return_streaming_state = bool(stream_cfg.return_streaming_state)

    tail_finish_controls = _resolve_tail_finish_controls(
        getattr(stream_cfg, "tail_finish", None),
        adaptive=adaptive,
        stop_based_on=stop_based_on,
    )

    return StreamingControls(
        forward_window_size_in_tokens=forward_window_size_in_tokens,
        sliding_context_tokens=validation_sliding_context_tokens,
        initial_context_tokens=validation_initial_context_tokens,
        validation_horizon_tokens=validation_horizon_tokens,
        num_sampling_steps=num_sampling_steps,
        num_sampling_noise_levels=num_sampling_noise_levels,
        schedule=schedule,
        mode=mode,
        adaptive=adaptive,
        should_bake_in=should_bake_in,
        stride_in_tokens=stride_in_tokens,
        resolved_inner_steps_per_emit=resolved_inner_steps_per_emit,
        conditioning_mode=conditioning_mode,
        stabilization_level=stabilization_level,
        stop_based_on=stop_based_on,
        emit_eps=emit_eps,
        frame_done_eps=frame_done_eps,
        emit_noise_level=emit_noise_level,
        frame_done_noise_level=frame_done_noise_level,
        max_cycles_without_emit=max_cycles_without_emit,
        initial_noise_profile=initial_noise_profile,
        should_collect_readout_noise_level=should_collect_readout_noise_level,
        return_streaming_state=return_streaming_state,
        tail_finish=tail_finish_controls,
    )


def _resolve_tail_finish_controls(
    tail_cfg,
    *,
    adaptive: bool,
    stop_based_on: str,
) -> TailFinishControls:
    """
    Parse / validate `tasks.prediction.streaming.tail_finish` config.
    When the config is absent or `enabled=false`, returns a disabled instance.

    The cleanup-entry threshold (`sigma_thresh`) reuses `emit_noise_level`; we
    do not introduce a separate threshold here. Once a frame's `sigma_hat` falls
    to/below it, the frame runs `n_cleanup_steps` finishing steps inline in the
    same loop before being marked done.
    """
    if tail_cfg is None:
        return TailFinishControls()

    enabled = bool(getattr(tail_cfg, "enabled", False))
    if not enabled:
        return TailFinishControls()

    # Cleanup is an adaptive-only feature: in fixed mode every frame already runs
    # the full denoising depth, so there is no early cleanup phase to enter.
    if not adaptive:
        raise ValueError(
            "tail_finish.enabled requires mode='adaptive' "
            "(the unified cleanup is an adaptive-only feature)."
        )
    if stop_based_on not in {"gradnorm", "readout"}:
        raise ValueError(
            f"Invalid stop_based_on `{stop_based_on}` for tail_finish; "
            "expected 'gradnorm' or 'readout'."
        )

    n_cleanup_steps = int(getattr(tail_cfg, "n_cleanup_steps", 4))
    if n_cleanup_steps < 1:
        raise ValueError(
            f"tail_finish.n_cleanup_steps must be >= 1, got {n_cleanup_steps}."
        )

    step_multiplier = float(getattr(tail_cfg, "step_multiplier", 1.0))
    if step_multiplier <= 0:
        raise ValueError(
            f"tail_finish.step_multiplier must be > 0, got {step_multiplier}."
        )

    return TailFinishControls(
        enabled=True,
        n_cleanup_steps=n_cleanup_steps,
        disable_momentum=bool(getattr(tail_cfg, "disable_momentum", True)),
        step_multiplier=step_multiplier,
    )


def resolve_tail_finish_controls(
    tail_cfg,
    *,
    adaptive: bool,
    stop_based_on: str,
) -> TailFinishControls:
    """Parse streaming tail-finish controls."""
    return _resolve_tail_finish_controls(
        tail_cfg,
        adaptive=adaptive,
        stop_based_on=stop_based_on,
    )


@dataclass
class StreamingState:
    """
    Persistent global state for streaming inference.

    Shapes
    ------
    x_act: (B, active_horizon_tokens, *x_shape)
    n_act: (B, active_horizon_tokens) long in [0, num_sampling_steps], the denoising state of the token. When it hits 0 --> clean.
    committed: (B, T_done, *x_shape)
    committed_is_generated: (B, T_done) bool
    momentum: (B, active_horizon_tokens, *x_shape)
    """

    x_act: Tensor
    n_act: Tensor
    committed: Tensor
    committed_is_generated: Tensor
    momentum: Tensor
    initial_context_tokens: int
    raw_nfe: Tensor
    # Token-level NFE proxy: counts how many active tokens were updated per inner step
    # (weighted by nfe_per_step), per sample.
    raw_token_nfe: Tensor
    # Unified adaptive cleanup ("tail finish") per-token state, shape (B, active_horizon_tokens):
    # - tail_mask: whether the token has entered the cleanup phase (sigma_hat <= sigma_thresh).
    # - tail_j: number of cleanup steps already taken for that token.
    # Both default to all-zeros; only used when streaming.tail_finish is enabled.
    tail_mask: Tensor
    tail_j: Tensor

    @classmethod
    def init_from_context(
        cls,
        context: Tensor,
        active_horizon_tokens: int,
        num_sampling_steps: int,
        clip_noise: float = 20.0,
        generator: Optional[torch.Generator] = None,
        raw_nfe_init: Optional[Union[float, Tensor]] = None,
        raw_token_nfe_init: Optional[Union[float, Tensor]] = None,
        initial_context_tokens: int = 25,
    ) -> "StreamingState":
        """
        Initialize from clean committed context and noisy active horizon state.
        """

        batch_size = context.shape[0]
        device = context.device
        dtype = context.dtype
        x_shape = list(context.shape[2:])

        x_act = torch.randn(
            batch_size,
            active_horizon_tokens,
            *x_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        ).clamp(-clip_noise, clip_noise)

        # Initialize state tracking of each token
        # Warm starts are disabled, so initialization is always flat.
        n_act = torch.full(
                (batch_size, active_horizon_tokens),
                int(num_sampling_steps),
                dtype=torch.long,
                device=device,
            )

        committed = context.clone() # committed video starts as context frames.
        committed_len = int(committed.shape[1])

        committed_is_generated = torch.zeros(
            (batch_size, committed_len),
            dtype=torch.bool,
            device=device,
        )

        momentum = torch.zeros(
            (batch_size, active_horizon_tokens, *x_shape),
            device=device,
            dtype=dtype,
        )

        tail_mask = torch.zeros(
            (batch_size, active_horizon_tokens),
            dtype=torch.bool,
            device=device,
        )
        tail_j = torch.zeros(
            (batch_size, active_horizon_tokens),
            dtype=torch.long,
            device=device,
        )
        
        # Can be None, a scalar, or a tensor
        if raw_nfe_init is None:
            raw_nfe = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        elif isinstance(raw_nfe_init, Tensor):
            if raw_nfe_init.shape != (batch_size,):
                raise ValueError(
                    "raw_nfe_init shape must be "
                    f"{(batch_size,)}, got {tuple(raw_nfe_init.shape)}"
                )
            raw_nfe = raw_nfe_init.to(device=device, dtype=torch.float32)
        else:
            raw_nfe = torch.full(
                (batch_size,),
                float(raw_nfe_init),
                device=device,
                dtype=torch.float32,
            )

        # Can be None, a scalar, or a tensor
        if raw_token_nfe_init is None:
            raw_token_nfe = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        elif isinstance(raw_token_nfe_init, Tensor):
            if raw_token_nfe_init.shape != (batch_size,):
                raise ValueError(
                    "raw_token_nfe_init shape must be "
                    f"{(batch_size,)}, got {tuple(raw_token_nfe_init.shape)}"
                )
            raw_token_nfe = raw_token_nfe_init.to(device=device, dtype=torch.float32)
        else:
            raw_token_nfe = torch.full(
                (batch_size,),
                float(raw_token_nfe_init),
                device=device,
                dtype=torch.float32,
            )

        return cls(
            x_act=x_act,
            n_act=n_act,
            committed=committed,
            committed_is_generated=committed_is_generated,
            momentum=momentum,
            raw_nfe=raw_nfe,
            raw_token_nfe=raw_token_nfe,
            initial_context_tokens=initial_context_tokens,
            tail_mask=tail_mask,
            tail_j=tail_j,
        )

    @property
    def B(self) -> int:
        return int(self.x_act.shape[0])

    @property
    def active_horizon_tokens(self) -> int:
        return int(self.x_act.shape[1])

    @property
    def total_committed(self) -> int:
        return int(self.committed.shape[1])
    
    @property
    def global_window_start_index(self) -> int:
        """
        "Global" in assuming that there is no history worth counting before the initial committed context.
        Assumes that initial_context_tokens == sliding_context_tokens and that the context is always committed.
        """
        return int(self.committed.shape[1] - self.initial_context_tokens)
    
    @property
    def global_window_end_index(self) -> int:
        """
        "Global" in assuming that there is no history worth counting before the initial committed context.
        """
        return int(self.committed.shape[1] + self.active_horizon_tokens)

    @property
    def x_shape(self) -> list[int]:
        return list(self.x_act.shape[2:])

    def get_context(
        self,
        sliding_context_tokens: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Return (x_ctx, is_generated, is_valid), each shape
        (B, sliding_context_tokens, ...)/(B, sliding_context_tokens).

        `is_valid` is False for left-padding slots when requested context
        exceeds available committed history.
        """
        sliding_context_tokens = int(sliding_context_tokens)

        if sliding_context_tokens == 0:
            x_ctx = self.x_act.new_empty((self.B, 0, *self.x_shape))
            flags = self.committed_is_generated.new_empty((self.B, 0))
            return x_ctx, flags, flags

        total_committed_tokens = self.total_committed
        valid_context_tokens = min(sliding_context_tokens, total_committed_tokens)
        x_ctx = self.committed[:, -valid_context_tokens:]
        is_generated = self.committed_is_generated[:, -valid_context_tokens:]
        is_valid = torch.ones(
            (self.B, valid_context_tokens),
            dtype=torch.bool,
            device=self.x_act.device,
        )

        if valid_context_tokens < sliding_context_tokens:
            left_pad_tokens = sliding_context_tokens - valid_context_tokens
            x_pad = self.x_act.new_zeros((self.B, left_pad_tokens, *self.x_shape))
            g_pad = torch.zeros(
                (self.B, left_pad_tokens),
                dtype=torch.bool,
                device=self.x_act.device,
            )
            x_ctx = torch.cat([x_pad, x_ctx], dim=1)
            is_generated = torch.cat([g_pad, is_generated], dim=1)
            is_valid = torch.cat([g_pad, is_valid], dim=1)

        return x_ctx, is_generated, is_valid

    def advance(
        self,
        emit_tokens: int,
        num_sampling_steps: int,
        clip_noise: float = 20.0,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Commit oldest `emit_tokens` active frames and append the same number
        of fresh noisy frames.
        """

        if emit_tokens == 0:
            return self

        device = self.x_act.device
        dtype = self.x_act.dtype

        x_commit = self.x_act[:, :emit_tokens].detach()
        commit_flag = torch.ones(
            (self.B, emit_tokens),
            dtype=torch.bool,
            device=device,
        )
        committed = torch.cat([self.committed, x_commit], dim=1)
        committed_is_generated = torch.cat(
            [self.committed_is_generated, commit_flag], dim=1
        )

        x_keep = self.x_act[:, emit_tokens:]
        n_keep = self.n_act[:, emit_tokens:]
        m_keep = self.momentum[:, emit_tokens:]
        tail_mask_keep = self.tail_mask[:, emit_tokens:]
        tail_j_keep = self.tail_j[:, emit_tokens:]

        x_new = torch.randn(
            (self.B, emit_tokens, *self.x_shape),
            device=device,
            dtype=dtype,
            generator=generator,
        ).clamp(-clip_noise, clip_noise)
        n_new = torch.full(
            (self.B, emit_tokens),
            int(num_sampling_steps),
            dtype=torch.long,
            device=device,
        )
        m_new = torch.zeros(
            (self.B, emit_tokens, *self.x_shape),
            device=device,
            dtype=dtype,
        )
        # Freshly appended tokens start outside the cleanup phase.
        tail_mask_new = torch.zeros(
            (self.B, emit_tokens),
            dtype=torch.bool,
            device=device,
        )
        tail_j_new = torch.zeros(
            (self.B, emit_tokens),
            dtype=torch.long,
            device=device,
        )

        self.x_act = torch.cat([x_keep, x_new], dim=1)
        self.n_act = torch.cat([n_keep, n_new], dim=1)
        self.committed = committed
        self.committed_is_generated = committed_is_generated
        self.momentum = torch.cat([m_keep, m_new], dim=1)
        self.tail_mask = torch.cat([tail_mask_keep, tail_mask_new], dim=1)
        self.tail_j = torch.cat([tail_j_keep, tail_j_new], dim=1)

    def add_nfe(self, active_mask: Tensor, nfe_per_step: float = 1.0) -> None:
        """
        Update per-sample raw NFE counters.
        """
        if active_mask.ndim != 2 or active_mask.shape[0] != self.B:
            raise ValueError(
                "active_mask must have shape (B, T) with B matching state batch size; "
                f"got {tuple(active_mask.shape)} for B={self.B}."
            )
        # Per-sample "model eval" count: if anything is active, count 1 eval for that sample.
        sample_active = active_mask.to(dtype=torch.bool).any(dim=1).to(dtype=torch.float32)
        # Per-sample token-update count: how many tokens were updated on this inner step.
        token_active = active_mask.to(dtype=torch.float32).sum(dim=1)
        self.raw_nfe = self.raw_nfe + float(nfe_per_step) * sample_active
        self.raw_token_nfe = self.raw_token_nfe + float(nfe_per_step) * token_active
