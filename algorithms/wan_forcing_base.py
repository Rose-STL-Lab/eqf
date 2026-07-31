from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from omegaconf import DictConfig

from algorithms.backbones import create_main_model
from algorithms.denoising import create_flow_algo
from algorithms.denoising.flow_matching import FlowMatching
from algorithms.denoising_video import DenoisingVideo
from algorithms.streaming_state import resolve_streaming_controls
from algorithms.wan.modules.t5 import umt5_xxl
from algorithms.wan.modules.tokenizers import HuggingfaceTokenizer
from algorithms.wan.text_utils import load_single_prompt_embed
from utils.ckpt_utils import smart_load_state_dict
from utils.distributed_utils import is_rank_zero, rank_zero_print
from utils.logging_utils import log_video
from utils.print_utils import cyan, red


class PromptConditionedDenoiser(torch.nn.Module):
    """
    Shared prompt-embedding storage/selection for Wan denoiser adapters.

    During sampling the denoiser is invoked once per CFG branch with a
    `prompt_embed_key` (`"positive"` or `"null"`); the embeddings themselves
    are stashed here via `set_prompt_embeds` so the generic denoising loop does
    not need to know about text conditioning.
    """

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self._positive_prompt_embeds: Any = None
        self._null_prompt_embeds: Any = None

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.backbone.parameters()).dtype

    def set_prompt_embeds(
        self,
        positive_prompt_embeds: Any,
        null_prompt_embeds: Any = None,
    ) -> None:
        self._positive_prompt_embeds = positive_prompt_embeds
        self._null_prompt_embeds = null_prompt_embeds

    def _select_prompt_embeds(self, prompt_embed_key: str) -> Any:
        key = str(prompt_embed_key).lower()
        if key == "positive":
            context = self._positive_prompt_embeds
        elif key == "null":
            context = self._null_prompt_embeds
        else:
            raise ValueError(
                f"Invalid prompt_embed_key={prompt_embed_key!r}. "
                "Expected 'positive' or 'null'."
            )

        if context is None:
            raise ValueError(f"{type(self).__name__} requires {key} prompt embeddings.")
        return context


class BaseWanForcingVideo(DenoisingVideo):
    """
    Shared Wan training/inference shell for repo-native diffusion forcing.

    Wan versions differ in text-embedding shape and backbone call signature, but
    share prompt loading, checkpoint loading, VAE handling, and training hooks.
    """

    denoising_algo: FlowMatching

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _resolve_path(self, path: str | None) -> str | None:
        if path is None or path == "null":
            return None
        resolved = Path(path)
        if resolved.is_absolute():
            return str(resolved)
        return str((Path.cwd() / resolved).resolve())

    def _load_pretrained_backbone(self, backbone: torch.nn.Module, path: str):
        return backbone.from_pretrained(path)

    def _make_denoising_model(self, backbone: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError

    def _null_prompt_embed_path(self) -> str | None:
        return self._resolve_path(getattr(self.cfg, "null_prompt_embed_path", None))

    @property
    def _force_null_prompt(self) -> bool:
        return bool(getattr(self.cfg, "force_null_prompt", False))

    def _load_null_prompt_embed_if_configured(self) -> None:
        self.null_prompt_embed = None
        path = self._null_prompt_embed_path()
        if path is None:
            return
        self.null_prompt_embed = load_single_prompt_embed(path).to(dtype=self.model_dtype)
        rank_zero_print(
            cyan(
                "Loaded null-caption T5 embedding from "
                f"{path} with shape {tuple(self.null_prompt_embed.shape)}."
            )
        )

    def _load_text_components(self) -> None:
        """
        Set up the text-conditioning inputs. Three independent modes:
          - `force_null_prompt`: every sample uses the one cached null embedding
            (e.g. RE10K, whose captions are all ""). No T5/tokenizer loaded.
          - `load_prompt_embed`: the dataset provides a precomputed embedding
            per sample. No T5/tokenizer loaded.
          - otherwise: load the live T5 encoder + tokenizer and encode captions.

        Note `force_null_prompt`/`load_prompt_embed` are distinct from
        `null_prompt_embed_path`: the latter loads a single shared null embedding
        (always loaded here if configured) used for CFG dropout and the
        unconditional branch, independent of which mode is active.
        """
        self._load_null_prompt_embed_if_configured()

        force_null_prompt = self._force_null_prompt
        load_prompt_embed = bool(getattr(self.cfg, "load_prompt_embed", False))
        # for re10k (no caption datasets where there are no nonnull prompt embeds)
        # we assume that load_prompt_embed is set to false.
        if force_null_prompt and load_prompt_embed:
            raise ValueError(
                "`load_prompt_embed=true` cannot be combined with "
                "`force_null_prompt=true`."
            )

        # Both the force-null-prompt path (re10k) and the cached-prompt-embed
        # path run the model without a live T5 text encoder/tokenizer.
        if force_null_prompt or load_prompt_embed:
            if force_null_prompt:
                self._require_null_prompt_embed("`force_null_prompt=true`")
                message = (
                    "force_null_prompt=true: using the cached null prompt embedding "
                    "for every sample; tokenizer/T5 text encoder will not be loaded."
                )
            else:
                message = (
                    "Using precomputed prompt embeddings; tokenizer/T5 text encoder "
                    "will not be loaded."
                )
            self.__dict__["text_encoder"] = None
            self.text_encoder_device = torch.device("cpu")
            self.tokenizer = None
            rank_zero_print(cyan(message))
            return

        tokenizer_name = self.cfg.text_encoder.name
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_name,
            seq_len=int(self.cfg.text_encoder.text_len),
            clean="whitespace",
        )

        text_dtype = (
            torch.bfloat16
            if str(getattr(self.cfg.text_encoder, "dtype", "bfloat16")).lower()
            == "bfloat16"
            else torch.float32
        )
        text_device = (
            torch.device("cuda")
            if bool(self.cfg.text_encoder.load_to_gpu)
            else torch.device("cpu")
        )
        text_encoder = (
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=text_dtype,
                device=text_device,
            )
            .eval()
            .requires_grad_(False)
        )
        if self.cfg.text_encoder.pretrained_ckpt_path is not None:
            text_encoder.load_state_dict(
                torch.load(
                    self._resolve_path(self.cfg.text_encoder.pretrained_ckpt_path),
                    map_location="cpu",
                    weights_only=True,
                )
            )
        if bool(getattr(self.cfg.text_encoder, "compile", False)):
            text_encoder = torch.compile(text_encoder)

        self.__dict__["text_encoder"] = text_encoder
        self.text_encoder_device = text_device

    def encode_text(self, texts: list[str]):
        raise NotImplementedError

    def _prepare_cached_prompt_embeds(
        self,
        prompt_embeds: Any,
        prompt_embed_lens: Any | None = None,
    ) -> Any:
        """
        Convert a raw `(B, L, D)` prompt-embedding tensor (plus optional
        per-sample lengths) into the backbone's expected context format.

        This is the single place that knows the per-version context layout, and
        is reused for both the real (positive) embeddings and the null
        embedding. Overridden per Wan version:
          - Wan2.1 returns a list of variable-length `[L_i, D]` tensors.
          - Wan2.2 returns a zero-padded `(B, L, D)` tensor.
        """
        del prompt_embeds, prompt_embed_lens
        raise NotImplementedError(
            f"{type(self).__name__}, the base file, does not support cached prompt embeddings."
        )

    def _get_cached_prompt_embeds_from_metadata(self, metadata: Any) -> Any:
        # Positive branch (cached): adapt the per-sample embeddings carried on
        # the batch `metadata` into backbone context via the shared formatter.
        if not isinstance(metadata, dict) or "prompt_embeds" not in metadata:
            raise ValueError(
                f"{type(self).__name__} expected `metadata.prompt_embeds` when "
                "`algorithm.load_prompt_embed=true`."
            )
        return self._prepare_cached_prompt_embeds(
            metadata["prompt_embeds"],
            metadata.get("prompt_embed_len"),
        )

    def _get_prompts_from_metadata(self, metadata: Any) -> list[str]:
        if isinstance(metadata, dict) and "caption" in metadata:
            captions = metadata["caption"]
            if isinstance(captions, str):
                return [captions]
            return [str(x) for x in captions]
        raise ValueError(f"{type(self).__name__} requires `metadata.caption` prompts.")

    def _require_null_prompt_embed(self, reason: str) -> torch.Tensor:
        null_prompt_embed = getattr(self, "null_prompt_embed", None)
        if null_prompt_embed is None:
            raise ValueError(
                f"{reason} requires `algorithm.null_prompt_embed_path` to point "
                "to a cached null-caption T5 embedding."
            )
        return null_prompt_embed

    def _get_null_prompt_embeds(self, batch_size: int) -> Any:
        """
        Build the backbone context for the unconditional (null) CFG branch:
        broadcast the single cached null embedding to `batch_size` and run it
        through the same `_prepare_cached_prompt_embeds` formatter as the
        positive embeddings. Uses the null embedding's own length (unlike
        `_null_embed_for_dropout`, which matches the batch's sequence length).
        """
        null_prompt_embed = self._require_null_prompt_embed(
            "Classifier-free guidance"
        )
        prompt_embeds = null_prompt_embed.unsqueeze(0).expand(batch_size, -1, -1)
        prompt_embed_lens = torch.full(
            (batch_size,),
            null_prompt_embed.shape[0],
            dtype=torch.long,
            device=prompt_embeds.device,
        )
        return self._prepare_cached_prompt_embeds(prompt_embeds, prompt_embed_lens)

    def _null_embed_for_dropout(
        self,
        prompt_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """
        Shape the null embedding for in-place training dropout: pad/truncate it
        to the *current batch's* sequence length `L` so it can be written
        directly into masked rows of the raw `(B, L, D)` `metadata.prompt_embeds`
        tensor (before `_prepare_cached_prompt_embeds` runs). Returns
        `(padded [L, D], null_len)`.
        """
        null_prompt_embed = self._require_null_prompt_embed(
            "Text-embedding dropout"
        ).to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        if null_prompt_embed.shape[-1] != prompt_embeds.shape[-1]:
            raise ValueError(
                "Null prompt embedding dim does not match batch prompt "
                f"embedding dim: {null_prompt_embed.shape[-1]} vs "
                f"{prompt_embeds.shape[-1]}."
            )

        target_len = prompt_embeds.shape[1]
        null_len = min(null_prompt_embed.shape[0], target_len)
        padded = prompt_embeds.new_zeros((target_len, prompt_embeds.shape[-1]))
        padded[:null_len] = null_prompt_embed[:null_len]
        return padded, null_len

    def _apply_text_embed_dropout_to_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        dropout_prob = float(getattr(self.cfg, "text_embed_dropout_prob", 0.0))
        if dropout_prob < 0.0 or dropout_prob > 1.0:
            raise ValueError("algorithm.text_embed_dropout_prob must be in [0.0, 1.0].")
        if dropout_prob == 0.0 or not self.trainer.training:
            return metadata
        self._require_null_prompt_embed("Text-embedding dropout")

        prompt_embeds = metadata.get("prompt_embeds", None)
        if not torch.is_tensor(prompt_embeds):
            raise TypeError(
                "Text-embedding dropout requires tensor `metadata.prompt_embeds`."
            )
        if prompt_embeds.ndim != 3:
            raise ValueError(
                "Text-embedding dropout expects `metadata.prompt_embeds` with shape "
                f"(B, L, D), got {tuple(prompt_embeds.shape)}."
            )

        dropout_mask = torch.rand(
            prompt_embeds.shape[0],
            device=prompt_embeds.device,
        ) < dropout_prob
        if not bool(dropout_mask.any()):
            return metadata

        null_prompt_embed, null_len = self._null_embed_for_dropout(prompt_embeds)
        metadata = dict(metadata)
        prompt_embeds = prompt_embeds.clone()
        prompt_embeds[dropout_mask] = null_prompt_embed
        metadata["prompt_embeds"] = prompt_embeds

        if "prompt_embed_len" in metadata:
            prompt_embed_lens = torch.as_tensor(
                metadata["prompt_embed_len"],
                device=prompt_embeds.device,
                dtype=torch.long,
            ).clone()
            prompt_embed_lens[dropout_mask] = null_len
            metadata["prompt_embed_len"] = prompt_embed_lens

        return metadata

    @contextmanager
    def _prompt_context(self, metadata: Any) -> Iterator[None]:
        """
        Treatment of positive prompts, which are the only ones used during training
            If we are forcing null, we cannot also load_cached_prompt embeds. This is because 
            load cached gets them from metadata, assuming that prompt_embeds is an entry in the 
            batch. On the other hand, forcing null intends to use prompts = ['', ...] and 
            within encode text we repeat the null embed along the batch dimension.
        Treatment of null prompts
            We only bring in null prompts when doing CFG, because during training without prompts 
            we treat those as positive via encode_text.
            Get it from the null embedding.
        """
        load_cached_prompt_embeds = bool(getattr(self.cfg, "load_prompt_embed", False))
        prompts = self._get_prompts_from_metadata(metadata)
        if load_cached_prompt_embeds:
            positive_prompt_embeds = self._get_cached_prompt_embeds_from_metadata(metadata)
        else:
            positive_prompt_embeds = self.encode_text(prompts)
        null_prompt_embeds = None
        if float(getattr(self.denoising_cfg, "guidance_scale", 0.0)) > 0.0:
            # The unconditional branch always uses the cached null embedding
            # (`null_prompt_embed_path`), independent of how the positive
            # embeddings were produced (cached or live T5). `_get_null_prompt_embeds`
            # raises a clear error if the null embedding was not configured.
            null_prompt_embeds = self._get_null_prompt_embeds(len(prompts))
        self.denoising_model.set_prompt_embeds(
            positive_prompt_embeds,
            null_prompt_embeds,
        )
        try:
            yield
        finally:
            self.denoising_model.set_prompt_embeds(None)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        """
        Process batch specially for prompt loading. 
        If not loading prompt embeds, then we will encode text downstream with the tokenizer and encoder
        If prompt embeds are not in the batch, then we don't have them and will either be encoding text
        or forcing null prompts.

        If we are loading prompt embeds, we will apply dropout below by replacing some positive examples
        with the null prompt embedding.
        """
        processed = super().on_after_batch_transfer(batch, dataloader_idx)
        source_batch = batch[dataloader_idx] if isinstance(batch, list) else batch
        if not (
            bool(getattr(self.cfg, "load_prompt_embed", False))
            and isinstance(source_batch, dict)
            and "prompt_embeds" in source_batch
        ):
            return processed

        xs, conditions, masks, gt_videos, metadata = processed
        if not isinstance(metadata, dict):
            raise ValueError(
                f"{type(self).__name__} expected dict metadata when using cached "
                "prompt embeddings."
            )
        metadata = dict(metadata)
        metadata["prompt_embeds"] = source_batch["prompt_embeds"]
        if "prompt_embed_len" in source_batch:
            metadata["prompt_embed_len"] = source_batch["prompt_embed_len"]
        metadata = self._apply_text_embed_dropout_to_metadata(metadata)
        return xs, conditions, masks, gt_videos, metadata

    def create_denoising_algo(self) -> torch.nn.Module:
        return create_flow_algo(
            num_sampling_steps=self.num_sampling_steps,
            num_noise_levels=self.num_noise_levels,
            mean_type=self.denoising_cfg.mean_type,
            cfg=self.denoising_cfg,
            logger=self.logger,
        )

    def configure_model(self) -> None:
        _, image_h, image_w = map(int, list(self.cfg.x_shape))
        vae_stride_t, vae_stride_h, vae_stride_w = map(
            int, list(self.cfg.vae.downsampling_factor)
        )
        self.vae_stride = [vae_stride_t, vae_stride_h, vae_stride_w]
        self.lat_h = image_h // vae_stride_h
        self.lat_w = image_w // vae_stride_w
        self.latent_size = [self.lat_w, self.lat_h]

        self.model_dtype = (
            torch.bfloat16
            if str(getattr(self.cfg, "model_dtype", "bfloat16")).lower() == "bfloat16"
            else torch.float32
        )
        self._load_text_components()

        backbone = create_main_model(self.backbone_cfg)
        pretrained_ckpt_path = self._resolve_path(self.cfg.model.pretrained_ckpt_path)
        if pretrained_ckpt_path is None:
            raise ValueError("Wan pretrained backbone checkpoint path is required.")

        backbone = self._load_pretrained_backbone(backbone, pretrained_ckpt_path)

        backbone = backbone.to(self.model_dtype)
        gradient_checkpointing_rate = float(
            getattr(self.cfg, "gradient_checkpointing_rate", 0.0)
        )
        if gradient_checkpointing_rate > 0:
            backbone.gradient_checkpointing_enable(p=gradient_checkpointing_rate)
        if bool(getattr(self.cfg.model, "compile", False)):
            backbone = torch.compile(backbone)

        self.denoising_model = self._make_denoising_model(backbone)

        if self.cfg.load_model_state:
            model_dict = torch.load(
                self._resolve_path(self.cfg.load_model_state),
                map_location="cpu",
                weights_only=False,
            )
            if self.cfg.model_state_key in model_dict:
                model_dict = model_dict[self.cfg.model_state_key]
            else:
                state_keys = list(model_dict.keys())
                rank_zero_print(
                    red(
                        f'\n !!! No model state found in {self.cfg.load_model_state}, while loading using "{self.cfg.model_state_key}" to load model state !!! \n'
                    )
                )
                rank_zero_print(
                    red(
                        f"Available keys: {state_keys}. You should use one of them to load the state for the model ! \n"
                    )
                )
                raise ValueError(
                    f'No model state found in {self.cfg.load_model_state}, while loading using "{self.cfg.model_state_key}" to load model state'
                )

            model_dict = {
                k.replace("denoising_model.", ""): v for k, v in model_dict.items()
            }
            load_model = self.cfg.load_model_state_mode
            if load_model == "strict":
                self.denoising_model.load_state_dict(model_dict, strict=True)
            elif load_model == "partial":
                self.denoising_model.load_state_dict(model_dict, strict=False)
            elif load_model == "smart":
                if hasattr(self.cfg, "handlers"):
                    handlers = self.cfg.handlers
                    if handlers == "depth_fine_tuning":
                        from utils.ckpt_utils import depth_fine_tuning_handlers

                        smart_load_state_dict(
                            self.denoising_model,
                            model_dict,
                            custom_handlers=depth_fine_tuning_handlers,
                            verbose=True,
                        )
                    else:
                        raise ValueError(f"Invalid handler: {handlers}")
                else:
                    smart_load_state_dict(self.denoising_model, model_dict, verbose=True)
            else:
                raise ValueError(f"Invalid load_model: {load_model}")

            rank_zero_print(
                cyan(
                    f"\n ==== Successfully loaded {len(model_dict)} parameters from {self.cfg.load_model_state}[{self.cfg.model_state_key}], load model mode: {load_model} ===="
                )
            )

        self.denoising_algo = self.create_denoising_algo().to(self.denoising_model.dtype)

        if self.is_latent_diffusion and not self.use_preprocessed_latents:
            self._load_vae()

        self._build_metrics()

    def training_step(
        self, batch, batch_idx, dataloader_idx=0, namespace="training"
    ):
        *_, metadata = batch
        with self._prompt_context(metadata):
            return super().training_step(
                batch,
                batch_idx,
                dataloader_idx=dataloader_idx,
                namespace=namespace,
            )

    def _sample_all_videos(self, batch, batch_idx, namespace="validation"):
        *_, metadata = batch
        with self._prompt_context(metadata):
            return super()._sample_all_videos(batch, batch_idx, namespace=namespace)

    def validation_step(
        self, batch, batch_idx, dataloader_idx=0, namespace="validation"
    ):
        output = super().validation_step(
            batch,
            batch_idx,
            dataloader_idx=dataloader_idx,
            namespace=namespace,
        )
        self._maybe_log_prompt_validation(
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            namespace=namespace,
        )
        return output

    @torch.no_grad()
    def _maybe_log_prompt_validation(
        self,
        *,
        batch_idx: int,
        dataloader_idx: int,
        namespace: str,
    ) -> None:
        prompt_cfg = getattr(self.cfg, "prompt_validation", None)
        if prompt_cfg is None or not bool(getattr(prompt_cfg, "enabled", False)):
            return
        if batch_idx != int(getattr(prompt_cfg, "batch_idx", 0)):
            return
        if dataloader_idx != int(getattr(prompt_cfg, "dataloader_idx", 0)):
            return
        if self.trainer.sanity_checking and not bool(self.logging_cfg.sanity_generation):
            return
        if not is_rank_zero:
            return

        prompts = [str(prompt) for prompt in getattr(prompt_cfg, "prompts", [])]
        if not prompts:
            return

        batch_size = int(getattr(prompt_cfg, "batch_size", 1))
        outputs: list[torch.Tensor] = []
        for start in range(0, len(prompts), batch_size):
            outputs.append(self.sample_prompts(prompts[start : start + batch_size]).cpu())
        generated = torch.cat(outputs, dim=0).float().clamp(0.0, 1.0)

        should_log_to_logger = bool(self.logger) and is_rank_zero
        log_video(
            generated,
            namespace=f"{namespace}_prompt_validation",
            prefix="generated",
            captions=prompts,
            logger=self.logger.experiment if should_log_to_logger else None,
            raw_dir=getattr(prompt_cfg, "raw_dir", None),
            fps=int(self.logging_cfg.fps),
            log_to_logger=should_log_to_logger,
        )

    def _prompt_latent_shape(self) -> tuple[int, int, int, int]:
        prompt_cfg = self.cfg.prompt_inference
        vae_stride = list(self.cfg.vae.stride)
        return (
            int(self.cfg.vae.z_dim),
            1 + (int(prompt_cfg.n_frames) - 1) // int(vae_stride[0]),
            int(prompt_cfg.height) // int(vae_stride[1]),
            int(prompt_cfg.width) // int(vae_stride[2]),
        )

    @torch.no_grad()
    def sample_prompts(self, prompts: list[str]) -> torch.Tensor:
        """
        Prompt-only sampling through the repo streaming sampler.
        """
        if bool(getattr(self.cfg, "load_prompt_embed", False)):
            raise ValueError(
                f"{type(self).__name__}.sample_prompts requires a live text encoder; "
                "disable `algorithm.load_prompt_embed` for prompt-only sampling."
            )

        prompts = list(prompts)
        if not prompts:
            raise ValueError("sample_prompts requires at least one prompt.")

        prompt_cfg = self.cfg.prompt_inference
        latent_channels, latent_frames, latent_h, latent_w = self._prompt_latent_shape()
        batch_size = len(prompts)
        prompt_embeds = self.encode_text(prompts)
        null_prompt_embeds = self.encode_text([""] * batch_size)

        stream_cfg = self.cfg.tasks.prediction.streaming
        controls = resolve_streaming_controls(
            stream_cfg,
            self.forward_window_size_in_tokens,
            self.num_sampling_steps,
            self.validation_n_sliding_context_tokens,
            self.validation_n_initial_context_tokens,
            self._n_frames_to_n_tokens(stream_cfg.stride_in_frames),
        )
        self._validate_readout_controls(controls)

        if controls.initial_context_tokens != 0:
            raise ValueError(
                f"Prompt-only {type(self).__name__}.sample_prompts requires "
                "algorithm.tasks.prediction.streaming.initial_context_frames=0."
            )

        context = torch.empty(
            batch_size,
            controls.initial_context_tokens,
            latent_channels,
            latent_h,
            latent_w,
            device=self.device,
            dtype=self.denoising_model.dtype,
        )

        self.denoising_model.set_prompt_embeds(prompt_embeds, null_prompt_embeds)
        try:
            latents, _ = self._run_streaming_sampler(
                context=context,
                total_length=latent_frames,
                controls=controls,
                conditions=None,
                prediction_kwargs={},
            )
        finally:
            self.denoising_model.set_prompt_embeds(None)

        return self._decode(latents, desired_length=int(prompt_cfg.n_frames))


__all__ = ["BaseWanForcingVideo"]
