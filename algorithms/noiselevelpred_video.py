import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import Optimizer
from omegaconf import DictConfig

from algorithms.common.noiselevelpred_inference_mixin import NoiseLevelPredInferenceMixin
from algorithms.denoising_video import DenoisingVideo
from algorithms.readout import NoiseLevelReadout
from utils.print_utils import cyan, red
from utils.distributed_utils import rank_zero_print


class NoiseLevelPredVideo(NoiseLevelPredInferenceMixin, DenoisingVideo):
    """
    Training-first noise-level-prediction variant of DenoisingVideo.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.readout_cfg = getattr(cfg, "readout", None)
        self.readout_enabled = False
        self.readout_only = False
        self.readout_head = None
        self.noise_level_finetune_step = 0
        super().__init__(cfg)
        self._validate_readout_config()

    def _validate_readout_config(self) -> None:
        if not getattr(self.readout_cfg, "enabled", False):
            raise ValueError("NoiseLevelPredVideo requires `algorithm.readout.enabled=true`.")
        if not self.use_continuous_timesteps:
            raise ValueError(
                "NoiseLevelPredVideo requires continuous denoising timesteps for noise-level readout."
            )
        if not getattr(self.readout_cfg, "only_noise_level", False):
            raise ValueError("NoiseLevelPredVideo only supports noise-level readout loss.")
        # Noise-level prediction always runs in readout-only mode.
        self.readout_only = True

    def configure_model(self) -> None:
        super().configure_model()
        self._freeze_denoising_model()
        self._configure_readout_head()

    def _freeze_denoising_model(self) -> None:
        for param in self.denoising_model.parameters():
            param.requires_grad = False

    def _configure_readout_head(self) -> None:
        tap_every = max(1, int(self.readout_cfg.tap_every))
        num_layers = int(self.backbone_cfg.num_layers)
        num_taps = (num_layers + tap_every - 1) // tap_every
        feature_dim = int(self.backbone_cfg.num_attention_heads) * int(
            self.backbone_cfg.attention_head_dim
        )

        self.readout_head = NoiseLevelReadout(
            feature_dim=feature_dim,
            num_layers=num_taps,
            adapter_dim=int(self.readout_cfg.adapter_dim),
            num_mlp_layers=int(self.readout_cfg.num_mlp_layers),
            pooling=self.readout_cfg.pooling,
            per_layer_adapter=bool(self.readout_cfg.per_layer_adapter),
            mix_layers=bool(self.readout_cfg.mix_layers),
            mix_method=self.readout_cfg.mix_method,
            target_type=self.readout_cfg.target,
            loss_type=self.readout_cfg.loss,
            eps=float(self.readout_cfg.eps),
        )
        self.readout_enabled = True

    def _readout_model_kwargs(self):
        return {
            "return_hidden_states": True,
            "hidden_states_tap_every": max(1, int(self.readout_cfg.tap_every)),
        }

    def _denoising_ema_parameter_keys(self) -> list[str]:
        prefix = (
            f"{self.main_model_prefix}._orig_mod."
            if self.cfg.compile
            else f"{self.main_model_prefix}."
        )
        return [prefix + name for name, _ in self.denoising_model.named_parameters()]

    def _readout_ema_parameter_keys(self) -> list[str]:
        return [f"readout_head.{name}" for name, _ in self.readout_head.named_parameters()]

    def _combined_ema_parameter_keys(self) -> list[str]:
        return self._denoising_ema_parameter_keys() + self._readout_ema_parameter_keys()

    def _bootstrap_denoiser_from_ema(self) -> bool:
        return bool(getattr(self.readout_cfg, "bootstrap_denoiser_from_ema", True))

    def _checkpoint_readout_bootstrap_source(self, checkpoint: dict) -> str | None:
        return checkpoint.get("noiselevelpred_readout_bootstrap_source")

    def _pin_denoiser_to_checkpoint_ema(self, checkpoint: dict) -> None:
        """
        Keep the frozen denoiser on checkpoint EMA weights throughout noiselevelpred runs.
        """
        if not self._bootstrap_denoiser_from_ema():
            return
        if (
            checkpoint.get("pretrained_ema", False)
            and len(checkpoint.get("optimizer_states", [])) == 0
        ):
            return

        optimizer_states = checkpoint.get("optimizer_states", [])
        if not optimizer_states or "ema" not in optimizer_states[0]:
            return

        ema_weights = optimizer_states[0]["ema"]
        denoising_keys = self._denoising_ema_parameter_keys()
        readout_keys = self._readout_ema_parameter_keys()
        checkpoint_has_readout = any(
            key.startswith("readout_head.") for key in checkpoint.get("state_dict", {}).keys()
        )

        if len(ema_weights) == len(denoising_keys) and not checkpoint_has_readout:
            denoiser_ema_weights = ema_weights[: len(denoising_keys)]
        elif len(ema_weights) == len(denoising_keys) + len(readout_keys):
            denoiser_ema_weights = ema_weights[: len(denoising_keys)]
        else:
            raise ValueError(
                "EMA weight count is invalid for this checkpoint. "
                f"Base denoiser bootstrap expects {len(denoising_keys)} EMA params with no readout_head.* keys; "
                f"noiselevelpred checkpoints require {len(denoising_keys) + len(readout_keys)} EMA params. "
                f"Got {len(ema_weights)} EMA params with checkpoint_has_readout={checkpoint_has_readout}."
            )

        for key, weight in zip(denoising_keys, denoiser_ema_weights):
            checkpoint["state_dict"][key] = weight

    def _load_ema_weights_to_state_dict(self, checkpoint: dict) -> None:
        """Allow base-denoiser bootstrap, but require combined EMA once readout weights exist."""
        if (
            checkpoint.get("pretrained_ema", False)
            and len(checkpoint.get("optimizer_states", [])) == 0
        ):
            rank_zero_print(
                cyan(
                    "EMA weights are already baked into this release checkpoint's "
                    "state_dict; loading them directly."
                )
            )
            return

        optimizer_states = checkpoint.get("optimizer_states", [])
        if not optimizer_states or "ema" not in optimizer_states[0]:
            rank_zero_print(
                red("No EMA weights found in the checkpoint, so using the initialized value for the EMA weights.")
            )
            return

        ema_weights = optimizer_states[0]["ema"]
        denoising_keys = self._denoising_ema_parameter_keys()
        readout_keys = self._readout_ema_parameter_keys()
        checkpoint_has_readout = any(
            key.startswith("readout_head.") for key in checkpoint.get("state_dict", {}).keys()
        )

        if len(ema_weights) == len(denoising_keys) and not checkpoint_has_readout:
            target_keys = denoising_keys
        elif len(ema_weights) == len(denoising_keys) + len(readout_keys):
            target_keys = denoising_keys + readout_keys
        else:
            raise ValueError(
                "EMA weight count is invalid for this checkpoint. "
                f"Base denoiser bootstrap expects {len(denoising_keys)} EMA params with no readout_head.* keys; "
                f"noiselevelpred checkpoints require {len(denoising_keys) + len(readout_keys)} EMA params. "
                f"Got {len(ema_weights)} EMA params with checkpoint_has_readout={checkpoint_has_readout}."
            )

        for key, weight in zip(target_keys, ema_weights):
            checkpoint["state_dict"][key] = weight

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.noise_level_finetune_step = int(checkpoint.get("noise_level_finetune_step", 0))
        checkpoint_has_readout = any(
            key.startswith("readout_head.") for key in checkpoint.get("state_dict", {}).keys()
        )
        saved_bootstrap_source = self._checkpoint_readout_bootstrap_source(checkpoint)
        requested_bootstrap_source = (
            "ema" if self._bootstrap_denoiser_from_ema() else "online"
        )

        if (
            checkpoint_has_readout
            and saved_bootstrap_source is not None
            and saved_bootstrap_source != requested_bootstrap_source
        ):
            raise ValueError(
                "NoiseLevelPred checkpoint bootstrap source mismatch: "
                f"checkpoint was trained with `{saved_bootstrap_source}` but current "
                f"`algorithm.readout.bootstrap_denoiser_from_ema` requests `{requested_bootstrap_source}`."
            )

        self._pin_denoiser_to_checkpoint_ema(checkpoint)
        super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["noise_level_finetune_step"] = int(self.noise_level_finetune_step)
        checkpoint["noiselevelpred_readout_bootstrap_source"] = (
            "ema" if self._bootstrap_denoiser_from_ema() else "online"
        )
        super().on_save_checkpoint(checkpoint)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_closure=None,
    ) -> None:
        super().optimizer_step(
            epoch=epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_closure=optimizer_closure,
        )
        self.noise_level_finetune_step += 1
        self.log(
            "training/noise_level_finetune_step",
            float(self.noise_level_finetune_step),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=False,
            logger=True,
        )

    def training_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
        namespace="training",
        noise_level_override: torch.Tensor | None = None,
        show_noise_level_override: torch.Tensor | None = None,
        ablation_eval_mode: bool = False,
    ) -> STEP_OUTPUT:
        """
        Readout-only training on detached denoiser hidden states. Also supports
        the ablation eval mode used by `_run_eval_denoising_once`.
        """
        denoising_cfg = self.denoising_cfg

        xs, conds, masks, gt_videos, metadata = batch

        # in ablation mode, set k to provided noise level
        if ablation_eval_mode:
            if noise_level_override is None:
                raise ValueError("ablation_eval_mode requires `noise_level_override`.")
            k = noise_level_override.to(device=xs.device, dtype=xs.dtype)
            if k.shape != xs.shape[:2]:
                raise ValueError(
                    "noise_level_override must have shape (B, T) matching the cropped eval batch. "
                    f"Got {tuple(k.shape)} vs {tuple(xs.shape[:2])}."
                )
            attn_mask = None
            masks_for_loss = masks
        # in readout training, sample k randomly
        else:
            k, masks_for_loss, attn_mask = self._get_training_noise_levels(
                xs, masks, strategy=denoising_cfg.strategy
            )

        # (typically) in ablation mode, optionally override the noise level shown to the model
        # so that it differs from the true noise level of the data
        show_k = None
        if show_noise_level_override is not None:
            show_k = show_noise_level_override.to(device=xs.device, dtype=xs.dtype)
            if show_k.shape != xs.shape[:2]:
                raise ValueError(
                    "show_noise_level_override must have shape (B, T) matching the cropped eval batch. "
                    f"Got {tuple(show_k.shape)} vs {tuple(xs.shape[:2])}."
                )

        if conds is not None:
            conds = conds.to(device=xs.device, dtype=xs.dtype)

        if denoising_cfg.strategy == "diffusion-forcing":
            cfg_dropping_strategy = "frame_wise"
        else:
            raise ValueError(f"Invalid strategy: {denoising_cfg.strategy}")

        model_kwargs = dict(
            strategy=denoising_cfg.strategy,
            noise_abs_max=denoising_cfg.noise_abs_max,
            masks=masks_for_loss,
            cfg_dropping_strategy=cfg_dropping_strategy,
            attention_mask=attn_mask,
        )
        model_kwargs.update(self._readout_model_kwargs())

        # Execute forward pass and compute loss on data with noise level k
        # having shown the model noise level show_k (potentially None, where we default to
        # some other policy depending on self.denoising_algo)
        loss_dict = self.denoising_algo.training_loss(
            self.denoising_model,
            x_start=xs,
            conditions=conds,
            masks=masks_for_loss,
            k=k,
            model_kwargs=model_kwargs,
            show_k=show_k,
        )

        loss_main = loss_dict["loss"].mean()
        hidden_states = loss_dict.pop("hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "NoiseLevelPredVideo requested hidden states but the backbone did not return them."
            )

        pred_raw = self.readout_head(hidden_states)
        (
            readout_loss,
            readout_loss_per_sample,
            readout_noise_level_target,
        ) = self.readout_head.compute_error_from_raw(
            pred_raw,
            k,
            masks=masks_for_loss,
            return_details=True,
        )

        readout_loss_weight = float(getattr(self.readout_cfg, "readout_loss_weight", 0.0))
        loss = readout_loss_weight * readout_loss

        predicted_x_start = loss_dict["predicted_x_start"]
        original_x = loss_dict["original_x"]
        output_dict = {
            "loss": loss,
            "loss_main": loss_main,
            "loss_readout": readout_loss,
            "predicted_x_start": predicted_x_start,
            "original_x": original_x,
            "noise_level": k,
        }
        self._maybe_update_noise_level_ablation_output(
            output_dict,
            ablation_eval_mode=ablation_eval_mode,
            pred_raw=pred_raw,
            true_noise_level=k,
            show_noise_level=show_k,
            masks=masks,
            # losses in the dict from the denoising backbone
            v_loss=loss_dict["loss"],
            x_loss=loss_dict["x_mse"],
        )

        if "throughput" in loss_dict:
            output_dict["throughput"] = loss_dict["throughput"]

        if (not ablation_eval_mode) and batch_idx % self.logging_cfg.loss_freq == 0:
            metrics_dict = {
                f"{namespace}/loss": loss,
                f"{namespace}/loss_main": loss_main,
                f"{namespace}/loss_readout": readout_loss,
            }

            if getattr(self.logging_cfg, "loss_bucket_count", 0) > 0:
                noise_level_bucket_means, _, noise_level_bucket_labels = self._bucketed_loss(
                    k=readout_noise_level_target,
                    loss_per_token=readout_loss_per_sample,
                    masks=None,
                    num_buckets=self.logging_cfg.loss_bucket_count,
                )
                if (
                    noise_level_bucket_means is not None
                    and noise_level_bucket_labels is not None
                ):
                    metrics_dict.update(
                        {
                            f"{namespace}/noise_level_loss_bucket_{noise_level_bucket_labels[i]}": noise_level_bucket_means[i]
                            for i in range(noise_level_bucket_means.shape[0])
                        }
                    )

            self.log_dict(
                metrics_dict,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
                prog_bar=True,
            )

        return output_dict
