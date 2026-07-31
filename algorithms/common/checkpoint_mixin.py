import torch
from torch import Tensor
from typing import Dict, Any
from utils.print_utils import cyan, red
from utils.distributed_utils import rank_zero_print
from utils.torch_utils import freeze_model

class CheckpointMixin:

    # ---------------------------------------------------------------------
    # Checkpoint Utils
    # ---------------------------------------------------------------------

    def _uncompile_checkpoint(self, checkpoint: Dict[str, Any]):
        """Converts the state_dict if self.main_model is compiled, to uncompiled."""
        if self.cfg.compile:
            checkpoint["state_dict"] = {
                k.replace(f"{self.main_model_prefix}._orig_mod.", f"{self.main_model_prefix}."): v
                for k, v in checkpoint["state_dict"].items()
            }

    def _compile_checkpoint(self, checkpoint: Dict[str, Any]):
        """Converts the state_dict to the format expected by the compiled model."""
        if self.cfg.compile:
            checkpoint["state_dict"] = {
                k.replace(f"{self.main_model_prefix}.", f"{self.main_model_prefix}._orig_mod."): v
                for k, v in checkpoint["state_dict"].items()
            }

    def _should_include_in_checkpoint(self, key: str) -> bool:
        if self.main_model_prefix in key:
            return True
        if key.startswith("readout_head."):
            return True
        return False

    def _checkpoint_has_readout_keys(self, checkpoint: Dict[str, Any]) -> bool:
        return any(key.startswith("readout_head.") for key in checkpoint["state_dict"].keys())

    def _algorithm_cls_has_readout_keys(self) -> bool:
        return any(key.startswith("readout_head.") for key in self.state_dict().keys())
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # 1. (Optionally) uncompile the model's state_dict before saving
        self._uncompile_checkpoint(checkpoint)
        # 2. Only save the meaningful keys defined by self._should_include_in_checkpoint
        # by default, only the model's state_dict is saved and metrics & registered buffes (e.g. diffusion schedule) are not discarded
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            if not self._should_include_in_checkpoint(key):
                del state_dict[key]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # 1. (Optionally) compile the model's state_dict before loading
        self._compile_checkpoint(checkpoint)
        checkpoint_has_readout = self._checkpoint_has_readout_keys(checkpoint)
        alg_cls_has_readout = self._algorithm_cls_has_readout_keys()
        if checkpoint_has_readout and not alg_cls_has_readout:
            algo_name = getattr(self.cfg, "_name", self.__class__.__name__)
            rank_zero_print(
                cyan(
                    "Loaded a checkpoint with `readout_head.*` weights, but the "
                    f"current algorithm ({algo_name}) does not define a readout head. "
                    "Ignoring the readout weights and loading the denoising model only."
                )
            )
        # 2. (Optionally) swap the state_dict of the model with the EMA weights for inference
        super().on_load_checkpoint(checkpoint)
        # 3. (Optionally) reset the optimizer states - for fresh finetuning or resuming training
        if "deepspeed" in self.cfg.training.strategy and self.training:
            return
        if "deepspeed" in self.cfg.validation.strategy and not self.training:
            return
        
        if self.cfg.checkpoint.reset_optimizer:
            # Start optimizer-backed training state from scratch when the resumed run
            # intentionally changes the optimizer layout relative to the source checkpoint.
            #
            # This is critical for readout-only NoiseLevelPred runs, which resume from a
            # base denoiser checkpoint but rebuild the optimizer with a new param-group
            # structure: frozen denoiser group + trainable readout group. If we only clear
            # `optimizer_states` and keep the old `lr_schedulers`, Lightning can restore a
            # scheduler state that was created for the old optimizer shape. In practice,
            # the old scheduler can bind correctly to the inherited denoiser group while
            # leaving the new readout group with an effective learning rate of 0.0, so the
            # readout receives gradients but never updates.
            #
            # The same stale scheduler failure mode applies to any full-checkpoint
            # resume where optimizer parameter groups differ from the checkpoint.
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []

        # 4. Rewrite the state_dict of the checkpoint, only leaving meaningful keys
        # defined by self._should_include_in_checkpoint
        # also print out warnings when the checkpoint does not exactly match the expected format

        new_state_dict = {}
        for key, value in self.state_dict().items():
            if (
                self._should_include_in_checkpoint(key)
                and key in checkpoint["state_dict"]
            ):
                new_state_dict[key] = checkpoint["state_dict"][key]
            else:
                new_state_dict[key] = value

        # print keys that are ignored from the checkpoint
        ignored_keys = [
            key
            for key in checkpoint["state_dict"].keys()
            if not self._should_include_in_checkpoint(key)
        ]
        if ignored_keys:
            rank_zero_print(
                cyan("The following keys are ignored from the checkpoint:"),
                ignored_keys,
            )
        # print keys that are not found in the checkpoint
        missing_keys = [
            key
            for key in self.state_dict().keys()
            if self._should_include_in_checkpoint(key)
            and key not in checkpoint["state_dict"]
        ]
        if missing_keys:
            rank_zero_print(
                cyan("The following keys are not found in the checkpoint:"),
                missing_keys,
            )
            import sys
            print(f"\n\n=== MISSING KEYS ({len(missing_keys)} total) ===", file=sys.stderr)
            for key in missing_keys:
                print(f"  - {key}", file=sys.stderr)
            print("=" * 50 + "\n", file=sys.stderr)
            if self.cfg.checkpoint.strict:
                raise ValueError(
                    f"Found {len(missing_keys)} missing keys in checkpoint. Thus, the checkpoint cannot be loaded. To ignore this error, turn off strict checkpoint loading by setting `algorithm.checkpoint.strict=False`."
                )
            else:
                rank_zero_print(
                    cyan(
                        "Strict checkpoint loading is turned off, so using the initialized value for the missing keys."
                    )
                )
        checkpoint["state_dict"] = new_state_dict


    def _load_ema_weights_to_state_dict(self, checkpoint: Dict[str, Any]) -> None:
        if (
            checkpoint.get("pretrained_ema", False) and len(checkpoint["optimizer_states"]) == 0
        ):
            # NOTE: for lightweight EMA-only ckpts for releasing pretrained models,
            # we already have EMA weights in the state_dict
            rank_zero_print(
                cyan(
                    "EMA weights are already baked into this release checkpoint's "
                    "state_dict; loading them directly."
                )
            )
            return
        if "ema" not in checkpoint.get("optimizer_states", [{}])[0]:
            rank_zero_print(
                red(
                    "No EMA weights found in the checkpoint, so using the initialized value for the EMA weights."
                )
            )
            return
        ema_weights = checkpoint["optimizer_states"][0]["ema"]
        parameter_keys = [
            f"{self.main_model_prefix}." + k for k, _ in getattr(self, self.main_model_prefix).named_parameters()
        ]
        if len(ema_weights) != len(parameter_keys):
            checkpoint_has_readout = self._checkpoint_has_readout_keys(checkpoint)
            alg_cls_has_readout = self._algorithm_cls_has_readout_keys()
            if (
                checkpoint_has_readout
                and not alg_cls_has_readout
                and len(ema_weights) > len(parameter_keys)
            ):
                # NoiseLevelPred stores EMA tensors in denoiser-then-readout order.
                # A base algorithm needs only the denoiser prefix.
                ema_weights = ema_weights[: len(parameter_keys)]
            else:
                raise ValueError(
                    "Number of model parameters and EMA weights do not match: "
                    f"expected {len(parameter_keys)}, found {len(ema_weights)}."
                )
        for key, weight in zip(parameter_keys, ema_weights):
            state_value = checkpoint["state_dict"].get(key)
            if state_value is None or state_value.shape != weight.shape:
                raise ValueError(
                    f"EMA tensor for `{key}` does not match the checkpoint state_dict."
                )
            checkpoint["state_dict"][key] = weight
