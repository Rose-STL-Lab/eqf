"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from abc import ABC
from typing import Any, Optional, Union, Dict, List
import pathlib
import hydra
import torch
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy
from lightning.pytorch.strategies.fsdp import FSDPStrategy
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)

def custom_auto_wrap_policy(module, recurse, nonwrapped_numel: int):
    if isinstance(module, nn.Conv2d):
        print(module)
        return False
    return size_based_auto_wrap_policy(module, recurse, nonwrapped_numel)

import torch.nn as nn
import os
import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from tqdm import tqdm
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.print_utils import cyan
from utils.run_utils import replace_symlink
from utils.distributed_utils import rank_zero_print, rank_zero_only
from utils.lightning_utils import EMA, OneShotFwBwFLOPsDispatch
from .data_modules import BaseDataModule
import numpy as np
from lightning.pytorch.callbacks import DeviceStatsMonitor

import signal
import sys
from utils.distributed_utils import rank_zero_print, is_rank_zero
from pathlib import Path
from omegaconf import OmegaConf

torch.set_float32_matmul_precision("high")

def collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    # Helper function to handle numpy arrays
    def to_writable_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.copy())  # Make a writable copy
        return x

    # Process each item in the batch
    processed_batch = []
    for item in batch:
        if isinstance(item, dict):
            # Handle dictionary items
            processed_item = {k: to_writable_tensor(v) for k, v in item.items()}
            processed_batch.append(processed_item)
        else:
            # Handle non-dictionary items
            processed_batch.append(to_writable_tensor(item))

    return torch.utils.data.default_collate(processed_batch)


def register_signal_handlers(trainer=None, datamodule=None):
    """
    Handles sigterm interrupt
    Will save a checkpoint at save_dir/checkpoints/slurm_preempt_{wandb_id}_step_{global_step}.ckpt
    And then also a metadata file at save_dir/requeue_meta.yaml
    This metadata file will include the absolute path to the checkpoint, the information about the wandb run
        and eventually information about the dataset resuming too
    Finally, to handle the sigterm, a job id that gets requeued will be placed back into the queue with the same
        command as the original, and also the same slurm job id. So we will map from the slurm job id to the 
        directory where we can find the requeue metadata, so we can do the proper resuming.
    """
    def _handle_signal(signum, frame):
        if not is_rank_zero:
            return
        
        if signum == signal.SIGTERM:
            print("Caught SIGTERM — saving checkpoint and requeue metadata...")

            if trainer is None:
                print("No trainer provided — skipping checkpoint save.")
                sys.exit(0)
            
            wandb_id = trainer.logger.version
            # save dir will be the date and hour and minute and second
            output_dir = Path(trainer.logger.save_dir)
            ckpt_file_name = f"slurm_preempt_{wandb_id}_step_{trainer.global_step}.ckpt"
            ckpt_path = output_dir / "checkpoints" / ckpt_file_name
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            trainer.save_checkpoint(str(ckpt_path))
            rank_zero_print(cyan(f"Checkpoint saved at {ckpt_path}"))

            meta = {
                "resume": wandb_id,
                "checkpoint_path": str(ckpt_path),
            }

            # Save metadata to YAML
            meta_path = output_dir / "requeue_meta.yaml"
            meta_path.write_text(OmegaConf.to_yaml(meta))
            rank_zero_print(cyan(f"Requeue metadata saved at {meta_path}"))

            # Create symlink outputs/.requeue_links/<SLURM_JOB_ID> pointing to <output_dir>
            job_id = os.environ.get("SLURM_JOB_ID", "manual_requeue_location")
            if job_id:
                link_path = Path("outputs/.requeue_links") / job_id
                # overwrite if symlink already exists, for the case of manual_requeue_location. SLURM_JOB_ID should not ever end up replicated
                replace_symlink(link_path, output_dir)
                rank_zero_print(cyan(f"Created symlink for requeue: {link_path} -> {output_dir}"))
            sys.exit(0)

        else:
            print(f"Caught signal {signum}")

    signal.signal(signal.SIGTERM, _handle_signal)


class BaseExperiment(ABC):
    """
    Abstract class for an experiment. This generalizes the pytorch lightning Trainer & lightning Module to more
    flexible experiments that doesn't fit in the typical ml loop, e.g. multi-stage reinforcement learning benchmarks.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Constructor

        Args:
            cfg: configuration file that contains everything about the experiment
            logger: a pytorch-lightning WandbLogger instance
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.root_cfg = root_cfg
        self.cfg = root_cfg.experiment
        self.debug = root_cfg.debug
        self.logger = logger if logger else False
        is_requeue = self.root_cfg.get("requeue", None)
        
        if self.root_cfg.algorithm.load_model_state and not is_requeue:
            self.ckpt_path = None
            rank_zero_print(cyan("Loading weights only from ckpt_path, when we don't want to resume steps and epochs: "), self.root_cfg.algorithm.load_model_state)
        else:
            if is_requeue and self.root_cfg.algorithm.load_model_state:
                rank_zero_print(cyan("Originally loaded from weights only, now requeueing using optimizer state as well"))
            self.ckpt_path = ckpt_path
        self.algo = None

    def _build_algo(self):
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        algo_name = self.root_cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this Experiment class. "
                "Make sure you define compatible_algorithms correctly and make sure that each key has "
                "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"
            )
        return self.compatible_algorithms[algo_name](self.root_cfg.algorithm)

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """

        if hasattr(self, task) and callable(getattr(self, task)):
            rank_zero_print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            getattr(self, task)() # NOTE: training, validation, test
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )
            
class BaseLightningExperiment(BaseExperiment):
    """
    Abstract class for pytorch lightning experiments. Useful for computer vision & nlp where main components are
    simply models, datasets and train loop.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    # each key has to be a yaml file under '[project_root]/configurations/dataset' without .yaml suffix
    compatible_datasets: Dict = NotImplementedError
    data_module_cls = BaseDataModule

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        super().__init__(root_cfg, logger, ckpt_path)
        self.data_module = self.data_module_cls(root_cfg, self.compatible_datasets)


    def _build_common_callbacks(self):
        if ("deepspeed" in self.cfg.training.strategy or "fsdp" in self.cfg.training.strategy) and self.training:
            return []
        else:
            if self.cfg.ema.enable:
                return [EMA(**self.cfg.ema)]
            else:
                return []

    def _slug(self, value) -> str:
        return "".join(
            c if (c.isalnum() or c in "._-") else "_" for c in str(value)
        ).strip("_")

    def _ckpt_prefix(self) -> str:
        algo_name = getattr(getattr(self.root_cfg, "algorithm", None), "_name", "algo")
        dataset_name = getattr(getattr(self.root_cfg, "dataset", None), "name", None) or getattr(
            getattr(self.root_cfg, "dataset", None), "_name", "dataset"
        )
        run_id = (
            getattr(self.logger, "version", None)
            if self.logger
            else None
        ) or "run"
        return f"{self._slug(algo_name)}_{self._slug(dataset_name)}_{self._slug(run_id)}"

    def _readout_training_mode(self) -> Optional[str]:
        readout_cfg = getattr(getattr(self.root_cfg, "algorithm", None), "readout", None)
        if readout_cfg is None or not getattr(readout_cfg, "enabled", False):
            return None
        return "readout_only"

    def _has_multiple_validation_dataloaders(self) -> bool:
        validation_cfg = getattr(self.cfg, "validation", None)
        splits_included = getattr(validation_cfg, "splits_included", None)
        if splits_included is None:
            return False
        resolved = OmegaConf.to_container(splits_included, resolve=True)
        return isinstance(resolved, list) and len(resolved) > 1

    def _resolve_checkpoint_monitor_name(self, monitor_name: str) -> str:
        if (
            not self._has_multiple_validation_dataloaders()
            or "/dataloader_idx_" in monitor_name
        ):
            return monitor_name

        if monitor_name.startswith("validation/") or monitor_name.startswith("prediction/"):
            return f"{monitor_name}/dataloader_idx_0"

        return monitor_name

    def _build_last_checkpoint_callback(
        self,
        save_path: pathlib.Path,
        checkpointing_kwargs: Dict[str, Any],
    ) -> ModelCheckpoint:
        """
        Save `last.ckpt` on the configured train-step/epoch/time schedule,
        independent of validation metrics.
        """
        last_checkpoint_kwargs = {
            "monitor": None,
            "save_top_k": 0,
            "save_last": True,
        }
        for key in (
            "every_n_train_steps",
            "every_n_epochs",
            "train_time_interval",
            "save_on_train_epoch_end",
            "save_weights_only",
            "auto_insert_metric_name",
            "enable_version_counter",
        ):
            if key in checkpointing_kwargs:
                last_checkpoint_kwargs[key] = checkpointing_kwargs[key]
        last_checkpoint_kwargs.setdefault("auto_insert_metric_name", False)
        last_checkpoint_kwargs.setdefault("enable_version_counter", False)
        return ModelCheckpoint(
            dirpath=save_path,
            **last_checkpoint_kwargs,
        )

    def _build_base_checkpoint_callbacks(self, save_path: pathlib.Path) -> List:
        callbacks = []

        # Checkpointing
        # - Always keep exactly 2 local checkpoints by default:
        #   1) best (by monitored metric) and 2) most recent ("last")
        # - Use stable filenames (no version counter) so these overwrite in-place.
        checkpointing_cfg = self.cfg.training.checkpointing
        raw_checkpointing_kwargs = OmegaConf.to_container(checkpointing_cfg, resolve=True)
        checkpointing_kwargs: Dict[str, Any] = (
            {str(k): v for k, v in raw_checkpointing_kwargs.items()}
            if isinstance(raw_checkpointing_kwargs, dict)
            else {}
        )
        checkpointing_kwargs.setdefault("monitor", "validation/loss")
        checkpointing_kwargs["monitor"] = self._resolve_checkpoint_monitor_name(
            str(checkpointing_kwargs["monitor"])
        )
        checkpointing_kwargs.setdefault("mode", "min")
        checkpointing_kwargs.setdefault("save_top_k", 1)
        checkpointing_kwargs.setdefault("save_last", True)
        save_last = bool(checkpointing_kwargs.pop("save_last", True))
        checkpointing_kwargs["filename"] = f"{self._ckpt_prefix()}_epoch{{epoch}}_step{{step}}_lowest_loss"
        checkpointing_kwargs.setdefault("auto_insert_metric_name", False)
        checkpointing_kwargs.setdefault("enable_version_counter", False)
        checkpointing_kwargs["save_last"] = False
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_path,
                **checkpointing_kwargs,
            )
        )
        if save_last:
            callbacks.append(
                self._build_last_checkpoint_callback(save_path, checkpointing_kwargs)
            )

        additional_checkpointing_cfg = getattr(
            self.cfg.training, "additional_checkpointing", None
        )
        if additional_checkpointing_cfg:
            # Optional extra checkpoints (e.g. best-FVD) should be conditional on metrics being enabled.
            algo_logging_cfg = getattr(
                getattr(self.root_cfg, "algorithm", None), "logging", None
            )
            algo_metrics = OmegaConf.to_container(
                getattr(algo_logging_cfg, "metrics", []), resolve=True
            ) or []
            enabled_metrics = {str(m).lower() for m in algo_metrics}
            strategy_name = str(self.cfg.training.strategy).lower()

            additional_checkpointing_list = OmegaConf.to_container(
                additional_checkpointing_cfg, resolve=True
            )
            if isinstance(additional_checkpointing_list, dict):
                additional_checkpointing_list = [additional_checkpointing_list]

            for extra_ckpt_cfg in additional_checkpointing_list or []:
                extra_ckpt_cfg = extra_ckpt_cfg or {}
                monitor_name = str(extra_ckpt_cfg.get("monitor", "")).lower()
                if "fvd" in monitor_name:
                    if "fsdp" in strategy_name:
                        rank_zero_print(
                            cyan(
                                "[BaseExp] Skipping FVD checkpoint because FSDP strategy is enabled."
                            )
                        )
                        continue
                extra_ckpt_cfg["monitor"] = self._resolve_checkpoint_monitor_name(
                    str(extra_ckpt_cfg.get("monitor", ""))
                )
                monitor_name = str(extra_ckpt_cfg.get("monitor", "")).lower()
                if monitor_name.split("/")[-1] not in enabled_metrics:
                    rank_zero_print(
                        cyan(
                            f"[BaseExp] Skipping {monitor_name} checkpoint because it is not enabled in algorithm.logging.metrics."
                        )
                    )
                    continue

                extra_ckpt_cfg.setdefault("enable_version_counter", False)
                extra_ckpt_cfg["filename"] = (
                    f"{self._ckpt_prefix()}_epoch{{epoch}}_step{{step}}_best_{monitor_name.replace('/', '-')}"
                )
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=save_path,
                        **extra_ckpt_cfg,
                    )
                )

        return callbacks

    def _build_readout_checkpoint_callbacks(
        self, save_path: pathlib.Path, *, readout_only: bool
    ) -> List:
        checkpointing_cfg = self.cfg.training.checkpointing
        raw_checkpointing_kwargs = OmegaConf.to_container(checkpointing_cfg, resolve=True)
        checkpointing_kwargs: Dict[str, Any] = (
            {str(k): v for k, v in raw_checkpointing_kwargs.items()}
            if isinstance(raw_checkpointing_kwargs, dict)
            else {}
        )
        checkpointing_kwargs["monitor"] = "validation/loss_readout"
        checkpointing_kwargs["monitor"] = self._resolve_checkpoint_monitor_name(
            str(checkpointing_kwargs["monitor"])
        )
        checkpointing_kwargs.setdefault("mode", "min")
        checkpointing_kwargs.setdefault("save_top_k", 1)
        save_last = bool(readout_only)
        checkpointing_kwargs["save_last"] = False
        suffix = "lowest_readout_loss" if readout_only else "lowest_base_and_readout_loss"
        checkpointing_kwargs["filename"] = (
            f"{self._ckpt_prefix()}_epoch{{epoch}}_step{{step}}_{suffix}"
        )
        checkpointing_kwargs.setdefault("auto_insert_metric_name", False)
        checkpointing_kwargs.setdefault("enable_version_counter", False)
        callbacks = [
            ModelCheckpoint(
                dirpath=save_path,
                **checkpointing_kwargs,
            )
        ]
        if save_last:
            callbacks.append(
                self._build_last_checkpoint_callback(save_path, checkpointing_kwargs)
            )
        return callbacks
            
        
    def training(self) -> None:
        """
        All training happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)
            

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if self.cfg.calculate_throughput:
            throughput_cfg = getattr(self.cfg, "throughput", None)
            if isinstance(throughput_cfg, DictConfig):
                throughput_cfg = OmegaConf.to_container(throughput_cfg, resolve=True)
            throughput_cfg = throughput_cfg or {}
            rank_zero_print(cyan(f"[BaseExp] throughput_cfg after conversion: {throughput_cfg}"))

            kwargs = {
                "log_wandb_table": throughput_cfg.get("log_wandb_table", True),
                "warmup_steps": int(throughput_cfg.get("warmup_steps", 63)),
                "module_specs": throughput_cfg.get("module_specs"),
            }
            if throughput_cfg.get("target_attr") is not None:
                kwargs["target_attr"] = throughput_cfg.get("target_attr")
            if throughput_cfg.get("core_module_attr") is not None:
                kwargs["core_module_attr"] = throughput_cfg.get("core_module_attr")

            rank_zero_print(cyan(f"[BaseExp] Initializing OneShotFwBwFLOPsDispatch with kwargs: {kwargs}"))
            callbacks.append(OneShotFwBwFLOPsDispatch(**kwargs))

        save_path = pathlib.Path(
                        hydra.core.hydra_config.HydraConfig.get()["runtime"][
                            "output_dir"
                        ]
                    ) / "checkpoints"
        readout_mode = self._readout_training_mode()
        if readout_mode == "readout_only":
            callbacks += self._build_readout_checkpoint_callbacks(
                save_path, readout_only=True
            )
        else:
            callbacks += self._build_base_checkpoint_callbacks(save_path)
            if readout_mode == "joint":
                callbacks += self._build_readout_checkpoint_callbacks(
                    save_path, readout_only=False
                )
        callbacks += self._build_common_callbacks()

        match self.cfg.training.strategy:
            case "ddp":
                strategy = DDPStrategy(find_unused_parameters=self.cfg.find_unused_parameters) \
                if torch.cuda.device_count() > 1 \
                else "auto"
            case "deepspeed_stage_1":
                strategy = "deepspeed_stage_1"
            case "deepspeed_stage_2":
                strategy = DeepSpeedStrategy(stage=2, overlap_comm=False)
            case "deepspeed_stage_3":
                strategy = DeepSpeedStrategy(stage=3, overlap_comm=False)
            case "fsdp":
                strategy = FSDPStrategy(
                    auto_wrap_policy=custom_auto_wrap_policy,
                    # ignored_modules=[self.algo.vae.encoder, self.algo.vae.patch_embed]
                )
            case "auto":
                strategy = "auto"
            case _:
                strategy = "auto"
        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices=self.cfg.training.devices if not self.cfg.debug else 1,
            num_nodes=self.cfg.num_nodes,
            strategy=strategy,
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
            val_check_interval=self.cfg.validation.val_every_n_step,
            limit_train_batches=self.cfg.training.limit_batch,
            limit_val_batches=self.cfg.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
            precision=self.cfg.training.precision,
            detect_anomaly=self.cfg.debug, 
            num_sanity_val_steps=(
                int(self.cfg.debug)
                if self.cfg.validation.num_sanity_val_steps is None
                else self.cfg.validation.num_sanity_val_steps
            ),
            max_epochs=self.cfg.training.max_epochs,
            max_steps=self.cfg.training.max_steps,
            max_time=self.cfg.training.max_time,
            reload_dataloaders_every_n_epochs=self.cfg.reload_dataloaders_every_n_epochs,
            fast_dev_run=self.cfg.debug,
            use_distributed_sampler=True,
            enable_checkpointing=True,
            log_every_n_steps=self.algo.cfg.logging.loss_freq,
            # barebones=True
        )
        
        # Create signal handler to save a checkpoint on sigterm (slurm preempt)
        register_signal_handlers(trainer)

        trainer.fit(
            self.algo,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
            weights_only=False
        )

    def validation(self) -> None:
        """
        All validation happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        callbacks = [] + self._build_common_callbacks()
        devices = "auto" if not self.cfg.validation.devices else self.cfg.validation.devices

        match self.cfg.validation.strategy:
            case "ddp":
                strategy = DDPStrategy(find_unused_parameters=self.cfg.find_unused_parameters) \
                if torch.cuda.device_count() > 1 \
                else "auto"
            case "deepspeed_stage_1":
                strategy = "deepspeed_stage_1"
            case "deepspeed_stage_2":
                strategy = DeepSpeedStrategy(stage=2, overlap_comm = False)
            case "fsdp":
                strategy = FSDPStrategy(
                    sharding_strategy="FULL_SHARD",
                    cpu_offload=True
                )
            case _:
                strategy = "auto"

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices=devices,
            num_nodes=self.cfg.num_nodes,
            strategy=strategy,
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,
            inference_mode=self.cfg.validation.inference_mode,
        )

        trainer.validate(
            self.algo,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
            weights_only=False,
        )

    def test(self) -> None:
        """
        All testing happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.test.compile:
            self.algo = torch.compile(self.algo)

        callbacks = [] + self._build_common_callbacks()

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=(
                DDPStrategy(find_unused_parameters=self.cfg.find_unused_parameters)
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            limit_test_batches=self.cfg.test.limit_batch,
            precision=self.cfg.test.precision,
            detect_anomaly=False,
            inference_mode=self.cfg.test.inference_mode,
        )

        # Only load the checkpoint if only testing. Otherwise, it will have been loaded
        # and further trained during train.
        trainer.test(
            self.algo,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
        )
