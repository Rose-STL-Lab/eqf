from abc import ABC, abstractmethod
import warnings
from typing import Any, Optional, Union, Sequence, Dict, List, Tuple, cast
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.apply_func import apply_to_collection
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.trainer.states import TrainerFn
import torch
import numpy as np
from PIL import Image
import wandb
import einops
from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print
from utils.distributed_utils import is_rank_zero
from algorithms.common.metrics.video import filter_excluded_video_metrics
from utils.fvd_eval_utils import handle_curve_metrics


class BasePytorchAlgo(pl.LightningModule, ABC):
    """
    A base class for Pytorch algorithms using Pytorch Lightning.
    See https://lightning.ai/docs/pytorch/stable/starter/introduction.html for more details.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.debug = self.cfg.debug
        self.should_validate_ema_weights = False
        self._wandb_group_keys_logged = False

        # self.configure_model()

    @abstractmethod
    def configure_model(self):
        """
        Create all pytorch nn.Modules here.
        """
        raise NotImplementedError

    @abstractmethod
    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        r"""Here you compute and return the training loss and some additional metrics for e.g. the progress bar or
        logger.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: (only if multiple dataloaders used) The index of the dataloader that produced this batch.

        Return:
            Any of these options:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
            - ``None`` - Skip to the next batch. This is only supported for automatic optimization.
                This is not supported for multi-GPU, TPU, IPU, or DeepSpeed.

        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.

        Example::

            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss(out, x)
                return loss

        To use multiple optimizers, you can switch to 'manual optimization' and control their stepping:

        .. code-block:: python

            def __init__(self):
                super().__init__()
                self.automatic_optimization = False


            # Multiple optimizers (e.g.: GANs)
            def training_step(self, batch, batch_idx):
                opt1, opt2 = self.optimizers()

                # do training_step with encoder
                ...
                opt1.step()
                # do training_step with decoder
                ...
                opt2.step()

        Note:
            When ``accumulate_grad_batches`` > 1, the loss returned here will be automatically
            normalized by ``accumulate_grad_batches`` internally.

        """
        return super().training_step(*args, **kwargs)

    def configure_optimizers(self):
        """
        Return an optimizer. If you need to use more than one optimizer, refer to pytorch lightning documentation:
        https://lightning.ai/docs/pytorch/stable/common/optimization.html
        """
        parameters = self.parameters()
        return torch.optim.Adam(parameters, lr=self.cfg.lr)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.should_validate_ema_weights:
            self._load_ema_weights_to_state_dict(checkpoint)

    def _load_ema_weights_to_state_dict(self, checkpoint: dict) -> None:
        """
        Load EMA weights to state dict.
        """
        rank_zero_print(
            cyan(
                "WARNING: should_validate_ema_weights is set to True, but cannot validate with EMA weights."
            ),
            "Please implement '_load_ema_weights_to_state_dict' in your LightningModule to validate with EMA weights.",
        )

    def log_video(
        self,
        key: str,
        video: Union[np.ndarray, torch.Tensor],
        mean: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        std: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        fps: int = 12,
        format: str = "mp4",
    ):
        """
        Log video to wandb. WandbLogger in pytorch lightning does not support video logging yet, so we call wandb directly.

        Args:
            video: a numpy array or tensor, either in form (time, channel, height, width) or in the form
                (batch, time, channel, height, width). The content must be be in 0-255 if under dtype uint8
                or [0, 1] otherwise.
            mean: optional, the mean to unnormalize video tensor, assuming unnormalized data is in [0, 1].
            std: optional, the std to unnormalize video tensor, assuming unnormalized data is in [0, 1].
            key: the name of the video.
            fps: the frame rate of the video.
            format: the format of the video. Can be either "mp4" or "gif".
        """

        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()

        expand_shape = [1] * (len(video.shape) - 2) + [3, 1, 1]
        if std is not None:
            if isinstance(std, (float, int)):
                std = [std] * 3
            if isinstance(std, torch.Tensor):
                std = std.detach().cpu().numpy()
            std = np.array(std).reshape(*expand_shape)
            video = video * std
        if mean is not None:
            if isinstance(mean, (float, int)):
                mean = [mean] * 3
            if isinstance(mean, torch.Tensor):
                mean = mean.detach().cpu().numpy()
            mean = np.array(mean).reshape(*expand_shape)
            video = video + mean

        if video.dtype != np.uint8:
            video = np.clip(video, a_min=0, a_max=1) * 255
            video = video.astype(np.uint8)

        self._wandb_log(
            {
                key: wandb.Video(video, fps=fps, format=format),
            },
            commit=False,
        )

    def log_image(
        self,
        key: str,
        image: Union[np.ndarray, torch.Tensor, Image.Image, Sequence[Image.Image]],
        mean: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        std: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        **kwargs: Any,
    ):
        """
        Log image(s) using WandbLogger.
        Args:
            key: the name of the video.
            image: a single image or a batch of images. If a batch of images, the shape should be (batch, channel, height, width).
            mean: optional, the mean to unnormalize image tensor, assuming unnormalized data is in [0, 1].
            std: optional, the std to unnormalize tensor, assuming unnormalized data is in [0, 1].
            kwargs: optional, WandbLogger log_image kwargs, such as captions=xxx.
        """
        if isinstance(image, Image.Image):
            image = [image]
        elif len(image) and not isinstance(image[0], Image.Image):
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()

            if len(image.shape) == 3:
                image = image[None]

            if image.shape[1] == 3:
                if image.shape[-1] == 3:
                    warnings.warn(
                        f"Two channels in shape {image.shape} have size 3, assuming channel first."
                    )
                image = einops.rearrange(image, "b c h w -> b h w c")

            if std is not None:
                if isinstance(std, (float, int)):
                    std = [std] * 3
                if isinstance(std, torch.Tensor):
                    std = std.detach().cpu().numpy()
                std = np.array(std)[None, None, None]
                image = image * std
            if mean is not None:
                if isinstance(mean, (float, int)):
                    mean = [mean] * 3
                if isinstance(mean, torch.Tensor):
                    mean = mean.detach().cpu().numpy()
                mean = np.array(mean)[None, None, None]
                image = image + mean

            if image.dtype != np.uint8:
                image = np.clip(image, a_min=0.0, a_max=1.0) * 255
                image = image.astype(np.uint8)
                image = [img for img in image]
            # WandbLogger expects a list of images even if already uint8.
            image = [img for img in image]

        captions = kwargs.pop("captions", None)
        if kwargs:
            warnings.warn(
                f"log_image received unsupported kwargs and will ignore them: {list(kwargs.keys())}"
            )

        if isinstance(captions, str):
            captions = [captions] * len(image)

        wandb_images = []
        for i, img in enumerate(image):
            caption = captions[i] if isinstance(captions, Sequence) and i < len(captions) else None
            wandb_images.append(wandb.Image(img, caption=caption))

        payload = {key: wandb_images[0] if len(wandb_images) == 1 else wandb_images}
        self._wandb_log(payload, commit=False)

    def _manual_wandb_step(self) -> Optional[int]:
        """
        Return the step to use for manual wandb logging.
        - During training (`fit`), pin logs to `global_step`.
        - During validation-only/test runs, let wandb auto-step to avoid step=0 collisions.
        """
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        try:
            if trainer.state.fn == TrainerFn.FITTING:
                return int(self.global_step)
        except Exception:
            return None
        return None

    def _wandb_log(self, payload: Dict[str, Any], commit: bool = False) -> None:
        """Manual wandb logging helper with consistent step semantics."""
        if not self.logger or not is_rank_zero:
            return
        log_kwargs: Dict[str, Any] = {"commit": commit}
        step = self._manual_wandb_step()
        if step is not None:
            log_kwargs["step"] = step
        self.logger.experiment.log(payload, **log_kwargs)


        
    def gather_data(
        self, data: Union[torch.Tensor, Dict, List, Tuple], batch_dim: int = 0
    ):
        """
        Gather tensors or collections of tensors from all devices,
        and stack them along the batch dimension.
        Args:
            data: tensor or collection of tensors to gather
            batch_dim: the batch dimension of the original tensor
        """
        # if not ddp, skip gathering and return the original data
        if self.trainer.world_size == 1:
            return apply_to_collection(data, torch.Tensor, lambda x: x.to(self.device))

        # synchronize before gathering
        torch.distributed.barrier()
        gathered_data = self.all_gather(data)

        # (r ... b ...) -> (... (r b) ...)
        rearrange_fn = (
            lambda x: x.permute(
                list(range(1, batch_dim + 1))
                + [0]
                + list(range(batch_dim + 1, x.dim()))
            )
            .reshape(*x.shape[1 : batch_dim + 1], -1, *x.shape[batch_dim + 2 :])
            .contiguous()
        )

        return apply_to_collection(gathered_data, torch.Tensor, rearrange_fn)

    def on_validation_epoch_end(self, namespace="validation") -> None:
        self.generator = None
        # If using preprocessed latents, we don't keep the vae in memory during training. need 
        # to offload it at the end of validation.
        if self.is_latent_diffusion and self.use_preprocessed_latents:
            self.vae = None
        self.num_logged_videos = [0] * len(self.trainer.val_dataloaders)
        self._num_logged_videos_by_task = {}

        if self.trainer.sanity_checking:
            if not self.cfg.logging.sanity_generation:
                rank_zero_print(f"sanity checking, and we set {self.cfg.logging.sanity_generation=} to False, thus skip logging the validation results to logger")
            return

        # Log a combined grouping key into W&B config once per run.
        # Controlled by `cfg.logging.wandb_config_group_by`, e.g.:
        #   ["algorithm.denoising.default_eta_multiple", "algorithm._name"]
        if (not self._wandb_group_keys_logged) and is_rank_zero and self.logger:
            try:
                paths = list(getattr(self.cfg.logging, "wandb_config_group_by", []) or [])
                values: list[str] = []
                kv: Dict[str, object] = {}
                for p in paths:
                    p = str(p)
                    # Accept paths with or without "algorithm." prefix.
                    p_sel = p[len("algorithm.") :] if p.startswith("algorithm.") else p
                    v = OmegaConf.select(self.cfg, p_sel)
                    v_str = "" if v is None else str(v)
                    values.append(v_str)
                    kv[p] = v_str
                if paths:
                    group_key = "_".join([str(p).split(".")[-1] for p in paths])
                    kv[f"group/{group_key}"] = "__".join(values)
                    self.logger.experiment.config.update(kv, allow_val_change=True)
                self._wandb_group_keys_logged = True
            except Exception:
                self._wandb_group_keys_logged = True

        # Log inference settings only for validation-only runs (not during training/fit).
        if self.trainer.state.fn != TrainerFn.FITTING and hasattr(self, "_calc_total_nfes"):
            total_nfes_allowed, num_sampling_steps = self._calc_total_nfes()

            # Log inference settings as scalars via Lightning so they show up in W&B history
            # exactly like other metrics (and can be plotted against them).
            self.log(
                "inference/num_sampling_steps",
                torch.as_tensor(float(num_sampling_steps), device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "inference/max_possible_nfes",
                torch.as_tensor(float(total_nfes_allowed), device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        num_val_loaders = len(getattr(self.trainer, "val_dataloaders", []) or [])
        loader_indices = range(num_val_loaders) if num_val_loaders > 1 else [None]

        for loader_idx in loader_indices:
            for task in self.tasks:
                metric_module = self._metrics(task, dataloader_idx=loader_idx)
                if metric_module is None:
                    continue
                metrics_dict = metric_module.log(task)

                # split into what to log vs what to handle separately
                to_log, to_log_manual = filter_excluded_video_metrics(metrics_dict)

                # Persist curve metrics (e.g. fvd_over_time): rank-zero JSON
                # dump under raw_dir + a wandb line plot via log_image.
                if to_log_manual:
                    logger = cast(Any, self.logger) if self.logger else None
                    wandb_experiment = (
                        logger.experiment if logger is not None else None
                    )
                    handle_curve_metrics(
                        to_log_manual,
                        task=task,
                        loader_idx=loader_idx,
                        raw_dir=OmegaConf.select(
                            self.cfg, "logging.raw_dir", default=None
                        ),
                        step=self._manual_wandb_step(),
                        log_image=self.log_image if self.logger else None,
                        wandb_experiment=wandb_experiment,
                    )

                # Compute local (per-rank) scalars.
                local_log_dict: Dict[str, torch.Tensor] = {}
                for k, v in to_log.items():
                    metric = cast(Any, v)
                    compute = getattr(metric, "compute", None)
                    x = compute() if callable(compute) else metric
                    if not isinstance(x, torch.Tensor):
                        x = torch.as_tensor(x)
                    metric_key = (
                        f"{k}/dataloader_idx_{loader_idx}"
                        if loader_idx is not None
                        else k
                    )
                    local_log_dict[metric_key] = x.to(self.device)

                # Reset after compute.
                for v in to_log.values():
                    reset = getattr(cast(Any, v), "reset", None)
                    if callable(reset):
                        reset()

                # Standard Lightning logging: call on ALL ranks; sync_dist performs the collective.
                # (The logger itself is rank-zero-only; other ranks participate in the sync.)
                self.log_dict(
                    local_log_dict,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

                # Build an aggregated metrics table (single writer), using Lightning's strategy reduce.
                # NOTE: W&B tables are not reducible artifacts; we still log them once on rank 0,
                # but the VALUES are globally aggregated across ranks.
                reduced_for_table = {
                    k: self.trainer.strategy.reduce(v, reduce_op="mean").detach().cpu().item()
                    for k, v in local_log_dict.items()
                }
                if is_rank_zero and self.logger:
                    columns = ["name"] + list(reduced_for_table.keys())
                    row = [self.logger.experiment.name] + list(reduced_for_table.values())
                    table = wandb.Table(columns=columns, data=[row])

                    # In validation-only runs, Lightning doesn't advance `global_step`, and
                    # other logs (e.g. sampling progress) may have already advanced W&B's
                    # internal step. Avoid forcing `step=0` which triggers out-of-order warnings.
                    step = self._manual_wandb_step()
                    step_kw = {"step": step} if step is not None else {}
                    table_key = (
                        f"validation_metrics_table/{task}/dataloader_idx_{loader_idx}"
                        if loader_idx is not None
                        else f"validation_metrics_table/{task}"
                    )
                    wandb.log({table_key: table}, **step_kw, commit=False)
        # Ensure any pending media/table logs are committed so they appear in W&B charts.
        # This is especially important for validation-only runs where no other commit occurs.
        if is_rank_zero and self.logger:
            try:
                if wandb.run is not None:
                    # Ensure at least one committed scalar lands in history so charts/media render.
                    step = self._manual_wandb_step()
                    step_kw = {"step": step} if step is not None else {}
                    wandb.log({"_flush": 0, "validation/commit": 1}, **step_kw, commit=True)
            except Exception:
                pass
        return
