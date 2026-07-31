from typing import Callable, Optional, Dict, Tuple
from omegaconf import DictConfig, open_dict
import torch
from torch import Tensor
from lightning.pytorch.utilities.types import STEP_OUTPUT
from einops import rearrange, reduce
from algorithms.common import BasePytorchAlgo, BaseWMMixin
from abc import abstractmethod

from algorithms.denoising.noise_schedule import build_timestep_rand_fn

from utils.print_utils import cyan, blue, red
from utils.distributed_utils import rank_zero_print
from utils.torch_utils import bernoulli_tensor


class DenoisingVideo(BaseWMMixin, BasePytorchAlgo):
    """
    An algorithm for training and evaluating
    World Models' memory ability on video datasets on different tasks
    This file is meant to be overridden with a particular implementation strategy for the type of denoising.
    """
    def __init__(self, cfg: DictConfig) -> None:
        
        # 1. Shape
        self.x_shape = list(cfg.x_shape)
        self.frame_skip = cfg.frame_skip
        self.chunk_size = cfg.chunk_size
        self.external_cond_dim = cfg.external_cond_dim * (
            cfg.frame_skip if cfg.external_cond_stack else 1
        )

        # 2. Latent
        self.is_latent_diffusion = cfg.latent.enable
        self.use_preprocessed_latents = cfg.latent.enable and cfg.latent.type.startswith("pre_")
        rank_zero_print(
            blue(
                f"latent information: \n - Using latent diffusion: {self.is_latent_diffusion}, \n - use preprocessed latents: {self.use_preprocessed_latents}"
            )
        )
        self.temporal_downsampling_factor = cfg.latent.downsampling_factor[0]
        self.is_latent_video_vae = self.temporal_downsampling_factor > 1

        self.x_shape = [cfg.latent.num_channels] + [
            d // cfg.latent.downsampling_factor[1] for d in self.x_shape[1:]
        ]
        if self.is_latent_video_vae:
            # In the original DFOT repo, there was a check for this. if we revert to using video vae latents, then double check this.
            # self.check_video_vae_compatibility(cfg)
            pass 

        # 3. Diffusion
        self.use_causal_mask = cfg.denoising.use_causal_mask
        self.clip_noise = cfg.denoising.clip_noise
        
        self.denoising_cfg = cfg.denoising
        self.backbone_cfg = cfg.backbone
        self.num_noise_levels = self.denoising_cfg.num_noise_levels
        self.use_continuous_timesteps = cfg.denoising.is_continuous

        self.attn_mask = None

        # 4. Logging
        self.logging_cfg = cfg.logging
            
        self.tasks = [
            task
            for task in ["prediction", "reconstruction"]
            if getattr(cfg.tasks, task).enabled
        ]
        self.num_logged_videos = 0
        self.generator = None
        self.latent_size = None

        self.main_model_prefix = "denoising_model"

        super().__init__(cfg)


    # ---------------------------------------------------------------------
    # Model & Metrics building
    # ---------------------------------------------------------------------
    def configure_model(self) -> None:
        """
        Build the model
        """
        
        # Diffusion Model
        vae_patch_size = self.cfg.vae.downsampling_factor[1]
        _, H, W = map(int, list(self.cfg.x_shape))  
        self.latent_size = [W // vae_patch_size, H // vae_patch_size]
        
        backbone_cfg = self.backbone_cfg

        from algorithms.backbones import create_main_model
        channel_num = backbone_cfg.in_channels

        # Update DictConfig with new keys
        with open_dict(backbone_cfg):
            backbone_cfg.in_channels = channel_num
            backbone_cfg.out_channels = channel_num
            backbone_cfg.sample_width = self.latent_size[0]
            backbone_cfg.sample_height = self.latent_size[1]

            
        self.denoising_model = create_main_model(
            backbone_cfg=backbone_cfg
        )

        if self.cfg.load_model_state:
            model_dict = torch.load(self.cfg.load_model_state, map_location="cpu", weights_only=False)
            if self.cfg.model_state_key in model_dict:
                model_dict = model_dict[self.cfg.model_state_key]
            else:
                state_keys = list(model_dict.keys())
                rank_zero_print(red(f"\n !!! No model state found in {self.cfg.load_model_state}, while loading using \"{self.cfg.model_state_key}\" to load model state !!! \n"))
                rank_zero_print(red(f"Available keys: {state_keys}. You should use one of them to load the state for the model ! \n"))
                raise ValueError(f"No model state found in {self.cfg.load_model_state}, while loading using \"{self.cfg.model_state_key}\" to load model state")

            readout_keys = [
                key for key in model_dict.keys() if key.startswith("readout_head.")
            ]
            if readout_keys:
                rank_zero_print(
                    cyan(
                        f"{self.cfg.load_model_state} contains `readout_head.*` weights, "
                        "but `algorithm.load_model_state` only initializes the denoising "
                        f"model. Ignoring {len(readout_keys)} readout weights."
                    )
                )
                model_dict = {
                    key: value
                    for key, value in model_dict.items()
                    if not key.startswith("readout_head.")
                }

            # remove "denoising_model." prefix
            model_dict = {k.replace("denoising_model.", ""): v for k, v in model_dict.items()}
            load_model = self.cfg.load_model_state_mode
            if load_model == "strict":
                missing_keys, unexpected_keys = self.denoising_model.load_state_dict(model_dict, strict = True)
            elif load_model == "partial":
                missing_keys, unexpected_keys = self.denoising_model.load_state_dict(model_dict, strict = False)
            elif load_model == "smart":
                from utils.ckpt_utils import smart_load_state_dict
                if hasattr(self.cfg, "handlers"):
                    handlers = self.cfg.handlers
                    if handlers == "depth_fine_tuning":
                        from utils.ckpt_utils import depth_fine_tuning_handlers
                        skipped_keys = smart_load_state_dict(self.denoising_model, model_dict, custom_handlers = depth_fine_tuning_handlers, verbose = True)
                    else:
                        raise ValueError(f"Invalid handler: {handlers}")
            else:
                raise ValueError(f"Invalid load_model: {load_model}")
            
            rank_zero_print(cyan(f"\n ==== Successfully loaded {len(model_dict)} parameters from {self.cfg.load_model_state}[{self.cfg.model_state_key}], load model mode: {load_model} ===="))

        self.denoising_algo = self.create_denoising_algo().to(self.denoising_model.dtype)

        # VAE
        if self.is_latent_diffusion and not self.use_preprocessed_latents:
            self._load_vae()
        
        # Build metrics
        self._build_metrics()
        
    
    @abstractmethod
    def create_denoising_algo(self) -> torch.nn.Module:
        """
        Create denoising algorithm, abstract method to be overridden by downstream classes
        """
        

    def training_step(self, batch, batch_idx, dataloader_idx = 0, namespace="training") -> STEP_OUTPUT:
        """
        Training step
        """
        
        denoising_cfg = self.denoising_cfg
        
        xs, conds, masks, gt_videos, metadata = batch
        
        k, masks, attn_mask = self._get_training_noise_levels(
            xs, masks, strategy=denoising_cfg.strategy
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
            masks=masks,
            cfg_dropping_strategy=cfg_dropping_strategy,
            attention_mask=attn_mask,
        )
        loss_dict = self.denoising_algo.training_loss(
            self.denoising_model, x_start = xs, conditions = conds, masks = masks,
            k = k, model_kwargs = model_kwargs, )

        loss = loss_dict["loss"].mean()

        predicted_x_start = loss_dict["predicted_x_start"]
        original_x = loss_dict["original_x"]
        output_dict = {
            "loss": loss,
            "predicted_x_start": predicted_x_start,
            "original_x": original_x,
            "time": k
        }

        if "throughput" in loss_dict:
            output_dict["throughput"] = loss_dict["throughput"]

        if batch_idx % self.logging_cfg.loss_freq == 0:
            metrics_dict = {
                f"{namespace}/loss": loss,
            }

            # Bucketed Loss
            self._update_bucketed_loss_metrics(
                metrics_dict=metrics_dict,
                namespace=namespace,
                k=k,
                loss_per_token=loss_dict.get("x_mse"),
                masks=masks,
            )

            self.log_dict(
                metrics_dict,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
                prog_bar=True
            )

        return output_dict

    def _get_training_noise_levels(
        self, xs: Tensor, masks: Tensor = None, strategy="diffusion-forcing"
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if strategy != "diffusion-forcing":
            raise ValueError(f"Invalid strategy: {strategy}")
        return self._get_training_noise_levels_diffusion_forcing(xs, masks)

    def _get_random_length_history_noise_levels(
        self,
        *,
        batch_size: int,
        n_tokens: int,
        rand_fn: Callable[[tuple[int, ...]], Tensor],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        cfg = getattr(self.cfg, "random_length_history", None)
        if cfg is None:
            raise ValueError(
                "algorithm.random_length_history must be configured when "
                "algorithm.noise_level=random_length_history."
            )

        min_history_tokens = int(getattr(cfg, "min_history_tokens", 0))
        max_history_tokens = int(getattr(cfg, "max_history_tokens", 6))
        clean_history_prob = float(getattr(cfg, "clean_history_prob", 0.5))
        if min_history_tokens < 0:
            raise ValueError("random_length_history.min_history_tokens must be >= 0.")
        if max_history_tokens < min_history_tokens:
            raise ValueError(
                "random_length_history.max_history_tokens must be >= "
                "min_history_tokens."
            )
        if not 0.0 <= clean_history_prob <= 1.0:
            raise ValueError(
                "random_length_history.clean_history_prob must be in [0, 1]."
            )

        max_history_tokens = min(max_history_tokens, n_tokens)
        min_history_tokens = min(min_history_tokens, max_history_tokens)
        history_lens = torch.randint(
            min_history_tokens,
            max_history_tokens + 1,
            (batch_size, 1),
            device=device,
            generator=self.generator,
        )

        history_noise = rand_fn((batch_size, 1))
        future_noise = rand_fn((batch_size, 1))
        clean_history = torch.rand(
            (batch_size, 1),
            device=device,
            generator=self.generator,
        ) < clean_history_prob

        token_indices = torch.arange(n_tokens, device=device).unsqueeze(0)
        history_mask = token_indices < history_lens
        context_mask = history_mask & clean_history
        
        # context_mask is used later to both zero the loss for frames that are chosen to be 
        # 'clean_history', and to set their actual noise levels to 0.
        return torch.where(history_mask, history_noise, future_noise), context_mask


    def _get_training_noise_levels_diffusion_forcing(
        self, xs: Tensor, masks: Tensor = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Generate random noise levels for training. If we enable a context config, early frames or futre frames can get special noise levels.
        With no context config, all frames (context and future) get random noise levels without distinction.

        Context config options:
            - Variable: each frame gets a binary assignment of being context or future with a certain probability
            - Fixed: a first determined set of frames are always context


        Parameters:
            masks: from data loader / on_after_batch_transfer, one indicates not padding, so loss need to be calculated on that
        
        """
        batch_size, n_tokens, *_ = xs.shape # (b, forward_window_size, ...)

        rand_fn = build_timestep_rand_fn(
            use_continuous_timesteps=self.use_continuous_timesteps,
            num_noise_levels=self.num_noise_levels,
            device=xs.device,
            generator=self.generator,
        )

        # Below, build the context_mask which has shape (B, forward_window_size).
        # Baseline training (SD: fixed_context, BD: variable_context)
        context_mask_base = None
        if self.cfg.variable_context.enabled:
            assert (
                not self.cfg.fixed_context.enabled
            ), "Cannot use both fixed and variable context"
            context_mask_base = bernoulli_tensor(
                (batch_size, n_tokens),
                self.cfg.variable_context.prob, # NOTE: variable length is randomly decided, but this looks weird, as it implies context don't need to be consecutive frames
                device=self.device,
                generator=self.generator,
            ).bool()
        elif self.cfg.fixed_context.enabled:
            context_indices = self.cfg.fixed_context.indices or list(
                range(self.training_n_context_tokens)
            )
            context_mask_base = torch.zeros(
                (batch_size, n_tokens), dtype=torch.bool, device=xs.device
            )
            context_mask_base[:, context_indices] = True

        context_mask = context_mask_base
        if self.cfg.noise_level == "random_length_history" and (
            context_mask is not None or self.cfg.uniform_future.enabled
        ):
            raise ValueError(
                "algorithm.noise_level=random_length_history cannot be combined with "
                "fixed_context, variable_context, or uniform_future."
            )

        # decided for the entire video forward_window_size
        match self.cfg.noise_level:
            case "random_independent":  # independent noise levels (Diffusion Forcing)
                noise_levels = rand_fn((batch_size, n_tokens))
            case "random_uniform":  # uniform noise levels shared (Typical Video Diffusion)
                noise_levels = rand_fn((batch_size, 1)).repeat(1, n_tokens)
            case "random_length_history":
                noise_levels, context_mask = self._get_random_length_history_noise_levels(
                    batch_size=batch_size,
                    n_tokens=n_tokens,
                    rand_fn=rand_fn,
                    device=xs.device,
                )
            case _:
                raise ValueError(f"Invalid cfg noise level setting: {self.cfg.noise_level}")

        if self.cfg.uniform_future.enabled:  # simplified training (Dfot Appendix A.5)
            noise_levels[:, self.training_n_context_tokens :] = rand_fn((batch_size, 1)).repeat(
                1, n_tokens - self.training_n_context_tokens
            ) # overwrite future at the same noise level

        # treat frames that are not available as "full noise"
        noise_levels = torch.where(
            reduce(masks.bool(), "b t ... -> b t", torch.any),
            noise_levels,
            torch.full_like(
                noise_levels,
                1 if self.use_continuous_timesteps else self.num_noise_levels - 1,
            ),
        )

        if context_mask is not None:
            # Never apply context masking to padding tokens.
            valid_tokens = reduce(masks.bool(), "b t ... -> b t", torch.any)
            context_mask = torch.logical_and(context_mask, valid_tokens)

            # Baseline context behavior: binary dropout training to enable guidance
            dropout = (
                (
                    self.cfg.variable_context
                    if self.cfg.variable_context.enabled
                    else self.cfg.fixed_context
                ).dropout
                if self.trainer.training
                else 0.0
            ) # always zero
            if dropout != 0.0:
                raise ValueError("Dropout should always be 0.0 during training.")
            context_noise_levels = bernoulli_tensor(
                (batch_size, 1),
                dropout,
                device=xs.device,
                generator=self.generator,
            )
            if not self.use_continuous_timesteps:
                context_noise_levels = context_noise_levels.long() * (
                    self.num_noise_levels - 1
                )
            noise_levels = torch.where(context_mask, context_noise_levels, noise_levels)

            # modify masks to exclude context frames from loss computation
            context_mask_exp = rearrange(
                context_mask, "b t -> b t" + " 1" * len(masks.shape[2:])
            )
            masks = torch.where(context_mask_exp, False, masks)

        return noise_levels, masks, None


    def on_after_batch_transfer(
        self, batch: Dict, dataloader_idx: int
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        """
        Preprocess the batch before training/validation.

        Args:
            batch (Dict): The batch of data. Contains "videos" or "latents", (optional) "conditions", and "masks".
            dataloader_idx (int): The index of the dataloader.
        Returns:
            xs (Tensor, "B n_tokens *x_shape"): Tokens to be processed by the model.
            conditions (Optional[Tensor], "B n_tokens d"): External conditions for the tokens.
            masks (Tensor, "B n_tokens"): Masks for the tokens.
            gt_videos (Optional[Tensor], "B n_frames *x_shape"): Optional ground truth videos, used for validation in latent diffusion.
        """
        # 1. Tokenize the videos and optionally prepare the ground truth videos
        if type(batch) == list:
            batch = batch[dataloader_idx]
            
        device = self.denoising_model.device
        
        if self.is_latent_diffusion:
            if self.use_preprocessed_latents:
                xs = batch["latents"]
            else:
                xs = self._encode(batch["videos"])
            if "videos" in batch:
                gt_videos = batch["videos"].to(device)
            else:
                if hasattr(self, "vae") and self.vae is not None:
                    # If we only have latents, decode to RGB for metrics/logging.
                    # Important: for non-causal VideoVAEs, decoding may return a padded length (e.g. 20)
                    # even when the intended clip length is shorter (e.g. 17). Keep GT aligned to cfg.n_frames.
                    desired_len = int(getattr(self.cfg, "n_frames", 0) or 0) or None
                    gt_videos = self._decode(xs, desired_length=desired_len)
                else:
                    gt_videos = None
        else:
            xs = batch["videos"]

        xs = xs.to(device)
        # NOTE: When using preprocessed latents, they are often stored as float16 on disk.
        # In standalone validation/inference with `Trainer(precision=32)`, Lightning does
        # not enable autocast, so float16 inputs can hit dtype-mismatch errors inside
        # components that expect full-float inputs (e.g. some embedding MLPs).
        # During `trainer.fit(...)`, validation uses the *training* precision plugin, so
        # this issue may not reproduce there (e.g. bf16 autocast).
        trainer_precision = getattr(self.trainer, "precision", None)
        if trainer_precision is not None:
            tp = str(trainer_precision)
            if tp.startswith("32") and xs.dtype != torch.float32:
                xs = xs.to(dtype=torch.float32)
        # 2. Prepare external conditions
        conditions = batch.get("conds", None)
        if conditions is not None:
            conditions = conditions.to(device=device, dtype=xs.dtype)
        else:
            # TODO: See if there is a better fix for this
            # Unconditional / no-conditioning runs use external_cond_dim == 0.
            # Represent "no conditioning" as an empty feature tensor (B, T, 0).
            if self.external_cond_dim == 0:
                conditions = xs.new_empty((xs.shape[0], xs.shape[1], 0))
            else:
                raise ValueError(
                    "Batch is missing key `conds` but config has external_cond_dim > 0. "
                    "Either provide conditions in the dataset or set dataset.external_cond_dim=0."
                )
        conditions = self._encode_conditions(conditions)
        
        # 3. Prepare the masks
        if "masks" in batch:
            assert (
                not self.is_latent_video_vae
            ), "Masks should not be provided from the dataset when using VideoVAE. " 
        else:
            masks = torch.ones(*xs.shape[:2]).bool().to(device)

        return xs, conditions, masks, gt_videos, batch["metadata"]
    

    # ---------------------------------------------------------------------
    # Validation & Test
    # ---------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        # If using preprocessed latents, we don't keep the vae in memory during training. need 
        # to reload it during validation (only if we plan to decode/log RGB videos).
        if self.is_latent_diffusion and self.use_preprocessed_latents:
            load_vae_for_eval = getattr(self.cfg.latent, "load_vae_for_eval", True)
            if load_vae_for_eval:
                self._load_vae()
        self._init_global_grid_epoch_accumulator()
        self.num_logged_videos = [0] * len(self.trainer.val_dataloaders)
        self._num_logged_videos_by_task = {}
        if self.cfg.logging.deterministic is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(
                self.cfg.logging.deterministic
            )

    def on_validation_epoch_end(self, namespace="validation") -> None:
        # Flush accumulated global-grid tensors once per epoch so means are
        # aggregated across validation batches (batch-size agnostic).
        self._flush_global_grid_epoch_accumulator(namespace=namespace)
        super().on_validation_epoch_end(namespace=namespace)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0, namespace="validation") -> STEP_OUTPUT:
        """Validation step"""
        # 1. If running validation while training a model, directly evaluate
        # the denoising performance to detect overfitting, etc.
        # Logs the "denoising_vis" visualization as well as "validation/loss" metric.
        
        if self.trainer.sanity_checking:
            if "reconstruction" in self.tasks:
                self._eval_denoising(batch, batch_idx, dataloader_idx=dataloader_idx, namespace=namespace)

        # 2. Sample all videos (based on the specified tasks)
        # and log the generated videos and metrics
        
        elif not (
            self.trainer.sanity_checking and not self.logging_cfg.sanity_generation
        ):
            if "reconstruction" in self.tasks:
                self._eval_denoising(batch, batch_idx, dataloader_idx=dataloader_idx, namespace=namespace)

            if self.cfg.validation.sample_during_training:
                all_videos, video_metadata, other_results_by_task = self._sample_all_videos(
                    batch, batch_idx, namespace
                )

                # Log actual inference work (NFEs / touches) as scalar metrics.
                self._log_inference_step_counts(
                    other_results_by_task=other_results_by_task,
                    dataloader_idx=dataloader_idx,
                )

                # Log per-threshold empirical (readout-driven) step counts to
                # reach each noise level, plus the scheduled baseline.
                self._log_inference_step_to_noise_level_thresholds(
                    other_results_by_task=other_results_by_task,
                    dataloader_idx=dataloader_idx,
                )
    
                # Log global grids
                self._log_global_grids(other_results_by_task, namespace, dataloader_idx=dataloader_idx, video_metadata=video_metadata)

                # Standard visualizations
                self._update_metrics(all_videos, dataloader_idx=dataloader_idx)
                self._log_videos(all_videos, namespace, dataloader_idx=dataloader_idx, video_metadata=video_metadata)
        return
