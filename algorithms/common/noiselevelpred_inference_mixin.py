from __future__ import annotations

from typing import Any

import numpy as np
import torch
import wandb
from einops import reduce
from torch import Tensor

from algorithms.denoising.noise_schedule import build_timestep_rand_fn
from utils.noise_level_eval_utils import (
    make_noise_level_abs_delta_error_curve,
    make_noise_level_grid_error_matrix,
    reduce_grid_abs_delta_curve,
)


class NoiseLevelPredInferenceMixin:
    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self._noise_level_ablation_accumulators: dict[int, dict[str, Any]] = {}

    def _ablation_error_metric(self) -> str:
        ablation_cfg = getattr(self.readout_cfg, "ablation", None)
        metric = getattr(ablation_cfg, "error_metric", None) if ablation_cfg is not None else None
        if metric is None:
            return str(self.readout_cfg.loss)
        return str(metric)

    def _should_run_noise_level_ablations(self) -> bool:
        if not getattr(self, "readout_enabled", False):
            return False
        ablation_cfg = getattr(self.readout_cfg, "ablation", None)
        if ablation_cfg is None or not bool(getattr(ablation_cfg, "enabled", False)):
            return False
        if self.trainer.sanity_checking:
            return False
        return True

    def _valid_noise_level_mask(self, masks: Tensor | None) -> Tensor | None:
        if masks is None:
            return None
        if masks.ndim > 2:
            return reduce(masks.bool(), "b t ... -> b t", torch.any)
        return masks.bool()

    def _coerce_ablation_per_frame_tensor(
        self,
        tensor: Tensor,
        *,
        reference: Tensor,
    ) -> Tensor:
        if tensor.shape == reference.shape:
            return tensor
        if tensor.ndim == 0:
            return torch.full_like(reference, tensor)
        if tensor.ndim == 1:
            if tensor.shape[0] == reference.shape[0]:
                return tensor[:, None].expand_as(reference)
            if tensor.numel() == reference.numel():
                return tensor.reshape_as(reference)
        if tensor.ndim >= 2 and tuple(tensor.shape[:2]) == tuple(reference.shape):
            return tensor.reshape(*reference.shape, -1).mean(dim=-1)
        raise ValueError(
            "Could not coerce ablation tensor to per-frame shape. "
            f"Got {tuple(tensor.shape)} with reference {tuple(reference.shape)}."
        )

    def _maybe_update_noise_level_ablation_output(
        self,
        output_dict: dict[str, Tensor],
        *,
        ablation_eval_mode: bool,
        pred_raw: Tensor,
        true_noise_level: Tensor,
        show_noise_level: Tensor | None,
        masks: Tensor | None,
        v_loss: Tensor,
        x_loss: Tensor,
    ) -> None:
        if not ablation_eval_mode:
            return

        fake_noise_level = true_noise_level if show_noise_level is None else show_noise_level
        valid_noise_level_mask = self._valid_noise_level_mask(masks)
        pred_noise_level = self.readout_head.predict_from_raw(pred_raw)
        _, noise_level_error_true, _ = self.readout_head.compute_error_from_raw(
            pred_raw,
            true_noise_level,
            masks=None,
            loss_type=self._ablation_error_metric(),
            return_details=True,
        )
        _, noise_level_error_fake, _ = self.readout_head.compute_error_from_raw(
            pred_raw,
            fake_noise_level,
            masks=None,
            loss_type=self._ablation_error_metric(),
            return_details=True,
        )
        v_loss_per_frame = self._coerce_ablation_per_frame_tensor(
            v_loss,
            reference=true_noise_level,
        )
        x_loss_per_frame = self._coerce_ablation_per_frame_tensor(
            x_loss,
            reference=true_noise_level,
        )
        output_dict.update(
            {
                "pred_noise_level": pred_noise_level,
                "true_noise_level": true_noise_level,
                "fake_noise_level": fake_noise_level,
                "noise_level_error_true": noise_level_error_true,
                "noise_level_error_fake": noise_level_error_fake,
                "v_loss_per_frame": v_loss_per_frame,
                "x_loss_per_frame": x_loss_per_frame,
                "valid_noise_level_mask": (
                    valid_noise_level_mask
                    if valid_noise_level_mask is not None
                    else torch.ones_like(true_noise_level, dtype=torch.bool)
                ),
            }
        )

    def _sample_ablation_noise_levels(self, xs: Tensor) -> Tensor:
        batch_size, n_tokens = xs.shape[:2]
        rand_fn = build_timestep_rand_fn(
            use_continuous_timesteps=self.use_continuous_timesteps,
            num_noise_levels=self.num_noise_levels,
            device=xs.device,
            generator=self.generator,
        )
        if self.cfg.noise_level == "random_independent":
            return rand_fn((batch_size, n_tokens))
        if self.cfg.noise_level == "random_uniform":
            return rand_fn((batch_size, 1)).repeat(1, n_tokens)
        raise ValueError(
            "Noise-level readout ablations only support `random_independent` and `random_uniform`."
        )

    def _build_random_ablation_specs(self) -> list[dict[str, Any]]:
        ablation_cfg = self.readout_cfg.ablation
        random_cfg = ablation_cfg.random
        if not bool(getattr(random_cfg, "enabled", False)):
            return []
        return [
            {
                "mode": "random",
                "repeat_idx": repeat_idx,
            }
            for repeat_idx in range(int(random_cfg.repeats))
        ]

    def _build_grid_ablation_specs(self) -> list[dict[str, Any]]:
        ablation_cfg = self.readout_cfg.ablation
        grid_cfg = ablation_cfg.grid
        if not bool(getattr(grid_cfg, "enabled", False)):
            return []
        grid_values = [float(value) for value in grid_cfg["values"]]
        if len(grid_values) == 0:
            raise ValueError("algorithm.readout.ablation.grid.values must be non-empty when grid is enabled.")
        specs: list[dict[str, Any]] = []
        for repeat_idx in range(int(grid_cfg.repeats)):
            for true_idx, true_value in enumerate(grid_values):
                for fake_idx, fake_value in enumerate(grid_values):
                    specs.append(
                        {
                            "mode": "grid",
                            "repeat_idx": repeat_idx,
                            "true_time_index": true_idx,
                            "fake_time_index": fake_idx,
                            "true_time_value": true_value,
                            "fake_time_value": fake_value,
                        }
                    )
        return specs

    def _materialize_ablation_noise_levels(
        self,
        spec: dict[str, Any],
        xs: Tensor,
        masks: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        valid = self._valid_noise_level_mask(masks)
        if spec["mode"] == "random":
            true_noise_level = self._sample_ablation_noise_levels(xs)
            fake_noise_level = self._sample_ablation_noise_levels(xs)
        elif spec["mode"] == "grid":
            true_noise_level = torch.full(
                xs.shape[:2],
                float(spec["true_time_value"]),
                dtype=xs.dtype,
                device=xs.device,
            )
            fake_noise_level = torch.full(
                xs.shape[:2],
                float(spec["fake_time_value"]),
                dtype=xs.dtype,
                device=xs.device,
            )
        else:
            raise ValueError(f"Unsupported ablation mode: {spec['mode']}")

        if valid is not None:
            fill_value = 1.0 if self.use_continuous_timesteps else float(self.num_noise_levels - 1)
            fill_tensor = torch.full_like(true_noise_level, fill_value)
            true_noise_level = torch.where(valid, true_noise_level, fill_tensor)
            fake_noise_level = torch.where(valid, fake_noise_level, fill_tensor)
        return true_noise_level, fake_noise_level

    def _build_noise_level_ablation_accumulator(self) -> dict[str, Any]:
        ablation_cfg = self.readout_cfg.ablation
        accumulator: dict[str, Any] = {}
        if bool(getattr(ablation_cfg.random, "enabled", False)):
            bins = int(ablation_cfg.random.abs_delta_bins)
            accumulator["random"] = {
                "sum_error_true_by_abs_delta_bin": torch.zeros(
                    bins, dtype=torch.float64, device=self.device
                ),
                "sum_error_fake_by_abs_delta_bin": torch.zeros(
                    bins, dtype=torch.float64, device=self.device
                ),
                "sum_v_loss_by_abs_delta_bin": torch.zeros(
                    bins, dtype=torch.float64, device=self.device
                ),
                "sum_x_loss_by_abs_delta_bin": torch.zeros(
                    bins, dtype=torch.float64, device=self.device
                ),
                "sum_pred_by_abs_delta_bin": torch.zeros(
                    bins, dtype=torch.float64, device=self.device
                ),
                "sum_true_by_abs_delta_bin": torch.zeros(
                    bins, dtype=torch.float64, device=self.device
                ),
                "sum_fake_by_abs_delta_bin": torch.zeros(
                    bins, dtype=torch.float64, device=self.device
                ),
                "count_by_abs_delta_bin": torch.zeros(
                    bins, dtype=torch.float64, device=self.device
                ),
            }
        if bool(getattr(ablation_cfg.grid, "enabled", False)):
            grid_values = [float(value) for value in ablation_cfg.grid["values"]]
            n_values = len(grid_values)
            accumulator["grid_values"] = grid_values
            accumulator["grid"] = {
                "sum_error_true_matrix": torch.zeros(
                    (n_values, n_values), dtype=torch.float64, device=self.device
                ),
                "sum_error_fake_matrix": torch.zeros(
                    (n_values, n_values), dtype=torch.float64, device=self.device
                ),
                "sum_v_loss_matrix": torch.zeros(
                    (n_values, n_values), dtype=torch.float64, device=self.device
                ),
                "sum_x_loss_matrix": torch.zeros(
                    (n_values, n_values), dtype=torch.float64, device=self.device
                ),
                "sum_pred_matrix": torch.zeros(
                    (n_values, n_values), dtype=torch.float64, device=self.device
                ),
                "count_matrix": torch.zeros(
                    (n_values, n_values), dtype=torch.float64, device=self.device
                ),
            }
        return accumulator

    def _get_noise_level_ablation_accumulator(self, dataloader_idx: int) -> dict[str, Any]:
        # Keep ablation summaries separate per validation dataloader.
        if dataloader_idx not in self._noise_level_ablation_accumulators:
            self._noise_level_ablation_accumulators[dataloader_idx] = (
                self._build_noise_level_ablation_accumulator()
            )
        return self._noise_level_ablation_accumulators[dataloader_idx]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0, namespace="validation") -> None:
        super().validation_step(
            batch,
            batch_idx,
            dataloader_idx=dataloader_idx,
            namespace=namespace,
        )
        if not self._should_run_noise_level_ablations():
            return
        self._run_noise_level_ablation_passes(
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            namespace=namespace,
        )

    def _run_noise_level_ablation_passes(
        self,
        *,
        batch,
        batch_idx: int,
        dataloader_idx: int,
        namespace: str,
    ) -> None:
        accumulator = self._get_noise_level_ablation_accumulator(dataloader_idx)
        xs, _conditions, masks, _gt_videos, _video_metadata = batch
        xs_eval = xs[:, : self.forward_window_size_in_tokens]
        masks_eval = masks[:, : self.forward_window_size_in_tokens]
        specs = self._build_random_ablation_specs() + self._build_grid_ablation_specs()
        for spec in specs:
            true_noise_level, fake_noise_level = self._materialize_ablation_noise_levels(
                spec=spec,
                xs=xs_eval,
                masks=masks_eval,
            )
            result = self._run_eval_denoising_once(
                batch,
                batch_idx,
                dataloader_idx,
                namespace=namespace,
                prepare_visuals=False,
                training_step_kwargs={
                    "noise_level_override": true_noise_level,
                    "show_noise_level_override": fake_noise_level,
                    "ablation_eval_mode": True,
                },
            )
            self._accumulate_noise_level_ablation_metrics(
                accumulator=accumulator,
                spec=spec,
                output=result["output"],
            )

    def _accumulate_noise_level_ablation_metrics(
        self,
        *,
        accumulator: dict[str, Any],
        spec: dict[str, Any],
        output: dict[str, Tensor],
    ) -> None:
        pred = output["pred_noise_level"].detach().to(dtype=torch.float64)
        true_noise_level = output["true_noise_level"].detach().to(dtype=torch.float64)
        fake_noise_level = output["fake_noise_level"].detach().to(dtype=torch.float64)
        error_true = output["noise_level_error_true"].detach().to(dtype=torch.float64)
        error_fake = output["noise_level_error_fake"].detach().to(dtype=torch.float64)
        v_loss = output["v_loss_per_frame"].detach().to(dtype=torch.float64)
        x_loss = output["x_loss_per_frame"].detach().to(dtype=torch.float64)
        valid = output["valid_noise_level_mask"].detach().bool()
        if not bool(valid.any()):
            return

        pred = pred[valid]
        true_noise_level = true_noise_level[valid]
        fake_noise_level = fake_noise_level[valid]
        error_true = error_true[valid]
        error_fake = error_fake[valid]
        v_loss = v_loss[valid]
        x_loss = x_loss[valid]

        if spec["mode"] == "random":
            random_acc = accumulator["random"]
            num_bins = int(random_acc["count_by_abs_delta_bin"].shape[0])
            abs_delta = torch.abs(fake_noise_level - true_noise_level).clamp(min=0.0, max=1.0)
            bucket_idx = torch.floor(abs_delta * num_bins).long().clamp(0, num_bins - 1)
            ones = torch.ones_like(pred, dtype=torch.float64)
            random_acc["sum_error_true_by_abs_delta_bin"].scatter_add_(0, bucket_idx, error_true)
            random_acc["sum_error_fake_by_abs_delta_bin"].scatter_add_(0, bucket_idx, error_fake)
            random_acc["sum_v_loss_by_abs_delta_bin"].scatter_add_(0, bucket_idx, v_loss)
            random_acc["sum_x_loss_by_abs_delta_bin"].scatter_add_(0, bucket_idx, x_loss)
            random_acc["sum_pred_by_abs_delta_bin"].scatter_add_(0, bucket_idx, pred)
            random_acc["sum_true_by_abs_delta_bin"].scatter_add_(0, bucket_idx, true_noise_level)
            random_acc["sum_fake_by_abs_delta_bin"].scatter_add_(0, bucket_idx, fake_noise_level)
            random_acc["count_by_abs_delta_bin"].scatter_add_(0, bucket_idx, ones)
            return

        if spec["mode"] == "grid":
            grid_acc = accumulator["grid"]
            true_idx = int(spec["true_time_index"])
            fake_idx = int(spec["fake_time_index"])
            grid_acc["sum_error_true_matrix"][true_idx, fake_idx] += error_true.sum()
            grid_acc["sum_error_fake_matrix"][true_idx, fake_idx] += error_fake.sum()
            grid_acc["sum_v_loss_matrix"][true_idx, fake_idx] += v_loss.sum()
            grid_acc["sum_x_loss_matrix"][true_idx, fake_idx] += x_loss.sum()
            grid_acc["sum_pred_matrix"][true_idx, fake_idx] += pred.sum()
            grid_acc["count_matrix"][true_idx, fake_idx] += float(pred.numel())
            return

        raise ValueError(f"Unsupported ablation mode: {spec['mode']}")

    def _reduce_noise_level_ablation_tensor(self, tensor: Tensor) -> Tensor:
        gathered = self.gather_data(tensor.unsqueeze(0), batch_dim=0)
        return gathered.sum(dim=0)

    def _noise_level_ablation_key(self, *, namespace: str, dataloader_idx: int, name: str) -> str:
        num_loaders = len(getattr(self.trainer, "val_dataloaders", []))
        if num_loaders <= 1:
            return f"{namespace}/noise_level_ablation/{name}"
        return f"{namespace}/noise_level_ablation/loader_id={dataloader_idx}/{name}"

    def _log_noise_level_ablation_table(
        self,
        *,
        key: str,
        columns: list[str],
        rows: list[list[float]],
    ) -> None:
        if len(rows) == 0:
            return
        table = wandb.Table(columns=columns, data=rows)
        self._wandb_log({key: table}, commit=False)

    def _flush_noise_level_ablation_metrics(self, *, namespace: str) -> None:
        for dataloader_idx, accumulator in self._noise_level_ablation_accumulators.items():
            if "random" in accumulator:
                random_acc = {
                    key: self._reduce_noise_level_ablation_tensor(value).detach().cpu().numpy()
                    for key, value in accumulator["random"].items()
                }
                counts = random_acc["count_by_abs_delta_bin"]
                if counts.sum() > 0:
                    bin_edges = np.linspace(
                        0.0,
                        1.0,
                        int(counts.shape[0]) + 1,
                        dtype=np.float32,
                    )
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    mean_error_true = np.divide(
                        random_acc["sum_error_true_by_abs_delta_bin"],
                        np.clip(counts, 1.0, None),
                    )
                    self.log_image(
                        key=self._noise_level_ablation_key(
                            namespace=namespace,
                            dataloader_idx=dataloader_idx,
                            name="random_abs_delta_curve",
                        ),
                        image=make_noise_level_abs_delta_error_curve(
                            x_values=bin_centers,
                            mean_error=mean_error_true,
                            counts=counts,
                            title="Random readout ablation",
                        ),
                    )
                    mean_v_loss = np.divide(
                        random_acc["sum_v_loss_by_abs_delta_bin"],
                        np.clip(counts, 1.0, None),
                    )
                    self.log_image(
                        key=self._noise_level_ablation_key(
                            namespace=namespace,
                            dataloader_idx=dataloader_idx,
                            name="random_abs_delta_v_loss_curve",
                        ),
                        image=make_noise_level_abs_delta_error_curve(
                            x_values=bin_centers,
                            mean_error=mean_v_loss,
                            counts=counts,
                            title="Random v loss ablation",
                            ylabel="v loss",
                        ),
                    )
                    mean_x_loss = np.divide(
                        random_acc["sum_x_loss_by_abs_delta_bin"],
                        np.clip(counts, 1.0, None),
                    )
                    self.log_image(
                        key=self._noise_level_ablation_key(
                            namespace=namespace,
                            dataloader_idx=dataloader_idx,
                            name="random_abs_delta_x_loss_curve",
                        ),
                        image=make_noise_level_abs_delta_error_curve(
                            x_values=bin_centers,
                            mean_error=mean_x_loss,
                            counts=counts,
                            title="Random x loss ablation",
                            ylabel="x loss",
                        ),
                    )
                    random_rows: list[list[float]] = []
                    for idx in range(int(counts.shape[0])):
                        count = float(counts[idx])
                        denom = max(count, 1.0)
                        random_rows.append(
                            [
                                float(bin_edges[idx]),
                                float(bin_edges[idx + 1]),
                                float(random_acc["sum_pred_by_abs_delta_bin"][idx] / denom),
                                float(random_acc["sum_true_by_abs_delta_bin"][idx] / denom),
                                float(random_acc["sum_fake_by_abs_delta_bin"][idx] / denom),
                                float(random_acc["sum_error_true_by_abs_delta_bin"][idx] / denom),
                                float(random_acc["sum_error_fake_by_abs_delta_bin"][idx] / denom),
                                float(random_acc["sum_v_loss_by_abs_delta_bin"][idx] / denom),
                                float(random_acc["sum_x_loss_by_abs_delta_bin"][idx] / denom),
                                count,
                            ]
                        )
                    self._log_noise_level_ablation_table(
                        key=self._noise_level_ablation_key(
                            namespace=namespace,
                            dataloader_idx=dataloader_idx,
                            name="random_table",
                        ),
                        columns=[
                            "abs_delta_bin_left",
                            "abs_delta_bin_right",
                            "mean_pred_time",
                            "mean_true_time",
                            "mean_fake_time",
                            "mean_error_to_true",
                            "mean_error_to_fake",
                            "mean_v_loss",
                            "mean_x_loss",
                            "count",
                        ],
                        rows=random_rows,
                    )

            if "grid" in accumulator:
                grid_acc = {
                    key: self._reduce_noise_level_ablation_tensor(value).detach().cpu().numpy()
                    for key, value in accumulator["grid"].items()
                }
                count_matrix = grid_acc["count_matrix"]
                if count_matrix.sum() > 0:
                    grid_values = accumulator["grid_values"]
                    mean_error_matrix = np.divide(
                        grid_acc["sum_error_true_matrix"],
                        np.clip(count_matrix, 1.0, None),
                    )
                    self.log_image(
                        key=self._noise_level_ablation_key(
                            namespace=namespace,
                            dataloader_idx=dataloader_idx,
                            name="grid_error_matrix",
                        ),
                        image=make_noise_level_grid_error_matrix(
                            grid_values=grid_values,
                            mean_error_matrix=mean_error_matrix,
                            title="Grid readout ablation",
                        ),
                    )
                    mean_v_loss_matrix = np.divide(
                        grid_acc["sum_v_loss_matrix"],
                        np.clip(count_matrix, 1.0, None),
                    )
                    self.log_image(
                        key=self._noise_level_ablation_key(
                            namespace=namespace,
                            dataloader_idx=dataloader_idx,
                            name="grid_v_loss_matrix",
                        ),
                        image=make_noise_level_grid_error_matrix(
                            grid_values=grid_values,
                            mean_error_matrix=mean_v_loss_matrix,
                            title="Grid v loss ablation",
                            colorbar_label="v loss",
                        ),
                    )
                    mean_x_loss_matrix = np.divide(
                        grid_acc["sum_x_loss_matrix"],
                        np.clip(count_matrix, 1.0, None),
                    )
                    self.log_image(
                        key=self._noise_level_ablation_key(
                            namespace=namespace,
                            dataloader_idx=dataloader_idx,
                            name="grid_x_loss_matrix",
                        ),
                        image=make_noise_level_grid_error_matrix(
                            grid_values=grid_values,
                            mean_error_matrix=mean_x_loss_matrix,
                            title="Grid x loss ablation",
                            colorbar_label="x loss",
                        ),
                    )
                    grid_curve_x, grid_curve_y, grid_curve_counts = reduce_grid_abs_delta_curve(
                        grid_values=grid_values,
                        sum_error_matrix=grid_acc["sum_error_true_matrix"],
                        count_matrix=count_matrix,
                    )
                    if grid_curve_counts.size > 0:
                        self.log_image(
                            key=self._noise_level_ablation_key(
                                namespace=namespace,
                                dataloader_idx=dataloader_idx,
                                name="grid_abs_delta_curve",
                            ),
                            image=make_noise_level_abs_delta_error_curve(
                                x_values=grid_curve_x,
                                mean_error=grid_curve_y,
                                counts=grid_curve_counts,
                                title="Grid readout ablation",
                            ),
                        )
                    (
                        grid_v_loss_curve_x,
                        grid_v_loss_curve_y,
                        grid_v_loss_curve_counts,
                    ) = reduce_grid_abs_delta_curve(
                        grid_values=grid_values,
                        sum_error_matrix=grid_acc["sum_v_loss_matrix"],
                        count_matrix=count_matrix,
                    )
                    if grid_v_loss_curve_counts.size > 0:
                        self.log_image(
                            key=self._noise_level_ablation_key(
                                namespace=namespace,
                                dataloader_idx=dataloader_idx,
                                name="grid_abs_delta_v_loss_curve",
                            ),
                            image=make_noise_level_abs_delta_error_curve(
                                x_values=grid_v_loss_curve_x,
                                mean_error=grid_v_loss_curve_y,
                                counts=grid_v_loss_curve_counts,
                                title="Grid v loss ablation",
                                ylabel="v loss",
                            ),
                        )
                    (
                        grid_x_loss_curve_x,
                        grid_x_loss_curve_y,
                        grid_x_loss_curve_counts,
                    ) = reduce_grid_abs_delta_curve(
                        grid_values=grid_values,
                        sum_error_matrix=grid_acc["sum_x_loss_matrix"],
                        count_matrix=count_matrix,
                    )
                    if grid_x_loss_curve_counts.size > 0:
                        self.log_image(
                            key=self._noise_level_ablation_key(
                                namespace=namespace,
                                dataloader_idx=dataloader_idx,
                                name="grid_abs_delta_x_loss_curve",
                            ),
                            image=make_noise_level_abs_delta_error_curve(
                                x_values=grid_x_loss_curve_x,
                                mean_error=grid_x_loss_curve_y,
                                counts=grid_x_loss_curve_counts,
                                title="Grid x loss ablation",
                                ylabel="x loss",
                            ),
                        )

                    grid_rows: list[list[float]] = []
                    for true_idx, true_value in enumerate(grid_values):
                        for fake_idx, fake_value in enumerate(grid_values):
                            count = float(count_matrix[true_idx, fake_idx])
                            denom = max(count, 1.0)
                            grid_rows.append(
                                [
                                    float(true_value),
                                    float(fake_value),
                                    float(abs(fake_value - true_value)),
                                    float(grid_acc["sum_pred_matrix"][true_idx, fake_idx] / denom),
                                    float(grid_acc["sum_error_true_matrix"][true_idx, fake_idx] / denom),
                                    float(grid_acc["sum_error_fake_matrix"][true_idx, fake_idx] / denom),
                                    float(grid_acc["sum_v_loss_matrix"][true_idx, fake_idx] / denom),
                                    float(grid_acc["sum_x_loss_matrix"][true_idx, fake_idx] / denom),
                                    count,
                                ]
                            )
                    self._log_noise_level_ablation_table(
                        key=self._noise_level_ablation_key(
                            namespace=namespace,
                            dataloader_idx=dataloader_idx,
                            name="grid_table",
                        ),
                        columns=[
                            "true_time",
                            "fake_time",
                            "abs_delta",
                            "mean_pred_time",
                            "mean_error_to_true",
                            "mean_error_to_fake",
                            "mean_v_loss",
                            "mean_x_loss",
                            "count",
                        ],
                        rows=grid_rows,
                    )

    def on_validation_epoch_end(self, namespace="validation") -> None:
        if self._should_run_noise_level_ablations():
            self._flush_noise_level_ablation_metrics(namespace=namespace)
        super().on_validation_epoch_end(namespace=namespace)
