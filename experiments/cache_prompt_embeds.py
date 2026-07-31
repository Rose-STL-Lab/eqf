from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, cast

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import open_dict
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from algorithms.wan.modules.t5 import umt5_xxl
from algorithms.wan.modules.tokenizers import HuggingfaceTokenizer
from datasets.video.droid import DroidVideoDataset
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan

from .base_exp import BaseExperiment


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _prompt_embed_meta_path(prompt_embed_path: Path) -> Path:
    return prompt_embed_path.with_suffix(".json")


def _atomic_torch_save(value: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: Dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
    os.replace(temporary, path)


class _PromptWorkDataset(IterableDataset):
    """Shard prompt-cache work exactly across Lightning ranks and workers."""

    def __init__(self, items: List[Tuple[str, str]]) -> None:
        super().__init__()
        self.items = items

    @staticmethod
    def _rank_and_world_size() -> Tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))

    def __len__(self) -> int:
        rank, world_size = self._rank_and_world_size()
        return max(0, (len(self.items) + world_size - 1 - rank) // world_size)

    def __iter__(self) -> Iterator[Dict[str, str]]:
        rank, world_size = self._rank_and_world_size()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        global_worker_id = rank * num_workers + worker_id
        global_worker_count = world_size * num_workers

        for index in range(global_worker_id, len(self.items), global_worker_count):
            prompt, output_path = self.items[index]
            yield {"prompt": prompt, "output_path": output_path}


class _PromptEmbeddingPreprocessor(pl.LightningModule):
    """One frozen UMT5 encoder per rank, writing disjoint cache files."""

    def __init__(
        self,
        algorithm_cfg: Any,
        output_records: List[Dict[str, Any]],
        output_metadata_path: str,
        total_items: int,
    ) -> None:
        super().__init__()
        self.algorithm_cfg = algorithm_cfg
        self.output_records = output_records
        self.output_metadata_path = Path(output_metadata_path)
        self.total_items = int(total_items)
        self.tokenizer: HuggingfaceTokenizer | None = None
        self.text_encoder: torch.nn.Module | None = None
        # DDP requires at least one trainable parameter even for validation-only
        # preprocessing. This scalar is never used or optimized.
        self._ddp_anchor = torch.nn.Parameter(torch.zeros(()))

    def on_validation_start(self) -> None:
        local_items = (
            self.total_items + self.trainer.world_size - 1 - self.global_rank
        ) // self.trainer.world_size
        print(
            "[cache_prompt_embeds] "
            f"rank={self.global_rank}/{self.trainer.world_size} "
            f"device={self.device} items={local_items}",
            flush=True,
        )

    @staticmethod
    def _resolve_path(path: str | None) -> str | None:
        if path is None or path == "null":
            return None
        resolved = Path(path)
        if resolved.is_absolute():
            return str(resolved)
        return str((Path.cwd() / resolved).resolve())

    def configure_model(self) -> None:
        if self.text_encoder is not None:
            return
        text_cfg = self.algorithm_cfg.text_encoder
        self.tokenizer = HuggingfaceTokenizer(
            name=text_cfg.name,
            seq_len=int(text_cfg.text_len),
            clean="whitespace",
        )
        text_dtype = (
            torch.bfloat16
            if str(getattr(text_cfg, "dtype", "bfloat16")).lower() == "bfloat16"
            else torch.float32
        )
        if self.device.type == "cpu" and text_dtype == torch.bfloat16:
            text_dtype = torch.float32
        text_encoder = cast(
            torch.nn.Module,
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=text_dtype,
                device=self.device,
            ),
        )
        text_encoder = text_encoder.eval().requires_grad_(False)
        checkpoint_path = self._resolve_path(
            getattr(text_cfg, "pretrained_ckpt_path", None)
        )
        if checkpoint_path is not None:
            text_encoder.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            )
        text_encoder = text_encoder.to(self.device)
        if bool(getattr(text_cfg, "compile", False)):
            text_encoder = cast(torch.nn.Module, torch.compile(text_encoder))
        self.text_encoder = text_encoder

    @torch.no_grad()
    def validation_step(
        self, batch: Dict[str, List[str]], batch_idx: int
    ) -> None:
        del batch_idx
        if self.tokenizer is None or self.text_encoder is None:
            raise RuntimeError("Prompt text encoder was not configured.")
        prompts = list(batch["prompt"])
        output_paths = [Path(path) for path in batch["output_path"]]
        ids, mask = self.tokenizer(
            prompts, return_mask=True, add_special_tokens=True
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        sequence_lengths = mask.gt(0).sum(dim=1).long().cpu()
        contexts = self.text_encoder(ids, mask).detach().cpu()
        for prompt, output_path, embedding, sequence_length in zip(
            prompts, output_paths, contexts, sequence_lengths
        ):
            _atomic_torch_save(
                embedding[: int(sequence_length)].clone(), output_path
            )
            _atomic_json_save(
                {
                    "prompt_hash": _prompt_hash(prompt),
                    "caption_normalization": (
                        DroidVideoDataset._CAPTION_NORMALIZATION_VERSION
                    ),
                },
                _prompt_embed_meta_path(output_path),
            )

    def on_validation_end(self) -> None:
        if self.trainer.world_size > 1:
            self.trainer.strategy.barrier()
        if self.global_rank != 0:
            return
        self.output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_metadata_path.with_name(
            f".{self.output_metadata_path.name}.{os.getpid()}.tmp"
        )
        pd.DataFrame.from_records(self.output_records).to_csv(
            temporary, index=False
        )
        os.replace(temporary, self.output_metadata_path)
        rank_zero_print(
            cyan(
                "Wrote prompt-embed metadata CSV to "
                f"{self.output_metadata_path}"
            )
        )


class CachePromptEmbedsExperiment(BaseExperiment):
    """
    Cache Wan UMT5 prompt embeddings for text-conditioned datasets.

    For Droid this reads `dataset.metadata_path`, writes one prompt embedding
    tensor per CSV row, and saves a new CSV with `prompt_embed_path` populated.
    The original CSV is left untouched by default.
    """

    compatible_datasets: Dict[str, Any] = {
        "droid": DroidVideoDataset,
    }

    def _dataset_save_dir(self) -> Path:
        return Path(str(self.root_cfg.dataset.save_dir))

    def _metadata_csv_path(self) -> Path:
        configured_path = getattr(
            self.cfg.cache_prompt_embeds, "input_metadata_path", None
        )
        if configured_path is None:
            configured_path = getattr(
                self.root_cfg.dataset, "metadata_path", "cleaned_metadata.csv"
            )
        metadata_path = Path(str(configured_path))
        if metadata_path.is_absolute():
            return metadata_path
        return self._dataset_save_dir() / metadata_path

    def _ensure_metadata_csv_exists(self) -> None:
        metadata_csv_path = self._metadata_csv_path()
        if metadata_csv_path.exists():
            return
        dataset_cfg = deepcopy(self.root_cfg.dataset)
        with open_dict(dataset_cfg):
            dataset_cfg.load_prompt_embed = False
        self.compatible_datasets[self.root_cfg.dataset._name](
            dataset_cfg,
            split="training",
            purpose="training",
        )

    @staticmethod
    def _caption_from_record(record: Dict[str, Any]) -> str:
        for key in ("gemini_caption", "caption", "original_caption"):
            value = record.get(key, "")
            caption = DroidVideoDataset.normalize_caption_value(value)
            if caption:
                return caption
        return ""

    @staticmethod
    def _relative_video_path(video_path: str, save_dir: Path) -> Path:
        path = Path(str(video_path))
        if not path.is_absolute():
            return DroidVideoDataset.normalize_video_path(video_path)
        try:
            return DroidVideoDataset.normalize_video_path(path.relative_to(save_dir))
        except ValueError:
            return Path(path.name)

    def _prompt_embed_path_for_record(
        self, record: Dict[str, Any], save_dir: Path, prompt: str
    ) -> Path:
        prompt_dir = Path(str(self.cfg.cache_prompt_embeds.prompt_embed_dir))
        rel_video_path = self._relative_video_path(str(record["video_path"]), save_dir)
        prompt_hash = self._prompt_hash(prompt)[:12]
        return (
            prompt_dir
            / rel_video_path.with_suffix("").with_name(
                f"{rel_video_path.stem}__prompt_{prompt_hash}.pt"
            )
        )

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        return _prompt_hash(prompt)

    @staticmethod
    def _prompt_embed_meta_path(prompt_embed_path: Path) -> Path:
        return _prompt_embed_meta_path(prompt_embed_path)

    def _prompt_embed_cache_matches(self, prompt_embed_path: Path, prompt: str) -> bool:
        if not prompt_embed_path.exists():
            return False
        meta_path = self._prompt_embed_meta_path(prompt_embed_path)
        if not meta_path.exists():
            return False
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            return False
        return (
            metadata.get("prompt_hash") == self._prompt_hash(prompt)
            and metadata.get("caption_normalization")
            == DroidVideoDataset._CAPTION_NORMALIZATION_VERSION
        )

    def validation(self) -> None:
        cfg = self.cfg.cache_prompt_embeds
        self._ensure_metadata_csv_exists()
        save_dir = self._dataset_save_dir()
        metadata_csv_path = self._metadata_csv_path()
        output_metadata_path = Path(str(cfg.output_metadata_path))
        if not output_metadata_path.is_absolute():
            output_metadata_path = save_dir / output_metadata_path

        records = pd.read_csv(
            metadata_csv_path, keep_default_na=False
        ).to_dict("records")
        max_records = getattr(cfg, "max_records", None)
        if max_records is not None:
            records = records[: int(max_records)]

        id_token = str(getattr(self.root_cfg.dataset, "id_token", "") or "")
        overwrite = bool(getattr(cfg, "overwrite_existing", False))
        batch_size = int(getattr(cfg, "batch_size", 8))

        new_records = [dict(record) for record in records]
        pending: List[Tuple[str, str]] = []
        for record in new_records:
            prompt = id_token + self._caption_from_record(record)
            rel_prompt_path = self._prompt_embed_path_for_record(record, save_dir, prompt)
            abs_prompt_path = save_dir / rel_prompt_path
            record["prompt_embed_path"] = rel_prompt_path.as_posix()
            record["caption"] = prompt[len(id_token) :] if id_token else prompt
            if overwrite or not self._prompt_embed_cache_matches(abs_prompt_path, prompt):
                pending.append((prompt, str(abs_prompt_path)))

        rank_zero_print(
            cyan(
                f"Caching {len(pending)} / {len(new_records)} prompt embeddings "
                f"from {metadata_csv_path}"
            )
        )

        if not pending:
            output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_metadata_path.with_name(
                f".{output_metadata_path.name}.{os.getpid()}.tmp"
            )
            pd.DataFrame.from_records(new_records).to_csv(temporary, index=False)
            os.replace(temporary, output_metadata_path)
            rank_zero_print(
                cyan(f"Wrote prompt-embed metadata CSV to {output_metadata_path}")
            )
            return

        dataloader = DataLoader(
            _PromptWorkDataset(pending),
            batch_size=batch_size,
            num_workers=int(self.cfg.validation.dataloader.num_workers),
            pin_memory=bool(self.cfg.validation.dataloader.pin_memory),
        )
        module = _PromptEmbeddingPreprocessor(
            algorithm_cfg=self.root_cfg.algorithm,
            output_records=new_records,
            output_metadata_path=str(output_metadata_path),
            total_items=len(pending),
        )

        requested_device = str(getattr(cfg, "device", "cuda"))
        use_cuda = requested_device.startswith("cuda") and torch.cuda.is_available()
        devices = self.cfg.validation.devices if use_cuda else 1
        strategy: str | DDPStrategy = str(self.cfg.validation.strategy)
        if strategy == "ddp" and (
            not use_cuda or int(devices) <= 1
        ):
            strategy = "auto"
        trainer = pl.Trainer(
            accelerator="gpu" if use_cuda else "cpu",
            devices=devices,
            num_nodes=int(self.cfg.num_nodes),
            strategy=strategy,
            logger=self.logger,
            precision=self.cfg.validation.precision if use_cuda else "32-true",
            enable_checkpointing=False,
            use_distributed_sampler=False,
            inference_mode=True,
        )
        trainer.validate(module, dataloaders=dataloader)
