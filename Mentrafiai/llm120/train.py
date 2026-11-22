"""Training: python -m llm120.train --help.

One GPU / CPU:   python -m llm120.train --config ... --device cuda
Several GPUs on one machine (e.g. Kaggle 2x T4), data-parallel DDP:
                 torchrun --standalone --nproc_per_node=2 -m llm120.train --config ...

All paths (including paths inside YAML) resolve against the current working
directory, NOT the YAML directory; --data-dir/--tokenizer/--output-dir override
them. Under DDP the effective batch is unchanged (gradient_accumulation_steps
is split across GPUs), so checkpoints can resume on a different GPU count.
An interrupted/failed update is not checkpointed: restart from the last atomic
latest.pt, possibly replaying work since that clean boundary. Exact restart
state is saved, but bit-exact results across different devices/PyTorch kernels
are not promised. Changing max_steps on resume explicitly replans the cosine
schedule, not its step count.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml

from .data import PretrainDataset, SFTDataset, collate_sft, validate_manifest
from .tokenization import IGNORE_INDEX, Tokenizer, file_sha256


@dataclass(frozen=True)
class TrainingConfig:
    stage: str
    data_dir: str
    tokenizer: str
    output_dir: str
    batch_size: int = 2
    gradient_accumulation_steps: int = 32
    max_steps: int = 40000
    warmup_steps: int = 500
    learning_rate: float = 0.0003
    min_lr: float = 0.00003
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    precision: str = "auto"
    gradient_checkpointing: bool = True
    eval_interval: int = 250
    eval_batches: int = 20
    save_interval: int = 250
    seed: int = 42
    max_overflow_retries: int = 8

    def __post_init__(self) -> None:
        if self.stage not in ("pretrain", "sft"):
            raise ValueError("training.stage must be pretrain or sft")
        for name in ("data_dir", "tokenizer", "output_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"training.{name} must be a nonempty path string")
            object.__setattr__(self, name, str(Path(value).resolve()))
        positive = ("batch_size", "gradient_accumulation_steps", "max_steps",
                    "eval_interval", "eval_batches", "save_interval")
        for name in (*positive, "warmup_steps", "seed", "max_overflow_retries"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in positive else 0):
                raise ValueError(f"training.{name} must be an integer >= {1 if name in positive else 0}")
        for name in ("learning_rate", "min_lr", "weight_decay", "grad_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"training.{name} must be finite")
            if value < 0 or (name in ("learning_rate", "grad_clip") and value == 0):
                raise ValueError(f"training.{name} is out of range")
        if self.min_lr > self.learning_rate:
            raise ValueError("training.min_lr must not exceed learning_rate")
        if self.precision not in ("auto", "fp32", "fp16", "bf16"):
            raise ValueError("training.precision must be auto, fp32, fp16, or bf16")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("training.gradient_checkpointing must be a boolean")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TrainingConfig:
        if not isinstance(values, Mapping) or not all(isinstance(key, str) for key in values):
            raise ValueError("training must be a mapping with string keys")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown training keys: {sorted(unknown, key=str)}")
        required = {"stage", "data_dir", "tokenizer", "output_dir"} - set(values)
        if required:
            raise ValueError(f"Missing training keys: {sorted(required)}")
        return cls(**values)


@dataclass(frozen=True)
class RunConfig:
    model: Any
    training: TrainingConfig

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model.to_dict(), "training": asdict(self.training)}


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path}: expected a mapping with string keys")
    return value


def model_config_from_dict(values: Mapping[str, Any]):
    from .config import ModelConfig

    if not isinstance(values, Mapping) or not all(isinstance(key, str) for key in values):
        raise ValueError("model must be a mapping with string keys")
    unknown = set(values) - {field.name for field in fields(ModelConfig)}
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown, key=str)}")
    return ModelConfig.from_dict(values)


def load_config(path: str | Path, *, model_config: str | Path | None = None,
                max_steps: int | None = None, overrides: Mapping[str, Any] | None = None) -> RunConfig:
    raw = read_yaml(path)
    unknown = set(raw) - {"model", "training"}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    if "training" not in raw:
        raise ValueError("Config requires a training mapping with an explicit stage")
    if model_config is not None and "model" in raw:
        raise ValueError("Use either config.model or --model-config, not both")
    model_values = read_yaml(model_config) if model_config is not None else raw.get("model", {})
    # A model-only YAML can be either the field mapping or {model: ...}.
    if model_config is not None and set(model_values) == {"model"}:
        model_values = model_values["model"]
    training_values = dict(raw["training"])
    for key, value in (overrides or {}).items():
        if key not in ("data_dir", "tokenizer", "output_dir"):
            raise ValueError(f"Only data_dir/tokenizer/output_dir can be overridden, not {key}")
        training_values[key] = value
    training = TrainingConfig.from_dict(training_values)
    if max_steps is not None:
        training = TrainingConfig.from_dict({**asdict(training), "max_steps": max_steps})
    return RunConfig(model_config_from_dict(model_values), training)


def reject_distributed() -> None:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ValueError("WORLD_SIZE must be an integer; only single-device training is supported") from exc
    if world_size != 1:
        raise ValueError("Only WORLD_SIZE=1 is supported: use one GPU (e.g. one T4) or CPU; no DDP")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise ValueError("An initialized distributed process group is not supported")


def resolve_device(name: str | None = None) -> torch.device:
    reject_distributed()
    name = name or ("cuda" if torch.cuda.is_available() else "cpu")
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(name)


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def launched_world_size() -> int:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ValueError("WORLD_SIZE must be an integer") from exc
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be >= 1")
    return world_size


def setup_distributed(name: str | None) -> tuple[torch.device, DistributedContext]:
    """Single device, or one process per GPU under torchrun (DDP).

    torchrun sets WORLD_SIZE/RANK/LOCAL_RANK. CUDA uses NCCL; CPU uses gloo
    (for tests only). The process group must not already be initialized.
    """
    world_size = launched_world_size()
    if world_size == 1:
        return resolve_device(name), DistributedContext()
    if not torch.distributed.is_available():
        raise ValueError("WORLD_SIZE > 1 but torch.distributed is unavailable")
    if torch.distributed.is_initialized():
        raise ValueError("An already initialized process group is not supported")
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    except (KeyError, ValueError) as exc:
        raise ValueError("Distributed launch requires integer RANK/LOCAL_RANK; use torchrun") from exc
    name = name or ("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if local_rank >= torch.cuda.device_count():
            raise ValueError(f"LOCAL_RANK {local_rank} has no GPU ({torch.cuda.device_count()} visible)")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    elif name == "cpu":
        device, backend = torch.device("cpu"), "gloo"
    else:
        raise ValueError("device must be cpu or cuda")
    from datetime import timedelta
    # Rank 0 writes ~1.5 GB checkpoints while the others wait at a barrier.
    torch.distributed.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    return device, DistributedContext(rank, world_size, local_rank)


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        torch.distributed.barrier()


def resolve_precision(precision: str, device: torch.device) -> str:
    if precision not in ("auto", "fp32", "fp16", "bf16"):
        raise ValueError("precision must be auto, fp32, fp16, or bf16")
    if precision == "auto":
        return "fp16" if device.type == "cuda" else "fp32"
    if device.type == "cpu" and precision != "fp32":
        raise ValueError("CPU training/benchmark requires fp32 (or auto)")
    if precision == "bf16" and (not torch.cuda.is_bf16_supported()
                                 or torch.cuda.get_device_capability()[0] < 8):
        # Newer torch may report True for software emulation on Turing/T4.
        raise ValueError("bf16 requested but this CUDA device lacks native bf16 support")
    return precision


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_scaler(precision: str):
    # torch.amp.GradScaler(device=...) was introduced after the minimum torch 2.1.
    scaler_class = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if scaler_class is not None:
        return scaler_class("cuda", enabled=precision == "fp16")
    return torch.cuda.amp.GradScaler(enabled=precision == "fp16")


def learning_rate_at(step: int, config: TrainingConfig) -> float:
    """LR for the next successful update; step is the number already completed."""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    # The first post-warmup update is peak LR, the last planned update is min_lr.
    span = max(1, config.max_steps - config.warmup_steps - 1)
    progress = min(1.0, max(0.0, (step - config.warmup_steps) / span))
    return config.min_lr + 0.5 * (config.learning_rate - config.min_lr) * (1 + math.cos(math.pi * progress))


def capture_rng(device: torch.device | None = None) -> dict[str, Any]:
    # Only the selected device: CPU jobs must not initialize CUDA contexts and
    # single-GPU jobs must not allocate contexts on other visible GPUs.
    cuda = [torch.cuda.get_rng_state(device)] if device is not None and device.type == "cuda" else []
    return {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": cuda}


def restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    cuda_states = state["cuda"]
    if not isinstance(cuda_states, list) or len(cuda_states) > 1:
        raise ValueError("Expected a CUDA RNG list for at most one training device")
    if device.type == "cuda" and cuda_states:
        torch.cuda.set_rng_state(cuda_states[0].cpu(), device)


def measured_memory(device: torch.device) -> dict[str, float | None]:
    """Measured process peak RSS and CUDA allocator peaks, not fit estimates."""
    rss = None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in
                ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                 "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            rss = counters.PeakWorkingSetSize / 2**20
    else:
        import resource
        import sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024)
    return {"cpu_process_peak_rss_mib": rss,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None}


def load_datasets(config: RunConfig, tokenizer: Tokenizer):
    training = config.training
    root = Path(training.data_dir).resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = validate_manifest(manifest_path, tokenizer)
    if manifest.get("manifest_version") != 1 or manifest.get("mode") != training.stage:
        raise ValueError("Manifest version/stage mismatch; use llm120.prepare for this stage")
    if training.stage == "sft" and manifest.get("context_length") != config.model.context_length:
        raise ValueError("SFT manifest context_length must match model.context_length; reprepare data")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "val"}:
        raise ValueError("Manifest must contain exactly train and val splits")
    datasets = []
    for name in ("train", "val"):
        info = splits[name]
        suffix = "bin" if training.stage == "pretrain" else "jsonl"
        expected = f"{name}.{suffix}"
        if not isinstance(info, dict) or info.get("file") != expected:
            raise ValueError(f"Manifest {name} file must be {expected}; arbitrary/legacy paths are not supported")
        path = (root / expected).resolve(strict=True)
        if path.parent != root or not path.is_file():
            raise ValueError(f"Manifest {name} path escapes data_dir or is not a file")
        if type(info.get("size_bytes")) is not int or path.stat().st_size != info["size_bytes"]:
            raise ValueError(f"Manifest {name} size_bytes mismatch")
        if file_sha256(path) != info.get("sha256"):
            raise ValueError(f"Manifest {name} sha256 mismatch; data changed since preparation")
        for key in ("documents", "tokens"):
            if type(info.get(key)) is not int or info[key] < 1:
                raise ValueError(f"Manifest {name}.{key} must be a positive integer")
        if training.stage == "pretrain":
            dataset = PretrainDataset(path, config.model.context_length, tokenizer.vocab_size)
            if dataset.token_count != info["tokens"]:
                raise ValueError(f"Manifest {name} token count mismatch")
        else:
            dataset = SFTDataset(path, tokenizer, config.model.context_length)
            if dataset.stats["total"] != info["documents"] or dataset.stats["accepted"] != info["documents"]:
                raise ValueError(f"Prepared {name} SFT rows are invalid/overlong or documents count differs")
        # Sampling is with replacement: even one full window/example is enough.
        if len(dataset) < 1:
            raise ValueError(f"{name} needs at least one complete training example")
        datasets.append(dataset)
    identity = {"manifest_sha256": file_sha256(manifest_path),
                "train_sha256": splits["train"]["sha256"], "val_sha256": splits["val"]["sha256"]}
    return datasets[0], datasets[1], identity


def sample_microbatches(dataset, config: TrainingConfig, generator: torch.Generator,
                        pad_id: int) -> tuple[list[dict[str, torch.Tensor]], int]:
    batches = []
    total = 0
    for _ in range(config.gradient_accumulation_steps):
        indices = torch.randint(len(dataset), (config.batch_size,), generator=generator).tolist()
        batch = collate_sft([dataset[index] for index in indices], pad_id)
        count = int(batch["labels"].ne(IGNORE_INDEX).sum())
        if count == 0:
            raise ValueError("Sampled a microbatch without supervised targets")
        batches.append(batch)
        total += count
    return batches, total


def checked_loss_sum(output, expected_targets: int) -> torch.Tensor:
    if output.num_targets != expected_targets or expected_targets < 1:
        raise ValueError("Model num_targets disagrees with already-shifted data labels")
    if output.loss_sum is None or output.loss is None:
        raise ValueError("Model must return loss and differentiable loss_sum when labels are supplied")
    if not bool(torch.isfinite(output.loss_sum).all()) or not bool(torch.isfinite(output.loss).all()):
        raise FloatingPointError("Nonfinite loss: possible corrupt model/data; no weights will be substituted")
    return output.loss_sum


@torch.no_grad()
def evaluate(model, dataset, config: TrainingConfig, device: torch.device,
             precision: str, pad_id: int, *, rank: int = 0, world_size: int = 1) -> float:
    """Token-weighted validation on a fixed, evenly spaced subset; no RNG use.

    With world_size > 1, eval batches are split round-robin across ranks and
    the sums are all-reduced, so every rank returns the same loss as a single
    device would (up to floating-point summation order).
    """
    size = len(dataset)
    count = min(size, config.eval_batches * config.batch_size)
    if count <= 1:
        indices = [(size - 1) // 2] if count else []
    else:
        indices = [index * (size - 1) // (count - 1) for index in range(count)]
    was_training = model.training
    model.eval()
    loss_sum, targets = 0.0, 0
    try:
        starts = list(range(0, count, config.batch_size))[rank::world_size]
        for start in starts:
            batch = collate_sft([dataset[index] for index in
                                 indices[start:start + config.batch_size]], pad_id)
            batch_targets = int(batch["labels"].ne(IGNORE_INDEX).sum())
            batch = {name: value.to(device) for name, value in batch.items()}
            with autocast_context(device, precision):
                output = model(**batch)
            loss_sum += float(checked_loss_sum(output, batch_targets).double().item())
            targets += batch_targets
    finally:
        model.train(was_training)
    if world_size > 1:
        totals = torch.tensor([loss_sum, float(targets)], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(totals)
        loss_sum, targets = float(totals[0].item()), int(round(totals[1].item()))
    if targets == 0:
        raise ValueError("Validation has no supervised targets")
    result = loss_sum / targets
    if not math.isfinite(result):
        raise FloatingPointError("Nonfinite validation loss")
    return result


def ensure_finite_parameters(model) -> None:
    flags = [torch.isfinite(parameter.detach()).all() for parameter in model.parameters()]
    if flags and not bool(torch.stack(flags).all()):
        raise FloatingPointError("Nonfinite model parameters; refusing to train/save corrupt weights")


def make_optimizer(model, config: TrainingConfig):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW([
        {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": config.weight_decay},
        {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
    ], lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8)


def validate_resume(payload: Mapping[str, Any], config: RunConfig, data_identity: dict[str, str],
                    precision: str) -> dict[str, Any]:
    state = payload.get("training_state")
    required = {"version", "config", "data_identity", "precision", "step", "attempts", "tokens_seen",
                "batches_seen", "sampler_generator_state", "rng", "best_val_loss", "last_val_loss"}
    if not isinstance(state, dict) or not required.issubset(state) or state["version"] != 1:
        raise ValueError("Resume requires a complete llm120 training state, not a weights-only checkpoint")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("stage") != config.training.stage:
        raise ValueError("Resume checkpoint stage mismatch")
    old, new = state["config"], config.to_dict()
    if (not isinstance(old, dict) or set(old) != {"model", "training"}
            or not isinstance(old["training"], dict)):
        raise ValueError("Invalid saved training config")
    # Locations may change across servers; dataset hashes below and tokenizer/model
    # checks in load_datasets/load_checkpoint remain strict content identities.
    allowed = {"output_dir", "data_dir", "tokenizer", "max_steps", "eval_interval", "save_interval"}
    old_training = {key: value for key, value in old["training"].items() if key not in allowed}
    new_training = {key: value for key, value in new["training"].items() if key not in allowed}
    if old["model"] != new["model"] or old_training != new_training:
        raise ValueError("Resume config changed meaningful settings (only output_dir/data_dir/tokenizer/max_steps/eval_interval/save_interval/device may change; input content must match)")
    if state["data_identity"] != data_identity:
        raise ValueError("Resume dataset manifest/hash identity changed")
    # auto can resolve differently on CPU/CUDA, an explicitly supported device change.
    if state["precision"] != precision and config.training.precision != "auto":
        raise ValueError("Resume precision changed")
    for name in ("step", "attempts", "tokens_seen", "batches_seen"):
        if type(state[name]) is not int or state[name] < 0:
            raise ValueError(f"Invalid resume counter {name}")
    if state["attempts"] < state["step"] or state["step"] > config.training.max_steps:
        raise ValueError("Resume step/attempts incompatible with max_steps")
    if state["batches_seen"] != state["step"] * config.training.gradient_accumulation_steps:
        raise ValueError("Resume batches_seen inconsistent with successful steps")
    for name in ("best_val_loss", "last_val_loss"):
        value = state[name]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"Invalid resume {name}")
    if (not torch.is_tensor(state["sampler_generator_state"])
            or not isinstance(state["rng"], dict)
            or set(state["rng"]) != {"python", "torch", "cuda"}):
        raise ValueError("Invalid resume sampler/RNG state")
    if not isinstance(payload.get("optimizer"), dict) or not isinstance(payload.get("scaler"), dict):
        raise ValueError("Resume requires optimizer and scaler states")
    return state




def run_training(config: RunConfig, *, init: str | Path | None = None,
                 resume: str | Path | None = None, device: str | None = None,
                 time_limit_hours: float | None = None) -> dict[str, Any]:
    """Train on one device, or on every GPU of one machine under torchrun.

    Under DDP each update still consumes exactly gradient_accumulation_steps
    microbatches of batch_size: they are drawn from ONE shared sampler (same
    seed and state on every rank) and dealt out round-robin, so the data order,
    effective batch, LR schedule and checkpoints are identical for 1 or N GPUs.
    time_limit_hours stops cleanly after the update that crosses the limit and
    writes latest.pt, so a Kaggle session can end before it is killed.
    """
    if time_limit_hours is not None and (not math.isfinite(time_limit_hours) or time_limit_hours <= 0):
        raise ValueError("time_limit_hours must be positive")
    wall_start = time.perf_counter()
    target, dist = setup_distributed(device)
    try:
        return _run_training(config, init=init, resume=resume, target=target, dist=dist,
                             time_limit_hours=time_limit_hours, wall_start=wall_start)
    finally:
        if dist.enabled and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _run_training(config: RunConfig, *, init, resume, target: torch.device, dist: DistributedContext,
                  time_limit_hours: float | None, wall_start: float) -> dict[str, Any]:
    from .checkpoint import load_checkpoint, save_checkpoint
    from .model import CausalLM

    training = config.training
    if training.gradient_accumulation_steps % dist.world_size:
        raise ValueError(f"gradient_accumulation_steps={training.gradient_accumulation_steps} must be divisible "
                         f"by the number of GPUs ({dist.world_size}) so every GPU gets equal work")
    log = print if dist.is_main else (lambda *args, **kwargs: None)
    precision = resolve_precision(training.precision, target)
    if init is not None and resume is not None:
        raise ValueError("--init and --resume are mutually exclusive")
    if training.stage == "sft" and init is None and resume is None:
        raise ValueError("SFT requires --init PRETRAIN.pt (or --resume an SFT checkpoint)")
    tokenizer = Tokenizer(training.tokenizer)
    tokenizer.validate_config(config.model)
    train_data, val_data, data_identity = load_datasets(config, tokenizer)
    output_dir = Path(training.output_dir)
    if resume is None and any((output_dir / name).exists() for name in ("latest.pt", "best.pt")):
        raise FileExistsError("output_dir already contains trainer checkpoints; use --resume or a new output_dir")
    # Every rank must finish the check above before rank 0 writes latest.pt.
    barrier(dist)
    random.seed(training.seed)
    torch.manual_seed(training.seed)
    sampler = torch.Generator(device="cpu").manual_seed(training.seed)
    payload = None
    if init is not None or resume is not None:
        # Loading on CPU keeps serialized RNG/sampler byte tensors on CPU. The
        # optimizer is constructed after model.to(), so load_state_dict casts its moments.
        model, payload = load_checkpoint(resume if resume is not None else init, tokenizer, device="cpu")
        if model.config.to_dict() != config.model.to_dict():
            raise ValueError("Checkpoint model config must exactly match requested model config")
        metadata = payload.get("metadata")
        if init is not None and (not isinstance(metadata, Mapping) or metadata.get("stage") != "pretrain"):
            raise ValueError("--init requires a checkpoint with metadata.stage=pretrain")
    else:
        model = CausalLM(config.model)
    model.to(target)
    model.gradient_checkpointing = training.gradient_checkpointing
    model.train()
    ensure_finite_parameters(model)
    if dist.enabled:
        from torch.nn.parallel import DistributedDataParallel
        # Rank 0's weights are broadcast at construction, so every rank starts identical.
        train_model = DistributedDataParallel(
            model, device_ids=[target.index] if target.type == "cuda" else None,
            broadcast_buffers=False)
    else:
        train_model = model
    optimizer = make_optimizer(model, training)
    scaler = make_scaler(precision)
    state = {"version": 1, "config": config.to_dict(), "data_identity": data_identity,
             "precision": precision, "step": 0, "attempts": 0, "tokens_seen": 0,
             "batches_seen": 0, "best_val_loss": None, "last_val_loss": None}
    if resume is not None:
        if not isinstance(payload, Mapping):
            raise ValueError("Checkpoint loader did not return a resume payload")
        saved = validate_resume(payload, config, data_identity, precision)
        optimizer.load_state_dict(payload["optimizer"])
        for slot in optimizer.state.values():
            for value in slot.values():
                if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise ValueError("Nonfinite saved optimizer state")
        if precision == saved["precision"]:
            scaler.load_state_dict(payload["scaler"])
        else:
            log("Device/auto precision changed: GradScaler is reset for the destination; cross-hardware bit-exactness is not promised.")
        sampler.set_state(saved["sampler_generator_state"].cpu())
        restore_rng(saved["rng"], target)
        for key in ("step", "attempts", "tokens_seen", "batches_seen", "best_val_loss", "last_val_loss"):
            state[key] = saved[key]
    if not math.isfinite(scaler.get_scale()) or scaler.get_scale() <= 0:
        raise ValueError("Invalid GradScaler scale; refusing to save a corrupt restart state")
    # Do not retain a second CPU copy of checkpoint model/optimizer tensors.
    del payload
    if dist.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / "latest.pt"

    def save(path: Path) -> None:
        # Weights, optimizer and sampler are identical on every rank; rank 0 writes.
        if dist.is_main:
            ensure_finite_parameters(model)
            snapshot = {**state, "sampler_generator_state": sampler.get_state(), "rng": capture_rng(target)}
            save_checkpoint(path, model, tokenizer, optimizer=optimizer, scaler=scaler,
                            training_state=snapshot,
                            metadata={"stage": training.stage, "val_loss": state["last_val_loss"],
                                      "best_val_loss": state["best_val_loss"], "precision": precision,
                                      "world_size": dist.world_size})
        barrier(dist)

    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    log(json.dumps({"event": "start", "device": str(target), "precision": precision,
                    "stage": training.stage, "parameters": model.num_parameters(),
                    "world_size": dist.world_size,
                    "microbatches_per_gpu_per_update": training.gradient_accumulation_steps // dist.world_size,
                    "step": state["step"], "max_steps": training.max_steps,
                    "train_rows": len(train_data), "val_rows": len(val_data),
                    "time_limit_hours": time_limit_hours,
                    "sampling": "random_with_replacement", "tokens_seen_unit": "supervised_targets"}))
    # A safe initial snapshot exists even if the very first update is interrupted.
    save(latest)
    stop_reason = "max_steps"
    session_start_step = state["step"]
    loop_start = time.perf_counter()
    try:
        while state["step"] < training.max_steps:
            started = time.perf_counter()
            batches, total_targets = sample_microbatches(train_data, training, sampler, tokenizer.pad_id)
            local_batches = batches[dist.rank::dist.world_size]
            lr = learning_rate_at(state["step"], training)
            for group in optimizer.param_groups:
                group["lr"] = lr
            retries = 0
            while True:
                state["attempts"] += 1
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                for index, batch in enumerate(local_batches):
                    count = int(batch["labels"].ne(IGNORE_INDEX).sum())
                    on_device = {name: value.to(target, non_blocking=True) for name, value in batch.items()}
                    # Gradients are all-reduced once per update, on the last local microbatch.
                    sync = nullcontext() if (not dist.enabled or index == len(local_batches) - 1) \
                        else train_model.no_sync()
                    with sync:
                        with autocast_context(target, precision):
                            output = train_model(**on_device)
                            loss_sum = checked_loss_sum(output, count)
                            # Crucially NOT loss / number_of_microbatches: SFT masks vary.
                            # DDP averages gradients over ranks, so multiply by world_size
                            # to keep the token-weighted global mean.
                            objective = loss_sum * (dist.world_size / total_targets)
                        accumulated_loss += float(loss_sum.detach().double().item())
                        scaler.scale(objective).backward()
                scaler.unscale_(optimizer)
                # Gradients are identical on every rank after all-reduce, so every
                # rank takes the same overflow / step decision below.
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip)
                if not bool(torch.isfinite(norm)):
                    optimizer.zero_grad(set_to_none=True)
                    if precision != "fp16":
                        raise FloatingPointError("Nonfinite gradients in non-FP16 training")
                    # No optimizer.step, LR advancement, new sample, or successful
                    # counters on overflow. update(new_scale) also clears scaler flags.
                    scale = scaler.get_scale() * 0.5
                    if not math.isfinite(scale) or scale <= 0:
                        raise FloatingPointError("FP16 GradScaler scale exhausted")
                    scaler.update(new_scale=scale)
                    retries += 1
                    log(json.dumps({"event": "amp_overflow", "step": state["step"],
                                    "attempts": state["attempts"], "retry": retries, "scale": scale}))
                    if retries > training.max_overflow_retries:
                        raise FloatingPointError("FP16 overflow retry limit exceeded; last safe latest.pt is unchanged")
                    continue
                scaler.step(optimizer)
                scaler.update()
                ensure_finite_parameters(model)
                optimizer.zero_grad(set_to_none=True)
                break
            state["step"] += 1
            state["tokens_seen"] += total_targets
            state["batches_seen"] += len(batches)
            out_of_time = time_limit_hours is not None and \
                (time.perf_counter() - wall_start) / 3600 >= time_limit_hours
            if dist.enabled:
                # One small all-reduce: global loss for logging, and rank 0's
                # time decision so that all ranks stop at the same step.
                shared = torch.tensor([accumulated_loss, float(out_of_time and dist.is_main)],
                                      dtype=torch.float64, device=target)
                torch.distributed.all_reduce(shared)
                accumulated_loss, out_of_time = float(shared[0].item()), bool(shared[1].item() > 0)
            elapsed = time.perf_counter() - started
            done_this_session = state["step"] - session_start_step
            seconds_per_step = (time.perf_counter() - loop_start) / done_this_session
            log(json.dumps({"event": "update", "step": state["step"], "attempts": state["attempts"],
                            "loss": accumulated_loss / total_targets, "lr": lr,
                            "tokens_seen": state["tokens_seen"], "batches_seen": state["batches_seen"],
                            "supervised_tokens_per_second": total_targets / max(elapsed, 1e-9),
                            "eta_hours": round((training.max_steps - state["step"]) * seconds_per_step / 3600, 2),
                            **measured_memory(target)}))
            at_end = state["step"] == training.max_steps
            if state["step"] % training.eval_interval == 0 or at_end:
                val_loss = evaluate(model, val_data, training, target, precision, tokenizer.pad_id,
                                    rank=dist.rank, world_size=dist.world_size)
                state["last_val_loss"] = val_loss
                log(json.dumps({"event": "evaluation", "step": state["step"], "val_loss": val_loss,
                                "val_perplexity": math.exp(min(val_loss, 50))}))
                if state["best_val_loss"] is None or val_loss < state["best_val_loss"]:
                    state["best_val_loss"] = val_loss
                    save(output_dir / "best.pt")
            if state["step"] % training.save_interval == 0 or at_end:
                save(latest)
            elif out_of_time:
                save(latest)
            if out_of_time and not at_end:
                stop_reason = "time_limit"
                log(json.dumps({"event": "time_limit", "step": state["step"], "max_steps": training.max_steps,
                                "resume_from": str(latest)}))
                break
    except KeyboardInterrupt:
        # An interrupt may arrive inside optimizer.step or CUDA execution; even
        # apparently clean in-memory counters cannot prove that update is atomic.
        log(f"Interrupted: not saving in-flight state. Resume the last clean atomic snapshot: {latest}")
        raise
    if stop_reason == "max_steps":
        save(latest)
    log(json.dumps({"event": "finished", "reason": stop_reason, "step": state["step"],
                    "best_val_loss": state["best_val_loss"], "output_dir": str(output_dir)}))
    return {**state, "stop_reason": stop_reason,
            "sampler_generator_state": sampler.get_state(), "rng": capture_rng(target)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="YAML with model and training; paths resolve against cwd")
    parser.add_argument("--model-config", type=Path, help="Model-only YAML; alternative to nested config.model")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init", type=Path, help="Pretraining weights only; reset optimizer/scaler/RNG/counters")
    source.add_argument("--resume", type=Path, help="Restore complete optimizer/scaler/RNG/sampler/config/counters")
    parser.add_argument("--max-steps", type=int, help="Override total successful steps (also replans cosine decay)")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Auto chooses CUDA when available; under torchrun each process uses its LOCAL_RANK GPU")
    parser.add_argument("--data-dir", help="Override training.data_dir (e.g. a read-only /kaggle/input path)")
    parser.add_argument("--tokenizer", help="Override training.tokenizer")
    parser.add_argument("--output-dir", help="Override training.output_dir (e.g. /kaggle/working/...)")
    parser.add_argument("--time-limit-hours", type=float,
                        help="Stop cleanly and save latest.pt after this much wall time (Kaggle sessions end at 12 h)")
    args = parser.parse_args(argv)
    try:
        overrides = {key: value for key, value in (("data_dir", args.data_dir), ("tokenizer", args.tokenizer),
                                                   ("output_dir", args.output_dir)) if value is not None}
        config = load_config(args.config, model_config=args.model_config, max_steps=args.max_steps,
                             overrides=overrides)
        run_training(config, init=args.init, resume=args.resume, device=args.device,
                     time_limit_hours=args.time_limit_hours)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError, TypeError, FloatingPointError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()