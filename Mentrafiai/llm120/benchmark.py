"""Measure a synthetic single-device forward/backward workload, not model quality.

No tokenizer, checkpoint, or real data is needed. Defaults describe the full
model, not a fit guarantee for any GPU (including a single T4). Lower batch or
context length if the measured workload runs out of memory. This measures
forward/backward only: optimizer state and checkpoint/evaluation overhead are
NOT included, so training may need more memory and have lower throughput.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .train import (autocast_context, checked_loss_sum, make_scaler, measured_memory,
                    model_config_from_dict, read_yaml, resolve_device, resolve_precision)


def benchmark(*, model_config: str | Path | None = None, steps: int = 10,
              batch_size: int = 1, context_length: int | None = None,
              device: str | None = None, precision: str = "auto",
              warmup_steps: int = 2, gradient_checkpointing: bool = False) -> dict[str, Any]:
    from .model import CausalLM

    for name, value in (("steps", steps), ("batch_size", batch_size), ("warmup_steps", warmup_steps)):
        if type(value) is not int or value < (0 if name == "warmup_steps" else 1):
            raise ValueError(f"{name} must be an integer >= {0 if name == 'warmup_steps' else 1}")
    target = resolve_device(device)
    selected_precision = resolve_precision(precision, target)
    values = read_yaml(model_config) if model_config is not None else {}
    if set(values) == {"model"}:
        values = values["model"]
    config = model_config_from_dict(values)
    context = config.context_length if context_length is None else context_length
    if type(context) is not int or not 1 <= context <= config.context_length:
        raise ValueError("context_length must be positive and <= model context_length")
    torch.manual_seed(42)
    model = CausalLM(config).to(target).train()
    model.gradient_checkpointing = gradient_checkpointing
    inputs = torch.randint(0, config.vocab_size, (batch_size, context), device=target)
    labels = torch.randint(0, config.vocab_size, inputs.shape, device=target)
    scaler = make_scaler(selected_precision)
    overflow_retries = 0

    def iteration() -> None:
        nonlocal overflow_retries
        for retry in range(9):
            model.zero_grad(set_to_none=True)
            with autocast_context(target, selected_precision):
                output = model(inputs, labels=labels)
                loss = checked_loss_sum(output, labels.numel()) / labels.numel()
            scaler.scale(loss).backward()
            # No optimizer state: explicitly unscale and inspect gradients.
            inverse_scale = 1.0 / scaler.get_scale() if scaler.is_enabled() else 1.0
            flags = []
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(inverse_scale)
                    flags.append(torch.isfinite(parameter.grad).all())
            if flags and bool(torch.stack(flags).all()):
                return
            if not scaler.is_enabled() or retry == 8:
                raise FloatingPointError("Nonfinite synthetic gradients after bounded retries; try --precision fp32")
            scaler.update(new_scale=scaler.get_scale() * 0.5)
            overflow_retries += 1

    for _ in range(warmup_steps):
        iteration()
    if target.type == "cuda":
        torch.cuda.synchronize(target)
        torch.cuda.reset_peak_memory_stats(target)
    started = time.perf_counter()
    for _ in range(steps):
        iteration()
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    elapsed = time.perf_counter() - started
    result = {"device": str(target), "precision": selected_precision, "parameters": model.num_parameters(),
              "steps": steps, "warmup_steps": warmup_steps, "batch_size": batch_size,
              "context_length": context, "gradient_checkpointing": gradient_checkpointing,
              "elapsed_seconds": elapsed, "tokens_per_second": steps * batch_size * context / elapsed,
              "overflow_retries_including_warmup": overflow_retries,
              **measured_memory(target),
              "scope": "synthetic forward/backward only; no optimizer state, checkpoint, or real data",
              "fit_guarantee": False}
    print(json.dumps(result, sort_keys=True))
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", type=Path, help="Model-only YAML; otherwise use ModelConfig defaults")
    parser.add_argument("--steps", type=int, default=10, help="Measured forward/backward iterations")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Unmeasured warmup iterations")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, help="Sequence length, at most model context_length")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args(argv)
    try:
        benchmark(**vars(args))
    except (OSError, ValueError, TypeError, FloatingPointError, RuntimeError) as exc:
        parser.error(f"{exc}\nMeasured workloads provide no training fit guarantee; reduce batch/context if out of memory.")


if __name__ == "__main__":
    main()
