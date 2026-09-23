"""Checkpointing, logging, seed control, GPU-hour tracking."""

import os
import time
import random
import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slow)")
    return device


class GPUHourTracker:
    """Tracks cumulative GPU-hours used, so progress against a 30-40hr budget is visible."""

    def __init__(self, budget_hours: float = 40.0, state_path: str = "./checkpoints/gpu_hours.txt"):
        self.budget_hours = budget_hours
        self.state_path = state_path
        self.session_start = time.time()
        self.prior_hours = self._load_prior()

    def _load_prior(self) -> float:
        if os.path.exists(self.state_path):
            with open(self.state_path, "r") as f:
                try:
                    return float(f.read().strip())
                except ValueError:
                    return 0.0
        return 0.0

    def elapsed_hours(self) -> float:
        return self.prior_hours + (time.time() - self.session_start) / 3600.0

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                f.write(str(self.elapsed_hours()))
        except (OSError, PermissionError) as e:
            # Read-only filesystem (Lightning AI, etc.) — silently skip
            # GPU hours tracking is not critical to training
            pass

    def status_str(self) -> str:
        used = self.elapsed_hours()
        return f"GPU-hours used: {used:.2f} / {self.budget_hours:.0f} budget ({used/self.budget_hours*100:.1f}%)"


def save_checkpoint(path, model, optimizer, step, val_loss, extra=None):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    ckpt = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    if extra:
        ckpt.update(extra)
    # Replace only after the complete archive has been written.
    tmp_path = f"{path}.tmp"
    try:
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)
    except (OSError, PermissionError, RuntimeError):
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def load_checkpoint(path, model, optimizer=None, device="cuda"):
    # Load checkpoint without requiring safe_globals (newer PyTorch versions don't need this)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    has_opt_state = False
    if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            has_opt_state = True
        except ValueError:
            print("[WARN] Optimizer state dict is empty or invalid. Proceeding without it.")
    return ckpt.get("step", 0), ckpt.get("val_loss", float("inf")), has_opt_state


def cosine_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + np.cos(np.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)
