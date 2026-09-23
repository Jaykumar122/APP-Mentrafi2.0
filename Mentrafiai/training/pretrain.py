"""
Pretraining loop: next-token prediction over the mixed general+domain corpus.

Auto-detects your GPU and applies the right config automatically:
  GTX 1650 (4 GB)  → configs/pretrain_config.yaml      (fp32, batch=1, grad_accum=128)
  T4      (16 GB)  → configs/pretrain_config_t4.yaml   (bf16, batch=8, grad_accum=16)
  Any other GPU    → falls back to GTX 1650 config (safe default)

You can also force a GPU profile manually:
  --gpu gtx1650   use the GTX 1650 config
  --gpu t4        use the T4 config

Usage — GTX 1650 (auto or explicit):
    python training/pretrain.py
    python training/pretrain.py --gpu gtx1650

Usage — T4 on Colab/Kaggle (auto or explicit):
    python training/pretrain.py --gpu t4

Both GPUs share the same checkpoint dir so you can switch mid-run.
The script auto-resumes from the latest checkpoint on every restart.
"""

import argparse
import contextlib
import os
import pickle
import re
import sys
import threading
import time
from pathlib import Path

# Ensure project root is on sys.path so `model`, `training`, `tokenizer` are importable
# regardless of which directory the script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from model import ModelConfig, MentraFiAI
from training.dataset import PretrainDataset
from training.utils import (
    load_config, seed_everything, setup_device, GPUHourTracker,
    save_checkpoint, load_checkpoint, cosine_lr,
)


# ---------------------------------------------------------------------------
# GPU profile detection
# ---------------------------------------------------------------------------
_GPU_CONFIGS = {
    "gtx1650": "configs/pretrain_config.yaml",
    "t4":      "configs/pretrain_config_t4.yaml",
}

# Keywords in torch.cuda.get_device_name() that identify each profile.
# Checked in order — first match wins.
_GPU_NAME_MAP = [
    ("1650",          "gtx1650"),
    ("1660",          "gtx1650"),  # similar VRAM/no tensor cores
    ("1070",          "gtx1650"),
    ("1080",          "gtx1650"),
    ("T4",            "t4"),
    ("A100",          "t4"),       # A100 has tensor cores — use bf16 config
    ("A10",           "t4"),
    ("V100",          "t4"),
    ("3080",          "t4"),
    ("3090",          "t4"),
    ("4090",          "t4"),
    ("4080",          "t4"),
    ("4070",          "t4"),
    ("3070",          "t4"),
    ("3060",          "t4"),
]


def detect_gpu_profile() -> str:
    """Return 'gtx1650' or 't4' based on the detected GPU name."""
    if not torch.cuda.is_available():
        return "gtx1650"  # CPU fallback — use safe low-memory config
    name = torch.cuda.get_device_name(0)
    for keyword, profile in _GPU_NAME_MAP:
        if keyword.lower() in name.lower():
            return profile
    # Unknown GPU — default to gtx1650 (conservative, always works)
    print(f"[WARN] Unknown GPU '{name}' — using GTX 1650 config (safe default). "
          "Pass --gpu t4 if your card has >8 GB VRAM and tensor cores.")
    return "gtx1650"


def build_model_config(cfg_dict: dict) -> ModelConfig:
    return ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})


def _copy_tensors_to_cpu(value):
    """Detach and clone nested optimizer state so async saves cannot race training."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy_tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_tensors_to_cpu(item) for item in value)
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", default="configs/model_config.yaml")
    ap.add_argument("--train_config", default=None,
                    help="Override train config path. Auto-detected from GPU if not set.")
    ap.add_argument("--gpu", default=None, choices=["gtx1650", "t4"],
                    help="Force GPU profile (gtx1650 or t4). Auto-detected if not set.")
    ap.add_argument("--train_bin", default="data/processed/corpus_v2/train.bin")
    ap.add_argument("--val_bin",   default="data/processed/corpus_v2/val.bin")
    ap.add_argument("--budget_hours", type=float, default=200.0)
    ap.add_argument("--checkpoint_dir", default=None,
                    help="Override checkpoint directory. Useful for cloud platforms.")
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp = local_rank != -1

    if is_ddp:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_main_process = (rank == 0)
    else:
        device = setup_device()
        rank = 0
        world_size = 1
        is_main_process = True

    def print_rank0(*p_args, **p_kwargs):
        if is_main_process:
            print(*p_args, **p_kwargs)

    # --- resolve which config to use ---
    if args.train_config:
        train_config_path = args.train_config
        profile = args.gpu or "custom"
    else:
        profile = args.gpu or detect_gpu_profile()
        train_config_path = _GPU_CONFIGS.get(profile, _GPU_CONFIGS["gtx1650"])

    print_rank0(f"GPU profile : {profile}")
    print_rank0(f"Train config: {train_config_path}")
    if is_ddp:
        print_rank0(f"Distributed : DDP enabled across {world_size} GPUs (local_rank={local_rank})")

    model_cfg_dict = load_config(args.model_config)
    full_train_cfg = load_config(train_config_path)
    train_cfg = full_train_cfg["pretrain"]
    seed_everything(full_train_cfg.get("seed", 42) + rank)

    use_amp = train_cfg["use_mixed_precision"]
    if use_amp:
        dev_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else ""
        major_cap = torch.cuda.get_device_capability(device)[0] if torch.cuda.is_available() else 0
        if "T4" in dev_name.upper() or "1650" in dev_name or "1660" in dev_name or "2080" in dev_name or (0 < major_cap < 8):
            amp_dtype = torch.float16
        else:
            amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        print_rank0(f"Mixed precision: {amp_dtype}")
    else:
        amp_dtype = torch.float32
    cfg = build_model_config(model_cfg_dict)
    raw_model = MentraFiAI(cfg).to(device)
    if train_cfg.get("gradient_checkpointing"):
        raw_model.gradient_checkpointing = True
        print_rank0("Gradient checkpointing: ON")
    print_rank0(f"Model parameters: {raw_model.num_parameters():,} "
                f"({raw_model.num_parameters(non_embedding=True):,} non-embedding)")

    nw = train_cfg.get("num_workers",
                        full_train_cfg.get("hardware", {}).get("num_workers", 0))
    train_ds = PretrainDataset(args.train_bin, cfg.block_size)
    val_ds   = PretrainDataset(args.val_bin,   cfg.block_size)
    pin_mem = full_train_cfg.get("hardware", {}).get("pin_memory", False)

    if is_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                                  sampler=train_sampler, num_workers=nw, drop_last=True, pin_memory=pin_mem)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                                  num_workers=nw, drop_last=True, pin_memory=pin_mem)

    val_loader   = DataLoader(val_ds,   batch_size=train_cfg["batch_size"], shuffle=False,
                               num_workers=nw, drop_last=True, pin_memory=pin_mem)
    print_rank0(f"Train examples: {len(train_ds):,} | Val examples: {len(val_ds):,}")

    if "target_effective_batch" in train_cfg:
        target_tokens = train_cfg["target_effective_batch"]
        tokens_per_micro = train_cfg["batch_size"] * cfg.block_size * world_size
        grad_accum = max(1, round(target_tokens / tokens_per_micro))
    else:
        grad_accum = train_cfg["gradient_accumulation_steps"]

    eff_batch = train_cfg["batch_size"] * grad_accum * cfg.block_size * world_size
    print_rank0(f"Effective batch: {eff_batch:,} tokens | grad_accum: {grad_accum} | GPUs: {world_size} | num_workers: {nw}")

    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Handle checkpoint directory — allow override for cloud platforms (Lightning AI, Colab, etc.)
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
    else:
        ckpt_dir = Path(train_cfg["checkpoint_dir"])
        if Path("/teamspace/uploads").exists() and not ckpt_dir.parent.exists():
            ckpt_dir = Path("/teamspace/outputs/checkpoints/pretrain_v2")
            print_rank0(f"[INFO] Detected Lightning AI — redirecting checkpoints to {ckpt_dir}")
        elif Path("/kaggle/input").exists() and not ckpt_dir.parent.exists():
            ckpt_dir = Path("/kaggle/working/checkpoints/pretrain_v2")
            print_rank0(f"[INFO] Detected Kaggle — redirecting checkpoints to {ckpt_dir}")
        elif Path("/content").exists() and not ckpt_dir.parent.exists():
            ckpt_dir = Path("/content/checkpoints/pretrain_v2")
            print_rank0(f"[INFO] Detected Colab — redirecting checkpoints to {ckpt_dir}")
    
    # Ensure directory is writable (only on main process)
    if is_main_process:
        try:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            test_file = ckpt_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
        except (OSError, PermissionError) as e:
            print_rank0(f"[ERROR] Cannot write to {ckpt_dir}: {e}")
            print_rank0(f"[INFO] Using fallback directory: ./checkpoints_fallback/")
            ckpt_dir = Path("./checkpoints_fallback/pretrain_v2")
            ckpt_dir.mkdir(parents=True, exist_ok=True)

    if is_ddp:
        dist.barrier()

    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best_model.pt"

    step = 0
    best_val_loss = float("inf")
    resumed = False

    all_ckpts = list(ckpt_dir.glob("*.pt"))
    if all_ckpts:
        # Inspect candidates and sort by (step_number, mtime) descending
        valid_candidates = []
        for ckpt_path in all_ckpts:
            if ckpt_path.name == "best_model.pt":
                continue
            try:
                # Quickly inspect step without loading full model into memory if possible, or read step dict
                meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                ckpt_step = meta.get("step", 0)
                valid_candidates.append((ckpt_step, ckpt_path.stat().st_mtime, ckpt_path))
            except Exception as e:
                print_rank0(f"[WARN] Could not inspect checkpoint {ckpt_path.name}: {e}")

        # Sort highest step first
        valid_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        for ckpt_step, mtime, candidate in valid_candidates:
            try:
                step, best_val_loss, has_opt_state = load_checkpoint(
                    str(candidate), raw_model, optimizer, device
                )
                print_rank0(f"Resumed from {candidate.name} at step {step}, "
                            f"best_val_loss={best_val_loss:.4f}")
                resumed = True
                break
            except (OSError, RuntimeError, EOFError, KeyError, ValueError,
                    pickle.UnpicklingError) as exc:
                print_rank0(f"[WARN] Cannot load checkpoint {candidate.name}: "
                            f"{exc}", flush=True)

        if not resumed:
            print_rank0("[WARN] No readable checkpoint found; starting pretraining "
                        "from step 0.", flush=True)

    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(raw_model, device_ids=[local_rank])
    else:
        model = raw_model

    gpu_tracker = GPUHourTracker(budget_hours=args.budget_hours,
                                  state_path=str(ckpt_dir / "gpu_hours.txt"))

    def get_batch_iter():
        epoch = 0
        while True:
            if is_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                yield batch
            epoch += 1

    batch_iter = get_batch_iter()
    max_steps = train_cfg["max_steps"]

    save_every_seconds = 30 * 60
    if resumed:
        last_save_time = time.time() - save_every_seconds
        print_rank0(f"[INFO] Forcing checkpoint save at next save_interval to preserve resumed state")
    else:
        last_save_time = time.time()

    heartbeat_file = "training_heartbeat.txt"

    # Background checkpoint saver — writes to disk in a thread so the
    # training loop never stalls waiting for a 400 MB file to flush.
    _save_lock = threading.Lock()
    _save_thread: threading.Thread | None = None

    def async_save(path: str, step_n: int, loss: float):
        """Copy model/optimizer state to CPU and save in a background thread."""
        nonlocal _save_thread
        # Wait for any previous save to finish before starting a new one
        if _save_thread and _save_thread.is_alive():
            _save_thread.join()
        # Snapshot state dicts on CPU so training can keep using GPU immediately
        with _save_lock:
            cpu_model = {
                k: v.detach().cpu().clone()
                for k, v in raw_model.state_dict().items()
            }
            cpu_opt = _copy_tensors_to_cpu(optimizer.state_dict())
            snap_step = step_n
            snap_loss = loss
        def _write():
            import torch as _torch
            ckpt = {
                "step": snap_step,
                "model_state_dict": cpu_model,
                "optimizer_state_dict": cpu_opt,
                "val_loss": snap_loss,
            }
            tmp = path + ".tmp"
            try:
                _torch.save(ckpt, tmp)
                os.replace(tmp, path)  # atomic rename
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print_rank0(f"  [CKPT SAVED] Successfully wrote {Path(path).name} ({size_mb:.1f} MB)", flush=True)
            except (OSError, PermissionError, RuntimeError) as e:
                print_rank0(f"\n[CRITICAL ERROR] Failed to save checkpoint to {path}: {e}\n", flush=True)
                print_rank0(f"[INFO] Training continues (checkpoint save not fatal)", flush=True)
                # Cleanup temp file if it exists
                if Path(tmp).exists():
                    try:
                        os.unlink(tmp)
                    except:
                        pass
        _save_thread = threading.Thread(target=_write, daemon=True)
        _save_thread.start()

    model.train()
    # t0 and pause_total track *training* time, excluding time spent saving
    t0 = time.time()
    pause_total = 0.0   # seconds spent in checkpoint saves (excluded from tok/s)
    tokens_seen = 0
    step_t0 = time.time()
    consecutive_errors = 0   # track repeated CUDA failures
    MAX_CONSECUTIVE_ERRORS = 5

    # Periodic cache-clear every N steps to prevent VRAM fragmentation
    # GTX 1650 (4GB) needs more aggressive clearing than larger GPUs
    CACHE_CLEAR_INTERVAL = 50

    while step < max_steps:
        lr = cosine_lr(step, train_cfg["warmup_steps"], max_steps,
                        train_cfg["learning_rate"], train_cfg["min_learning_rate"])
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ── Forward + backward with full CUDA error recovery ──────────────────
        step_ok = False
        try:
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for micro_step in range(grad_accum):
                x, y = next(batch_iter)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype if use_amp else torch.float32):
                    _, loss = model(x, y)
                    loss = loss / grad_accum

                # Preemptive NaN check BEFORE backward — catch explosions early
                if not torch.isfinite(loss):
                    print_rank0(f"  [SKIP] step {step+1} micro_step {micro_step} — loss became NaN/Inf during forward, aborting batch", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    accumulated_loss = float('nan')  # mark this step as bad
                    break

                is_accumulating = (micro_step < grad_accum - 1)
                sync_ctx = model.no_sync() if (is_ddp and is_accumulating) else contextlib.nullcontext()
                with sync_ctx:
                    scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                tokens_seen += x.numel() * world_size

            # Skip step if loss is NaN/Inf (bad batch) — don't update weights
            if not (accumulated_loss == accumulated_loss) or accumulated_loss > 1e6:
                print_rank0(f"  [SKIP] step {step+1} — NaN/Inf loss ({accumulated_loss:.4f}), skipping weight update", flush=True)
                # Full GPU state reset after NaN — clear corrupted memory
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Reset scaler — internal state may be corrupted
                scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
                consecutive_errors += 1
            else:
                scaler.unscale_(optimizer)
                # Check for NaN in gradients before clipping
                has_nan_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in raw_model.parameters()
                )
                if has_nan_grad:
                    print_rank0(f"  [SKIP] step {step+1} — NaN gradient detected, skipping", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    consecutive_errors += 1
                else:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train_cfg["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    consecutive_errors = 0  # reset on clean step
                    step_ok = True

        except (RuntimeError, torch.cuda.OutOfMemoryError) as cuda_err:
            err_str = str(cuda_err)
            is_oom      = "out of memory" in err_str.lower() or isinstance(cuda_err, torch.cuda.OutOfMemoryError)
            is_illegal  = "illegal memory access" in err_str.lower() or "cudaErrorIllegalAddress" in err_str or "CUBLAS_STATUS" in err_str

            print(f"\n  [CUDA ERROR] step {step+1}: {cuda_err.__class__.__name__}: {err_str[:120]}", flush=True)

            # ----------------------------------------------------------------
            # cudaErrorIllegalAddress / CUBLAS corruption = NON-RECOVERABLE.
            # The GPU context is dead. Any further CUDA call (zero_grad,
            # empty_cache, synchronize, model.cpu()) will also throw.
            # Strategy: grab tensor data via .data.to('cpu') which bypasses
            # most driver calls, save weights-only checkpoint, and exit.
            # ----------------------------------------------------------------
            if is_illegal:
                print(f"  [FATAL] Illegal memory access = non-recoverable CUDA context corruption.", flush=True)
                print(f"  [FATAL] Attempting emergency save BEFORE any further CUDA calls...", flush=True)

                # Wait for any in-flight async save to finish first
                try:
                    if _save_thread and _save_thread.is_alive():
                        _save_thread.join(timeout=30)
                except Exception:
                    pass

                emergency_path = ckpt_dir / f"checkpoint_step{step}_emergency.pt"
                emergency_saved = False

                # Bypass .cpu() (which makes a CUDA call) — pull each tensor
                # individually so partial failures don't abort the whole save.
                try:
                    cpu_state = {}
                    for k, v in model.state_dict().items():
                        try:
                            cpu_state[k] = v.data.to("cpu")
                        except Exception:
                            # If even this fails, store zeros so the file is loadable
                            cpu_state[k] = torch.zeros(v.shape, dtype=v.dtype, device="cpu")
                    ckpt = {
                        "step": step,
                        "model_state_dict": cpu_state,
                        "optimizer_state_dict": {},   # optimizer moments lost — acceptable
                        "val_loss": best_val_loss,
                    }
                    torch.save(ckpt, str(emergency_path))
                    print(f"  [FATAL] Emergency checkpoint saved (weights only, no optimizer state): {emergency_path}", flush=True)
                    print(f"  [FATAL] NOTE: optimizer state not saved — LR warmup will re-run for ~{train_cfg['warmup_steps']} steps after resume", flush=True)
                    emergency_saved = True
                except Exception as e1:
                    print(f"  [FATAL] Emergency save failed: {e1}", flush=True)
                    print(f"  [FATAL] Resume from last good timed checkpoint: checkpoint_step{step}.pt", flush=True)

                print(f"\n  [ACTION] Restart training — it will auto-resume from the emergency checkpoint.", flush=True)
                print(f"  [ACTION] If this error recurs on the same step, set CUDA_LAUNCH_BLOCKING=1 to get a proper traceback.", flush=True)
                sys.exit(1)

            # ----------------------------------------------------------------
            # OOM and other (potentially) recoverable errors — retry logic
            # ----------------------------------------------------------------
            # Safely clear CUDA state — wrap everything individually because
            # empty_cache / synchronize themselves can throw when GPU is hot.
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

            # Reset GradScaler — its internal state is invalid after a CUDA error
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            except Exception:
                pass

            consecutive_errors += 1

            # Skip the corrupted batch by consuming it from the iterator.
            # Without this we retry the exact same batch forever.
            print(f"  [RECOVER] Skipping batch, will retry next batch ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS} errors)", flush=True)
            try:
                for _ in range(grad_accum):
                    _ = next(batch_iter)
            except Exception:
                print(f"  [FATAL] Cannot advance data iterator - dataset may be corrupted", flush=True)
                consecutive_errors = MAX_CONSECUTIVE_ERRORS  # force exit

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"\n[FATAL] {consecutive_errors} consecutive CUDA errors. Saving emergency checkpoint and exiting.", flush=True)
                if _save_thread and _save_thread.is_alive():
                    _save_thread.join()

                emergency_saved = False
                emergency_path = ckpt_dir / f"checkpoint_step{step}_emergency.pt"

                # Strategy 1: Move model to CPU first (avoids CUDA access during save)
                try:
                    model_cpu = model.cpu()
                    opt_cpu = _copy_tensors_to_cpu(optimizer.state_dict())
                    ckpt = {
                        "step": step,
                        "model_state_dict": model_cpu.state_dict(),
                        "optimizer_state_dict": opt_cpu,
                        "val_loss": best_val_loss,
                    }
                    torch.save(ckpt, str(emergency_path))
                    print(f"[FATAL] Emergency checkpoint saved: {emergency_path}", flush=True)
                    emergency_saved = True
                except Exception as e1:
                    print(f"[FATAL] CPU-based save failed: {e1}", flush=True)

                # Strategy 2: Fallback to regular save
                if not emergency_saved:
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        save_checkpoint(str(emergency_path), model, optimizer, step, best_val_loss)
                        print(f"[FATAL] Emergency checkpoint saved (fallback): {emergency_path}", flush=True)
                        emergency_saved = True
                    except Exception as e2:
                        print(f"[FATAL] Fallback save failed: {e2}", flush=True)

                if not emergency_saved:
                    print(f"[FATAL] All save attempts failed. Resume from step {step-1} checkpoint.", flush=True)
                    print(f"[FATAL] Last known good checkpoint: checkpoint_step{step-1}.pt", flush=True)
                else:
                    print(f"[INFO] To resume, restart training - will auto-load the emergency checkpoint", flush=True)

                sys.exit(1)

            # Don't increment step counter - retry with the next batch
            step_t0 = time.time()  # reset timer so skip doesn't skew tok/s
            continue  # skip all logging for this step, go to next

        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            print(f"\n[FATAL] {consecutive_errors} consecutive errors (NaNs). Saving emergency checkpoint and exiting.", flush=True)
            if _save_thread and _save_thread.is_alive():
                _save_thread.join()
            emergency_path = ckpt_dir / f"checkpoint_step{step}_emergency.pt"
            try:
                model_cpu = model.cpu()
                opt_cpu = _copy_tensors_to_cpu(optimizer.state_dict())
                ckpt = {
                    "step": step,
                    "model_state_dict": model_cpu.state_dict(),
                    "optimizer_state_dict": opt_cpu,
                    "val_loss": best_val_loss,
                }
                torch.save(ckpt, str(emergency_path))
                print(f"[FATAL] Emergency checkpoint saved: {emergency_path}", flush=True)
            except Exception as e1:
                print(f"[FATAL] CPU-based save failed: {e1}", flush=True)
            sys.exit(1)
        
        step += 1

        # ── Periodic VRAM cache clear (prevents fragmentation over long runs) ──
        if step % CACHE_CLEAR_INTERVAL == 0:
            torch.cuda.empty_cache()

        # Print compact loss line every step — visible in terminal live
        now = time.time()
        step_time = now - step_t0
        step_t0 = now
        training_elapsed = max(now - t0 - pause_total, 1e-6)  # exclude save time
        toks_per_sec = tokens_seen / training_elapsed
        pct = 100.0 * step / max_steps
        step_flag = "" if step_ok else " [SKIP]"
        print_rank0(f"step {step:>5}/{max_steps} ({pct:4.1f}%) | loss {accumulated_loss:.4f} | lr {lr:.2e} | {toks_per_sec:,.0f} tok/s | step {step_time:.1f}s{step_flag}", flush=True)

        if is_main_process:
            if step % train_cfg["log_interval"] == 0:
                gpu_str = gpu_tracker.status_str()
                # Also show GPU memory usage
                if torch.cuda.is_available():
                    mem_used = torch.cuda.memory_allocated() / 1024**3
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_str += f" | VRAM {mem_used:.1f}/{mem_total:.0f}GB"
                print(f"  [STATUS] step {step} | {gpu_str}", flush=True)
                # Update heartbeat file so you can check progress without tailing log
                try:
                    with open(heartbeat_file, "w") as hb:
                        import datetime
                        hb.write(f"step {step}/{max_steps} ({pct:.1f}%)\n")
                        hb.write(f"loss {accumulated_loss:.4f} | lr {lr:.2e}\n")
                        hb.write(f"tok/s {toks_per_sec:,.0f}\n")
                        hb.write(f"{gpu_str}\n")
                        hb.write(f"updated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                except Exception:
                    pass

            if step % train_cfg["eval_interval"] == 0:
                val_loss = evaluate(raw_model, val_loader, device, train_cfg["eval_iters"],
                                     use_amp, amp_dtype)
                print(f"  [eval] step {step} | val_loss {val_loss:.4f}")
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    save_checkpoint(str(best_path), raw_model, optimizer, step, val_loss)
                    print(f"  [BEST] New best model saved! val_loss: {val_loss:.4f}")
                raw_model.train()

            time_to_save = (time.time() - last_save_time) > save_every_seconds
            if step % train_cfg["save_interval"] == 0 or time_to_save:
                save_t0 = time.time()
                # latest.pt is always kept so resume is always safe
                async_save(str(latest_path), step, best_val_loss)
                # Timestamped checkpoint every save_interval OR every 2 hours (whichever comes first)
                ts_path = ckpt_dir / f"checkpoint_step{step}.pt"
                async_save(str(ts_path), step, best_val_loss)
                print(f"  [SAVE] Checkpoint queued: {ts_path.name}", flush=True)
                # Prune old timestamped checkpoints (wait for thread so files exist)
                if _save_thread:
                    _save_thread.join(timeout=30)
                keep_n = train_cfg.get("keep_last_n_checkpoints", 2)
                def _extract_step(p):
                    m = re.search(r'checkpoint_step(\d+)\.pt', p.name)
                    return int(m.group(1)) if m else 0
                ts_ckpts = sorted(ckpt_dir.glob("checkpoint_step*.pt"), key=_extract_step)
                for old in ts_ckpts[:-keep_n]:
                    old.unlink(missing_ok=True)
                gpu_tracker.save()
                save_elapsed = time.time() - save_t0
                pause_total += save_elapsed          # exclude save time from tok/s
                last_save_time = time.time()         # reset so next 2-hr window is clean
                step_t0 = time.time()               # don't penalise next step_time either

        if is_ddp and (step % train_cfg["eval_interval"] == 0 or step % train_cfg["save_interval"] == 0):
            dist.barrier()

        if gpu_tracker.elapsed_hours() > gpu_tracker.budget_hours:
            print_rank0(f"[WARN] GPU-hour budget ({gpu_tracker.budget_hours}h) reached - stopping pretraining.")
            break

    # Final save: wait for any background thread, then do a blocking write
    if is_main_process:
        if _save_thread and _save_thread.is_alive():
            _save_thread.join()
        save_checkpoint(str(latest_path), raw_model, optimizer, step, best_val_loss)
        gpu_tracker.save()
        print(f"[DONE] Pretraining finished at step {step}. Best val_loss: {best_val_loss:.4f}")
        print(gpu_tracker.status_str())

    if is_ddp:
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(model, val_loader, device, eval_iters, use_amp, amp_dtype=torch.float32):
    model.eval()
    losses = []
    it = iter(val_loader)
    for _ in range(eval_iters):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(val_loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype if use_amp else torch.float32):
            _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


if __name__ == "__main__":
    main()
