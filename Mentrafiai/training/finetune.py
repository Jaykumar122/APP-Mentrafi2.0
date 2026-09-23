"""
Supervised fine-tuning (SFT) phase: continues training the pretrained model on
chat-formatted instruction data (<user>...<assistant>...), with loss masked
so it's only computed on the assistant's response.

Usage:
    python training/finetune.py \
        --model_config configs/model_config.yaml \
        --train_config configs/train_config.yaml \
        --sft_data data/processed/sft_train.jsonl \
        --sft_val data/processed/sft_val.jsonl \
        --tokenizer tokenizer/tokenizer.model
"""

import argparse
import sys
import pickle
import time
from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader

# Force unbuffered output and UTF-8 console on Windows to fix encoding/mojibake characters
if sys.platform == "win32":
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.system("")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import ModelConfig, MentraFiAI
from tokenizer.tokenizer_utils import Tokenizer
from training.dataset import SFTDataset, sft_collate_fn
from training.utils import (
    load_config, seed_everything, setup_device,
    save_checkpoint, load_checkpoint, cosine_lr,
)


def build_model_config(cfg_dict: dict) -> ModelConfig:
    return ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", default="configs/model_config.yaml")
    ap.add_argument("--train_config", default="configs/finetune_config.yaml")
    ap.add_argument("--sft_data", default=None)
    ap.add_argument("--sft_val", default=None)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer_v2.model")
    args = ap.parse_args()

    full_cfg = load_config(args.train_config)
    model_cfg_dict = load_config(args.model_config)
    train_cfg = full_cfg["finetune"]

    sft_data_path = args.sft_data or train_cfg.get("train_data", "data/processed/sft_train_v3.jsonl")
    sft_val_path = args.sft_val or train_cfg.get("val_data", "data/processed/sft_val_v3.jsonl")
    seed_everything(full_cfg.get("seed", 42))

    device = setup_device()
    cfg = build_model_config(model_cfg_dict)
    model = MentraFiAI(cfg).to(device)

    tok = Tokenizer(args.tokenizer)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best_model.pt"

    step = 0
    best_val_loss = float("inf")
    resumed = False
    if latest_path.exists():
        candidates = [latest_path] + sorted(
            ckpt_dir.glob("checkpoint_step*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            try:
                step, best_val_loss, _ = load_checkpoint(
                    str(candidate), model, optimizer, device
                )
                print(f"Resumed SFT from {candidate.name} at step {step}")
                resumed = True
                break
            except (OSError, RuntimeError, EOFError, KeyError, ValueError,
                    pickle.UnpicklingError) as exc:
                print(f"[WARN] Cannot load checkpoint {candidate.name}: "
                      f"{exc}", flush=True)
        if not resumed:
            print("[WARN] No readable SFT checkpoint found; starting from "
                  "step 0.", flush=True)
    elif Path(train_cfg["init_from"]).exists():
        load_checkpoint(train_cfg["init_from"], model, optimizer=None, device=device)
        print(f"[OK] Initialized from pretrained checkpoint: {train_cfg['init_from']}")
    else:
        print("[WARN] No pretrained checkpoint found — SFT will start from random init (not recommended)")

    # Enable multi-GPU support via DataParallel
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)

    # Dynamic padding: pad each batch to its own longest sequence instead of the
    # full block_size. Numerically identical (pad targets are ignored and, under
    # causal attention, pad tokens never affect real ones) but skips the wasted
    # compute on padding. Toggle off via `dynamic_padding: false` in train_config.
    dynamic_padding = train_cfg.get("dynamic_padding", True)
    collate = sft_collate_fn(tok.pad_id) if dynamic_padding else None

    train_ds = SFTDataset(sft_data_path, tok, cfg.block_size, dynamic_padding=dynamic_padding)
    val_ds = SFTDataset(sft_val_path, tok, cfg.block_size, dynamic_padding=dynamic_padding)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                              drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
                            drop_last=True, collate_fn=collate)

    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg["use_mixed_precision"])
    grad_accum = train_cfg["gradient_accumulation_steps"]
    max_steps = train_cfg["max_steps"]

    def get_batch_iter():
        while True:
            for batch in train_loader:
                yield batch

    batch_iter = get_batch_iter()
    save_every_seconds = 30 * 60       # 30 minutes
    last_save_time = time.time() - save_every_seconds if resumed else time.time()

    model.train()
    while step < max_steps:
        lr = cosine_lr(step, train_cfg["warmup_steps"], max_steps,
                        train_cfg["learning_rate"], train_cfg["min_learning_rate"])
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(grad_accum):
            x, y = next(batch_iter)
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=train_cfg["use_mixed_precision"]):
                _, loss = model(x, y)
                loss = loss.mean() / grad_accum
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        step += 1

        pct = 100.0 * step / max_steps
        print(f"[SFT] step {step:>4}/{max_steps} ({pct:4.1f}%) | loss {accumulated_loss:.4f} | lr {lr:.2e}", flush=True)
        if step % train_cfg["log_interval"] == 0:
            print(f"  [STATUS] step {step} -> checkpoint_dir: {ckpt_dir}", flush=True)

        if step % train_cfg["eval_interval"] == 0:
            val_loss = evaluate(model, val_loader, device, train_cfg["eval_iters"],
                                 train_cfg["use_mixed_precision"])
            print(f"  [SFT eval] step {step} | val_loss {val_loss:.4f}", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(str(best_path), model, optimizer, step, val_loss)
            model.train()

        time_to_save = (time.time() - last_save_time) > save_every_seconds
        if step % train_cfg["save_interval"] == 0 or time_to_save:
            save_checkpoint(str(latest_path), model, optimizer, step, best_val_loss)
            if time_to_save:
                ts_path = ckpt_dir / f"checkpoint_step{step}.pt"
                save_checkpoint(str(ts_path), model, optimizer, step, best_val_loss)
                print(f"  [SAVE] 2-hour checkpoint: {ts_path.name}", flush=True)
            last_save_time = time.time()

    save_checkpoint(str(latest_path), model, optimizer, step, best_val_loss)
    print(f"[DONE] SFT finished at step {step}. Best val_loss: {best_val_loss:.4f}", flush=True)


@torch.no_grad()
def evaluate(model, val_loader, device, eval_iters, use_amp):
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
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)
            loss = loss.mean()
        losses.append(loss.item())
    return sum(losses) / len(losses)


if __name__ == "__main__":
    main()
