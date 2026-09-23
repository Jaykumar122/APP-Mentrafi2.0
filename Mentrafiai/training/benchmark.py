"""
Measure real training throughput on THIS GPU for a given model config.

Everything downstream (corpus size, step budget, the 1-week plan) is derived from
numbers this script produces, so it deliberately measures the full training step
(forward + backward + optimizer + grad clip), not just a forward pass.

Each cell runs in its OWN SUBPROCESS. That is not defensive over-engineering: on
this card, pushing past ~3 GB of allocated memory does not raise a clean
OutOfMemoryError. It either spills to host RAM through WDDM (measured 5x
slowdown) or kills the CUDA context with an illegal memory access, which poisons
the process and takes every remaining cell down with it. Isolation means one bad
cell costs one cell.

Usage:
    python training/benchmark.py                              # default sweep
    python training/benchmark.py --override n_layer=12,n_embd=640 --label 62M
    python training/benchmark.py --seq 512 --modes ckpt+chunk

Reported per (dtype, mode, micro_batch) cell:
    step_s     median wall time of one end-to-end training step
    tok/s      micro_bs * seq / step_s
    peak MiB   torch.cuda.max_memory_allocated during the cell
    status     ok | SPILL | OOM | CRASH | TIMEOUT

Modes:
    plain       stock forward, unchunked loss
    chunk       chunked LM head + loss
    ckpt        gradient checkpointing
    ckpt+chunk  both
"""

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model import ModelConfig, MentraFiAI
from model.architecture import CausalSelfAttention, apply_rope
from training.utils import load_config

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
MODES = {"plain": (False, False), "chunk": (False, True),
         "ckpt": (True, False), "ckpt+chunk": (True, True)}

# Peak allocated above this on a 4096 MiB card means WDDM is paging to host RAM.
# It does not fail loudly; it just runs ~5x slower, so it has to be treated as a
# hard ceiling rather than a warning.
DEFAULT_MEM_CAP = 3000
_MARK = "@@CELL@@"


def patch_sdpa_gqa():
    """Let SDPA handle the GQA head expansion instead of materializing it.

    The stock forward calls k.repeat_interleave(n_rep, dim=1), which allocates a
    full n_head-wide K and V (3x the real KV size at n_head=12/n_kv_head=4) and
    writes it to memory before attention even starts. SDPA's enable_gqa=True
    broadcasts internally, so the copy disappears. Patched here rather than in
    architecture.py so the benchmark can measure the delta before it is adopted.
    """
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True, enable_gqa=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(out))

    CausalSelfAttention._orig_forward = CausalSelfAttention.forward
    CausalSelfAttention.forward = forward


# ---------------------------------------------------------------------------
# Wrappers applied non-destructively so the benchmark can probe efficiency
# techniques before they are committed to model/architecture.py.
# ---------------------------------------------------------------------------
class _CkptBlock(nn.Module):
    """Recompute this block's activations during backward instead of storing them.

    Trades ~30% extra compute for a large activation-memory saving, which on a
    4GB card usually buys a bigger micro-batch — whether that is a net win is
    exactly what this benchmark decides.
    """

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, cos, sin):
        return checkpoint(self.block, x, cos, sin, use_reentrant=False)


def enable_grad_checkpointing(model):
    model.blocks = nn.ModuleList([_CkptBlock(b) for b in model.blocks])
    return model


def chunked_loss(model, idx, targets, chunk=256):
    """Run the LM head + cross-entropy in slices over the time axis.

    The unchunked path materializes logits of shape (B, T, vocab) and then a
    float32 copy inside cross_entropy: at B=8/T=1024/vocab=16000 that is ~0.8GB
    of transient memory on a 4GB card. Slicing over T keeps the peak to
    chunk/T of that, at no cost in the computed value (sum of per-slice
    token-weighted losses == full loss).
    """
    B, T = idx.shape
    x = model.dropout(model.tok_embed(idx))
    cos, sin = model.rope_cos.to(x.device), model.rope_sin.to(x.device)
    for block in model.blocks:
        x = block(x, cos, sin)
    x = model.final_norm(x)

    total = x.new_zeros(())
    n_valid = 0
    for i in range(0, T, chunk):
        xs = x[:, i:i + chunk, :]
        ts = targets[:, i:i + chunk]
        logits = model.head(xs)
        valid = int((ts != model.cfg.pad_token_id).sum())
        if valid == 0:
            continue
        ls = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ts.reshape(-1),
            ignore_index=model.cfg.pad_token_id, reduction="sum",
        )
        total = total + ls
        n_valid += valid
    return total / max(n_valid, 1)


# ---------------------------------------------------------------------------


def _isfloat(v):
    try:
        float(v); return True
    except ValueError:
        return False


def build_cfg(base_path, overrides):
    d = load_config(base_path)
    for kv in overrides:
        if not kv:
            continue
        k, v = kv.split("=", 1)
        d[k] = int(v) if v.isdigit() else (float(v) if _isfloat(v) else v)
    return ModelConfig(**{k: v for k, v in d.items() if k in ModelConfig.__dataclass_fields__})


def bench_cell(cfg, device, micro_bs, seq, iters, warmup, use_ckpt, use_chunk,
               dtype="fp32", compile_model=False, fused_adam=True):
    """Time `iters` full training steps. Returns dict or {'oom': True}."""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); gc.collect()
    amp_dtype = DTYPES[dtype]
    use_amp = dtype != "fp32"
    model = opt = None
    try:
        model = MentraFiAI(cfg).to(device)
        if use_ckpt:
            enable_grad_checkpointing(model)
        model.train()
        if compile_model:
            model = torch.compile(model)
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01,
                                    betas=(0.9, 0.95), fused=fused_adam)
        except (RuntimeError, TypeError):
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01,
                                    betas=(0.9, 0.95))
            fused_adam = False
        # GradScaler is only meaningful for fp16; bf16 and fp32 need no loss scaling.
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "fp16"))

        x = torch.randint(4, cfg.vocab_size, (micro_bs, seq), device=device)
        y = torch.randint(4, cfg.vocab_size, (micro_bs, seq), device=device)

        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        times = []
        t_start = None
        for i in range(warmup + iters):
            if i == warmup:
                torch.cuda.synchronize(); t_start = time.perf_counter()
            step_t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                if use_chunk:
                    loss = chunked_loss(inner, x, y)
                else:
                    _, loss = model(x, y)
            scaler.scale(loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.perf_counter() - step_t0)

        wall = time.perf_counter() - t_start
        peak = torch.cuda.max_memory_allocated() / 2**20
        med = statistics.median(times)
        return {
            "micro_bs": micro_bs, "seq": seq, "ckpt": use_ckpt, "chunk": use_chunk,
            "dtype": dtype, "compiled": compile_model, "fused": fused_adam,
            "step_s": med,
            "tok_s": micro_bs * seq / med,
            "tok_s_wall": micro_bs * seq * iters / wall,
            "peak_mib": peak,
            "loss_finite": bool(torch.isfinite(loss).item()),
        }
    except torch.cuda.OutOfMemoryError:
        return {"micro_bs": micro_bs, "seq": seq, "ckpt": use_ckpt, "dtype": dtype,
                "chunk": use_chunk, "status": "OOM"}


# --------------------------------------------------------------------------- #
# Child process: run exactly one cell and hand the result back over stdout.
# --------------------------------------------------------------------------- #
def run_child(args):
    torch.backends.cudnn.benchmark = True
    if args.gqa:
        patch_sdpa_gqa()
    dtype, mode, mbs = args.cell.split(":")
    cfg = build_cfg(args.model_config, args.override.split(","))
    if args.dropout is not None:
        cfg.dropout = args.dropout
    seq = args.seq or cfg.block_size
    use_ckpt, use_chunk = MODES[mode]
    r = bench_cell(cfg, torch.device("cuda"), int(mbs), seq, args.iters, args.warmup,
                   use_ckpt, use_chunk, dtype=dtype, compile_model=args.compile)
    r["mode"] = mode
    sys.stdout.write(_MARK + json.dumps(r) + "\n")
    sys.stdout.flush()


def spawn_cell(args, dtype, mode, mbs, timeout):
    """Run one cell in a fresh process. A dead CUDA context costs one cell."""
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
           "--cell", f"{dtype}:{mode}:{mbs}",
           "--model_config", args.model_config, "--override", args.override,
           "--iters", str(args.iters), "--warmup", str(args.warmup)]
    if args.seq:
        cmd += ["--seq", str(args.seq)]
    if args.dropout is not None:
        cmd += ["--dropout", str(args.dropout)]
    if args.gqa:
        cmd += ["--gqa"]
    if args.compile:
        cmd += ["--compile"]

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    base = {"micro_bs": mbs, "seq": args.seq, "mode": mode, "dtype": dtype}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=env, cwd=str(Path(__file__).resolve().parent.parent))
    except subprocess.TimeoutExpired:
        return {**base, "status": "TIMEOUT"}
    for line in (p.stdout or "").splitlines():
        if line.startswith(_MARK):
            r = json.loads(line[len(_MARK):])
            r.setdefault("status", "ok")
            return r
    tail = ((p.stderr or "").strip().splitlines() or [""])[-1][:160]
    return {**base, "status": "CRASH", "error": tail, "returncode": p.returncode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", default="configs/model_config.yaml")
    ap.add_argument("--override", default="", help="comma-separated k=v model-config overrides")
    ap.add_argument("--seq", type=int, default=None, help="default: cfg.block_size")
    ap.add_argument("--micro_bs", default="1,2,4,8", help="comma-separated sweep")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--modes", default="plain,chunk,ckpt,ckpt+chunk")
    ap.add_argument("--dtypes", default="fp32", help="comma list of fp32,fp16,bf16")
    ap.add_argument("--gqa", action="store_true", help="use SDPA enable_gqa (no KV copy)")
    ap.add_argument("--dropout", type=float, default=None, help="override cfg.dropout")
    ap.add_argument("--compile", action="store_true", help="wrap in torch.compile")
    ap.add_argument("--mem_cap", type=float, default=DEFAULT_MEM_CAP,
                    help="peak MiB above which a cell is treated as host-spilling")
    ap.add_argument("--cell_timeout", type=float, default=900.0)
    ap.add_argument("--out", default=None, help="write results JSON here")
    ap.add_argument("--label", default="current")
    ap.add_argument("--cell", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable — this benchmark is meaningless on CPU."); sys.exit(1)

    if args.cell:
        run_child(args)
        return

    cfg = build_cfg(args.model_config, args.override.split(","))
    if args.dropout is not None:
        cfg.dropout = args.dropout
    seq = args.seq or cfg.block_size

    probe = MentraFiAI(cfg)
    n_all = probe.num_parameters()
    n_emb = probe.num_parameters(non_embedding=True)
    del probe

    name = torch.cuda.get_device_name(0)
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    print("=" * 100)
    print(f"BENCHMARK  label={args.label}  gpu={name}  vram={total_mib:,.0f} MiB  "
          f"torch={torch.__version__}")
    print(f"  config: n_layer={cfg.n_layer} n_embd={cfg.n_embd} n_head={cfg.n_head} "
          f"n_kv_head={cfg.n_kv_head} mlp_hidden={cfg.hidden_dim} vocab={cfg.vocab_size} "
          f"seq={seq} dropout={cfg.dropout}")
    print(f"  params: {n_all:,} total | {n_emb:,} non-embedding"
          f"   | sdpa_gqa={args.gqa} compile={args.compile} mem_cap={args.mem_cap:.0f} MiB")
    print("=" * 100)
    print(f'{"mode":12} {"dtype":6} {"mbs":>4} {"step_s":>8} {"tok/s":>9} {"peak MiB":>9}  status')
    print("-" * 100)

    results = []
    for dtype in [d.strip() for d in args.dtypes.split(",") if d.strip()]:
        for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
            for mbs in [int(b) for b in args.micro_bs.split(",")]:
                r = spawn_cell(args, dtype, mode, mbs, args.cell_timeout)
                if r.get("status") == "ok" and r["peak_mib"] > args.mem_cap:
                    r["status"] = "SPILL"
                results.append(r)
                if r.get("status") == "ok":
                    print(f'{mode:12} {dtype:6} {mbs:4} {r["step_s"]:8.3f} '
                          f'{r["tok_s"]:9,.0f} {r["peak_mib"]:9,.0f}  ok', flush=True)
                else:
                    peak = f'{r["peak_mib"]:9,.0f}' if "peak_mib" in r else f'{"":>9}'
                    extra = f'  {r.get("error","")}' if r.get("error") else ""
                    print(f'{mode:12} {dtype:6} {mbs:4} {"":>8} {"":>9} {peak}  '
                          f'{r["status"]}{extra}', flush=True)
                    # Memory pressure is monotonic in micro_bs: once a cell spills,
                    # crashes or OOMs, every larger batch in this mode will too.
                    print(f'{"":12} {"":6} {"":>4} skipping larger micro_bs '
                          f'for {dtype}/{mode}', flush=True)
                    break

    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        best = max(ok, key=lambda r: r["tok_s"])
        print("-" * 100)
        print(f'BEST (peak <= {args.mem_cap:.0f} MiB): mode={best["mode"]} '
              f'dtype={best["dtype"]} micro_bs={best["micro_bs"]} -> '
              f'{best["tok_s"]:,.0f} tok/s ({best["peak_mib"]:,.0f} MiB peak, '
              f'{args.mem_cap - best["peak_mib"]:,.0f} MiB headroom)')
        for days, label in ((7, "7 days"), (6.5, "6.5 days (eval/ckpt slack)")):
            tot = best["tok_s"] * days * 86400
            print(f'  {label:26} -> {tot/1e9:.3f}B tokens  '
                  f'({tot/max(n_emb,1):.1f} tokens/non-emb-param)')
    else:
        print("-" * 100)
        print("no cell completed within the memory cap")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"label": args.label, "gpu": name, "vram_mib": total_mib,
             "params_total": n_all, "params_non_embedding": n_emb,
             "seq": seq, "sdpa_gqa": args.gqa, "compile": args.compile,
             "mem_cap": args.mem_cap, "config": cfg.__dict__,
             "results": results}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
