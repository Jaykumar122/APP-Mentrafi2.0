"""
Benchmark candidate architectures head to head and solve for the right size.

The question this answers: at 102.5M params the measured throughput gives about
0.95B tokens in a 6.5-day budget, which is ~10 tokens per non-embedding
parameter — roughly half of the ~20 that Chinchilla-style scaling says is
compute-optimal. A smaller model would see proportionally more tokens AND run
faster, so there is a fixed point somewhere below 102.5M. Theory cannot locate it
on this GPU because throughput does not scale cleanly with parameter count at
this size (small models underutilize the SMs, and memory pressure is a cliff
rather than a curve). So: measure several, then solve.

Each config is benchmarked by training/benchmark.py in isolated subprocesses and
cached to benchmarks/arch_<label>.json, so the sweep is resumable.

Usage:
    python training/arch_bench.py
    python training/arch_bench.py --only 640_16 --force
    python training/arch_bench.py --report            # aggregate cached results only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.arch_search import count, flops_per_token, validate

OUT_DIR = Path("benchmarks")
DAYS = 6.5           # 7-day wall clock minus slack for validation and checkpoints
CHINCHILLA = 20.0    # target tokens per non-embedding parameter

# (label, n_embd, n_layer, n_head, n_kv_head, mlp_ratio)
# Chosen to separate the variables rather than to cover a grid:
#   768_14  the incumbent, for a like-for-like reference point
#   768_12  same width, less depth  -> isolates depth
#   640_16  narrower and deeper at similar params -> isolates aspect ratio
#   576_20  narrower still, deeper still
#   640_12  same shape as 640_16, ~25% fewer params -> isolates size alone
#   640_12f same size params as 640_16 but spent on MLP width, not layers
#   512_12  clearly below the target, to bracket the fixed point from underneath
CONFIGS = [
    ("768_14", 768, 14, 12, 4, 2.67),
    ("768_12", 768, 12, 12, 4, 2.67),
    ("640_16", 640, 16, 10, 2, 2.67),
    ("576_20", 576, 20, 9, 3, 2.67),
    ("640_12", 640, 12, 10, 2, 2.67),
    ("640_12f", 640, 12, 10, 2, 3.50),
    ("512_12", 512, 12, 8, 2, 2.67),

    # --- round 2: SHAPE at the size round 1 solved for --------------------
    # Round 1 answered "how big" (N_opt = 62.3M non-emb, D_opt = 1.25B tokens)
    # and incidentally showed depth is expensive here: 576_20 managed only
    # 1,529 tok/s against 640_16's 2,007 at similar parameter count. Layers are
    # sequential, so depth cannot be hidden behind the GPU's parallelism, while
    # width lands in larger GEMMs that this card runs proportionally better.
    #
    # These four sit at 60-67M non-embedding and 0.47-0.49 GFLOP/token -- near
    # enough to identical compute that throughput differences between them are
    # attributable to SHAPE alone. Aspect ratio spans 8.9 to 21.9 layers per
    # 1000 dim, which brackets the plausible range in both directions.
    #
    # Throughput is not the only criterion. Very shallow models lose quality at
    # fixed parameter count, so this measures the price of depth in tokens and
    # the final pick trades it against the known cost in representational depth
    # rather than simply taking the fastest row.
    ("896_8", 896, 8, 14, 2, 2.67),     # very wide, very shallow   (8.9 L/kdim)
    ("768_10", 768, 10, 12, 4, 2.67),   # wide, shallow            (13.0 L/kdim)
    ("704_12", 704, 12, 11, 2, 2.67),   # balanced                 (17.0 L/kdim)
    ("640_14", 640, 14, 10, 2, 2.67),   # narrow, deep             (21.9 L/kdim)

    # --- round 3: KV heads, the last free variable ------------------------
    # Round 2 left two things unresolved. First a confound of my own making:
    # 768_10 ran with n_kv_head=4 while every other round-2 config used 2, so
    # its 2,188 tok/s is not comparable to the rest. 768_10_kv2 fixes that.
    #
    # Second, n_kv_head was never varied deliberately. It is cheap in parameters
    # but it trades throughput against attention quality, and 10 query heads
    # over 2 KV heads is a 5:1 ratio -- more aggressive than Llama-3's 4:1 at
    # vastly larger scale. n_head=10 admits only 1, 2, 5 or 10, so these three
    # cover MQA, the current setting, and 2:1.
    #
    # Note kv changes the parameter count: kv1 62.4M, kv2 63.4M, kv5 66.4M.
    # kv10 (full MHA) would reach 71.3M, past the measured optimum, so it is
    # excluded rather than benchmarked.
    ("640_12f_kv1", 640, 12, 10, 1, 3.50),
    ("640_12f_kv5", 640, 12, 10, 5, 3.50),
    ("768_10_kv2", 768, 10, 12, 2, 2.67),
]


def override_str(n_embd, n_layer, n_head, n_kv_head, mlp_ratio):
    return (f"n_embd={n_embd},n_layer={n_layer},n_head={n_head},"
            f"n_kv_head={n_kv_head},mlp_ratio={mlp_ratio}")


def run_one(label, spec, args):
    bad = validate(*spec)
    if bad:
        print(f"[invalid] {label}: {bad} — skipping (not benchmarkable)")
        return None
    out = OUT_DIR / f"arch_{label}.json"
    if out.exists() and not args.force:
        print(f"[cached] {label}")
        return out
    cmd = [sys.executable, "-u", "training/benchmark.py",
           "--label", label,
           "--override", override_str(*spec),
           "--dtypes", "fp32",
           "--modes", args.modes,
           "--micro_bs", args.micro_bs,
           "--gqa", "--dropout", "0.0",
           "--iters", str(args.iters), "--warmup", str(args.warmup),
           "--mem_cap", str(args.mem_cap),
           "--out", str(out)]
    print(f"\n[bench] {label}  {override_str(*spec)}", flush=True)
    p = subprocess.run(cmd, text=True)
    if p.returncode != 0:
        print(f"  benchmark.py exited {p.returncode} for {label}")
    return out if out.exists() else None


def report(args):
    rows = []
    for label, spec in [(c[0], c[1:]) for c in CONFIGS]:
        f = OUT_DIR / f"arch_{label}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        ok = [r for r in d["results"] if r.get("status") == "ok"]
        if not ok:
            rows.append((label, spec, d, None))
            continue
        best = max(ok, key=lambda r: r["tok_s"])
        rows.append((label, spec, d, best))

    if not rows:
        print("no cached results yet"); return

    print("\n" + "=" * 118)
    print(f"ARCHITECTURE COMPARISON   budget = {DAYS} days   "
          f"Chinchilla target = {CHINCHILLA:.0f} tokens/non-emb-param")
    print("=" * 118)
    print(f'{"label":9} {"non-emb":>11} {"total":>11} {"GF/tok":>7} {"best cell":>16} '
          f'{"tok/s":>8} {"peak":>7} {"tokens":>9} {"tok/par":>8} {"vs opt":>7}')
    print("-" * 118)

    scored = []
    for label, spec, d, best in rows:
        c = count(spec[0], spec[1], spec[2], spec[3], spec[4], d["config"]["vocab_size"])
        gf = flops_per_token(spec[0], spec[1], spec[2], spec[3], spec[4], d["seq"]) / 1e9
        if best is None:
            print(f'{label:9} {c["non_emb"]:11,} {c["total"]:11,} {gf:7.2f} '
                  f'{"no cell fit":>16}')
            continue
        toks = best["tok_s"] * DAYS * 86400
        ratio = toks / c["non_emb"]
        cell = f'{best["mode"]}/mbs{best["micro_bs"]}'
        print(f'{label:9} {c["non_emb"]:11,} {c["total"]:11,} {gf:7.2f} {cell:>16} '
              f'{best["tok_s"]:8,.0f} {best["peak_mib"]:7,.0f} {toks/1e9:8.2f}B '
              f'{ratio:8.1f} {ratio/CHINCHILLA:6.2f}x')
        scored.append({"label": label, "spec": spec, "non_emb": c["non_emb"],
                       "total": c["total"], "gflop_tok": gf, "tok_s": best["tok_s"],
                       "peak_mib": best["peak_mib"], "mode": best["mode"],
                       "micro_bs": best["micro_bs"], "tokens": toks, "ratio": ratio})

    if not scored:
        return
    print("-" * 118)

    # The compute budget K = N * D is roughly conserved across configs, because a
    # model half the size runs about twice as fast. Measure K per config, then the
    # Chinchilla-optimal size for that budget is N = sqrt(K / 20).
    print("\nSolving for the compute-optimal size from the MEASURED budget:")
    print(f'{"label":9} {"N (non-emb)":>13} {"D (tokens)":>12} {"K = N*D":>11} '
          f'{"N_opt=sqrt(K/20)":>18} {"D_opt":>10}')
    print("-" * 80)
    ks = []
    for s in scored:
        k = s["non_emb"] * s["tokens"]
        n_opt = (k / CHINCHILLA) ** 0.5
        ks.append((s["label"], k, n_opt))
        print(f'{s["label"]:9} {s["non_emb"]:13,} {s["tokens"]/1e9:11.2f}B '
              f'{k:11.2e} {n_opt/1e6:17.1f}M {CHINCHILLA*n_opt/1e9:9.2f}B')

    med_k = sorted(k for _, k, _ in ks)[len(ks) // 2]
    n_opt = (med_k / CHINCHILLA) ** 0.5
    print("-" * 80)
    print(f"median K = {med_k:.2e}  ->  N_opt = {n_opt/1e6:.1f}M non-embedding, "
          f"D_opt = {CHINCHILLA*n_opt/1e9:.2f}B tokens")

    closest = min(scored, key=lambda s: abs(s["non_emb"] - n_opt))
    print(f"\nclosest benchmarked config: {closest['label']} "
          f"({closest['non_emb']/1e6:.1f}M non-emb, {closest['ratio']:.1f} tok/param, "
          f"{closest['tok_s']:,.0f} tok/s, {closest['peak_mib']:,.0f} MiB peak)")

    print("\nNote: K is only approximately conserved. If K rises as models shrink, "
          "\nsmall models are running MORE efficiently than 1/N scaling predicts; "
          "\nif it falls, they are underutilizing the GPU and the optimum sits higher.")

    (OUT_DIR / "arch_summary.json").write_text(json.dumps(
        {"days": DAYS, "chinchilla": CHINCHILLA, "configs": scored,
         "median_K": med_k, "n_opt": n_opt}, indent=2))
    print(f"\nwrote {OUT_DIR / 'arch_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of labels")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true", help="skip benching, aggregate only")
    ap.add_argument("--modes", default="chunk,ckpt+chunk")
    ap.add_argument("--micro_bs", default="1,2,4,6,8")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mem_cap", type=float, default=2600.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.report:
        only = {s for s in args.only.split(",") if s}
        for c in CONFIGS:
            if only and c[0] not in only:
                continue
            run_one(c[0], c[1:], args)
    report(args)


if __name__ == "__main__":
    main()
