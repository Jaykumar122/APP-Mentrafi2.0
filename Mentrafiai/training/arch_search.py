"""
Enumerate candidate architectures and their exact parameter counts.

Runs on the CPU in milliseconds, so the GPU benchmark only has to measure the
handful of configs that are actually the right size — rather than discovering
mid-sweep that a config was 40M params off target.

The formula is verified against the current config, which is known to be
102,455,040 total / 90,167,040 non-embedding.

Usage:
    python training/arch_search.py --target 65 --tol 12
    python training/arch_search.py --explain 640 14 10 2 2.67
"""

import argparse
import itertools
import sys


def hidden_dim(n_embd, mlp_ratio, multiple=64):
    return (int(mlp_ratio * n_embd) + multiple - 1) // multiple * multiple


def validate(n_embd, n_layer, n_head, n_kv_head, mlp_ratio):
    """Reject shapes model/architecture.py cannot build. Returns a reason or None.

    These are the same asserts CausalSelfAttention makes at construction time.
    Checking them here matters because a parameter count for an unbuildable shape
    looks perfectly reasonable on paper: 704/12/11/2 counts out at 62.7M
    non-embedding, sits right on the compute-optimal target, and then dies on
    `assert cfg.n_head % cfg.n_kv_head == 0` after the GPU stage has already been
    scheduled. Failing at the CPU tier costs nothing; failing at the GPU tier
    costs a slot in a multi-hour sweep.
    """
    if n_embd % n_head:
        return f"n_embd {n_embd} not divisible by n_head {n_head}"
    if n_head % n_kv_head:
        return f"n_head {n_head} not divisible by n_kv_head {n_kv_head}"
    if n_embd // n_head not in (32, 64, 128):
        return f"head_dim {n_embd // n_head} is not an efficient size (want 64)"
    return None


def count(n_embd, n_layer, n_head, n_kv_head, mlp_ratio, vocab=16000):
    """Exact counts for the MentraFiAI block: GQA attention + SwiGLU + RMSNorm.

    Weight tying means the LM head contributes nothing beyond the embedding.
    """
    head_dim = n_embd // n_head
    h = hidden_dim(n_embd, mlp_ratio)
    attn = 2 * n_embd * n_embd + 2 * n_embd * (n_kv_head * head_dim)
    mlp = 3 * n_embd * h
    norms = 2 * n_embd
    per_layer = attn + mlp + norms
    non_emb = per_layer * n_layer + n_embd          # + final_norm
    emb = vocab * n_embd
    return {"non_emb": non_emb, "emb": emb, "total": non_emb + emb,
            "per_layer": per_layer, "hidden": h, "head_dim": head_dim,
            "attn_share": attn / per_layer, "mlp_share": mlp / per_layer}


def flops_per_token(n_embd, n_layer, n_head, n_kv_head, mlp_ratio, seq):
    """Forward+backward FLOPs per token, counting attention's seq-dependent term.

    2 FLOPs per MAC, x3 for fwd+bwd. Attention scores and the value-weighted sum
    each cost seq*head_dim per query head, which is why this is not just 6*N.
    """
    head_dim = n_embd // n_head
    h = hidden_dim(n_embd, mlp_ratio)
    proj = 2 * n_embd * n_embd + 2 * n_embd * (n_kv_head * head_dim) + 3 * n_embd * h
    attn = 2 * seq * n_embd                        # QK^T and AV, summed over heads
    return 3 * 2 * n_layer * (proj + attn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=65.0, help="target non-emb params, millions")
    ap.add_argument("--tol", type=float, default=12.0, help="+/- tolerance, millions")
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--explain", nargs=5, default=None,
                    metavar=("N_EMBD", "N_LAYER", "N_HEAD", "N_KV", "MLP_RATIO"))
    args = ap.parse_args()

    if args.explain:
        e, l, hd, kv, mr = (int(args.explain[0]), int(args.explain[1]),
                            int(args.explain[2]), int(args.explain[3]),
                            float(args.explain[4]))
        c = count(e, l, hd, kv, mr, args.vocab)
        f = flops_per_token(e, l, hd, kv, mr, args.seq)
        print(f"n_embd={e} n_layer={l} n_head={hd} n_kv_head={kv} mlp_ratio={mr}")
        print(f"  head_dim      {c['head_dim']}")
        print(f"  mlp_hidden    {c['hidden']}")
        print(f"  per layer     {c['per_layer']:,}  "
              f"(attn {c['attn_share']:.0%} / mlp {c['mlp_share']:.0%})")
        print(f"  non-embedding {c['non_emb']:,}")
        print(f"  embedding     {c['emb']:,}  ({args.vocab} x {e}, tied)")
        print(f"  TOTAL         {c['total']:,}")
        print(f"  fwd+bwd FLOPs/token at seq={args.seq}: {f/1e9:.2f} GFLOP")
        print(f"  depth/width ratio {l/e*1000:.1f} layers per 1000 dim")
        return

    lo = (args.target - args.tol) * 1e6
    hi = (args.target + args.tol) * 1e6
    rows = []
    for n_embd in (448, 512, 576, 640, 704, 768):
        for n_head in range(4, 17):
            if n_embd % n_head:
                continue
            head_dim = n_embd // n_head
            if head_dim not in (32, 48, 64, 80, 96, 128):
                continue
            for n_kv_head in (1, 2, 3, 4, 6, 8, n_head):
                if n_head % n_kv_head or n_kv_head > n_head:
                    continue
                for mlp_ratio in (2.67, 3.0, 3.5, 4.0):
                    for n_layer in range(8, 33):
                        c = count(n_embd, n_layer, n_head, n_kv_head, mlp_ratio, args.vocab)
                        if not (lo <= c["non_emb"] <= hi):
                            continue
                        f = flops_per_token(n_embd, n_layer, n_head, n_kv_head,
                                            mlp_ratio, args.seq)
                        rows.append((n_embd, n_layer, n_head, n_kv_head, mlp_ratio, c, f))

    # De-duplicate near-identical shapes: keep the cheapest per (embd, layer, ratio).
    seen = {}
    for r in rows:
        key = (r[0], r[1], r[4])
        if key not in seen or r[6] < seen[key][6]:
            seen[key] = r
    rows = sorted(seen.values(), key=lambda r: r[6])

    print(f"candidates with non-embedding params in "
          f"[{lo/1e6:.0f}M, {hi/1e6:.0f}M], vocab={args.vocab}, seq={args.seq}")
    print(f"sorted by FLOPs/token ascending (cheapest = fastest, all else equal)\n")
    print(f'{"embd":>5} {"L":>3} {"H":>3} {"KV":>3} {"ratio":>6} {"hidden":>7} '
          f'{"non-emb":>12} {"total":>12} {"GFLOP/tok":>10} {"L/1kd":>6}')
    print("-" * 88)
    for n_embd, n_layer, n_head, n_kv, mr, c, f in rows[:args.top]:
        print(f'{n_embd:5} {n_layer:3} {n_head:3} {n_kv:3} {mr:6.2f} {c["hidden"]:7} '
              f'{c["non_emb"]:12,} {c["total"]:12,} {f/1e9:10.2f} '
              f'{n_layer/n_embd*1000:6.1f}')
    print(f"\n{len(rows)} distinct shapes in range")

    cur = count(768, 14, 12, 4, 2.67, args.vocab)
    curf = flops_per_token(768, 14, 12, 4, 2.67, args.seq)
    print(f"\nfor reference, the CURRENT config (768/14/12/4/2.67):")
    print(f'  non-emb {cur["non_emb"]:,}  total {cur["total"]:,}  '
          f'{curf/1e9:.2f} GFLOP/token')


if __name__ == "__main__":
    main()
