"""
CPU-only analysis of SFT example token lengths.

Used to justify dynamic/bucketed padding vs fixed block_size=1024:
prints the length distribution so we can (a) estimate the padding-waste
speedup and (b) pick a safe truncation cap for the fast config.

Run: python data_prep/analyze_sft_lengths.py
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizer.tokenizer_utils import Tokenizer


def lengths_for(path, tok):
    lens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "user" in obj and "assistant" in obj:
                ids, _ = tok.encode_chat_turn(obj["user"], obj["assistant"])
                lens.append(len(ids))
    return lens


def pct(sorted_lens, p):
    if not sorted_lens:
        return 0
    i = min(len(sorted_lens) - 1, int(round(p / 100.0 * (len(sorted_lens) - 1))))
    return sorted_lens[i]


def report(name, lens, block_size=1024):
    lens = sorted(lens)
    n = len(lens)
    total_real = sum(lens)
    # cost of fixed padding: every example padded to block_size
    fixed_tokens = n * block_size
    # cost of per-batch dynamic padding is data-dependent; here we show the
    # theoretical floor (no padding at all) to bound the win.
    print(f"\n=== {name} (n={n}) ===")
    print(f"  min={lens[0]}  p50={pct(lens,50)}  p90={pct(lens,90)}  "
          f"p95={pct(lens,95)}  p99={pct(lens,99)}  max={lens[-1]}")
    print(f"  mean={total_real/n:.1f} tokens/example")
    over = sum(1 for l in lens if l > block_size)
    print(f"  examples exceeding block_size={block_size}: {over} ({100*over/n:.1f}%)")
    print(f"  real tokens: {total_real:,}   fixed-pad tokens: {fixed_tokens:,}")
    print(f"  padding waste at fixed {block_size}: "
          f"{100*(1 - total_real/fixed_tokens):.1f}%  "
          f"(=> up to {fixed_tokens/total_real:.1f}x compute on padding)")
    return lens


def main():
    tok = Tokenizer("tokenizer/tokenizer.model")
    train = lengths_for("data/processed/sft_train.jsonl", tok)
    val = lengths_for("data/processed/sft_val.jsonl", tok)
    report("sft_train", train)
    report("sft_val", val)

    # Suggest a cap that keeps ~99% of examples intact.
    all_lens = sorted(train + val)
    p99 = pct(all_lens, 99)
    # round up to next multiple of 64 for clean shapes
    suggested = ((p99 + 63) // 64) * 64
    print(f"\nSuggested block_size cap (p99 rounded to /64): {suggested}")


if __name__ == "__main__":
    main()
