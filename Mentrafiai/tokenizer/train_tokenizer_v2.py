"""
Train the v2 SentencePiece BPE tokenizer on the CLEANED v2 corpus.

Why this is separate from train_tokenizer.py rather than a flag on it:

  1. train_tokenizer.py globs *.jsonl and hands the raw file to SentencePiece.
     For the v1 .txt corpus that was fine. For v2 every line is a JSON object,
     so SentencePiece would learn merges for `{"text":`, `","source":"` and
     `"domain":"finance"}` -- thousands of vocabulary slots spent on framing
     that never appears at training time. This script extracts the text field.

  2. SentencePiece samples input sentences uniformly, so a source's influence on
     the vocabulary is proportional to its BYTES. fineweb_edu and cosmopedia are
     3.3 GiB of the 5.2 GiB corpus, so an unweighted sample would let general web
     prose decide almost the entire vocabulary and leave finance terminology to
     be spelled out in pieces. Since every finance token costs extra positions in
     a 1024-token context forever, that is a permanent tax on the domain the
     model exists to serve.

     This script samples by the TARGET MIXTURE instead of by file size: the
     tokenizer sees finance in roughly the proportion the model will actually be
     trained on. That is the correct reference distribution -- a tokenizer should
     be efficient on the data it will encode, not on the data that happened to be
     easiest to download.

  3. It reports compression (bytes/token) PER SOURCE afterwards. A vocabulary
     that compresses fineweb well and finance badly is a silent failure that only
     shows up later as a shorter effective context on finance text, so it is
     measured here rather than assumed.

Vocabulary size stays at 16,000: it is what the benchmarked architectures were
sized against (embedding = 16000 x n_embd, and at 63M non-embedding params the
embedding is already ~14% of the model), so changing it would invalidate the
architecture sweep.

Usage:
    python tokenizer/train_tokenizer_v2.py
    python tokenizer/train_tokenizer_v2.py --sample_mib 1200 --vocab_size 16000
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MIB = 1024 ** 2

# Sampling weights = the target training mixture from configs/data_mix_v2.yaml,
# collapsed to the sources that exist on disk. Deliberately NOT read from the
# YAML: the mix file is about token budgets over a 6.5-day run, this is about
# which text decides the vocabulary. They start equal and should be free to
# diverge (e.g. numeracy is 4% of training but its digit patterns are already
# well covered by BPE, so it needs no extra vocabulary weight).
SAMPLE_WEIGHTS = {
    "fineweb_edu":       0.28,
    "cosmopedia":        0.17,
    "wikipedia":         0.08,
    "gutenberg":         0.02,
    "edgar":             0.13,
    "finance_instruct":  0.16,
    "wikipedia_finance": 0.05,
    "sujet_finance":     0.03,
    "finance_alpaca":    0.03,
    "finance_numeracy":  0.03,
    "gsm8k":             0.01,
    "dolly":             0.01,
    "regulatory":        0.00,
}


def sample_source(path, budget_bytes, rng, out_fh):
    """Stream a JSONL source and write text lines until budget_bytes is reached.

    Reservoir sampling would be the textbook choice but needs the whole file in
    memory or two passes over 2 GiB. Instead this takes every document with
    probability p, estimated from the file size, which is unbiased for this
    purpose and single-pass.
    """
    total = path.stat().st_size
    if total <= 0:
        return 0
    p = min(1.0, budget_bytes / (total * 0.85))   # 0.85 ~ text fraction of JSON
    written = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if written >= budget_bytes:
                break
            if p < 1.0 and rng.random() > p:
                continue
            try:
                text = json.loads(line).get("text", "")
            except Exception:
                continue
            if not text:
                continue
            # SentencePiece treats each input line as a sentence, so newlines
            # inside a document must go or one document becomes many samples of
            # wildly uneven length.
            text = " ".join(text.split())
            out_fh.write(text + "\n")
            written += len(text.encode("utf-8"))
    return written


def build_sample(in_dir, sample_path, sample_mib, seed):
    rng = random.Random(seed)
    budget = sample_mib * MIB
    files = {p.stem: p for p in sorted(Path(in_dir).glob("*.jsonl"))}
    missing = [s for s in SAMPLE_WEIGHTS if s not in files and SAMPLE_WEIGHTS[s] > 0]
    if missing:
        print(f"  note: not on disk, weight redistributed: {', '.join(missing)}")

    present = {s: w for s, w in SAMPLE_WEIGHTS.items() if s in files and w > 0}
    tot_w = sum(present.values())
    if not present:
        raise SystemExit(f"no known sources under {in_dir}")

    print(f"\nbuilding {sample_mib} MiB training sample "
          f"from {len(present)} sources\n")
    print(f'{"source":22} {"target":>9} {"got":>9} {"docs/s":>9}')
    print("-" * 54)
    got = {}
    with open(sample_path, "w", encoding="utf-8") as out:
        for s, w in sorted(present.items(), key=lambda kv: -kv[1]):
            want = int(budget * w / tot_w)
            t0 = time.time()
            n = sample_source(files[s], want, rng, out)
            got[s] = n
            print(f"{s:22} {want/MIB:8.1f}M {n/MIB:8.1f}M "
                  f"{n/MIB/max(time.time()-t0, 1e-9):8.1f}M/s", flush=True)
    print("-" * 54)
    print(f'{"TOTAL":22} {budget/MIB:8.1f}M {sum(got.values())/MIB:8.1f}M')
    short = {s: n for s, n in got.items()
             if n < 0.9 * budget * present[s] / tot_w}
    if short:
        print("\n  UNDER TARGET (source exhausted, weight lost to others):")
        for s, n in short.items():
            print(f"    {s:22} {n/MIB:.1f} MiB of "
                  f"{budget*present[s]/tot_w/MIB:.1f} MiB")
    return got


def measure_compression(model_path, in_dir, per_source_docs=300):
    """Bytes per token per source. The number that decides effective context."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    print(f"\n{'source':22} {'bytes/token':>12} {'tokens/doc':>12}")
    print("-" * 50)
    rows = []
    for p in sorted(Path(in_dir).glob("*.jsonl")):
        nb = nt = nd = 0
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if nd >= per_source_docs:
                    break
                try:
                    text = json.loads(line).get("text", "")
                except Exception:
                    continue
                if not text:
                    continue
                nb += len(text.encode("utf-8"))
                nt += len(sp.encode(text))
                nd += 1
        if nt:
            rows.append((p.stem, nb / nt, nt / max(nd, 1)))
            print(f"{p.stem:22} {nb/nt:12.3f} {nt/max(nd,1):12,.0f}", flush=True)
    if rows:
        avg = sum(r[1] for r in rows) / len(rows)
        worst = min(rows, key=lambda r: r[1])
        print("-" * 50)
        print(f"mean {avg:.3f} bytes/token | worst-compressed: "
              f"{worst[0]} at {worst[1]:.3f}")
        print("\nHigher bytes/token is better (more text per context window).")
        print("A finance source far below the mean means finance text is being")
        print("spelled out in pieces and the effective context is shorter there.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/processed/corpus_v2/clean",
                    help="cleaned corpus; falls back to raw if absent")
    ap.add_argument("--sample_mib", type=int, default=1200)
    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--output_prefix", default="tokenizer/tokenizer_v2")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--keep_sample", action="store_true")
    ap.add_argument("--measure_only", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists() or not any(in_dir.glob("*.jsonl")):
        alt = Path("data/raw/corpus_v2")
        print(f"{in_dir} empty — falling back to {alt} (UNCLEANED)")
        in_dir = alt

    model = Path(args.output_prefix + ".model")
    if args.measure_only:
        if not model.exists():
            raise SystemExit(f"{model} not found")
        measure_compression(model, in_dir)
        return

    sample = Path("data/processed/tokenizer_sample_v2.txt")
    sample.parent.mkdir(parents=True, exist_ok=True)
    build_sample(in_dir, sample, args.sample_mib, args.seed)

    import sentencepiece as spm
    print(f"\ntraining BPE vocab={args.vocab_size:,} "
          f"(this takes a while on {args.sample_mib} MiB)", flush=True)
    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=str(sample),
        model_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        input_sentence_size=0,          # the sample IS the budget; do not re-cap
        shuffle_input_sentence=True,
        pad_id=0, unk_id=3, bos_id=1, eos_id=2,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
        user_defined_symbols=["<user>", "<assistant>"],
        # Digits split individually so arithmetic is learned over a fixed
        # alphabet of 10 symbols rather than over merged chunks like "000" that
        # differ between "1,000" and "10,00". This matters specifically because
        # finance_numeracy is teaching arithmetic.
        split_digits=True,
        byte_fallback=True,             # no <unk> on unseen bytes
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=True,
    )
    print(f"trained in {time.time()-t0:,.0f}s -> {args.output_prefix}.model")

    measure_compression(model, in_dir)

    if not args.keep_sample:
        sample.unlink(missing_ok=True)
        print(f"\nremoved {sample} (pass --keep_sample to retain)")


if __name__ == "__main__":
    main()
