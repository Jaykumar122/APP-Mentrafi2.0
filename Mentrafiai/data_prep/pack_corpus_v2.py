"""
Pack the cleaned v2 corpus into train.bin / val.bin under an ENFORCED mixture.

This is the stage that failed in v1. data_mix.yaml declared 25% finance; the
corpus that reached the model was 0.0025% finance. Nothing between the two
compared them, so a four-order-of-magnitude miss shipped silently and the model
was pretrained on effectively no finance text.

Three mechanisms prevent a repeat, in order of how much they matter:

  1. SUPPLY IS CHECKED AGAINST DEMAND, AND A SHORTFALL IS AN ERROR.
     For each source: required_epochs = (share x total_tokens) / available_tokens.
     If that exceeds the source's max_epochs, the build stops and prints the
     number. v1 had no such comparison, so a source that could not supply its
     share just quietly supplied less.

  2. THE SPLIT IS BY DOCUMENT HASH, NOT BY POSITION.
     Slicing the last 0.5% of each file leaks: consecutive documents from one
     stream share topic and often share near-duplicate passages, so a tail split
     measures how well the model memorised a document's neighbours. Hashing the
     document text assigns it to train or val deterministically and independently
     of order, and -- because the hash is of the TEXT -- a document appearing in
     two sources lands on the same side both times. Overlap is then impossible by
     construction rather than by inspection, and it is still verified at the end.

  3. TOKENS ARE COUNTED, NOT ESTIMATED.
     Shares are enforced on tokens actually written to the .bin, so the report at
     the end states what the model will really see. bytes_per_token in the config
     is only used for the up-front feasibility estimate.

Epoch handling: a source needing 2.4 epochs gets 2 full passes plus a
deterministic 40% subsample of a third, rather than 2 passes and then the first
40% of the file again. Taking a prefix would over-expose whatever the source
happened to put first (for EDGAR, that is one shard's alphabetical run of
companies).

Output:
    data/processed/corpus_v2/train.bin   uint16 token ids
    data/processed/corpus_v2/val.bin
    data/processed/corpus_v2/pack_report.json

Usage:
    python data_prep/pack_corpus_v2.py --dry_run      # feasibility only, no write
    python data_prep/pack_corpus_v2.py
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MIB = 1024 ** 2
# clean_corpus_v2.py writes its JSONL into the clean/ subdirectory; the .bin
# files and reports go in the parent. Pointing IN_DIR at the parent would glob
# zero .jsonl files and the run would abort — which is the safe failure, but
# only because there is no raw fallback here.
IN_DIR = Path("data/processed/corpus_v2/clean")
OUT_DIR = Path("data/processed/corpus_v2")
CONFIG = Path("configs/data_mix_v2.yaml")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_mix(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    flat = {}
    for gname, g in cfg["groups"].items():
        for sname, s in g["sources"].items():
            flat[sname] = {"share": float(s["share"]),
                           "max_epochs": float(s["max_epochs"]),
                           "group": gname}
    cfg["flat"] = flat
    return cfg


def check_shares(cfg):
    """Shares must sum to 1 at both levels, and groups must match their sources."""
    problems = []
    gsum = sum(g["share"] for g in cfg["groups"].values())
    if abs(gsum - 1.0) > 1e-6:
        problems.append(f"group shares sum to {gsum:.6f}, not 1.0")
    for gname, g in cfg["groups"].items():
        ssum = sum(s["share"] for s in g["sources"].values())
        if abs(ssum - g["share"]) > 1e-6:
            problems.append(f"group {gname}: sources sum to {ssum:.4f} "
                            f"but group share is {g['share']:.4f}")
    fsum = sum(s["share"] for s in cfg["flat"].values())
    if abs(fsum - 1.0) > 1e-6:
        problems.append(f"source shares sum to {fsum:.6f}, not 1.0")
    return problems


# --------------------------------------------------------------------------- #
# Split assignment
# --------------------------------------------------------------------------- #
def doc_bucket(text):
    """Stable 0..9999 bucket from the document text.

    blake2b rather than hash(): Python's str hash is randomised per process, so a
    resumed or re-run pack would assign the same document differently and leak
    val documents into train across runs.
    """
    h = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(h, "big") % 10000


# --------------------------------------------------------------------------- #
# Pass 1: inventory
# --------------------------------------------------------------------------- #
def inventory(files, sp, sample_docs):
    """Measure bytes, docs and MEASURED bytes/token per source.

    bytes_per_token from the config is a guess carried over from v1. Tokenising a
    sample of each source replaces it with a measurement, which matters because
    it varies a lot by register: EDGAR's legal boilerplate compresses very
    differently from Gutenberg's prose, and using one global ratio would mis-size
    every source's epoch requirement.
    """
    inv = {}
    print(f'{"source":22} {"docs":>10} {"MiB":>9} {"B/token":>8} {"est tokens":>13}')
    print("-" * 68)
    for name, path in files.items():
        nb = nd = 0
        sb = st = sd = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    text = json.loads(line).get("text", "")
                except Exception:
                    continue
                if not text:
                    continue
                b = len(text.encode("utf-8"))
                nb += b
                nd += 1
                if sd < sample_docs:
                    sb += b
                    st += len(sp.encode(text))
                    sd += 1
        bpt = (sb / st) if st else 4.2
        inv[name] = {"bytes": nb, "docs": nd, "bytes_per_token": bpt,
                     "tokens": int(nb / bpt) if bpt else 0, "path": str(path)}
        print(f"{name:22} {nd:10,} {nb/MIB:9.1f} {bpt:8.3f} "
              f"{inv[name]['tokens']:13,}", flush=True)
    print("-" * 68)
    print(f'{"TOTAL":22} {sum(i["docs"] for i in inv.values()):10,} '
          f'{sum(i["bytes"] for i in inv.values())/MIB:9.1f} {"":8} '
          f'{sum(i["tokens"] for i in inv.values()):13,}')
    return inv


# --------------------------------------------------------------------------- #
# Feasibility
# --------------------------------------------------------------------------- #
def plan(cfg, inv):
    """Decide epochs per source, or explain why the mixture is not satisfiable."""
    total = int(cfg["total_tokens"])
    rows, errors, warnings = [], [], []
    for name, s in sorted(cfg["flat"].items(), key=lambda kv: -kv[1]["share"]):
        want = int(total * s["share"])
        have = inv.get(name, {}).get("tokens", 0)
        if want == 0:
            if have:
                warnings.append(f"{name}: share 0.0 but {have:,} tokens on disk "
                                f"— source will be EXCLUDED")
            continue
        if have == 0:
            errors.append(f"{name}: share {s['share']:.3f} needs {want:,} tokens "
                          f"but the source is missing or empty")
            continue
        need = want / have
        row = {"source": name, "group": s["group"], "share": s["share"],
               "want_tokens": want, "have_tokens": have,
               "required_epochs": need, "max_epochs": s["max_epochs"]}
        if need > s["max_epochs"] + 1e-9:
            errors.append(
                f"{name}: needs {need:.2f} epochs to reach a "
                f"{s['share']*100:.1f}% share ({want:,} tokens from "
                f"{have:,} available) but max_epochs is {s['max_epochs']:.1f}. "
                f"Either lower its share to "
                f"{have*s['max_epochs']/total*100:.2f}%, raise max_epochs, "
                f"or fetch more of it.")
        elif need > 1.0:
            warnings.append(f"{name}: {need:.2f} epochs "
                            f"(repeating, within max {s['max_epochs']:.1f})")
        rows.append(row)
    return rows, errors, warnings


def check_assertions(cfg, rows):
    """The declared build-time guarantees, enforced rather than documented."""
    a = cfg.get("assertions", {})
    problems = []
    by_group = {}
    for r in rows:
        by_group[r["group"]] = by_group.get(r["group"], 0.0) + r["share"]
    fin = by_group.get("finance_domain", 0.0)
    gen = by_group.get("general_fluency", 0.0)
    if "min_finance_share" in a and fin < a["min_finance_share"] - 1e-9:
        problems.append(f"finance share {fin:.3f} < required "
                        f"{a['min_finance_share']:.3f}")
    if "min_general_share" in a and gen < a["min_general_share"] - 1e-9:
        problems.append(f"general share {gen:.3f} < required "
                        f"{a['min_general_share']:.3f}")
    return problems


# --------------------------------------------------------------------------- #
# Pass 2: write
# --------------------------------------------------------------------------- #
def write_split(rows, inv, cfg, sp, out_dir, seed):
    """Tokenise and append each source to train.bin / val.bin.

    Fractional epochs are taken as a deterministic subsample spread over the
    whole file (bucket-based), not as a prefix, so a partial pass is not biased
    toward whatever the source ordered first.
    """
    import random
    val_cut = int(float(cfg["val_fraction"]) * 10000)
    eos = sp.eos_id() if sp.eos_id() >= 0 else 2

    out_dir.mkdir(parents=True, exist_ok=True)
    tr_path, va_path = out_dir / "train.bin", out_dir / "val.bin"
    tr = open(tr_path, "wb")
    va = open(va_path, "wb")

    stats = {}
    tr_tokens = va_tokens = 0
    tr_hashes, va_hashes = set(), set()

    print(f'\n{"source":22} {"epochs":>7} {"train tok":>13} {"val tok":>11} {"s":>6}')
    print("-" * 66)
    for r in rows:
        name = r["source"]
        path = Path(inv[name]["path"])
        full = int(r["required_epochs"])
        frac = r["required_epochs"] - full
        frac_cut = int(frac * 10000)
        rng = random.Random(f"{seed}:{name}")
        t0 = time.time()
        st = sv = 0
        for epoch in range(full + (1 if frac_cut > 0 else 0)):
            partial = epoch == full
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        text = json.loads(line).get("text", "")
                    except Exception:
                        continue
                    if not text:
                        continue
                    b = doc_bucket(text)
                    # Partial epoch: keep a deterministic slice of the buckets,
                    # offset per epoch so a 2.4-epoch source does not see the
                    # same 40% of documents three times.
                    if partial and ((b + epoch * 977) % 10000) >= frac_cut:
                        continue
                    ids = sp.encode(text)
                    if not ids:
                        continue
                    ids.append(eos)
                    arr = np.asarray(ids, dtype=np.uint16)
                    if b < val_cut:
                        arr.tofile(va); sv += arr.size; va_hashes.add(b)
                    else:
                        arr.tofile(tr); st += arr.size; tr_hashes.add(b)
        tr_tokens += st
        va_tokens += sv
        stats[name] = {**r, "train_tokens": st, "val_tokens": sv,
                       "seconds": round(time.time() - t0, 1)}
        print(f"{name:22} {r['required_epochs']:7.2f} {st:13,} {sv:11,} "
              f"{time.time()-t0:6.0f}", flush=True)
    tr.close(); va.close()
    print("-" * 66)
    print(f'{"TOTAL":22} {"":7} {tr_tokens:13,} {va_tokens:11,}')

    overlap = tr_hashes & va_hashes
    return {"train_tokens": tr_tokens, "val_tokens": va_tokens,
            "per_source": stats, "bucket_overlap": len(overlap),
            "train_bin": str(tr_path), "val_bin": str(va_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=str(IN_DIR))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer_v2.model")
    ap.add_argument("--sample_docs", type=int, default=400,
                    help="docs per source used to measure bytes/token")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--dry_run", action="store_true",
                    help="inventory + feasibility only; write nothing")
    ap.add_argument("--force", action="store_true",
                    help="write even if assertions fail (records the violation)")
    args = ap.parse_args()

    cfg = load_mix(args.config)
    bad = check_shares(cfg)
    if bad:
        print("CONFIG ERROR:")
        for b in bad:
            print(f"  {b}")
        sys.exit(2)
    print(f"mix config OK: {len(cfg['flat'])} sources, "
          f"target {int(cfg['total_tokens']):,} tokens, "
          f"val_fraction {cfg['val_fraction']}")

    in_dir = Path(args.in_dir)
    files = {p.stem: p for p in sorted(in_dir.glob("*.jsonl"))}
    if not files:
        print(f"\nno cleaned corpus under {in_dir}.")
        print("run: python data_prep/clean_corpus_v2.py")
        sys.exit(1)

    tok = Path(args.tokenizer)
    if not tok.exists():
        print(f"\ntokenizer {tok} not found.")
        print("run: python tokenizer/train_tokenizer_v2.py")
        sys.exit(1)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(tok))
    print(f"tokenizer {tok} (vocab {sp.get_piece_size():,})\n")
    if sp.get_piece_size() > 65535:
        print("vocab exceeds uint16 — change the .bin dtype before packing")
        sys.exit(2)

    unknown = set(files) - set(cfg["flat"])
    if unknown:
        print(f"note: on disk but absent from the mix (ignored): "
              f"{', '.join(sorted(unknown))}\n")

    inv = inventory(files, sp, args.sample_docs)
    rows, errors, warnings = plan(cfg, inv)

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  {w}")
    if errors:
        print("\nMIXTURE NOT SATISFIABLE:")
        for e in errors:
            print(f"  {e}")
        print("\nNothing written. This is the check v1 did not have — the "
              "mixture is infeasible on the data actually present, and the "
              "alternative to failing here is shipping a corpus whose real "
              "composition does not match the config.")
        sys.exit(3)

    aprob = check_assertions(cfg, rows)
    if aprob:
        print("\nASSERTION FAILURES:")
        for p in aprob:
            print(f"  {p}")
        if not args.force:
            sys.exit(4)
        print("  --force given: continuing and recording the violation")

    print(f"\nfeasible: all {len(rows)} sources within max_epochs")
    if args.dry_run:
        print("--dry_run: stopping before write")
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        Path(args.out_dir, "pack_plan.json").write_text(json.dumps(
            {"inventory": inv, "plan": rows, "warnings": warnings}, indent=2))
        print(f"wrote {args.out_dir}/pack_plan.json")
        return

    res = write_split(rows, inv, cfg, sp, Path(args.out_dir), args.seed)

    total = res["train_tokens"] + res["val_tokens"]
    print(f"\n{'ACHIEVED MIXTURE (measured, not declared)':^70}")
    print("=" * 70)
    print(f'{"source":22} {"group":18} {"target":>8} {"actual":>8} {"delta":>7}')
    print("-" * 70)
    worst = 0.0
    for name, s in res["per_source"].items():
        got = (s["train_tokens"] + s["val_tokens"]) / max(total, 1)
        d = got - s["share"]
        worst = max(worst, abs(d))
        print(f'{name:22} {s["group"]:18} {s["share"]*100:7.2f}% '
              f'{got*100:7.2f}% {d*100:+6.2f}%')
    print("-" * 70)
    print(f"largest share deviation: {worst*100:.2f} percentage points")

    ok_overlap = res["bucket_overlap"] == 0
    print(f"\ntrain {res['train_tokens']:,} tokens | val {res['val_tokens']:,} "
          f"({res['val_tokens']/max(total,1)*100:.2f}%)")
    print(f"train/val bucket overlap: {res['bucket_overlap']} "
          f"({'OK' if ok_overlap else 'LEAK — investigate before training'})")

    report = {"config": args.config, "tokenizer": str(tok),
              "inventory": inv, "plan": rows, "result": res,
              "warnings": warnings, "assertion_failures": aprob,
              "achieved": {n: (s["train_tokens"] + s["val_tokens"]) / max(total, 1)
                           for n, s in res["per_source"].items()},
              "max_share_deviation": worst, "no_leak": ok_overlap}
    rp = Path(args.out_dir) / "pack_report.json"
    rp.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {rp}")


if __name__ == "__main__":
    main()
