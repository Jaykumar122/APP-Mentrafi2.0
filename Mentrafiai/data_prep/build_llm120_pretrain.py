"""Build the llm120 pretraining dataset (data/llm120_pretrain) without re-tokenizing.

Inputs (all produced earlier by the corpus_v2 pipeline, same tokenizer_v2.model):
  data/processed/corpus_v2/train.bin   = packed base mix + 4 appended Indian-MF files
  data/processed/corpus_v2/val.bin     = held-out base-mix split
  data/raw/corpus_v2/<mf source>.jsonl = Indian mutual-fund synthetic text

Fixes applied relative to the old packed train.bin:
  * The 4 appended Indian-MF segments contained 70-86% exact-duplicate documents
    (templated generators). They are cut off and re-added after exact dedup
    (NFKC + collapsed whitespace, same rule as llm120.prepare).
  * 5% of the unique Indian-MF documents go to val (content-hash split), so the
    validation loss actually measures the mutual-fund domain.
  * A llm120 manifest.json (tokenizer hash, sizes, SHA-256) is written so
    llm120.train accepts the data.

NOTE: this was a one-time build. Its inputs (the old packed corpus_v2/train.bin
and val.bin) have since been deleted, so data/llm120_pretrain is now the only
copy of the tokenized base mix. To rebuild from scratch, re-run
fetch_corpus_v2.py -> clean_corpus_v2.py -> pack_corpus_v2.py first.

Usage (from Mentrafiai/):  python data_prep/build_llm120_pretrain.py [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm120.data import DISK_DTYPE, TOKEN_DTYPE  # noqa: E402
from llm120.prepare import SPLIT_VERSION, _fingerprint, _normalized_text, _publish, _split_for  # noqa: E402
from llm120.tokenization import TEMPLATE_VERSION, Tokenizer, clean_content, file_sha256  # noqa: E402

PACKED = ROOT / "data" / "processed" / "corpus_v2"
RAW = ROOT / "data" / "raw" / "corpus_v2"
# Order in which pack_*_tokens.py appended them to train.bin.
APPENDED = ["advanced_finance_math", "conversational_dialogues",
            "indian_market_knowledge", "four_pillars_mastery"]
CHUNK = 64 * 1024 * 1024  # tokens per copy chunk


def count_eos(path: Path, start: int, stop: int, eos_id: int) -> int:
    tokens = np.memmap(path, dtype=DISK_DTYPE, mode="r")
    total = 0
    for pos in range(start, stop, CHUNK):
        total += int((tokens[pos:min(pos + CHUNK, stop)] == eos_id).sum())
    return total


def copy_tokens(src: Path, start: int, stop: int, dst) -> None:
    tokens = np.memmap(src, dtype=DISK_DTYPE, mode="r")
    for pos in range(start, stop, CHUNK):
        dst.write(np.ascontiguousarray(tokens[pos:min(pos + CHUNK, stop)]).tobytes())


def base_length(train: Path) -> int:
    """Token count of train.bin before the appended MF files; verifies the tails match."""
    tokens = np.memmap(train, dtype=DISK_DTYPE, mode="r")
    end = len(tokens)
    for name in reversed(APPENDED):
        segment = np.memmap(PACKED / f"{name}.bin", dtype=DISK_DTYPE, mode="r")
        start = end - len(segment)
        if start < 0 or not np.array_equal(tokens[start:end], segment):
            raise SystemExit(f"train.bin does not end with {name}.bin; refusing to guess the boundary")
        end = start
    return end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer" / "tokenizer_v2.model")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "llm120_pretrain")
    parser.add_argument("--mf-val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tok = Tokenizer(args.tokenizer)
    train_src, val_src = PACKED / "train.bin", PACKED / "val.bin"
    base_end = base_length(train_src)
    train_total = train_src.stat().st_size // 2
    print(f"train.bin: {train_total:,} tokens = base {base_end:,} + MF tail {train_total - base_end:,}")

    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        info = {name: {"file": f"{name}.bin", "documents": 0, "tokens": 0} for name in ("train", "val")}
        with (stage / "train.bin").open("wb") as train_out, (stage / "val.bin").open("wb") as val_out:
            copy_tokens(train_src, 0, base_end, train_out)
            copy_tokens(val_src, 0, val_src.stat().st_size // 2, val_out)
            info["train"]["documents"] = count_eos(train_src, 0, base_end, tok.eos_id)
            info["train"]["tokens"] = base_end
            val_tokens = val_src.stat().st_size // 2
            info["val"]["documents"] = count_eos(val_src, 0, val_tokens, tok.eos_id)
            info["val"]["tokens"] = val_tokens
            base_docs = dict((k, dict(v)) for k, v in info.items())

            seen: set[str] = set()
            mf = {"read": 0, "duplicates": 0, "rejected": 0, "kept": 0, "per_source": {}}
            outs = {"train": train_out, "val": val_out}
            for name in APPENDED:
                stats = {"read": 0, "unique": 0}
                with (RAW / f"{name}.jsonl").open("r", encoding="utf-8") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        mf["read"] += 1
                        stats["read"] += 1
                        try:
                            text = clean_content(json.loads(line)["text"])
                            ids = tok.encode(text)
                            if not ids:
                                raise ValueError("no tokens")
                        except (ValueError, KeyError, UnicodeError):
                            mf["rejected"] += 1
                            continue
                        fingerprint = _fingerprint(_normalized_text(text))
                        if fingerprint in seen:
                            mf["duplicates"] += 1
                            continue
                        seen.add(fingerprint)
                        ids.append(tok.eos_id)
                        split = _split_for(fingerprint, args.seed, args.mf_val_fraction)
                        outs[split].write(np.asarray(ids, dtype=DISK_DTYPE).tobytes())
                        info[split]["documents"] += 1
                        info[split]["tokens"] += len(ids)
                        mf["kept"] += 1
                        stats["unique"] += 1
                mf["per_source"][name] = stats
                print(f"  {name}: read {stats['read']:,}, unique {stats['unique']:,}")

        for name, split in info.items():
            path = stage / split["file"]
            split["size_bytes"] = path.stat().st_size
            if split["size_bytes"] != split["tokens"] * 2:
                raise SystemExit(f"{name}: size/token mismatch")
            split["sha256"] = file_sha256(path)

        manifest = {
            "manifest_version": 1, "mode": "pretrain", "dtype": TOKEN_DTYPE, "byte_order": "little",
            "tokenizer_sha256": tok.sha256, "vocab_size": tok.vocab_size,
            "template_version": TEMPLATE_VERSION, "seed": args.seed,
            "split_version": SPLIT_VERSION, "split_key": "base: pack_corpus_v2 document hash; "
            "indian_mf: normalized_document", "val_fraction": args.mf_val_fraction,
            "dedup_normalization": "NFKC + collapsed whitespace (applied to the Indian-MF appended text)",
            "sources": [{"path": str(train_src), "tokens_used": base_end},
                        {"path": str(val_src), "tokens_used": base_docs["val"]["tokens"]}]
                       + [{"path": str(RAW / f"{n}.jsonl")} for n in APPENDED],
            "counts": {"base_train_documents": base_docs["train"]["documents"],
                       "base_train_tokens": base_end,
                       "indian_mf_read": mf["read"], "indian_mf_unique": mf["kept"],
                       "indian_mf_duplicates_removed": mf["duplicates"],
                       "indian_mf_rejected": mf["rejected"],
                       "indian_mf_per_source": mf["per_source"]},
            "splits": info,
        }
        with (stage / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        _publish(stage, output, args.force)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(json.dumps({"train": info["train"], "val": info["val"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
