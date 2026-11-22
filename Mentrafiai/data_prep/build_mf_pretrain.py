"""Build the Indian-mutual-fund-focused pretraining set for llm120.

Mix (by training tokens):
  * Indian mutual fund text, repeated by source quality:
      - real prose (SEBI / AMFI investor pages, Wikipedia India finance)  x REAL
      - verified generated text (DB scheme facts, concepts)               x VERIFIED
      - templated synthetic text (old generators, calculator examples)    x SYNTH
  * FineWeb-Edu English (education-quality web text), each document once,
    filling the rest so the MF share is about --mf-share.
Documents naming real fund brands in the OLD synthetic generators are dropped
(their figures were invented). Exact duplicates are removed before repeating.

Output: data/llm120_pretrain/{train.bin,val.bin,manifest.json} (same format as
llm120.prepare; the previous dataset is moved to data/llm120_pretrain_previous).
Run: python -m data_prep.build_mf_pretrain
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import re
import shutil
import time
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from llm120.data import DISK_DTYPE, TOKEN_DTYPE
from llm120.tokenization import Tokenizer, file_sha256

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = ROOT / "tokenizer" / "tokenizer_v2.model"
RAW = ROOT / "data" / "raw"
OLD_SYNTHETIC = ["advanced_finance_math", "conversational_dialogues", "finance_numeracy",
                 "four_pillars_mastery", "indian_market_knowledge"]
REAL_BRAND = re.compile(r"\b(?:hdfc|sbi|icici|axis|kotak|nippon|parag parikh|mirae|quant|uti|dsp|tata|aditya birla|"
                        r"franklin|invesco|edelweiss|motilal|canara robeco|sundaram|hsbc|bandhan|baroda|pgim|"
                        r"mahindra|navi|groww|zerodha|whiteoak|360 one|lic mf|union)\b[^.]{0,30}\b(?:fund|mutual|amc)\b", re.I)

_tok = None


def _init(path: str) -> None:
    global _tok
    _tok = Tokenizer(path)


def _encode(texts: list[str]) -> list[np.ndarray]:
    out = []
    for text in texts:
        ids = _tok.encode(text)
        ids.append(_tok.eos_id)
        out.append(np.asarray(ids, dtype=DISK_DTYPE))
    return out


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def is_val(key: str, fraction: float) -> bool:
    digest = hashlib.sha256(("mf-pretrain-v1\0" + key).encode()).digest()
    return int.from_bytes(digest[:8], "big") < fraction * 2**64


INDIA_TERMS = re.compile(r"\b(india|indian|sebi|amfi|rupees?|rbi|reserve bank of india|nse|bse|sensex|nifty|"
                         r"mumbai|crore|lakh|income tax department|government of india)\b", re.I)
MF_TERMS = re.compile(r"\bmutual funds?\b|\bsystematic investment plan\b|\bnet asset value\b|\basset management compan|"
                      r"\bexpense ratio\b|\bindex funds?\b|\bexchange[- ]traded funds?\b", re.I)


def india_or_mf_article(text: str, seeds: set[str]) -> bool:
    """Wikipedia filter: hand-picked seed topics, or articles substantially about India / mutual funds."""
    title = text.split("\n", 1)[0].strip().casefold()
    if title in seeds:
        return True
    words = max(1, len(text.split()))
    india = len(INDIA_TERMS.findall(text))
    return india >= 5 and india / words >= 0.004 or len(MF_TERMS.findall(text)) >= 4


def load_mf_documents() -> dict[str, list[str]]:
    groups = {"real": [], "verified": [], "synth": []}
    stats = {}
    seen = set()

    def add(group, source, text):
        text = text.strip()
        key = normalized(text)
        if len(key) < 200 or key in seen:
            stats.setdefault(source, {"kept": 0, "dropped": 0})["dropped"] += 1
            return
        seen.add(key)
        groups[group].append(text)
        stats.setdefault(source, {"kept": 0, "dropped": 0})["kept"] += 1

    from data_prep.fetch_mf_web import WIKI_SEEDS

    seeds = {title.casefold() for title in WIKI_SEEDS}
    for path in sorted((RAW / "mf_corpus").glob("web_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            try:
                text = json.loads(line)["text"]
            except ValueError:  # a partially written last line
                continue
            if path.stem == "web_wikipedia" and not india_or_mf_article(text, seeds):
                stats.setdefault("web_wikipedia_off_topic", {"kept": 0, "dropped": 0})["dropped"] += 1
                continue
            add("real", path.stem, text)
    for name, group in (("db_funds", "verified"), ("db_families", "verified"), ("db_rankings", "verified"),
                        ("concepts", "verified"), ("calc", "synth")):
        for line in (RAW / "mf_corpus" / f"gen_{name}.jsonl").open(encoding="utf-8"):
            add(group, name, json.loads(line)["text"])
    for name in OLD_SYNTHETIC:
        for line in (RAW / "corpus_v2" / f"{name}.jsonl").open(encoding="utf-8"):
            text = json.loads(line).get("text") or ""
            if REAL_BRAND.search(text):
                stats.setdefault(name, {"kept": 0, "dropped": 0})["dropped"] += 1
                continue
            add("synth", name, text)
    return groups, stats


def fineweb_texts(min_score: float):
    import pyarrow.parquet as pq

    for path in sorted(glob.glob(str(RAW / "fineweb_edu" / "sample" / "10BT" / "*.parquet"))):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=4096, columns=["text", "score", "language_score"]):
            data = batch.to_pydict()
            for text, score, lang in zip(data["text"], data["score"], data["language_score"]):
                if score >= min_score and lang >= 0.9 and 500 <= len(text) <= 60000:
                    yield text


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-tokens", type=int, default=1_180_000_000)
    parser.add_argument("--mf-share", type=float, default=0.40)
    parser.add_argument("--max-repeat", type=int, default=8, help="Cap on repeats of any MF document")
    parser.add_argument("--min-score", type=float, default=3.0, help="FineWeb-Edu educational score threshold")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "llm120_pretrain")
    args = parser.parse_args()
    started = time.time()
    tok = Tokenizer(TOKENIZER)
    rng = random.Random(42)

    groups, source_stats = load_mf_documents()
    print(json.dumps({"mf_sources": source_stats}, indent=1), flush=True)
    mf = {"train": {}, "val": {}}
    with Pool(args.workers, initializer=_init, initargs=(str(TOKENIZER),)) as pool:
        for group, texts in groups.items():
            encoded = [ids for part in pool.imap(_encode, chunks(texts, 256)) for ids in part]
            for text, ids in zip(texts, encoded):
                split = "val" if is_val(normalized(text), 0.01) else "train"
                mf[split].setdefault(group, []).append(ids)
    unique = {g: sum(len(x) for x in docs) for g, docs in mf["train"].items()}
    print("MF unique train tokens:", {g: f"{v/1e6:.1f}M" for g, v in unique.items()}, flush=True)

    # Repeats: real prose is scarcest and most valuable; templated text least.
    target_mf = args.mf_share * args.train_tokens
    ratio = {"real": 3.0, "verified": 2.0, "synth": 1.0}
    base = target_mf / sum(ratio[g] * unique[g] for g in unique)
    repeats = {g: max(1, min(args.max_repeat, round(base * ratio[g]))) for g in unique}
    mf_tokens = sum(repeats[g] * unique[g] for g in unique)
    fw_target = args.train_tokens - mf_tokens
    print("repeats:", repeats, f"MF train tokens {mf_tokens/1e6:.0f}M, FineWeb target {fw_target/1e6:.0f}M", flush=True)

    queue = [(g, i) for g in mf["train"] for i in range(len(mf["train"][g])) for _ in range(repeats[g])]
    rng.shuffle(queue)

    stage = args.output.parent / (args.output.name + ".building")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    split_info = {name: {"file": f"{name}.bin", "documents": 0, "tokens": 0} for name in ("train", "val")}
    counts = {"fineweb_train_documents": 0, "fineweb_train_tokens": 0, "fineweb_val_documents": 0,
              "fineweb_val_tokens": 0, "mf_train_documents_with_repeats": 0, "mf_train_tokens_with_repeats": 0}
    with (stage / "train.bin").open("wb") as train_out, (stage / "val.bin").open("wb") as val_out:
        for docs in mf["val"].values():
            for ids in docs:
                val_out.write(ids.tobytes())
                split_info["val"]["documents"] += 1
                split_info["val"]["tokens"] += len(ids)
        mf_val_tokens = split_info["val"]["tokens"]
        position = 0
        fw_val_tokens = 0

        def write_mf_until(limit):
            nonlocal position
            while position < len(queue) and counts["mf_train_tokens_with_repeats"] < limit:
                g, i = queue[position]
                ids = mf["train"][g][i]
                train_out.write(ids.tobytes())
                counts["mf_train_documents_with_repeats"] += 1
                counts["mf_train_tokens_with_repeats"] += len(ids)
                position += 1

        with Pool(args.workers, initializer=_init, initargs=(str(TOKENIZER),)) as pool:
            buffer = []
            texts = fineweb_texts(args.min_score)

            def batches():
                batch = []
                for text in texts:
                    batch.append(text)
                    if len(batch) == 512:
                        yield batch
                        batch = []
                if batch:
                    yield batch

            for encoded in pool.imap(_encode, batches()):
                for ids in encoded:
                    if fw_val_tokens < mf_val_tokens and rng.random() < 0.01:
                        val_out.write(ids.tobytes())
                        fw_val_tokens += len(ids)
                        split_info["val"]["documents"] += 1
                        split_info["val"]["tokens"] += len(ids)
                        counts["fineweb_val_documents"] += 1
                        counts["fineweb_val_tokens"] += len(ids)
                        continue
                    train_out.write(ids.tobytes())
                    counts["fineweb_train_documents"] += 1
                    counts["fineweb_train_tokens"] += len(ids)
                    # Interleave MF documents in proportion to FineWeb progress.
                    write_mf_until(mf_tokens * min(1.0, counts["fineweb_train_tokens"] / fw_target))
                if counts["fineweb_train_tokens"] >= fw_target and fw_val_tokens >= mf_val_tokens * 0.9:
                    pool.terminate()
                    break
                if counts["fineweb_train_documents"] % 50000 < 512:
                    print(f"  fineweb {counts['fineweb_train_tokens']/1e6:.0f}M / {fw_target/1e6:.0f}M tokens, "
                          f"{(time.time()-started)/60:.1f} min", flush=True)
        write_mf_until(math.inf)
    split_info["train"]["documents"] = counts["fineweb_train_documents"] + counts["mf_train_documents_with_repeats"]
    split_info["train"]["tokens"] = counts["fineweb_train_tokens"] + counts["mf_train_tokens_with_repeats"]
    if counts["fineweb_train_tokens"] < fw_target * 0.95:
        print(f"WARNING: FineWeb ran out at {counts['fineweb_train_tokens']/1e6:.0f}M tokens "
              f"(target {fw_target/1e6:.0f}M); MF share is higher than requested.", flush=True)
    for name, info in split_info.items():
        path = stage / info["file"]
        info["size_bytes"] = path.stat().st_size
        info["sha256"] = file_sha256(path)
        assert info["size_bytes"] == 2 * info["tokens"]
    train_tokens = split_info["train"]["tokens"]
    manifest = {
        "manifest_version": 1, "mode": "pretrain", "tokenizer_sha256": tok.sha256, "vocab_size": tok.vocab_size,
        "dtype": TOKEN_DTYPE, "byte_order": "little", "seed": 42,
        "split_key": "sha256 of normalized document (mf-pretrain-v1), 1% MF to val; FineWeb val sampled to match",
        "builder": "data_prep/build_mf_pretrain.py", "builder_sha256": file_sha256(Path(__file__)),
        "mix": {
            "mf_share_of_train_tokens": counts["mf_train_tokens_with_repeats"] / train_tokens,
            "repeats_per_group": repeats,
            "mf_unique_train_tokens_per_group": unique,
            "mf_val_tokens": mf_val_tokens, "fineweb_val_tokens": fw_val_tokens,
            "fineweb_min_score": args.min_score,
            "groups": {"real": "SEBI/AMFI investor pages, Wikipedia India finance (CC BY-SA)",
                       "verified": "DB scheme facts from AMFI NAV data (no enriched columns), concepts",
                       "synth": "old template generators (real brand names removed), calculator examples"},
        },
        "mf_source_stats": source_stats, "counts": counts, "splits": split_info,
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output.exists():
        previous = args.output.parent / (args.output.name + "_previous")
        if previous.exists():
            shutil.rmtree(previous)
        args.output.rename(previous)
        print(f"Previous dataset moved to {previous}", flush=True)
    stage.rename(args.output)
    print(json.dumps({"splits": split_info, "mix": manifest["mix"], "minutes": round((time.time() - started) / 60, 1)},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
