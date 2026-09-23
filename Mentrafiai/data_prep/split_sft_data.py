"""
Merge all SFT JSONL files and split into train/val.

Sources (in priority order):
  1. data/raw/sft/mentrafiai_sft_v2.jsonl    — 3000 high-quality generated examples
  2. data/raw/sft/generated_recommendation_qa.jsonl — 800 examples
  3. data/raw/sft/vilavision_filtered.jsonl  — 2323 examples  
  4. data/raw/sft/conversational_greetings.jsonl — 26 examples

Deduplication: exact match on the "user" field (keep first occurrence).
Split: 90% train, 10% val (by document hash for reproducibility).
Output: data/processed/sft_train.jsonl, data/processed/sft_val.jsonl

Run from D:\\Mentrafiai:
    python data_prep/split_sft_data.py
"""

import hashlib
import json
import os
from pathlib import Path

SOURCES = [
    "data/raw/sft/mentrafiai_sft_v2.jsonl",
    "data/raw/sft/generated_recommendation_qa.jsonl",
    "data/raw/sft/vilavision_filtered.jsonl",
    "data/raw/sft/conversational_greetings.jsonl",
]
OUT_DIR = Path("data/processed")
TRAIN_PATH = OUT_DIR / "sft_train.jsonl"
VAL_PATH = OUT_DIR / "sft_val.jsonl"
VAL_FRACTION = 0.10


def doc_bucket(text: str) -> int:
    h = hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(h, "big") % 10000


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seen_users: set[str] = set()
    all_examples = []

    for src in SOURCES:
        if not os.path.exists(src):
            print(f"  SKIP (not found): {src}")
            continue
        with open(src, "r", encoding="utf-8") as f:
            lines = f.readlines()
        added = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            user = obj.get("user", "").strip()
            assistant = obj.get("assistant", "").strip()
            if not user or not assistant:
                continue
            # Quality filter: skip very short assistant responses
            if len(assistant) < 30:
                continue
            # Exact dedup on user text
            if user in seen_users:
                continue
            seen_users.add(user)
            all_examples.append({"user": user, "assistant": assistant})
            added += 1
        print(f"  {os.path.basename(src):45s} {added:5,} added  (total so far: {len(all_examples):,})")

    val_cut = int(VAL_FRACTION * 10000)
    train_count = val_count = 0

    with open(TRAIN_PATH, "w", encoding="utf-8") as ft, \
         open(VAL_PATH, "w", encoding="utf-8") as fv:
        for ex in all_examples:
            bucket = doc_bucket(ex["user"])
            if bucket < val_cut:
                fv.write(json.dumps(ex, ensure_ascii=False) + "\n")
                val_count += 1
            else:
                ft.write(json.dumps(ex, ensure_ascii=False) + "\n")
                train_count += 1

    print(f"\nTotal unique examples: {len(all_examples):,}")
    print(f"Train: {train_count:,}  ({train_count/max(len(all_examples),1)*100:.1f}%)")
    print(f"Val:   {val_count:,}  ({val_count/max(len(all_examples),1)*100:.1f}%)")
    print(f"Saved to: {TRAIN_PATH}, {VAL_PATH}")


if __name__ == "__main__":
    main()
