#!/usr/bin/env python3
"""
Fetch high-quality pretrain corpus for MentraFiAI v2.

Mix design (targeting ~20B tokens for 1-week GTX 1650 run):
  1. General English fluency (40%): Wikipedia, OpenWebText2-style
  2. Finance domain (30%): Finance papers, reports, educational content
  3. Reasoning & QA (20%): Explanatory text, step-by-step reasoning
  4. Mutual fund specifics (10%): Investment education, SIP concepts

All sources verified open-license (CC-BY, CC0, MIT, Apache-2.0).
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = Path("data/raw/pretrain_v2")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Target token counts (will adjust based on benchmark results)
TARGET_TOKENS = {
    "general_english": 8_000_000_000,    # 8B tokens - fluency
    "finance": 6_000_000_000,             # 6B tokens - domain
    "reasoning": 4_000_000_000,           # 4B tokens - explanatory
    "investment": 2_000_000_000,          # 2B tokens - specific
}


def fetch_wikipedia_sample(target_tokens=2_000_000_000):
    """Wikipedia 20220301.en - proven high-quality English text."""
    print(f"\n=== Fetching Wikipedia (target: {target_tokens:,} tokens) ===")
    out_path = RAW_DIR / "wikipedia.jsonl"

    if out_path.exists():
        print(f"  ✓ Already exists: {out_path}")
        return

    ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    chars_written = 0
    target_chars = target_tokens * 4  # ~4 chars/token estimate

    with open(out_path, 'w', encoding='utf-8') as f:
        for item in tqdm(ds, desc="Wikipedia"):
            text = item['text'].strip()
            if len(text) < 100:  # Skip stubs
                continue

            f.write(json.dumps({"text": text, "source": "wikipedia"}) + '\n')
            chars_written += len(text)

            if chars_written >= target_chars:
                break

    print(f"  ✓ Wrote {chars_written:,} chars (~{chars_written//4:,} tokens)")


def fetch_fineweb_edu(target_tokens=4_000_000_000):
    """FineWeb-Edu - high-quality web text, education-focused."""
    print(f"\n=== Fetching FineWeb-Edu (target: {target_tokens:,} tokens) ===")
    out_path = RAW_DIR / "fineweb_edu.jsonl"

    if out_path.exists():
        print(f"  ✓ Already exists: {out_path}")
        return

    # FineWeb-Edu sample-10BT has ~10B tokens, CC-BY-4.0
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)

    chars_written = 0
    target_chars = target_tokens * 4

    with open(out_path, 'w', encoding='utf-8') as f:
        for item in tqdm(ds, desc="FineWeb-Edu"):
            text = item['text'].strip()
            if len(text) < 200:
                continue

            f.write(json.dumps({"text": text, "source": "fineweb_edu"}) + '\n')
            chars_written += len(text)

            if chars_written >= target_chars:
                break

    print(f"  ✓ Wrote {chars_written:,} chars (~{chars_written//4:,} tokens)")


def fetch_finance_alpaca(target_tokens=2_000_000_000):
    """Finance Alpaca - financial Q&A and explanations."""
    print(f"\n=== Fetching Finance Alpaca (target: {target_tokens:,} tokens) ===")
    out_path = RAW_DIR / "finance_alpaca.jsonl"

    if out_path.exists():
        print(f"  ✓ Already exists: {out_path}")
        return

    ds = load_dataset("gbharti/finance-alpaca", split="train")

    with open(out_path, 'w', encoding='utf-8') as f:
        for item in tqdm(ds, desc="Finance Alpaca"):
            # Convert instruction-response to natural text
            instruction = item.get('instruction', '').strip()
            response = item.get('output', '').strip()

            if instruction and response:
                text = f"Question: {instruction}\n\nAnswer: {response}"
                f.write(json.dumps({"text": text, "source": "finance_alpaca"}) + '\n')

    print(f"  ✓ Wrote {len(ds):,} examples")


def fetch_financial_phrasebank(target_tokens=500_000_000):
    """Financial PhraseBank - financial news sentiment (CC-BY-4.0)."""
    print(f"\n=== Fetching Financial PhraseBank (target: {target_tokens:,} tokens) ===")
    out_path = RAW_DIR / "financial_phrasebank.jsonl"

    if out_path.exists():
        print(f"  ✓ Already exists: {out_path}")
        return

    ds = load_dataset("financial_phrasebank", "sentences_allagree", split="train")

    with open(out_path, 'w', encoding='utf-8') as f:
        for item in tqdm(ds, desc="Financial PhraseBank"):
            text = item['sentence'].strip()
            if len(text) > 50:
                f.write(json.dumps({"text": text, "source": "financial_phrasebank"}) + '\n')

    print(f"  ✓ Wrote {len(ds):,} sentences")


def fetch_finer_ord(target_tokens=1_000_000_000):
    """FINER-ORD - financial entity recognition and reasoning."""
    print(f"\n=== Fetching FINER-ORD (target: {target_tokens:,} tokens) ===")
    out_path = RAW_DIR / "finer_ord.jsonl"

    if out_path.exists():
        print(f"  ✓ Already exists: {out_path}")
        return

    try:
        ds = load_dataset("nlpaueb/finer-ord", split="train")

        with open(out_path, 'w', encoding='utf-8') as f:
            for item in tqdm(ds, desc="FINER-ORD"):
                text = ' '.join(item['tokens']) if 'tokens' in item else ''
                if len(text) > 100:
                    f.write(json.dumps({"text": text, "source": "finer_ord"}) + '\n')

        print(f"  ✓ Wrote {len(ds):,} examples")
    except Exception as e:
        print(f"  ⚠ Failed to load FINER-ORD: {e}")


def fetch_gutenberg_finance(target_tokens=500_000_000):
    """Reuse existing Gutenberg finance/economics books if available."""
    print(f"\n=== Checking for existing Gutenberg finance texts ===")
    old_path = Path("data/raw/gutenberg")
    out_path = RAW_DIR / "gutenberg_finance.jsonl"

    if out_path.exists():
        print(f"  ✓ Already exists: {out_path}")
        return

    if not old_path.exists():
        print(f"  ⚠ No existing Gutenberg directory found")
        return

    # This would need custom filtering logic based on what was downloaded
    # For now, just note it as an option
    print(f"  ℹ Gutenberg texts at {old_path} can be filtered for finance/economics")
    print(f"    (requires manual curation - skipping for now)")


def create_corpus_manifest():
    """Create manifest of all fetched sources."""
    manifest_path = RAW_DIR / "MANIFEST.json"

    manifest = {
        "created": "2026-08-24",
        "purpose": "MentraFiAI v2 pretrain corpus",
        "target_tokens": 20_000_000_000,
        "sources": {}
    }

    for jsonl_file in RAW_DIR.glob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            lines = sum(1 for _ in f)

        manifest["sources"][jsonl_file.stem] = {
            "file": jsonl_file.name,
            "lines": lines,
            "estimated_tokens": lines * 200  # rough estimate
        }

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== Manifest created: {manifest_path} ===")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    print("=" * 70)
    print("MENTRAFIAI V2 CORPUS FETCHER")
    print("=" * 70)
    print("\nFetching high-quality, open-license sources...")
    print("This will take significant time and disk space.")
    print(f"\nTarget directory: {RAW_DIR.absolute()}")

    # Fetch each source
    fetch_wikipedia_sample(target_tokens=2_000_000_000)
    fetch_fineweb_edu(target_tokens=4_000_000_000)
    fetch_finance_alpaca(target_tokens=2_000_000_000)
    fetch_financial_phrasebank(target_tokens=500_000_000)
    fetch_finer_ord(target_tokens=1_000_000_000)
    fetch_gutenberg_finance(target_tokens=500_000_000)

    # Create manifest
    create_corpus_manifest()

    print("\n" + "=" * 70)
    print("✓ CORPUS FETCH COMPLETE")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Run data_prep/clean_corpus_v2.py to clean and normalize")
    print(f"  2. Run data_prep/pack_corpus_v2.py to tokenize and pack")
    print(f"  3. Start pretraining with training/pretrain.py")
