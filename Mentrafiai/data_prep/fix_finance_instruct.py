"""
Strip instruction-format noise from finance_instruct.jsonl.

42.5% of finance_instruct docs start with system-prompt boilerplate like:
  "You are a helpful assistant."
  "As a finance expert, your role is to provide..."
  "Your task is to answer..."

During pretraining, the model learns to predict these prefixes token-by-token,
wasting capacity on repetitive noise. This script strips those prefixes, keeping
only the actual financial Q&A content that follows.

Also removes very short docs (<150 chars) that are just truncated prompts.

Run from D:\\Mentrafiai:
    python data_prep/fix_finance_instruct.py
"""

import json
import re
import sys
from pathlib import Path

SRC = Path("data/processed/corpus_v2/clean/finance_instruct.jsonl")
DST = Path("data/processed/corpus_v2/clean/finance_instruct.jsonl")
TMP = Path("data/processed/corpus_v2/clean/finance_instruct_fixed.jsonl")

# Patterns to strip from the START of a document
SYSTEM_PREFIXES = re.compile(
    r"^(You are a (helpful assistant|financial (expert|analyst)|finance expert)[.,\n]*"
    r"|As a finance expert,? your role is[^.]+\.[^.]+\."
    r"|Your (task|role|job) is to (answer|provide|classify|extract|analyze|evaluate)[^.]+\."
    r"|You will (use|analyze|evaluate|review|classify)[^\n]+\n+"
    r"|Context:[^\n]+\n+"
    r"|Answer the following question[^:]*:[^\n]*\n+"
    r")\s*",
    re.IGNORECASE | re.DOTALL,
)

# Also strip "Question:" / "Answer:" headers that are just noise
QA_HEADER = re.compile(r"^(Question|Q|Instruction|Input|Human):\s*", re.IGNORECASE)


def clean_doc(text: str) -> str:
    # Strip leading system prompt (up to 3 passes — some have nested prefixes)
    for _ in range(3):
        stripped = SYSTEM_PREFIXES.sub("", text).strip()
        if stripped == text:
            break
        text = stripped

    # Strip bare Q: / Question: headers
    text = QA_HEADER.sub("", text).strip()

    return text


def main():
    src = SRC
    if not src.exists():
        print(f"ERROR: {src} not found")
        sys.exit(1)

    with open(src, "r", encoding="utf-8") as fin, \
         open(TMP, "w", encoding="utf-8") as fout:

        total = kept = stripped = dropped = 0
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            text = obj["text"]

            cleaned = clean_doc(text)

            # Drop if too short after cleaning (was just a prompt header)
            if len(cleaned) < 150:
                dropped += 1
                continue

            if cleaned != text:
                stripped += 1

            obj["text"] = cleaned
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    # Replace original with fixed version
    DST.unlink()
    TMP.rename(DST)

    print(f"finance_instruct cleanup:")
    print(f"  Total docs:    {total:,}")
    print(f"  Kept:          {kept:,}")
    print(f"  Stripped noise:{stripped:,} ({100*stripped/max(total,1):.1f}%)")
    print(f"  Dropped short: {dropped:,} ({100*dropped/max(total,1):.1f}%)")
    print(f"  Saved to:      {DST}")
    print()
    print("Next step: repack the .bin files:")
    print("  python data_prep/pack_corpus_v2.py")


if __name__ == "__main__":
    main()
