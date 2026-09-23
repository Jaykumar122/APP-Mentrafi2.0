"""
Tokenize indian_market_knowledge.jsonl using tokenizer_v2.model and append into train.bin.
"""

import json
import time
from pathlib import Path
import numpy as np
import sentencepiece as spm

JSONL_PATH = Path("data/raw/corpus_v2/indian_market_knowledge.jsonl")
OUT_BIN = Path("data/processed/corpus_v2/indian_market_knowledge.bin")
TRAIN_BIN = Path("data/processed/corpus_v2/train.bin")
TOK_PATH = Path("tokenizer/tokenizer_v2.model")


def main():
    if not JSONL_PATH.exists():
        print(f"[ERROR] {JSONL_PATH} not found.")
        return

    sp = spm.SentencePieceProcessor(model_file=str(TOK_PATH))
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else 2
    print(f"Loaded tokenizer: {TOK_PATH} (vocab: {sp.get_piece_size():,})")

    token_chunks = []
    total_tokens = 0
    doc_count = 0
    t0 = time.time()

    print(f"Tokenizing {JSONL_PATH.name}...")
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if not text:
                continue
            
            ids = sp.encode(text, out_type=int)
            ids.append(eos_id)
            token_chunks.extend(ids)
            total_tokens += len(ids)
            doc_count += 1
            
            if doc_count % 10000 == 0:
                print(f"  {doc_count:,} articles tokenized | {total_tokens:,} tokens", flush=True)

    dt = time.time() - t0
    print(f"\n[DONE] Tokenized {doc_count:,} articles -> {total_tokens:,} tokens in {dt:.1f}s ({total_tokens / dt:,.0f} tok/s)")

    # Save standalone binary
    arr = np.array(token_chunks, dtype=np.uint16)
    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_BIN, "wb") as f:
        f.write(arr.tobytes())
    print(f"Saved standalone binary: {OUT_BIN} ({len(arr.tobytes()) / (1024*1024):.1f} MB)")

    # Append into train.bin so PretrainDataset immediately trains on it
    if TRAIN_BIN.exists():
        prev_size = TRAIN_BIN.stat().st_size
        prev_tokens = prev_size // 2
        with open(TRAIN_BIN, "ab") as f:
            f.write(arr.tobytes())
        new_size = TRAIN_BIN.stat().st_size
        new_tokens = new_size // 2
        print(f"\n[SUCCESS] Appended to {TRAIN_BIN}:")
        print(f"  - Previous Tokens: {prev_tokens:,} ({prev_size / (1024*1024):.1f} MB)")
        print(f"  - Added Tokens   : {total_tokens:,} ({len(arr.tobytes()) / (1024*1024):.1f} MB)")
        print(f"  - New Total Tokens: {new_tokens:,} ({new_size / (1024*1024):.1f} MB)")
    else:
        print(f"[WARN] {TRAIN_BIN} not found. Created standalone binary only.")


if __name__ == "__main__":
    main()
