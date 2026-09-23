"""
Train a SentencePiece BPE tokenizer for MentraFiAI.

Usage:
    python tokenizer/train_tokenizer.py --data_dir ./data/raw --vocab_size 16000

Special tokens included:
    <pad>=0, <bos>=1, <eos>=2, <unk>=3, <user>=4, <assistant>=5
The <user>/<assistant> tokens are used to delimit chat turns in the SFT phase.
"""

import argparse
import glob
import os
import sentencepiece as spm


def gather_input_files(data_dir: str):
    patterns = ["**/*.md", "**/*.txt", "**/*.jsonl"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(data_dir, p), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .md/.txt/.jsonl files found under {data_dir}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data/raw")
    ap.add_argument("--output_prefix", type=str, default="./tokenizer/tokenizer")
    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--input_sentence_size", type=int, default=3_000_000,
                     help="Cap sentences sampled for speed on large corpora (0 = no cap)")
    args = ap.parse_args()

    files = gather_input_files(args.data_dir)
    print(f"Found {len(files)} input files")

    spm.SentencePieceTrainer.train(
        input=",".join(files),
        model_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        input_sentence_size=args.input_sentence_size if args.input_sentence_size > 0 else 0,
        shuffle_input_sentence=True,
        pad_id=0, unk_id=3, bos_id=1, eos_id=2,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
        user_defined_symbols=["<user>", "<assistant>"],
        num_threads=os.cpu_count() or 4,
    )
    print(f"✓ Tokenizer trained: {args.output_prefix}.model / .vocab")


if __name__ == "__main__":
    main()
