"""
Script to create the complete, lightweight Kaggle upload package (mentrafiai_sft_package.zip).
Includes:
- Network 1 (6.6M Classifier Checkpoint)
- Network 2 (10M Query Parser Checkpoint)
- Network 3 (102M LLM Base Checkpoint)
- Clean SFT Datasets (sft_train_v3.jsonl, sft_val_v3.jsonl)
- Complete Source Code & Model Architectures
- Kaggle Execution Scripts
Excludes:
- 25GB pretraining binaries (corpus_v2/*.bin)
- Old intermediate step checkpoints
- __pycache__ and temporary logs
"""

import os
import sys
import zipfile
from pathlib import Path

BASE_DIR = Path(r"D:\Mentrafiai")
OUT_ZIP = BASE_DIR / "mentrafiai_sft_package.zip"

INCLUDED_ITEMS = [
    "chat.py",
    "kaggle_mentrafiai_run.py",
    "test_chat_integrated.py",
    "configs",
    "data_prep",
    "inference",
    "model",
    "tokenizer",
    "training",
]

# Explicit checkpoint files to include
CHECKPOINT_FILES = [
    "checkpoints/classifier/best_classifier.pt",
    "checkpoints/query_parser/best_parser_net.pt",
    "checkpoints/pretrain_v2/best_model.pt",
]

# Processed SFT data files
DATA_FILES = [
    "data/processed/sft_train_v3.jsonl",
    "data/processed/sft_val_v3.jsonl",
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".git",
    ".pyc",
    "corpus_v2",
    "train.bin",
    "val.bin",
    ".DS_Store",
]


def should_exclude(rel_path: str) -> bool:
    for pat in EXCLUDE_PATTERNS:
        if pat in rel_path:
            return True
    return False


def main():
    print(f"Creating Kaggle package at: {OUT_ZIP}...")
    temp_zip = BASE_DIR / "mentrafiai_sft_package.zip.tmp"
    if temp_zip.exists():
        temp_zip.unlink()

    total_files = 0
    with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Add included folders and root files
        for item_name in INCLUDED_ITEMS:
            src = BASE_DIR / item_name
            if not src.exists():
                print(f"[SKIP] Not found: {item_name}")
                continue

            if src.is_file():
                rel = item_name
                if not should_exclude(rel):
                    zf.write(src, rel)
                    total_files += 1
                    print(f"  + Added file: {rel}")
            elif src.is_dir():
                for root, dirs, files in os.walk(src):
                    for f in files:
                        full_p = Path(root) / f
                        rel_p = full_p.relative_to(BASE_DIR).as_posix()
                        if not should_exclude(rel_p):
                            zf.write(full_p, rel_p)
                            total_files += 1

        # 2. Add checkpoint files
        for ckpt in CHECKPOINT_FILES:
            full_p = BASE_DIR / ckpt
            if full_p.exists():
                rel_p = full_p.relative_to(BASE_DIR).as_posix()
                zf.write(full_p, rel_p)
                total_files += 1
                sz_mb = full_p.stat().st_size / (1024 * 1024)
                print(f"  + Added Checkpoint: {rel_p} ({sz_mb:.1f} MB)")
            else:
                print(f"  [ERROR] Checkpoint missing: {ckpt}")

        # 3. Add SFT datasets
        for df in DATA_FILES:
            full_p = BASE_DIR / df
            if full_p.exists():
                rel_p = full_p.relative_to(BASE_DIR).as_posix()
                zf.write(full_p, rel_p)
                total_files += 1
                sz_mb = full_p.stat().st_size / (1024 * 1024)
                print(f"  + Added Dataset: {rel_p} ({sz_mb:.1f} MB)")
            else:
                print(f"  [WARN] Data file missing: {df}")

    # Atomic replace
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    temp_zip.rename(OUT_ZIP)

    zip_size_gb = OUT_ZIP.stat().st_size / (1024 * 1024 * 1024)
    print(f"\n[DONE] Package created successfully!")
    print(f"  Location : {OUT_ZIP}")
    print(f"  Total files : {total_files}")
    print(f"  Archive Size: {zip_size_gb:.2f} GB ({OUT_ZIP.stat().st_size / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    main()
