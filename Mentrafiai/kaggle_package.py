"""Build the two folders to upload as Kaggle Datasets (run from Mentrafiai/).

    python kaggle_package.py

kaggle_upload/mentrafi-llm120-code/   small: llm120/, configs, tokenizer, launcher
kaggle_upload/mentrafi-llm120-data/   llm120_pretrain/ (~2.4 GB) and llm120_sft/

Data files are hard-linked when possible (no extra disk use on the same drive).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "kaggle_upload"
CODE_FILES = ["kaggle_train.py", "requirements-train.txt", "tokenizer/tokenizer_v2.model",
              "configs/kaggle_2xt4_pretrain.yaml", "configs/kaggle_2xt4_sft.yaml",
              "configs/llm120_pretrain.yaml", "configs/llm120_sft.yaml"]
DATA_DIRS = ["data/llm120_pretrain", "data/llm120_sft"]


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    code = OUT / "mentrafi-llm120-code"
    data = OUT / "mentrafi-llm120-data"
    if code.exists():
        shutil.rmtree(code)
    shutil.copytree(ROOT / "llm120", code / "llm120", ignore=shutil.ignore_patterns("__pycache__"))
    for name in CODE_FILES:
        link_or_copy(ROOT / name, code / name)
    for name in DATA_DIRS:
        source = ROOT / name
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        files = ["manifest.json"] + [info["file"] for info in manifest["splits"].values()]
        for file in files:
            link_or_copy(source / file, data / source.name / file)
    for folder in (code, data):
        size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        print(f"{folder}  ({size / 2**20:,.1f} MiB)")
    print("Upload each folder as a Kaggle Dataset (kaggle.com -> Datasets -> New Dataset, or "
          "`kaggle datasets create -p <folder> --dir-mode zip`). See KAGGLE.md.")


if __name__ == "__main__":
    main()
