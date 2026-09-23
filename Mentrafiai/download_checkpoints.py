"""
MentraFiAI: Download Best Checkpoints
Downloads the 3 networks directly into Mentrafiai/checkpoints/ from GitHub Releases or Hugging Face.
Zero external dependencies required (uses built-in Python urllib with Hugging Face fallback).
"""

import sys
import argparse
import urllib.request
from pathlib import Path

# Fix console encoding on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent

GITHUB_BASE_URL = "https://github.com/Jaykumar122/APP-Mentrafi2.0/releases/download/v1.0.0"
HF_REPO_ID = "singh0123/MentraFiAI-TriNetwork"

ASSETS = [
    # (Remote file name in GitHub release, local target path, HF relative path)
    ("best_classifier.pt", "checkpoints/classifier/best_classifier.pt", "checkpoints/classifier/best_classifier.pt"),
    ("best_parser_net.pt", "checkpoints/query_parser/best_parser_net.pt", "checkpoints/query_parser/best_parser_net.pt"),
    ("best_model.pt", "checkpoints/finetune_v5/best_model.pt", "checkpoints/finetune_v5/best_model.pt"),
    ("tokenizer_v2.model", "tokenizer/tokenizer_v2.model", "tokenizer/tokenizer_v2.model"),
    ("tokenizer_v2.vocab", "tokenizer/tokenizer_v2.vocab", "tokenizer/tokenizer_v2.vocab"),
    ("model_config.yaml", "configs/model_config.yaml", "configs/model_config.yaml"),
]

def reporthook(block_num, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    percent = min(100.0, (downloaded / total_size) * 100.0)
    mb_down = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    sys.stdout.write(f"\r  Progress: {percent:5.1f}% [{mb_down:6.1f} MB / {mb_total:6.1f} MB]")
    sys.stdout.flush()

def download_from_github(remote_name: str, local_path: Path) -> bool:
    url = f"{GITHUB_BASE_URL}/{remote_name}"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from GitHub Releases: {remote_name} ...")
    try:
        urllib.request.urlretrieve(url, str(local_path), reporthook=reporthook)
        print(f"\n[OK] Saved to: {local_path.relative_to(ROOT_DIR)}")
        return True
    except Exception as e:
        print(f"\n[FAIL] GitHub download error for {remote_name}: {e}")
        return False

def download_from_hf(hf_rel: str, local_path: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading from Hugging Face: {hf_rel} ...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_rel,
            local_dir=str(ROOT_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"[OK] Saved to: {local_path.relative_to(ROOT_DIR)}")
        return True
    except Exception as e:
        print(f"[FAIL] Hugging Face download error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download MentraFiAI Tri-Network models")
    parser.add_argument("--source", choices=["github", "hf"], default="github", help="Download source")
    args = parser.parse_args()

    print("=" * 65)
    print("MentraFiAI Tri-Network Model Downloader")
    print(f"Source: {args.source.upper()}")
    print("=" * 65)

    for remote_name, local_rel, hf_rel in ASSETS:
        local_path = ROOT_DIR / local_rel
        if local_path.exists() and local_path.stat().st_size > 1000:
            print(f"[EXISTS] {local_rel} already exists. Skipping.")
            continue

        success = False
        if args.source == "github":
            success = download_from_github(remote_name, local_path)
            if not success:
                print("Falling back to Hugging Face...")
                success = download_from_hf(hf_rel, local_path)
        else:
            success = download_from_hf(hf_rel, local_path)
            if not success:
                print("Falling back to GitHub Releases...")
                success = download_from_github(remote_name, local_path)

        if not success:
            print(f"[ERROR] Could not download {local_rel}")

    print("\n" + "=" * 65)
    print("All 3 Networks Ready!")
    print("Run the server with: python serve.py")
    print("=" * 65)

if __name__ == "__main__":
    main()
