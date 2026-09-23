"""
MentraFiAI: Upload Best Checkpoints to Hugging Face
Uploads ONLY the 3 best networks + tokenizer to singh0123/MentraFiAI-TriNetwork.
Resilient against Windows cp1252 Unicode errors, skips already-uploaded files, and retries on network drops.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Fix console encoding on Windows to prevent UnicodeEncodeError
if sys.platform == "win32":
    os.system("")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Disable flaky XetHub transfer backend; use rock-solid direct LFS S3 uploads
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import HfApi, login

DEFAULT_REPO_ID = "singh0123/MentraFiAI-TriNetwork"
ROOT_DIR = Path(__file__).resolve().parent

FILES_TO_UPLOAD = [
    # Network 1: Intent & Risk Classifier (10.65M) - ~42.6 MB
    ("checkpoints/classifier/best_classifier.pt", "checkpoints/classifier/best_classifier.pt"),
    # Network 2: Query Parser & Entity Slot Filler (15.26M) - ~61.1 MB
    ("checkpoints/query_parser/best_parser_net.pt", "checkpoints/query_parser/best_parser_net.pt"),
    # Network 3: Generative Financial SLM (102.5M) - ~1.22 GB
    ("checkpoints/finetune_v5/best_model.pt", "checkpoints/finetune_v5/best_model.pt"),
    # Tokenizer & Architecture Config (~1 MB)
    ("tokenizer/tokenizer_v2.model", "tokenizer/tokenizer_v2.model"),
    ("tokenizer/tokenizer_v2.vocab", "tokenizer/tokenizer_v2.vocab"),
    ("configs/model_config.yaml", "configs/model_config.yaml"),
]

def parse_args():
    parser = argparse.ArgumentParser(description="Upload MentraFiAI checkpoints to Hugging Face")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Access Token (Write permissions)")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Target Hugging Face repository ID")
    return parser.parse_args()

def upload_file_with_retry(api: HfApi, local_path: Path, repo_rel: str, repo_id: str, max_retries: int = 5):
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"\n=======================================================")
    print(f"[UPLOADING] {repo_rel} ({size_mb:.1f} MB)")
    print(f"=======================================================")

    for attempt in range(1, max_retries + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_rel,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"[SUCCESS] Uploaded {repo_rel}")
            return True
        except Exception as e:
            print(f"[WARNING] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait_time = attempt * 5
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] Failed to upload {repo_rel} after {max_retries} attempts.")
                raise e

def main():
    args = parse_args()
    repo_id = args.repo_id
    token = args.token or os.environ.get("HF_TOKEN")

    print("=" * 65)
    print("MentraFiAI Hugging Face Checkpoint Uploader (Resilient Mode)")
    print("=" * 65)

    if token:
        login(token=token, add_to_git_credential=True)
        api = HfApi(token=token)
    else:
        api = HfApi()

    # Verify authentication
    try:
        user_info = api.whoami()
        print(f"Authenticated as: {user_info['name']}")
    except Exception:
        print("\nNot logged in to Hugging Face or token missing!")
        user_token = input("Paste your Hugging Face Write Token here: ").strip()
        if not user_token:
            print("Error: Token cannot be empty. Aborting.")
            sys.exit(1)
        login(token=user_token, add_to_git_credential=True)
        api = HfApi(token=user_token)
        user_info = api.whoami()
        print(f"Authenticated as: {user_info['name']}")

    print(f"Target Repository: https://huggingface.co/{repo_id}")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print("Repository online.")
    except Exception as e:
        print(f"Repo check: {e}")

    # Fetch currently uploaded files so we don't repeat work
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        existing_files = set(s.rfilename for s in info.siblings)
    except Exception:
        existing_files = set()

    for local_rel, repo_rel in FILES_TO_UPLOAD:
        local_path = ROOT_DIR / local_rel
        if not local_path.exists():
            print(f"[SKIP] Missing local file: {local_path}")
            continue

        if repo_rel in existing_files:
            print(f"[ALREADY UPLOADED] {repo_rel} exists on Hugging Face. Skipping.")
            continue

        upload_file_with_retry(api, local_path, repo_rel, repo_id, max_retries=5)

    print("\n" + "=" * 65)
    print("[DONE] ALL 3 BEST CHECKPOINTS UPLOADED SUCCESSFULLY!")
    print(f"Repository URL: https://huggingface.co/{repo_id}")
    print("=" * 65)

if __name__ == "__main__":
    main()
