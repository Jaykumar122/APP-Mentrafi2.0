"""
MentraFiAI: Upload Best Checkpoints to Hugging Face
Uploads ONLY the 3 best networks + tokenizer to singh0123/MentraFiAI-TriNetwork.
"""

import sys
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "singh0123/MentraFiAI-TriNetwork"
ROOT_DIR = Path(__file__).resolve().parent

FILES_TO_UPLOAD = [
    # Network 1: Intent & Risk Classifier (10.65M)
    ("checkpoints/classifier/best_classifier.pt", "checkpoints/classifier/best_classifier.pt"),
    # Network 2: Query Parser & Entity Slot Filler (15.26M)
    ("checkpoints/query_parser/best_parser_net.pt", "checkpoints/query_parser/best_parser_net.pt"),
    # Network 3: Generative Financial SLM (102.5M)
    ("checkpoints/finetune_v5/best_model.pt", "checkpoints/finetune_v5/best_model.pt"),
    # Tokenizer & Architecture Config
    ("tokenizer/tokenizer_v2.model", "tokenizer/tokenizer_v2.model"),
    ("tokenizer/tokenizer_v2.vocab", "tokenizer/tokenizer_v2.vocab"),
    ("configs/model_config.yaml", "configs/model_config.yaml"),
]

def main():
    api = HfApi()
    print(f"Target Hugging Face Repository: https://huggingface.co/{REPO_ID}")
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note on repo creation: {e}")

    for local_rel, repo_rel in FILES_TO_UPLOAD:
        local_path = ROOT_DIR / local_rel
        if not local_path.exists():
            print(f"⚠️ Skipping missing file: {local_path}")
            continue

        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"Uploading {local_rel} ({size_mb:.1f} MB) -> {repo_rel} ...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_rel,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"✅ Uploaded {local_rel}")

    print("\n" + "=" * 60)
    print("🎉 All 3 Best Checkpoints Uploaded Successfully!")
    print(f"Repository URL: https://huggingface.co/{REPO_ID}")
    print("=" * 60)

if __name__ == "__main__":
    main()
