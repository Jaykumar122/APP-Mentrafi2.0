"""
MentraFiAI: Download Best Checkpoints from Hugging Face
Downloads the 3 networks directly into Mentrafiai/checkpoints/ so anyone can run the model.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "singh0123/MentraFiAI-TriNetwork"
ROOT_DIR = Path(__file__).resolve().parent

FILES = [
    # Network 1
    "checkpoints/classifier/best_classifier.pt",
    # Network 2
    "checkpoints/query_parser/best_parser_net.pt",
    # Network 3
    "checkpoints/finetune_v5/best_model.pt",
    # Tokenizer & Config
    "tokenizer/tokenizer_v2.model",
    "tokenizer/tokenizer_v2.vocab",
    "configs/model_config.yaml",
]

def main():
    print(f"Downloading MentraFiAI Tri-Network models from Hugging Face: {REPO_ID}...")
    for filename in FILES:
        print(f"Downloading {filename}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=str(ROOT_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"✅ Ready: {filename}")

    print("\n" + "=" * 60)
    print("🎉 All 3 Networks Downloaded Successfully!")
    print("You can now run: python serve.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
