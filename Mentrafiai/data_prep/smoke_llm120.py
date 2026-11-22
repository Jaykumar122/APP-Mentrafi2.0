"""Exercise real 120M pretrain -> SFT on configured data in a temporary directory.

This uses batch=1 and accumulation=2 for TWO updates per stage. It is a pipeline
smoke test, not a speed/quality assessment or a guarantee the full preset fits.
Temporary checkpoints are deleted; allow several GB of free temporary disk/RAM.
Run from Mentrafiai: python -m data_prep.smoke_llm120 --device cuda
"""
import argparse
import gc
import json
import platform
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from llm120.train import RunConfig, load_config, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "device": args.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      "scope": "temporary reduced-microbatch pipeline test, not quality or full-preset fit"}))
    with tempfile.TemporaryDirectory(prefix="llm120-smoke-") as tmp:
        for stage in ("pretrain", "sft"):
            config = load_config(f"configs/llm120_{stage}.yaml")
            training = replace(config.training, output_dir=str(Path(tmp) / stage), batch_size=1,
                               gradient_accumulation_steps=2, max_steps=2, warmup_steps=1,
                               eval_interval=2, eval_batches=2, save_interval=2)
            run_training(RunConfig(config.model, training), device=args.device,
                         init=Path(tmp) / "pretrain/best.pt" if stage == "sft" else None)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(json.dumps({"pipeline_smoke_passed": True, "stages": ["pretrain", "sft"],
                      "parameters": 120054528, "temporary_checkpoints_removed": True}))


if __name__ == "__main__":
    main()
