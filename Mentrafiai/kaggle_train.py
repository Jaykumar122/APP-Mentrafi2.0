"""One-command llm120 training on Kaggle (GPU T4 x2, also works on 1 GPU).

    python kaggle_train.py --stage pretrain      # run again in each new session
    python kaggle_train.py --stage sft           # after pretraining finished

What it does:
  * finds the prepared dataset (manifest.json with the right mode) and
    tokenizer_v2.model anywhere under /kaggle/input,
  * resumes automatically: from /kaggle/working/checkpoints/llm120_<stage>/latest.pt,
    or from a previous session's output attached as an input (it copies
    latest.pt/best.pt into /kaggle/working first, because inputs are read-only),
  * SFT initializes from the pretraining best.pt (working dir or an input),
  * runs one process per GPU with torchrun (DDP) and stops cleanly before
    Kaggle's 12-hour session limit, writing latest.pt for the next session.

Outputs: /kaggle/working/checkpoints/llm120_<stage>/{latest,best}.pt and
/kaggle/working/train_<stage>.log
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")


def find_files(base: Path, name: str) -> list[Path]:
    if not base.exists():
        return []
    return sorted(path for path in base.rglob(name) if path.is_file())


def find_data_dir(stage: str) -> Path:
    candidates = []
    for base in (INPUT, ROOT / "data"):
        for manifest in find_files(base, "manifest.json"):
            try:
                mode = json.loads(manifest.read_text(encoding="utf-8")).get("mode")
            except (OSError, ValueError):
                continue
            expected = "train.bin" if stage == "pretrain" else "train.jsonl"
            if mode == stage and (manifest.parent / expected).exists():
                candidates.append(manifest.parent)
    if not candidates:
        raise SystemExit(f"No prepared {stage} dataset found. Upload data/llm120_{stage}/ (with manifest.json) "
                         "as a Kaggle Dataset and add it to this notebook, or pass --data-dir.")
    if len(candidates) > 1:
        print(f"Several {stage} datasets found, using the first: {[str(c) for c in candidates]}")
    return candidates[0]


def find_tokenizer() -> Path:
    for base in (ROOT / "tokenizer", INPUT):
        found = find_files(base, "tokenizer_v2.model")
        if found:
            return found[0]
    raise SystemExit("tokenizer_v2.model not found; add the tokenizer folder or pass --tokenizer")


def previous_checkpoint(stage: str, name: str) -> Path | None:
    """A checkpoint from an earlier session's output attached under /kaggle/input."""
    matches = [path for path in find_files(INPUT, name) if path.parent.name == f"llm120_{stage}"]
    if not matches:
        return None
    import torch  # deferred: only needed to pick the most advanced checkpoint

    def step(path: Path) -> int:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return int(payload["training_state"]["step"])

    return max(matches, key=step) if len(matches) > 1 else matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("pretrain", "sft"), required=True)
    parser.add_argument("--hours", type=float, default=11.0,
                        help="Stop and save after this many hours of training (Kaggle kills sessions at 12 h; "
                             "leave time for saving the notebook output)")
    parser.add_argument("--max-steps", type=int, help="Override max_steps (also replans the LR schedule)")
    parser.add_argument("--config", type=Path, help="Defaults to configs/kaggle_2xt4_<stage>.yaml")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--init", type=Path, help="SFT only: pretraining best.pt (auto-detected by default)")
    parser.add_argument("--gpus", type=int, help="Number of GPUs to use (default: all visible)")
    parser.add_argument("--fresh", action="store_true", help="Do not resume; output dir must be empty")
    args = parser.parse_args()

    import torch

    config = args.config or ROOT / "configs" / f"kaggle_2xt4_{args.stage}.yaml"
    data_dir = args.data_dir or find_data_dir(args.stage)
    tokenizer = args.tokenizer or find_tokenizer()
    work = WORKING if WORKING.exists() else ROOT
    output = args.output_dir or work / "checkpoints" / f"llm120_{args.stage}"
    gpus = args.gpus if args.gpus is not None else torch.cuda.device_count()
    if gpus < 1:
        raise SystemExit("No GPU visible. In Kaggle: Settings -> Accelerator -> GPU T4 x2.")

    source: list[str] = []
    latest = output / "latest.pt"
    if not args.fresh and not latest.exists():
        found = previous_checkpoint(args.stage, "latest.pt")
        if found is not None:
            output.mkdir(parents=True, exist_ok=True)
            print(f"Copying previous session checkpoint {found} -> {output}")
            shutil.copy2(found, latest)
            best = found.parent / "best.pt"
            if best.exists():
                shutil.copy2(best, output / "best.pt")
    if latest.exists() and not args.fresh:
        source = ["--resume", str(latest)]
        if not args.max_steps:
            # Keep the schedule of the run being resumed (e.g. an earlier --max-steps 12000).
            saved = torch.load(latest, map_location="cpu", weights_only=True)["training_state"]
            args.max_steps = int(saved["config"]["training"]["max_steps"])
            print(f"Resuming at step {saved['step']} of {args.max_steps}")
            del saved
    elif args.stage == "sft":
        init = args.init
        if init is None:
            local = work / "checkpoints" / "llm120_pretrain" / "best.pt"
            init = local if local.exists() else previous_checkpoint("pretrain", "best.pt")
        if init is None:
            raise SystemExit("SFT needs the pretraining best.pt: attach the pretraining notebook output "
                             "as an input, or pass --init.")
        source = ["--init", str(init)]

    train_args = ["--config", str(config), "--data-dir", str(data_dir), "--tokenizer", str(tokenizer),
                  "--output-dir", str(output), "--time-limit-hours", str(args.hours), *source]
    if args.max_steps:
        train_args += ["--max-steps", str(args.max_steps)]
    if gpus > 1:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={gpus}", "-m", "llm120.train", *train_args]
    else:
        command = [sys.executable, "-m", "llm120.train", "--device", "cuda", *train_args]

    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
           "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"}
    log_path = work / f"train_{args.stage}.log"
    print(f"GPUs: {gpus} ({', '.join(torch.cuda.get_device_name(i) for i in range(gpus))})")
    print("Running:", " ".join(command))
    print(f"Log: {log_path}")
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    if code:
        raise SystemExit(f"Training exited with code {code}; see {log_path}. "
                         f"latest.pt in {output} is the last clean restart point.")
    import torch as _torch

    state = _torch.load(latest, map_location="cpu", weights_only=True)["training_state"]
    finished = state["step"] >= state["config"]["training"]["max_steps"]
    if not finished:
        print(f"Stopped at step {state['step']} (time limit). Next session: attach this notebook's output as an "
              f"input and run `python kaggle_train.py --stage {args.stage}` again; it resumes automatically.")
        return
    print(f"{args.stage} finished at step {state['step']}. Checkpoints in {output}.")
    if args.stage == "sft":
        from llm120.checkpoint import export_weights

        serving = export_weights(output / "best.pt", work / "llm120_sft_best_weights.pt")
        print(f"Serving weights: {serving} -> download it and save as Mentrafiai/checkpoints/llm120_sft/best.pt")
    else:
        print("Next: python kaggle_train.py --stage sft")


if __name__ == "__main__":
    main()
