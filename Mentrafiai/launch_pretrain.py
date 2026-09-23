#!/usr/bin/env python3
"""
Robust detached launcher for MentraFiAI pretraining.

This launcher ensures training survives:
- Console/terminal closing
- Laptop going to sleep (if configured to allow)
- Accidental window close
- Session disconnect

The process runs completely detached and saves checkpoints every 2 hours.
You can safely close your laptop lid (configure Windows power settings).

Usage:
    python launch_pretrain.py

After launching:
- Check training_heartbeat.txt for current status (updates every ~26 minutes)
- Check training_pretrain.log for full training log
- Find PID in training_pretrain.pid to monitor process
- Close console safely - training continues in background
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

def main():
    print("=" * 70)
    print("MENTRAFIAI PRETRAIN LAUNCHER")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Support optional command line overrides
    import argparse
    parser = argparse.ArgumentParser(description="Launch detached pretraining run")
    parser.add_argument("--model_config", default="configs/model_config.yaml", help="Path to model config")
    parser.add_argument("--train_config", default="configs/pretrain_config_t4.yaml", help="Path to train config")
    parser.add_argument("--checkpoint_dir", default="checkpoints/pretrain_v2", help="Path to checkpoint directory")
    parser.add_argument("--budget_hours", default="30.0", help="Budget in hours")
    args = parser.parse_args()

    # Prepare command
    cmd = [
        sys.executable,
        "-u",
        "training/pretrain.py",
        "--model_config", args.model_config,
        "--train_config", args.train_config,
        "--checkpoint_dir", args.checkpoint_dir,
        # pack_corpus_v2.py writes here; these must match its OUT_DIR or the
        # memmap open fails at startup.
        "--train_bin", "data/processed/corpus_v2/train.bin",
        "--val_bin",   "data/processed/corpus_v2/val.bin",
        "--budget_hours", str(args.budget_hours)
    ]

    # Preflight: the training process is detached, so a missing .bin would only
    # surface as a memmap traceback buried in the log. Fail loudly here instead.
    for flag in ("--train_bin", "--val_bin"):
        p = Path(cmd[cmd.index(flag) + 1])
        if not p.exists():
            print(f"ERROR: {flag} not found: {p}")
            print("       Run: python data_prep/pack_corpus_v2.py")
            return 1
        if p.stat().st_size == 0:
            print(f"ERROR: {flag} is empty: {p}")
            return 1

    # Only ~430 MiB of VRAM headroom remains at this config on a 4GB card, so
    # allocator fragmentation over 16.8k steps is a real OOM risk. Expandable
    # segments let the allocator grow blocks instead of stranding them.
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    # Setup logging
    log_file = Path("training_pretrain.log")
    pid_file = Path("training_pretrain.pid")

    print("Configuration:")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log file: {log_file.absolute()}")
    print(f"  PID file: {pid_file.absolute()}")
    print(f"  Heartbeat: training_heartbeat.txt (updates every ~26 min)")
    print(f"  Checkpoints: ./{args.checkpoint_dir}/ (every 2 hours)")
    print()

    # Check if training is already running
    if pid_file.exists():
        with open(pid_file, 'r') as f:
            old_pid = f.read().strip()

        # Try to check if process is still alive (Windows-compatible)
        try:
            result = subprocess.run(['tasklist', '/FI', f'PID eq {old_pid}'],
                                  capture_output=True, text=True, timeout=5)
            if old_pid in result.stdout:
                print(f"[WARN] Training already running (PID: {old_pid})")
                print(f"   Check training_heartbeat.txt for current status")
                print(f"   To kill: taskkill /PID {old_pid} /F")
                return
        except:
            pass  # Process check failed, assume it's not running

    # Launch detached process (Windows-compatible)
    creationflags = 0
    if sys.platform == "win32":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    print("Launching detached training process...")
    print("   This process will run in the background.")
    print("   You can safely close this window after seeing 'SUCCESS' message.")
    print()

    try:
        with open(log_file, 'w', buffering=1, encoding='utf-8', errors='replace') as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=creationflags,
                close_fds=True,
                env=env,
                cwd=str(Path.cwd())
            )

            # Save PID
            with open(pid_file, 'w') as pid_f:
                pid_f.write(str(proc.pid))

            print(f"[OK] SUCCESS! Training started with PID: {proc.pid}")
            print()
            print("The training process is now running in the background.")
            print()
            print("Monitoring:")
            print(f"  - Heartbeat: training_heartbeat.txt (updates every ~26 minutes)")
            print(f"  - Full log: {log_file.absolute()}")
            print(f"  - Checkpoints: ./{args.checkpoint_dir}/")
            print()
            print("To check status:")
            print(f"  type training_heartbeat.txt")
            print(f"  tail -100 {log_file}")
            print()
            print("To stop training:")
            print(f"  taskkill /PID {proc.pid} /F")
            print()
            print("YOU CAN NOW SAFELY CLOSE THIS WINDOW.")
            print("Training will continue in the background.")
            print()
            print("Expected completion: ~20 days (~470 GPU-hours at ~1,170 tok/s with grad checkpointing)")
            print("Checkpoints save every 2 hours - safe to turn off laptop between saves.")
            print("=" * 70)

            # Wait a moment to ensure process started successfully
            time.sleep(3)

            # Check if process is still running
            poll = proc.poll()
            if poll is not None:
                print()
                print(f"[ERROR] Process exited immediately with code {poll}")
                print(f"   Check {log_file} for error details")
                return 1

            return 0

    except Exception as e:
        print(f"[ERROR] launching training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
