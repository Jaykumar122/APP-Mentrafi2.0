"""Multi-process (torchrun, gloo/CPU) training equivalence and Kaggle time-limit resume.

A 2-process DDP run must produce the same weights as a 1-process run, because
both consume the same microbatches per update from one shared sampler.
Run: python -m pytest tests_llm120/test_distributed.py -q
"""

from __future__ import annotations

import gc
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from llm120.prepare import prepare_dataset
from llm120.tokenization import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer" / "tokenizer_v2.model"
MODEL = dict(vocab_size=16000, context_length=64, num_layers=2, d_model=32, num_heads=4,
             num_kv_heads=2, d_ff=64, dropout=0.0)


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class DistributedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="llm120-ddp-")
        cls.root = Path(cls.temporary.name)
        tokenizer = Tokenizer(TOKENIZER)
        source = write_jsonl(cls.root / "pretrain.jsonl", [
            {"text": f"Document {n}. A SIP invests a fixed amount every month. Number {n * 7}."}
            for n in range(120)])
        prepare_dataset([source], cls.root / "pretrain", tokenizer, mode="pretrain",
                        val_fraction=0.2, context_length=64)
        source = write_jsonl(cls.root / "sft.jsonl", [
            {"user": f"Question {n}?", "assistant": "Short." if n % 3 else "A much longer answer with more tokens."}
            for n in range(80)])
        prepare_dataset([source], cls.root / "sft", tokenizer, mode="sft", val_fraction=0.25, context_length=64)

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        cls.temporary.cleanup()

    def config(self, stage: str, name: str, **training) -> Path:
        values = dict(stage=stage, data_dir=str(self.root / stage), tokenizer=str(TOKENIZER),
                      output_dir=str(self.root / name), batch_size=2, gradient_accumulation_steps=4,
                      max_steps=4, warmup_steps=1, learning_rate=0.003, min_lr=0.0003, precision="fp32",
                      gradient_checkpointing=True, eval_interval=2, eval_batches=3, save_interval=2, seed=7)
        values.update(training)
        path = self.root / f"{name}.yaml"
        path.write_text(yaml.safe_dump({"model": MODEL, "training": values}), encoding="utf-8")
        return path

    @staticmethod
    def launch(config: Path, *extra: str, processes: int = 1):
        """Start ranks with the same env variables torchrun sets (torchrun's own
        rendezvous needs libuv, which Windows PyTorch builds lack)."""
        base = {**os.environ, "PYTHONPATH": str(ROOT), "USE_LIBUV": "0", "OMP_NUM_THREADS": "1"}
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            base.pop(key, None)
        command = [sys.executable, "-m", "llm120.train", "--config", str(config), "--device", "cpu", *extra]
        if processes == 1:
            return [subprocess.Popen(command, cwd=ROOT, env=base, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        procs = []
        for rank in range(processes):
            env = {**base, "WORLD_SIZE": str(processes), "RANK": str(rank), "LOCAL_RANK": str(rank),
                   "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)}
            procs.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True))
        return procs

    @staticmethod
    def collect(procs):
        results = []
        for proc in procs:
            try:
                stdout, stderr = proc.communicate(timeout=600)
            except subprocess.TimeoutExpired:
                for other in procs:
                    other.kill()
                raise
            results.append((proc.returncode, stdout, stderr))
        return results

    def run_train(self, config: Path, *extra: str, processes: int = 1) -> str:
        results = self.collect(self.launch(config, *extra, processes=processes))
        for rank, (code, stdout, stderr) in enumerate(results):
            if code:
                self.fail(f"Rank {rank} failed ({code}):\n{stdout[-3000:]}\n{stderr[-5000:]}")
        for _, stdout, _ in results[1:]:
            self.assertEqual(stdout.strip(), "", "only rank 0 should print")
        return results[0][1]

    @staticmethod
    def payload(path: Path):
        return torch.load(path, map_location="cpu", weights_only=True)

    def assert_same_weights(self, left: Path, right: Path):
        a, b = self.payload(left), self.payload(right)
        self.assertEqual(a["training_state"]["step"], b["training_state"]["step"])
        self.assertEqual(a["training_state"]["tokens_seen"], b["training_state"]["tokens_seen"])
        self.assertEqual(a["training_state"]["batches_seen"], b["training_state"]["batches_seen"])
        for name, tensor in a["model_state"].items():
            torch.testing.assert_close(tensor, b["model_state"][name], rtol=1e-4, atol=1e-5, msg=name)
        self.assertAlmostEqual(a["training_state"]["best_val_loss"], b["training_state"]["best_val_loss"], places=4)

    def test_two_process_pretrain_and_sft_match_single_process(self):
        self.run_train(self.config("pretrain", "pre1"))
        output = self.run_train(self.config("pretrain", "pre2"), processes=2)
        self.assertIn('"world_size": 2', output)
        self.assertEqual(output.count('"event": "start"'), 1, "only rank 0 should log")
        self.assert_same_weights(self.root / "pre1" / "latest.pt", self.root / "pre2" / "latest.pt")
        # SFT has variable assistant-token masks: checks global token weighting.
        init = str(self.root / "pre1" / "best.pt")
        self.run_train(self.config("sft", "sft1"), "--init", init)
        self.run_train(self.config("sft", "sft2"), "--init", init, processes=2)
        self.assert_same_weights(self.root / "sft1" / "latest.pt", self.root / "sft2" / "latest.pt")

    def test_time_limit_stop_then_resume_on_other_gpu_count_matches_uninterrupted(self):
        full = self.config("pretrain", "full")
        self.run_train(full)
        split = self.config("pretrain", "split")
        output = self.run_train(split, "--time-limit-hours", "0.0000001", processes=2)
        self.assertIn('"event": "time_limit"', output)
        stopped = self.payload(self.root / "split" / "latest.pt")["training_state"]["step"]
        self.assertEqual(stopped, 1)
        # Session 2 resumes on one device (e.g. Kaggle gave a single GPU).
        self.run_train(split, "--resume", str(self.root / "split" / "latest.pt"))
        self.assert_same_weights(self.root / "full" / "latest.pt", self.root / "split" / "latest.pt")

    def test_accumulation_must_divide_across_processes(self):
        config = self.config("pretrain", "odd", gradient_accumulation_steps=3)
        results = self.collect(self.launch(config, processes=2))
        for code, _, stderr in results:
            self.assertNotEqual(code, 0)
            self.assertIn("divisible", stderr)


if __name__ == "__main__":
    unittest.main()
