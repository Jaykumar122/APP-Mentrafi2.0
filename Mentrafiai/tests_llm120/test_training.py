"""Local CPU trainer integration tests; no downloads or GPU are required.

Run: python -m unittest discover -s tests_llm120 -p test_training.py -v
"""

from __future__ import annotations

import contextlib
import copy
import gc
import io
import json
import math
import os
import random
import shutil
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from llm120 import train
from llm120.benchmark import benchmark
from llm120.data import collate_sft
from llm120.prepare import prepare_dataset
from llm120.tokenization import IGNORE_INDEX, Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer" / "tokenizer_v2.model"


def training_config(root: Path, **overrides) -> train.TrainingConfig:
    values = dict(stage="pretrain", data_dir=str(root / "pretrain"), tokenizer=str(TOKENIZER),
                  output_dir=str(root / "output"), batch_size=1, gradient_accumulation_steps=2,
                  max_steps=2, warmup_steps=0, learning_rate=0.001, min_lr=0.0001,
                  precision="fp32", gradient_checkpointing=True, eval_interval=1,
                  eval_batches=2, save_interval=1, seed=42)
    return train.TrainingConfig(**{**values, **overrides})


def tiny_model_config(**overrides):
    values = dict(vocab_size=16000, context_length=64, num_layers=1, d_model=32,
                  num_heads=4, num_kv_heads=2, d_ff=64, dropout=0.0)
    return train.model_config_from_dict({**values, **overrides})


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def quiet_training(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return train.run_training(*args, **kwargs)


def load_payload(path: Path):
    # This must work without the unsafe weights_only=False fallback.
    return torch.load(path, map_location="cpu", weights_only=True)


class SimulatedScaler:
    """Exercise FP16 control flow on CPU, without pretending to measure CUDA."""

    def __init__(self):
        self.value = 1024.0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self, new_scale=None):
        if new_scale is not None:
            self.value = new_scale

    def get_scale(self):
        return self.value

    def state_dict(self):
        return {"scale": self.value}


class ConfigTests(unittest.TestCase):
    def test_training_defaults_and_strict_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = dict(stage="pretrain", data_dir="data", tokenizer="tok.model", output_dir="out")
            config = train.TrainingConfig.from_dict(base)
            self.assertEqual(config.batch_size * config.gradient_accumulation_steps * 1024, 65536)
            self.assertTrue(Path(config.data_dir).is_absolute())
            for update in ({"stage": "unknown"}, {"batch_size": True}, {"max_steps": 0},
                           {"warmup_steps": -1}, {"max_overflow_retries": -1},
                           {"precision": "tf32"}, {"gradient_checkpointing": "true"},
                           {"learning_rate": float("nan")}, {"grad_clip": 0},
                           {"min_lr": 1.0}, {"early_stopping": 3}, {"data_dir": ""}):
                with self.subTest(update=update), self.assertRaises(ValueError):
                    train.TrainingConfig.from_dict({**base, **update})
            with self.assertRaisesRegex(ValueError, "Missing"):
                train.TrainingConfig.from_dict({key: value for key, value in base.items() if key != "stage"})
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump({"training": base, "ignored": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown config"):
                train.load_config(path)

    def test_warmup_and_cosine_use_successful_update_count(self):
        config = training_config(Path.cwd(), max_steps=6, warmup_steps=2)
        self.assertAlmostEqual(train.learning_rate_at(0, config), config.learning_rate / 2)
        self.assertAlmostEqual(train.learning_rate_at(1, config), config.learning_rate)
        self.assertAlmostEqual(train.learning_rate_at(2, config), config.learning_rate)
        self.assertAlmostEqual(train.learning_rate_at(5, config), config.min_lr)
        self.assertAlmostEqual(train.learning_rate_at(100, config), config.min_lr)
        self.assertGreater(train.learning_rate_at(3, config), train.learning_rate_at(4, config))
        with self.assertRaises(ValueError):
            train.learning_rate_at(-1, config)

    def test_single_device_and_precision_validation(self):
        for value in ("2", "8", "0", "invalid"):
            with self.subTest(world_size=value), patch.dict(os.environ, WORLD_SIZE=value):
                with self.assertRaisesRegex(ValueError, "WORLD_SIZE"):
                    train.resolve_device("cpu")
        with patch.dict(os.environ, WORLD_SIZE="1"):
            self.assertEqual(str(train.resolve_device("cpu")), "cpu")
        self.assertEqual(train.resolve_precision("auto", torch.device("cpu")), "fp32")
        with self.assertRaisesRegex(ValueError, "CPU"):
            train.resolve_precision("fp16", torch.device("cpu"))
        with patch("torch.cuda.is_bf16_supported", return_value=False):
            with self.assertRaisesRegex(ValueError, "bf16"):
                train.resolve_precision("bf16", torch.device("cuda"))

    def test_sampler_generator_is_dedicated_and_restartable(self):
        dataset = [{"input_ids": torch.tensor([1, 6, 7]),
                    "labels": torch.tensor([IGNORE_INDEX, IGNORE_INDEX, 2])},
                   {"input_ids": torch.tensor([1, 8]), "labels": torch.tensor([8, 2])}]
        config = training_config(Path.cwd(), batch_size=2, gradient_accumulation_steps=3)
        generator = torch.Generator().manual_seed(99)
        state = generator.get_state()
        batches, count = train.sample_microbatches(dataset, config, generator, 0)
        self.assertEqual(count, sum(int(batch["labels"].ne(IGNORE_INDEX).sum()) for batch in batches))
        torch.rand(100)  # Global training/dropout RNG must not affect index selection.
        generator.set_state(state)
        repeated, repeated_count = train.sample_microbatches(dataset, config, generator, 0)
        self.assertEqual(count, repeated_count)
        for first, second in zip(batches, repeated):
            self.assertTrue(torch.equal(first["input_ids"], second["input_ids"]))

    def test_resume_validation_allows_only_explicit_overrides(self):
        config = train.RunConfig(SimpleNamespace(to_dict=lambda: {"context_length": 64}),
                                 training_config(Path.cwd()))
        identity = {"manifest_sha256": "test", "train_sha256": "train", "val_sha256": "val"}
        state = {"version": 1, "config": config.to_dict(), "data_identity": identity,
                 "precision": "fp32", "step": 1, "attempts": 2, "tokens_seen": 128,
                 "batches_seen": 2, "sampler_generator_state": torch.Generator().get_state(),
                 "rng": train.capture_rng(), "best_val_loss": 2.0, "last_val_loss": 2.0}
        payload = {"training_state": state, "metadata": {"stage": "pretrain"}, "optimizer": {}, "scaler": {}}
        relocated = replace(config, training=replace(config.training,
                            data_dir=str(Path.cwd() / "relocated" / "pretrain"),
                            tokenizer=str(Path.cwd() / "relocated" / "tokenizer.model")))
        self.assertIs(train.validate_resume(payload, relocated, identity, "fp32"), state)
        changed = replace(relocated, training=replace(relocated.training, max_steps=3, eval_interval=2,
                                                      save_interval=2, output_dir=str(Path.cwd() / "new")))
        self.assertIs(train.validate_resume(payload, changed, identity, "fp32"), state)
        changed = replace(config, training=replace(config.training, learning_rate=0.002))
        with self.assertRaisesRegex(ValueError, "meaningful"):
            train.validate_resume(payload, changed, identity, "fp32")
        for key in identity:
            with self.subTest(changed_hash=key), self.assertRaisesRegex(ValueError, "identity"):
                train.validate_resume(payload, relocated, {**identity, key: "changed"}, "fp32")
        with self.assertRaisesRegex(ValueError, "complete"):
            train.validate_resume({**payload, "training_state": {}}, config, identity, "fp32")
        with self.assertRaisesRegex(ValueError, "counter"):
            train.validate_resume({**payload, "training_state": {**state, "tokens_seen": -1}}, config, identity, "fp32")
        with self.assertRaisesRegex(ValueError, "stage"):
            train.validate_resume({**payload, "metadata": {"stage": "sft"}}, config, identity, "fp32")

    def test_evaluation_evenly_spans_dataset_without_consuming_rng(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen = []

            def forward(self, input_ids, labels):
                self.seen.extend(input_ids[:, 0].tolist())
                valid = labels.ne(IGNORE_INDEX)
                count = int(valid.sum())
                loss_sum = labels[valid].float().sum()
                return SimpleNamespace(num_targets=count, loss_sum=loss_sum, loss=loss_sum / count)

        cases = [(11, 2, 2, [0, 3, 6, 10]), (3, 2, 3, [0, 1, 2]),
                 (1, 2, 2, [0]), (10, 1, 1, [4])]
        for size, batch_size, eval_batches, expected in cases:
            with self.subTest(size=size, batch_size=batch_size, eval_batches=eval_batches):
                examples = [{"input_ids": torch.full((1 + index % 3,), index),
                             "labels": torch.full((1 + index % 3,), index + 1)}
                            for index in range(size)]
                config = training_config(Path.cwd(), batch_size=batch_size, eval_batches=eval_batches)
                model = RecordingModel()
                sampler = torch.Generator().manual_seed(99)
                sampler_state = sampler.get_state()
                python_state, torch_state = random.getstate(), torch.get_rng_state()
                expected_loss = (sum((index + 1) * (1 + index % 3) for index in expected)
                                 / sum(1 + index % 3 for index in expected))
                for was_training in (True, False):
                    model.train(was_training)
                    model.seen.clear()
                    measured = train.evaluate(model, examples, config, torch.device("cpu"), "fp32", 0)
                    self.assertEqual(model.seen, expected)
                    self.assertAlmostEqual(measured, expected_loss)
                    self.assertEqual(model.training, was_training)
                self.assertEqual(random.getstate(), python_state)
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
                self.assertTrue(torch.equal(sampler.get_state(), sampler_state))
        model = RecordingModel()
        with self.assertRaisesRegex(ValueError, "no supervised targets"):
            train.evaluate(model, [], training_config(Path.cwd()), torch.device("cpu"), "fp32", 0)
        self.assertTrue(model.training)

    def test_nonfinite_loss_is_corruption_not_sanitized(self):
        with self.assertRaises(FloatingPointError):
            train.checked_loss_sum(SimpleNamespace(loss=torch.tensor(float("nan")),
                                                   loss_sum=torch.tensor(float("nan")), num_targets=1), 1)
        with self.assertRaisesRegex(ValueError, "num_targets"):
            train.checked_loss_sum(SimpleNamespace(num_targets=2), 1)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(prefix="llm120-training-")
        cls.root = Path(cls.temporary.name)
        cls.tokenizer = Tokenizer(TOKENIZER)
        source = write_jsonl(cls.root / "pretrain_source.jsonl", [
            {"text": f"Training document number {number}. Hello world. The assistant explains good answers."}
            for number in range(100)])
        prepare_dataset([source], cls.root / "pretrain", cls.tokenizer,
                        mode="pretrain", val_fraction=0.3, context_length=64)
        source = write_jsonl(cls.root / "sft_source.jsonl", [
            {"user": f"Hi {number}?", "assistant": "Hello!" if number % 2 else "Hello world!"}
            for number in range(100)])
        prepare_dataset([source], cls.root / "sft", cls.tokenizer,
                        mode="sft", val_fraction=0.3, context_length=64)

    @classmethod
    def tearDownClass(cls):
        # Mock call histories/exception tracebacks can retain mmap-backed
        # datasets in cycles; Windows cannot unlink them until GC runs.
        gc.collect()
        try:
            cls.temporary.cleanup()
        finally:
            torch.set_num_threads(cls.old_threads)

    def config(self, name: str, **overrides):
        return train.RunConfig(tiny_model_config(), training_config(
            self.root, output_dir=str(self.root / name), **overrides))

    def test_presets_and_model_only_yaml_are_strict(self):
        for filename, stage in (("llm120_pretrain.yaml", "pretrain"), ("llm120_sft.yaml", "sft")):
            config = train.load_config(ROOT / "configs" / filename)
            self.assertEqual(config.training.stage, stage)
            self.tokenizer.validate_config(config.model)
            self.assertEqual(config.model.context_length, 1024)
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            train.model_config_from_dict({"made_up_dimension": 7})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model.yaml").write_text(yaml.safe_dump(tiny_model_config().to_dict()), encoding="utf-8")
            (path / "train.yaml").write_text(yaml.safe_dump({"training": asdict(training_config(self.root))}), encoding="utf-8")
            config = train.load_config(path / "train.yaml", model_config=path / "model.yaml", max_steps=3)
            self.assertEqual(config.model.d_model, 32)
            self.assertEqual(config.training.max_steps, 3)
            (path / "train.yaml").write_text(yaml.safe_dump({"model": {}, "training": asdict(config.training)}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "either"):
                train.load_config(path / "train.yaml", model_config=path / "model.yaml")

    def test_cpu_pretrain_to_sft_and_full_resume(self):
        pretrain = self.config("pretrain_run")
        state = quiet_training(pretrain, device="cpu")
        self.assertEqual(state["step"], 2)
        self.assertEqual(state["attempts"], 2)
        self.assertEqual(state["batches_seen"], 4)
        self.assertEqual(state["tokens_seen"], 2 * 2 * 64)
        checkpoint = Path(pretrain.training.output_dir) / "latest.pt"
        payload = load_payload(checkpoint)
        self.assertEqual(payload["metadata"]["stage"], "pretrain")
        self.assertIsInstance(payload["training_state"]["rng"]["python"], tuple)
        self.assertTrue((checkpoint.parent / "best.pt").is_file())
        self.assertTrue(math.isfinite(payload["metadata"]["val_loss"]))
        sft = self.config("sft_run", stage="sft", data_dir=str(self.root / "sft"), max_steps=1)
        with self.assertRaisesRegex(ValueError, "SFT requires"):
            quiet_training(sft, device="cpu")
        state = quiet_training(sft, init=checkpoint, device="cpu")
        self.assertEqual(state["step"], 1)
        self.assertEqual(state["attempts"], 1)  # Init did NOT inherit pretrain counters.
        self.assertLess(state["tokens_seen"], 128)  # User/padding tokens are not supervised.
        sft_checkpoint = Path(sft.training.output_dir) / "latest.pt"
        resumed = replace(sft, training=replace(sft.training, max_steps=2))
        state2 = quiet_training(resumed, resume=sft_checkpoint, device="cpu")
        self.assertEqual(state2["step"], 2)
        self.assertEqual(state2["attempts"], 2)
        self.assertGreater(state2["tokens_seen"], state["tokens_seen"])
        optimizer = load_payload(sft_checkpoint)["optimizer"]
        steps = [int(value["step"].item()) for value in optimizer["state"].values()]
        self.assertTrue(steps and all(step == 2 for step in steps))
        with self.assertRaisesRegex(ValueError, "stage=pretrain"):
            quiet_training(self.config("bad_init"), init=sft_checkpoint, device="cpu")
        changed = replace(resumed, training=replace(resumed.training, batch_size=2))
        with self.assertRaisesRegex(ValueError, "meaningful"):
            quiet_training(changed, resume=sft_checkpoint, device="cpu")
        changed = replace(resumed, model=replace(resumed.model, d_ff=96))
        with self.assertRaisesRegex(ValueError, "model config"):
            quiet_training(changed, resume=sft_checkpoint, device="cpu")

    def test_interrupted_update_preserves_snapshot_and_exact_cpu_restart(self):
        # Same schedule on both branches, with dropout: test the RNG, optimizer,
        # sampler and counters, not just that resume can deserialize a file.
        baseline = replace(self.config("baseline"), model=replace(tiny_model_config(), dropout=0.15))
        full_state = quiet_training(baseline, device="cpu")
        interrupted = replace(baseline, training=replace(baseline.training, output_dir=str(self.root / "interrupted")))
        original = train.sample_microbatches
        calls = 0

        def stop_second_update(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                # Simulate sampling/global RNG already advanced before interrupt.
                original(*args, **kwargs)
                random.random()
                torch.rand(9)
                raise KeyboardInterrupt
            return original(*args, **kwargs)

        with patch.object(train, "sample_microbatches", side_effect=stop_second_update):
            with self.assertRaises(KeyboardInterrupt):
                quiet_training(interrupted, device="cpu")
        path = Path(interrupted.training.output_dir) / "latest.pt"
        self.assertEqual(load_payload(path)["training_state"]["step"], 1)
        relocated_root = self.root / "relocated"
        shutil.copytree(self.root / "pretrain", relocated_root / "pretrain")
        relocated_tokenizer = relocated_root / "tokenizer.model"
        shutil.copyfile(TOKENIZER, relocated_tokenizer)
        relocated = replace(interrupted, training=replace(interrupted.training,
                            data_dir=str(relocated_root / "pretrain"), tokenizer=str(relocated_tokenizer)))
        restarted = quiet_training(relocated, resume=path, device="cpu")
        self.assertEqual(restarted["step"], full_state["step"])
        self.assertEqual(restarted["attempts"], full_state["attempts"])
        self.assertEqual(restarted["tokens_seen"], full_state["tokens_seen"])
        first = load_payload(Path(baseline.training.output_dir) / "latest.pt")
        second = load_payload(path)
        self.assertEqual(second["training_state"]["config"], relocated.to_dict())
        self.assertEqual(first["training_state"]["data_identity"], second["training_state"]["data_identity"])
        for name, tensor in first["model_state"].items():
            self.assertTrue(torch.equal(tensor, second["model_state"][name]), name)
        self.assertTrue(torch.equal(first["training_state"]["sampler_generator_state"],
                                    second["training_state"]["sampler_generator_state"]))
        self.assertTrue(torch.equal(first["training_state"]["rng"]["torch"],
                                    second["training_state"]["rng"]["torch"]))

    def test_token_weighted_gradient_accumulation_and_evaluation(self):
        from llm120.model import CausalLM

        torch.manual_seed(17)
        model = CausalLM(tiny_model_config())
        combined_model = copy.deepcopy(model)
        examples = [{"input_ids": torch.tensor([1, 6, 7]),
                     "labels": torch.tensor([IGNORE_INDEX, IGNORE_INDEX, 2])},
                    {"input_ids": torch.tensor([1, 8, 9, 10, 11]),
                     "labels": torch.tensor([8, 9, 10, 11, 2])}]
        total = 6
        expected_sum = 0.0
        for example in examples:
            output = model(**collate_sft([example], 0))
            expected_sum += float(output.loss_sum.detach())
            (output.loss_sum / total).backward()
        output = combined_model(**collate_sft(examples, 0))
        self.assertEqual(output.num_targets, total)
        output.loss.backward()
        for (name, p), (_, q) in zip(model.named_parameters(), combined_model.named_parameters()):
            self.assertTrue(torch.allclose(p.grad, q.grad, atol=2e-6, rtol=2e-5), name)
        config = training_config(self.root, batch_size=1, eval_batches=2)
        measured = train.evaluate(model, examples, config, torch.device("cpu"), "fp32", 0)
        self.assertAlmostEqual(measured, expected_sum / total, places=6)
        self.assertTrue(model.training)

    def test_simulated_amp_overflow_retries_without_advancing_step_lr_or_sampler(self):
        config = self.config("overflow")
        scaler = SimulatedScaler()
        original_clip = torch.nn.utils.clip_grad_norm_
        calls = 0

        def overflow_once(parameters, maximum):
            nonlocal calls
            calls += 1
            if calls == 1:
                return torch.tensor(float("inf"))
            return original_clip(parameters, maximum)

        with patch.object(train, "resolve_precision", return_value="fp16"), \
             patch.object(train, "autocast_context", side_effect=lambda *args: contextlib.nullcontext()), \
             patch.object(train, "make_scaler", return_value=scaler), \
             patch.object(train, "learning_rate_at", wraps=train.learning_rate_at) as schedule, \
             patch("torch.nn.utils.clip_grad_norm_", side_effect=overflow_once):
            state = quiet_training(config, device="cpu")
        self.assertEqual(state["step"], 2)
        self.assertEqual(state["attempts"], 3)
        self.assertEqual(state["batches_seen"], 4)
        self.assertEqual(state["tokens_seen"], 256)
        self.assertEqual(schedule.call_count, 2)
        self.assertEqual(scaler.value, 512.0)
        sampler = torch.Generator().manual_seed(config.training.seed)
        train_data, _, _ = train.load_datasets(config, self.tokenizer)
        for _ in range(2):
            train.sample_microbatches(train_data, config.training, sampler, 0)
        self.assertTrue(torch.equal(sampler.get_state(), state["sampler_generator_state"]))

    def test_overflow_limit_and_fp32_nonfinite_keep_last_safe_snapshot(self):
        for simulated_amp in (False, True):
            config = self.config(f"bad_grads_{simulated_amp}", max_overflow_retries=1)
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch("torch.nn.utils.clip_grad_norm_", return_value=torch.tensor(float("inf"))))
                if simulated_amp:
                    stack.enter_context(patch.object(train, "resolve_precision", return_value="fp16"))
                    stack.enter_context(patch.object(train, "autocast_context", side_effect=lambda *args: contextlib.nullcontext()))
                    stack.enter_context(patch.object(train, "make_scaler", return_value=SimulatedScaler()))
                with self.assertRaises(FloatingPointError):
                    quiet_training(config, device="cpu")
            payload = load_payload(Path(config.training.output_dir) / "latest.pt")
            self.assertEqual(payload["training_state"]["step"], 0)
            self.assertEqual(payload["training_state"]["attempts"], 0)

    def test_manifest_hashes_stage_and_no_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.root / "pretrain"
            for filename in ("manifest.json", "train.bin", "val.bin"):
                (root / filename).write_bytes((source / filename).read_bytes())
            config = self.config("unused", data_dir=str(root))
            train.load_datasets(config, self.tokenizer)
            path = root / "train.bin"
            contents = bytearray(path.read_bytes())
            contents[0] ^= 1
            path.write_bytes(contents)
            with self.assertRaisesRegex(ValueError, "sha256"):
                train.load_datasets(config, self.tokenizer)
            (root / "manifest.json").unlink()
            with self.assertRaises(FileNotFoundError):
                train.load_datasets(config, self.tokenizer)

    def test_cpu_synthetic_benchmark_and_help(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "tiny.yaml"
            config_path.write_text(yaml.safe_dump(tiny_model_config().to_dict()), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                result = benchmark(model_config=config_path, device="cpu", steps=2, warmup_steps=1,
                                   batch_size=1, context_length=8)
            self.assertGreater(result["tokens_per_second"], 0)
            self.assertFalse(result["fit_guarantee"])
            self.assertIsNone(result["cuda_peak_allocated_mib"])
            with self.assertRaisesRegex(ValueError, "context_length"):
                benchmark(model_config=config_path, device="cpu", steps=1, context_length=65)
        from llm120.benchmark import main as benchmark_main
        for main in (train.main, benchmark_main):
            with contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as error:
                main(["--help"])
            self.assertEqual(error.exception.code, 0)
            self.assertIn("--device", output.getvalue())


if __name__ == "__main__":
    unittest.main()
