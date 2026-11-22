"""Tiny CPU model/checkpoint regressions; no downloads or disk fixtures."""

from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import create_autospec, patch

import torch
from torch.nn import functional as F

from llm120.checkpoint import FORMAT_VERSION, load_checkpoint
from llm120.config import ModelConfig
from llm120.model import CausalLM
from llm120.tokenization import IGNORE_INDEX, TEMPLATE_VERSION, Tokenizer


def tiny_config():
    return ModelConfig(vocab_size=64, context_length=96, num_layers=3, d_model=32,
                       num_heads=4, num_kv_heads=2, d_ff=64, dropout=0.0)


class SmallModelCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def setUp(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(17)


class ModelTests(SmallModelCase):
    @torch.no_grad()
    def test_future_tokens_cannot_change_prefix_logits(self):
        model = CausalLM(tiny_config()).eval()
        inputs = torch.randint(6, model.config.vocab_size, (2, 12))
        changed = inputs.clone()
        changed[:, 6:] = (changed[:, 6:] - 6 + 1) % (model.config.vocab_size - 6) + 6
        original_logits = model(inputs).logits
        changed_logits = model(changed).logits
        torch.testing.assert_close(original_logits[:, :6], changed_logits[:, :6], atol=1e-6, rtol=1e-5)
        self.assertFalse(torch.allclose(original_logits[:, 6:], changed_logits[:, 6:]))

    @torch.no_grad()
    def test_single_and_multi_token_caches_match_full_logits(self):
        model = CausalLM(tiny_config()).eval()
        inputs = torch.randint(6, model.config.vocab_size, (2, 12))
        expected = model(inputs).logits
        for lengths in ((1,) * 12, (4, 1, 3, 4)):
            with self.subTest(chunk_lengths=lengths):
                cache, pieces, offset = None, [], 0
                for length in lengths:
                    output = model(inputs[:, offset:offset + length], use_cache=True, past_key_values=cache)
                    cache = output.past_key_values
                    pieces.append(output.logits)
                    offset += length
                    self.assertEqual(len(cache), model.config.num_layers)
                    for key, value in cache:
                        shape = (2, model.config.num_kv_heads, offset, model.config.head_dim)
                        self.assertEqual(tuple(key.shape), shape)
                        self.assertEqual(tuple(value.shape), shape)
                torch.testing.assert_close(torch.cat(pieces, dim=1), expected, atol=1e-6, rtol=1e-5)

    def test_loss_uses_explicitly_aligned_labels_without_another_shift(self):
        model = CausalLM(tiny_config())
        sequence = torch.randint(6, model.config.vocab_size, (2, 71))
        inputs, labels = sequence[:, :-1], sequence[:, 1:].clone()
        labels[0, :7] = IGNORE_INDEX
        labels[1, 10:15] = IGNORE_INDEX
        labels[0, 9] = model.config.pad_id  # Only IGNORE_INDEX, not pad ID, masks a target.
        labels[:, -1] = model.config.eos_id
        output = model(inputs, labels=labels)
        # 140 positions exercise the model's 128-position CE chunk boundary.
        expected_sum = F.cross_entropy(output.logits.reshape(-1, model.config.vocab_size).float(),
                                       labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum")
        targets = int(labels.ne(IGNORE_INDEX).sum())
        self.assertEqual(output.num_targets, targets)
        torch.testing.assert_close(output.loss_sum, expected_sum)
        torch.testing.assert_close(output.loss, expected_sum / targets)
        actual_grad = torch.autograd.grad(output.loss_sum, output.logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected_sum, output.logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_multilayer_gradient_checkpointing_matches_with_dropout(self):
        config = replace(tiny_config(), dropout=0.15)
        plain = CausalLM(config).train()
        checkpointed = copy.deepcopy(plain)
        checkpointed.gradient_checkpointing = True
        inputs = torch.randint(6, config.vocab_size, (2, 12))
        labels = torch.randint(6, config.vocab_size, inputs.shape)
        labels[0, :3] = IGNORE_INDEX
        rng = torch.get_rng_state()
        plain_output = plain(inputs, labels=labels)
        plain_output.loss.backward()
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        checkpointed_output = checkpointed(inputs, labels=labels)
        checkpointed_output.loss.backward()
        torch.testing.assert_close(checkpointed_output.loss, plain_output.loss)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for (name, parameter), (other_name, other) in zip(plain.named_parameters(), checkpointed.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertIsNotNone(other.grad, name)
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=2e-5, msg=name)


class CheckpointTests(SmallModelCase):
    def setUp(self):
        super().setUp()
        self.config = tiny_config()
        self.model = CausalLM(self.config)
        self.tokenizer = create_autospec(Tokenizer, instance=True)
        self.tokenizer.sha256 = "test-tokenizer-sha256"
        self.tokenizer.validate_config.side_effect = lambda config: self.assertEqual(config, self.config)
        self.payload = {
            "format_version": FORMAT_VERSION,
            "model_config": self.config.to_dict(),
            "model_state": self.model.state_dict(),
            "tokenizer_sha256": self.tokenizer.sha256,
            "template_version": TEMPLATE_VERSION,
            "optimizer": None,
            "scaler": None,
            "training_state": None,
            "metadata": {},
        }

    def load(self, payload):
        # Exercise loader validation without writing a checkpoint fixture to disk.
        with patch("llm120.checkpoint.torch.load", return_value=payload) as reader:
            result = load_checkpoint("in-memory.pt", self.tokenizer)
            reader.assert_called_once_with(Path("in-memory.pt"), map_location="cpu", weights_only=True)
            return result

    def test_valid_tiny_checkpoint_loads_with_tied_weights(self):
        loaded, _ = self.load(self.payload)
        self.assertFalse(loaded.training)
        self.assertIs(loaded.embedding.weight, loaded.lm_head.weight)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], value)

    def test_checkpoint_rejects_tokenizer_and_template_mismatches(self):
        for field, message in (("tokenizer_sha256", "tokenizer_sha256 mismatch"),
                               ("template_version", "template mismatch")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.load({**self.payload, field: "changed"})

    def test_checkpoint_rejects_unequal_tied_weights(self):
        state = {**self.payload["model_state"],
                 "lm_head.weight": self.payload["model_state"]["lm_head.weight"] + 1}
        with self.assertRaisesRegex(ValueError, "identical tied embedding/head weights"):
            self.load({**self.payload, "model_state": state})


if __name__ == "__main__":
    unittest.main()
