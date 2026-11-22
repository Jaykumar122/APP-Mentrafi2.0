"""Small, local-only pipeline tests using the repository's actual tokenizer."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from llm120.data import (
    PretrainDataset,
    SFTDataset,
    collate_sft,
    normalize_record,
    validate_manifest,
)
from llm120.prepare import main, prepare_dataset
from llm120.tokenization import IGNORE_INDEX, TEMPLATE_VERSION, Tokenizer, serialize

MODEL_PATH = Path(__file__).resolve().parents[1] / "tokenizer" / "tokenizer_v2.model"


def write_jsonl(path: Path, records: list) -> Path:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class PipelineCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer(MODEL_PATH)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llm120-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def prepare(self, inputs, output="dataset", **kwargs):
        return prepare_dataset(inputs, self.root / output, self.tokenizer, **kwargs)


class TokenizationTests(PipelineCase):
    def test_real_tokenizer_identity_and_config(self):
        tok = self.tokenizer
        self.assertEqual(tok.vocab_size, 16000)
        names = ("pad_id", "bos_id", "eos_id", "unk_id", "user_id", "assistant_id")
        self.assertEqual([getattr(tok, name) for name in names], list(range(6)))
        self.assertEqual(tok.sha256, hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest())
        config = {"vocab_size": 16000, **dict(zip(names, range(6)))}
        tok.validate_config(SimpleNamespace(**config))
        tok.validate_config(config)
        for name in ("vocab_size", *names):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                tok.validate_config({**config, name: config[name] + 1})
        self.assertEqual(tok.decode(tok.encode("Hello world!")), "Hello world!")
        self.assertNotIn(tok.bos_id, tok.encode("Hello"))
        self.assertIn(tok.user_id, tok.encode("<user>"))
        self.assertIn(tok.assistant_id, tok.encode("<assistant>"))

    def test_single_turn_exact_format_and_masks(self):
        tok = self.tokenizer
        user, answer = tok.encode("A question?"), tok.encode("An answer.")
        ids, labels = serialize([{"role": "user", "content": "A question?"},
                                 {"role": "assistant", "content": "An answer."}], tok)
        expected = [tok.bos_id, tok.user_id] + user + [tok.eos_id, tok.assistant_id] + answer + [tok.eos_id]
        self.assertEqual(ids, expected)
        self.assertEqual(labels, [IGNORE_INDEX] * (len(user) + 4) + answer + [tok.eos_id])
        for token, label in zip(ids, labels):
            if token in (tok.bos_id, tok.user_id, tok.assistant_id):
                self.assertEqual(label, IGNORE_INDEX)
        self.assertEqual(ids.count(tok.bos_id), 1)
        self.assertEqual([label for token, label in zip(ids, labels) if token == tok.eos_id],
                         [IGNORE_INDEX, tok.eos_id])

    def test_multiturn_system_and_generation_parity(self):
        tok = self.tokenizer
        messages = [{"role": "system", "content": "Be accurate."},
                    {"role": "user", "content": "One?"},
                    {"role": "assistant", "content": "First."},
                    {"role": "user", "content": "Two?"},
                    {"role": "assistant", "content": "Second."}]
        ids, labels = serialize(messages, tok)
        system = tok.encode("System:\nBe accurate.\n\n")
        self.assertEqual(ids[:len(system) + 1], [tok.bos_id] + system)
        self.assertEqual(labels[:len(system) + 1], [IGNORE_INDEX] * (len(system) + 1))
        expected_targets = tok.encode("First.") + [tok.eos_id] + tok.encode("Second.") + [tok.eos_id]
        self.assertEqual([value for value in labels if value != IGNORE_INDEX], expected_targets)
        prefix, prefix_labels = serialize(messages[:-1], tok, add_generation_prompt=True)
        self.assertEqual(ids[:len(prefix)], prefix)
        self.assertEqual(prefix[-1], tok.assistant_id)
        self.assertEqual(prefix_labels, labels[:len(prefix)])
        self.assertEqual(ids.count(tok.bos_id), 1)
        no_system, _ = serialize(messages[1:3], tok)
        self.assertEqual(no_system[:2], [tok.bos_id, tok.user_id])

    def test_content_whitespace_is_preserved_for_exact_prompt_parity(self):
        tok = self.tokenizer
        question, answer = "  indented question\r\n", "    return 42\n"
        record = normalize_record({"user": question, "assistant": answer})
        self.assertEqual(record["messages"][0]["content"], question)
        self.assertEqual(record["messages"][1]["content"], answer)
        ids, labels = serialize(record["messages"], tok)
        expected = ([tok.bos_id, tok.user_id] + tok.encode(question)
                    + [tok.eos_id, tok.assistant_id] + tok.encode(answer) + [tok.eos_id])
        self.assertEqual(ids, expected)
        self.assertEqual([value for value in labels if value != IGNORE_INDEX],
                         tok.encode(answer) + [tok.eos_id])

    def test_invalid_order_content_and_generation_rejected(self):
        u = {"role": "user", "content": "Question?"}
        a = {"role": "assistant", "content": "Answer."}
        s = {"role": "system", "content": "Rules."}
        invalid = [[], [s], [a], [u], [u, u, a], [s, s, u, a], [u, s, a],
                   [u, {"role": "tool", "content": "x"}], [u, {**a, "content": " \n\t"}],
                   [u, {**a, "content": None}], [u, {**a, "content": "<assistant>"}]]
        for messages in invalid:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                serialize(messages, self.tokenizer)
        with self.assertRaises(ValueError):
            serialize([u, a], self.tokenizer, add_generation_prompt=True)

    def test_normalize_supported_records_and_explicit_system(self):
        expected = {"messages": [{"role": "user", "content": "Summarize.\n\nSome text."},
                                  {"role": "assistant", "content": "Summary."}]}
        self.assertEqual(normalize_record({"instruction": "Summarize.", "input": "Some text.",
                                           "output": "Summary."}), expected)
        self.assertEqual(normalize_record({"user": "Summarize.\n\nSome text.",
                                           "assistant": "Summary."}), expected)
        self.assertEqual(normalize_record(expected), expected)
        record = normalize_record({"system": "Explicit only.", "user": "Q", "assistant": "A"})
        self.assertEqual(record["messages"][0]["role"], "system")
        self.assertEqual(len(expected["messages"]), 2)
        for record in ({}, None, {"instruction": "q", "input": [], "output": "a"},
                       {"user": "q", "assistant": ""}, {"messages": [], "system": "x"}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                normalize_record(record)


class DatasetTests(PipelineCase):
    def test_sft_shift_boundary_eos_and_collation(self):
        records = [{"user": "Question?", "assistant": "A detailed answer."},
                   {"user": "Q?", "assistant": "Yes."}]
        path = write_jsonl(self.root / "sft.jsonl", records)
        ids, labels = serialize(normalize_record(records[0])["messages"], self.tokenizer)
        ds = SFTDataset(path, self.tokenizer, len(ids) - 1)
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0]["input_ids"].tolist(), ids[:-1])
        self.assertEqual(ds[0]["labels"].tolist(), labels[1:])
        self.assertEqual(ds[0]["labels"][-1].item(), self.tokenizer.eos_id)
        self.assertEqual(ds[0]["input_ids"].dtype, torch.long)
        batch = collate_sft([ds[0], ds[1]], pad_id=7)
        short = len(ds[1]["input_ids"])
        self.assertEqual(tuple(batch["input_ids"].shape), (2, len(ids) - 1))
        self.assertTrue(torch.all(batch["input_ids"][1, short:] == 7))
        self.assertTrue(torch.all(batch["labels"][1, short:] == IGNORE_INDEX))
        self.assertTrue(torch.equal(batch["labels"][0], ds[0]["labels"]))
        with self.assertRaises(ValueError):
            collate_sft([])

    def test_sft_skips_invalid_and_overlong_without_truncation(self):
        path = write_jsonl(self.root / "mixed.jsonl", [
            {"user": "Q?", "assistant": "A."},
            {"user": "Q?", "assistant": "long text " * 200},
            {"user": "Q?", "assistant": "   "},
            {"messages": [{"role": "user", "content": "No assistant."}]},
        ])
        with path.open("a", encoding="utf-8") as stream:
            stream.write("{broken\n")
        with self.assertWarnsRegex(UserWarning, "skipped 4"):
            ds = SFTDataset(path, self.tokenizer, 32)
        self.assertEqual(ds.stats, {"total": 5, "accepted": 1, "rejected_invalid": 3,
                                    "rejected_overlong": 1})
        ids, _ = serialize(normalize_record({"user": "Q?", "assistant": "A."})["messages"], self.tokenizer)
        self.assertEqual(ds[0]["input_ids"].tolist(), ids[:-1])

    def test_sft_fails_when_all_overlong_or_all_ignored(self):
        path = write_jsonl(self.root / "one.jsonl", [{"user": "Q?", "assistant": "Answer."}])
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "No valid SFT"):
            SFTDataset(path, self.tokenizer, 1)
        with (
            patch("llm120.data.serialize", return_value=([1, 4, 8, 2], [IGNORE_INDEX] * 4)),
            self.assertWarnsRegex(UserWarning, "All shifted labels"),
            self.assertRaisesRegex(ValueError, "No valid SFT"),
        ):
            SFTDataset(path, self.tokenizer, 32)
        empty = self.root / "empty.jsonl"
        empty.touch()
        with self.assertRaisesRegex(ValueError, "No valid SFT"):
            SFTDataset(empty, self.tokenizer, 32)

    def test_pretrain_contiguous_shifted_windows_and_spawn_pickle(self):
        tokens = np.array([1, 10, 11, 2, 1, 20, 21, 2, 1, 30], dtype="<u2")
        path = self.root / "train.bin"
        tokens.tofile(path)
        ds = PretrainDataset(path, 3, self.tokenizer.vocab_size)
        self.assertIsInstance(ds._tokens, np.memmap)
        self.assertEqual(len(ds), 3)
        for index in range(3):
            self.assertEqual(ds[index]["input_ids"].tolist(), tokens[index * 3:index * 3 + 3].tolist())
            self.assertEqual(ds[index]["labels"].tolist(), tokens[index * 3 + 1:index * 3 + 4].tolist())
            self.assertEqual(ds[index]["input_ids"].dtype, torch.long)
        self.assertTrue(torch.equal(ds[-1]["labels"], ds[2]["labels"]))
        with self.assertRaises(IndexError):
            ds[3]
        restored = pickle.loads(pickle.dumps(ds))
        self.assertIsNone(restored._tokens)
        self.assertTrue(torch.equal(restored[1]["input_ids"], ds[1]["input_ids"]))

    def test_pretrain_binary_validation(self):
        path = self.root / "bad.bin"
        path.write_bytes(b"\x01")
        with self.assertRaisesRegex(ValueError, "divisible"):
            PretrainDataset(path, 2, 16000)
        np.array([1, 16000, 2], dtype="<u2").tofile(path)
        with self.assertRaisesRegex(ValueError, "outside vocab_size"):
            PretrainDataset(path, 2, 16000)
        np.array([1, 2], dtype="<u2").tofile(path)
        with self.assertRaisesRegex(ValueError, "at least"):
            PretrainDataset(path, 2, 16000)
        path.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "at least"):
            PretrainDataset(path, 2, 16000)
        with self.assertRaisesRegex(ValueError, "positive"):
            PretrainDataset(path, 0, 16000)
        with self.assertRaisesRegex(ValueError, "65536"):
            PretrainDataset(path, 2, 65537)

    def test_manifest_identity_dtype_and_template_validation(self):
        path = self.root / "manifest.json"
        base = {"mode": "sft", "tokenizer_sha256": self.tokenizer.sha256,
                "vocab_size": self.tokenizer.vocab_size, "dtype": "jsonl",
                "template_version": TEMPLATE_VERSION}
        path.write_text(json.dumps(base), encoding="utf-8")
        self.assertEqual(validate_manifest(path, self.tokenizer), base)
        for field, value in (("tokenizer_sha256", "0" * 64), ("vocab_size", 12),
                             ("template_version", "old"), ("dtype", "uint16"), ("mode", "other")):
            path.write_text(json.dumps({**base, field: value}), encoding="utf-8")
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                validate_manifest(path, self.tokenizer)
        pretrain = {**base, "mode": "pretrain", "dtype": "uint16", "byte_order": "little",
                    "template_version": "irrelevant-to-raw-text"}
        path.write_text(json.dumps(pretrain), encoding="utf-8")
        self.assertEqual(validate_manifest(path, self.tokenizer), pretrain)
        path.write_text(json.dumps({**pretrain, "dtype": "int64"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dtype"):
            validate_manifest(path, self.tokenizer)


class PreparationTests(PipelineCase):
    def test_pretrain_stream_dedup_eos_disjoint_splits_and_provenance(self):
        docs = [f"Unique document number {index}." for index in range(64)]
        txt = self.root / "paragraphs.txt"
        txt.write_text("\n\n".join(docs[:32]), encoding="utf-8")
        jsonl = write_jsonl(self.root / "documents.jsonl", [{"text": text} for text in docs[32:]]
                            + [{"text": "  Unique   document number 0.  "}])
        manifest = self.prepare([txt, jsonl], mode="pretrain", val_fraction=0.3)
        out = self.root / "dataset"
        self.assertEqual(manifest["counts"]["documents"], 64)
        self.assertEqual(manifest["counts"]["duplicates"], 1)
        self.assertEqual(manifest["counts"]["rejected"], 0)
        self.assertEqual(validate_manifest(out / "manifest.json", self.tokenizer), manifest)
        self.assertEqual(manifest["dtype"], "uint16")
        sets = {}
        token_total = 0
        for split in ("train", "val"):
            tokens = np.fromfile(out / f"{split}.bin", dtype="<u2").tolist()
            token_total += len(tokens)
            self.assertEqual(tokens[-1], self.tokenizer.eos_id)
            self.assertEqual(tokens.count(self.tokenizer.eos_id), manifest["splits"][split]["documents"])
            self.assertEqual(len(tokens), manifest["splits"][split]["tokens"])
            sets[split] = set()
            start = 0
            for end, token in enumerate(tokens):
                if token == self.tokenizer.eos_id:
                    sets[split].add(tuple(tokens[start:end]))
                    start = end + 1
            self.assertGreater(len(PretrainDataset(out / f"{split}.bin", 8, 16000)), 0)
        self.assertFalse(sets["train"] & sets["val"])
        self.assertEqual(len(sets["train"] | sets["val"]), 64)
        self.assertEqual(token_total, manifest["counts"]["tokens"])
        for source in manifest["sources"]:
            path = Path(source["path"])
            self.assertEqual(source["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(source["size_bytes"], path.stat().st_size)
        self.assertEqual(sorted(path.name for path in out.iterdir()), ["manifest.json", "train.bin", "val.bin"])

    def test_pretrain_split_reproducible_under_reorder_and_duplicates(self):
        records = [{"text": f"Stable document {index}."} for index in range(60)]
        one = write_jsonl(self.root / "one.jsonl", records)
        two = write_jsonl(self.root / "two.jsonl", list(reversed(records)) + records[:5])
        self.prepare([one], output="a", mode="pretrain", val_fraction=0.25, seed=7)
        manifest = self.prepare([two], output="b", mode="pretrain", val_fraction=0.25, seed=7)
        self.assertEqual(manifest["counts"]["duplicates"], 5)
        # Compare token documents, not order: partitioning is order independent.
        def chunks(directory, split):
            values = np.fromfile(directory / f"{split}.bin", dtype="<u2").tolist()
            result, current = set(), []
            for value in values:
                if value == self.tokenizer.eos_id:
                    result.add(tuple(current))
                    current = []
                else:
                    current.append(value)
            return result

        for split in ("train", "val"):
            self.assertEqual(chunks(self.root / "a", split), chunks(self.root / "b", split))

    def test_sft_dedup_question_grouping_rejections_and_normalized_output(self):
        records = []
        for index in range(40):
            question = f"Question {index}?"
            records.extend([{"user": question, "assistant": f"Answer {index} a."},
                            {"instruction": question, "input": "", "output": f"Answer {index} b."},
                            {"messages": [{"role": "system", "content": "Explicit rules."},
                                          {"role": "user", "content": question},
                                          {"role": "assistant", "content": "First reply."},
                                          {"role": "user", "content": "Follow up?"},
                                          {"role": "assistant", "content": "Follow-up reply."}]}])
        records.extend([records[0], {"instruction": "Question   0?", "output": "Answer 0 a."},
                        {"user": "Too long?", "assistant": "long text " * 1000},
                        {"user": "Blank?", "assistant": " "}])
        source = write_jsonl(self.root / "sft.jsonl", records)
        with self.assertWarnsRegex(UserWarning, "Rejected 2"):
            manifest = self.prepare([source], mode="sft", val_fraction=0.3, context_length=128)
        self.assertEqual(manifest["counts"]["documents"], 120)
        self.assertEqual(manifest["counts"]["duplicates"], 2)
        self.assertEqual(manifest["counts"]["rejected_overlong"], 1)
        self.assertEqual(manifest["counts"]["rejected_invalid"], 1)
        question_splits = {}
        for split in ("train", "val"):
            rows = read_jsonl(self.root / "dataset" / f"{split}.jsonl")
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual(set(row), {"messages"})
                question = next(msg["content"] for msg in row["messages"] if msg["role"] == "user")
                self.assertEqual(question_splits.setdefault(question, split), split)
            ds = SFTDataset(self.root / "dataset" / f"{split}.jsonl", self.tokenizer, 128)
            self.assertEqual(len(ds), len(rows))
        self.assertEqual(len(question_splits), 40)
        self.assertEqual(validate_manifest(self.root / "dataset" / "manifest.json", self.tokenizer), manifest)

    def test_atomic_failures_force_and_empty_split_policy(self):
        source = write_jsonl(self.root / "source.jsonl", [{"text": f"Document {i}."} for i in range(50)])
        self.prepare([source], mode="pretrain", val_fraction=0.3)
        output = self.root / "dataset"
        original = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaisesRegex(FileExistsError, "--force"):
            self.prepare([source], mode="pretrain", val_fraction=0.3)
        only = write_jsonl(self.root / "only.jsonl", [{"text": "Just one unique document."}] * 5)
        with self.assertRaisesRegex(ValueError, "split is empty"):
            self.prepare([only], mode="pretrain", val_fraction=0.3, force=True)
        self.assertEqual(original, {path.name: path.read_bytes() for path in output.iterdir()})
        self.assertFalse(list(self.root.glob(".dataset.staging-*")))
        replacement = self.prepare([source], mode="pretrain", val_fraction=0.4, seed=3, force=True)
        self.assertEqual(replacement["seed"], 3)
        self.assertFalse(list(self.root.glob(".dataset.backup-*")))
        with self.assertRaisesRegex(ValueError, "split is empty"):
            self.prepare([only], output="never-published", mode="pretrain", val_fraction=0.3)
        self.assertFalse((self.root / "never-published").exists())

    def test_failed_publication_restores_previous_dataset(self):
        source = write_jsonl(self.root / "source.jsonl", [{"text": f"Document {i}."} for i in range(50)])
        self.prepare([source], mode="pretrain", val_fraction=0.3)
        output = self.root / "dataset"
        original = {path.name: path.read_bytes() for path in output.iterdir()}
        rename = Path.rename

        def fail_stage(path, target):
            if path.name.startswith(".dataset.staging-"):
                raise OSError("simulated publication failure")
            return rename(path, target)

        with patch.object(Path, "rename", fail_stage), self.assertRaisesRegex(OSError, "simulated"):
            self.prepare([source], mode="pretrain", val_fraction=0.3, force=True)
        self.assertEqual(original, {path.name: path.read_bytes() for path in output.iterdir()})
        self.assertFalse(list(self.root.glob(".dataset.*-*")))

    def test_invalid_unicode_sft_is_counted_not_a_tokenizer_crash(self):
        source = self.root / "unicode.jsonl"
        source.write_text('{"user":"Q?","assistant":"\\ud800"}\n', encoding="utf-8")
        with self.assertWarnsRegex(UserWarning, "invalid=1"), self.assertRaisesRegex(ValueError, "split is empty"):
            self.prepare([source], mode="sft")
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "No valid SFT"):
            SFTDataset(source, self.tokenizer, 128)

    def test_reject_bad_documents_and_all_invalid_without_publish(self):
        source = write_jsonl(self.root / "invalid.jsonl", [{"text": ""}, {"wrong": "field"}, 42])
        with source.open("ab") as stream:
            stream.write(b"not json\n{\"text\":\"\xff\"}\n")
        with self.assertWarnsRegex(UserWarning, "Rejected 5"), self.assertRaisesRegex(ValueError, "split is empty"):
            self.prepare([source], mode="pretrain", val_fraction=0.3)
        self.assertFalse((self.root / "dataset").exists())
        all_long = write_jsonl(self.root / "long.jsonl", [{"user": "Q?", "assistant": "long text " * 1000}])
        with self.assertWarnsRegex(UserWarning, "overlong=1"), self.assertRaisesRegex(ValueError, "split is empty"):
            self.prepare([all_long], mode="sft", context_length=32)
        with self.assertRaisesRegex(ValueError, "between"):
            self.prepare([source], mode="pretrain", val_fraction=0)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            prepare_dataset([source], self.root, self.tokenizer, mode="pretrain", force=True)

    def test_cli_repeated_and_multi_value_inputs(self):
        paths = [write_jsonl(self.root / f"source{i}.jsonl", [{"text": f"Part {i} document {j}."}
                                                           for j in range(20)]) for i in range(3)]
        args = ["--mode", "pretrain", "--input", str(paths[0]), str(paths[1]), "--input", str(paths[2]),
                "--output", str(self.root / "cli"), "--tokenizer", str(MODEL_PATH),
                "--val-fraction", "0.3", "--seed", "42"]
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            main(args)
        manifest = json.loads(captured.getvalue())
        self.assertEqual(manifest["counts"]["documents"], 60)
        self.assertEqual(len(manifest["sources"]), 3)
        self.assertTrue((self.root / "cli" / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
