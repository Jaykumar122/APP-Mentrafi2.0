"""Local, validated next-token datasets. Dataset labels are ALREADY shifted."""

from __future__ import annotations

import json
import operator
import warnings
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenization import (
    IGNORE_INDEX,
    TEMPLATE_VERSION,
    Tokenizer,
    clean_content,
    serialize,
    validate_messages,
)

TOKEN_DTYPE = "uint16"
DISK_DTYPE = np.dtype("<u2")


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def normalize_record(record: Any) -> dict[str, list[dict[str, str]]]:
    """Normalize messages, user/assistant, or instruction/input/output records.

    Optional top-level system text is accepted for the two pair formats only.
    Instruction and nonblank input are joined with two newlines. No default
    system prompt is ever added; messages must describe a complete SFT exchange.
    """
    if not isinstance(record, Mapping):
        raise ValueError("Each record must be a JSON object")
    if "messages" in record:
        if "system" in record:
            raise ValueError("Put explicit system content inside messages, not both formats")
        messages = record["messages"]
    else:
        if "user" in record and "assistant" in record:
            question, answer = record["user"], record["assistant"]
        elif "instruction" in record and "output" in record:
            question, answer = clean_content(record["instruction"]), record["output"]
            extra = record.get("input", "")
            if extra is None:
                extra = ""
            if not isinstance(extra, str):
                raise ValueError("instruction input must be a string or null")
            if extra.strip():
                question += "\n\n" + clean_content(extra)
        else:
            raise ValueError("Expected messages, user/assistant, or instruction/input/output")
        messages = []
        if "system" in record:
            messages.append({"role": "system", "content": record["system"]})
        messages.extend([{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}])
    return {"messages": validate_messages(messages)}


def _validated_sft(record: Any, tokenizer: Tokenizer, context_length: int) -> tuple[list[int], list[int]]:
    ids, labels = serialize(normalize_record(record)["messages"], tokenizer)
    if len(ids) != len(labels) or len(ids) < 2:
        raise ValueError("Invalid serialization: IDs/labels must align and contain a target")
    if len(ids) > context_length + 1:
        raise OverflowError(f"Serialized length {len(ids)} exceeds context_length+1={context_length + 1}")
    if not any(value != IGNORE_INDEX for value in labels[1:]):
        raise ValueError("All shifted labels are ignored; no assistant targets")
    return ids, labels


class SFTDataset(Dataset):
    """Index valid JSONL byte offsets; tokenize one example at a time on access.

    Invalid/overlong rows are counted, warned about, and skipped, never truncated.
    stats holds total, accepted, rejected_invalid, rejected_overlong counts.
    DataLoader workers open their own source file for each access (no shared seek).
    """

    def __init__(self, jsonl_path: str | Path, tokenizer: Tokenizer, context_length: int):
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.context_length = _positive_int(context_length, "context_length")
        self._offsets = array("Q")
        self.stats = {"total": 0, "accepted": 0, "rejected_invalid": 0, "rejected_overlong": 0}
        examples = []
        with self.path.open("rb") as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                self.stats["total"] += 1
                try:
                    _validated_sft(json.loads(line), tokenizer, self.context_length)
                except (ValueError, UnicodeError, OverflowError) as exc:
                    key = "rejected_overlong" if isinstance(exc, OverflowError) else "rejected_invalid"
                    self.stats[key] += 1
                    if len(examples) < 3:
                        examples.append(f"line {self.stats['total']}: {exc}")
                    continue
                self._offsets.append(offset)
                self.stats["accepted"] += 1
        rejected = self.stats["rejected_invalid"] + self.stats["rejected_overlong"]
        if rejected:
            warnings.warn(f"{self.path}: skipped {rejected} SFT rows "
                          f"(invalid={self.stats['rejected_invalid']}, "
                          f"overlong={self.stats['rejected_overlong']}); " + "; ".join(examples),
                          UserWarning, stacklevel=2)
        if not self._offsets:
            raise ValueError(f"No valid SFT examples in {self.path}; check record format, assistant "
                             f"targets, and context_length={context_length}. Counts: {self.stats}")

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        offset = self._offsets[operator.index(index)]
        with self.path.open("rb") as source:
            source.seek(offset)
            record = json.loads(source.readline())
        ids, labels = _validated_sft(record, self.tokenizer, self.context_length)
        return {"input_ids": torch.tensor(ids[:-1], dtype=torch.long),
                "labels": torch.tensor(labels[1:], dtype=torch.long)}


def collate_sft(batch: Sequence[Mapping[str, torch.Tensor]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Right-pad shifted examples without changing alignment or target masks."""
    if not batch:
        raise ValueError("Cannot collate an empty SFT batch")
    for item in batch:
        x, y = item["input_ids"], item["labels"]
        if x.ndim != 1 or x.shape != y.shape or x.numel() == 0:
            raise ValueError("SFT items must have nonempty, aligned 1D input_ids and labels")
        if x.dtype != torch.long or y.dtype != torch.long:
            raise ValueError("SFT input_ids and labels must be torch.long")
    length = max(item["input_ids"].numel() for item in batch)
    x = torch.full((len(batch), length), pad_id, dtype=torch.long)
    y = torch.full_like(x, IGNORE_INDEX)
    for row, item in enumerate(batch):
        count = item["input_ids"].numel()
        x[row, :count] = item["input_ids"]
        y[row, :count] = item["labels"]
    return {"input_ids": x, "labels": y}


class PretrainDataset(Dataset):
    """Little-endian uint16 mmap, contiguous windows with stride context_length.

    The tail shorter than a full input/target window is unused. Pickling stores
    only metadata, not the mmap contents, so spawned workers reopen it lazily.
    """

    def __init__(self, bin_path: str | Path, context_length: int, vocab_size: int):
        self.path = Path(bin_path)
        self.context_length = _positive_int(context_length, "context_length")
        self.vocab_size = _positive_int(vocab_size, "vocab_size")
        if vocab_size > 65536:
            raise ValueError("uint16 data requires vocab_size <= 65536")
        size = self.path.stat().st_size
        if size % DISK_DTYPE.itemsize:
            raise ValueError(f"{self.path}: byte length {size} is not divisible by uint16 size 2")
        self.token_count = size // DISK_DTYPE.itemsize
        if self.token_count < context_length + 1:
            raise ValueError(f"{self.path}: need at least context_length+1={context_length + 1} "
                             f"tokens, found {self.token_count}")
        self._tokens = np.memmap(self.path, dtype=DISK_DTYPE, mode="r")
        for start in range(0, self.token_count, 1024 * 1024):
            block = self._tokens[start:start + 1024 * 1024]
            if int(block.max()) >= vocab_size:
                raise ValueError(f"{self.path}: token ID {int(block.max())} outside vocab_size={vocab_size}")

    def __len__(self) -> int:
        return (self.token_count - 1) // self.context_length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=DISK_DTYPE, mode="r")
        start = index * self.context_length
        window = torch.from_numpy(np.array(self._tokens[start:start + self.context_length + 1],
                                           dtype=np.int64, copy=True))
        return {"input_ids": window[:-1], "labels": window[1:]}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_tokens"] = None
        return state


def validate_manifest(manifest_path: str | Path, tokenizer: Tokenizer) -> dict[str, Any]:
    """Validate corpus/tokenizer identity and mode-specific on-disk format.

    Pretraining does not depend on the chat template. SFT must match exactly.
    Source hashes are provenance, not paths that need to remain mounted to train.
    """
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    if manifest.get("tokenizer_sha256") != tokenizer.sha256:
        raise ValueError("Manifest tokenizer_sha256 mismatch; reprepare with this tokenizer")
    if manifest.get("vocab_size") != tokenizer.vocab_size:
        raise ValueError("Manifest vocab_size mismatch")
    mode = manifest.get("mode")
    if mode not in ("pretrain", "sft"):
        raise ValueError("Manifest mode must be pretrain or sft")
    expected_dtype = TOKEN_DTYPE if mode == "pretrain" else "jsonl"
    if manifest.get("dtype") != expected_dtype:
        raise ValueError(f"Manifest dtype must be {expected_dtype!r} for {mode}")
    if mode == "pretrain" and manifest.get("byte_order", "little") != "little":
        raise ValueError("Pretrain manifest byte_order must be 'little'")
    if mode == "sft" and manifest.get("template_version") != TEMPLATE_VERSION:
        raise ValueError("Manifest template_version mismatch; reprepare SFT data")
    return manifest
