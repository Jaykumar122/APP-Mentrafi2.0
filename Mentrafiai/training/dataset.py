"""
Datasets for both training phases.

- PretrainDataset: plain next-token prediction over a flat token stream (.bin file
  of uint16 token ids), memory-mapped so multi-GB files don't need to fit in RAM.
- SFTDataset: chat-turn examples with a loss mask so loss is only computed on the
  assistant's response, not the user's message.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return max(0, (len(self.data) - self.block_size) // self.block_size)

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.data[start : start + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


class SFTDataset(Dataset):
    """
    Expects a .jsonl file where each line is {"user": "...", "assistant": "..."}.
    Tokenizes on the fly using the provided tokenizer.

    Two padding modes:
      - dynamic_padding=True (default): __getitem__ returns the variable-length
        (input_ids, loss_mask). Padding is deferred to `sft_collate_fn`, which
        pads each batch only to that batch's longest sequence. Because loss uses
        ignore_index=pad_token_id and attention is causal (pad tokens sit at the
        end and never influence real tokens), this is numerically identical to
        fixed padding but skips the wasted compute on pad positions.
      - dynamic_padding=False: legacy behavior — every example padded/truncated
        to block_size and returned as ready (x, y) tensors (default collate).

    Loss is masked on non-assistant tokens by setting those targets to pad_id,
    which the model's cross-entropy ignores via ignore_index.
    """

    def __init__(self, jsonl_path: str, tokenizer, block_size: int, dynamic_padding: bool = True):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.dynamic_padding = dynamic_padding
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "user" in obj and "assistant" in obj:
                    self.examples.append((obj["user"], obj["assistant"]))
                elif "messages" in obj:
                    u = next((m.get("content", "") for m in obj["messages"] if m.get("role") == "user"), "")
                    a = next((m.get("content", "") for m in obj["messages"] if m.get("role") == "assistant"), "")
                    if u and a:
                        self.examples.append((u, a))
        print(f"Loaded {len(self.examples)} SFT examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        user_text, assistant_text = self.examples[idx]
        input_ids, loss_mask = self.tokenizer.encode_chat_turn(user_text, assistant_text)

        input_ids = input_ids[: self.block_size]
        loss_mask = loss_mask[: self.block_size]

        if self.dynamic_padding:
            # Defer padding to sft_collate_fn (pads to per-batch max length).
            return input_ids, loss_mask

        pad_len = self.block_size - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
            loss_mask = loss_mask + [0] * pad_len

        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(input_ids[1:], dtype=torch.long)
        mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        # Encode "don't compute loss here" as pad_token_id target where mask==0,
        # matched by ignore_index=pad_token_id in the loss (see MentraFiAI.forward)
        y = torch.where(mask.bool(), y, torch.full_like(y, self.tokenizer.pad_id))
        return x, y


def sft_collate_fn(pad_id: int):
    """
    Build a collate_fn that pads a batch of (input_ids, loss_mask) pairs to the
    batch's longest sequence, then produces (x, y) with non-assistant / pad
    targets set to pad_id so the loss ignores them.

    Returns a closure so the pad_id can be bound from the tokenizer at wiring time.
    """
    def collate(batch):
        max_len = max(len(ids) for ids, _ in batch)
        xs, ys = [], []
        for input_ids, loss_mask in batch:
            pad_len = max_len - len(input_ids)
            ids = input_ids + [pad_id] * pad_len
            mask = loss_mask + [0] * pad_len

            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            m = torch.tensor(mask[1:], dtype=torch.long)
            y = torch.where(m.bool(), y, torch.full_like(y, pad_id))
            xs.append(x)
            ys.append(y)
        return torch.stack(xs), torch.stack(ys)

    return collate
