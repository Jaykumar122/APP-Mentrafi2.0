"""Tokenizer identity and the shared, explicit-system-only chat wire format."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sentencepiece as spm

TEMPLATE_VERSION = "llm120-role-chat-v1"
IGNORE_INDEX = -100


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Tokenizer:
    """SentencePiece adapter; encode never inserts BOS, EOS, or role markers."""

    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int
    user_id: int
    assistant_id: int

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path).resolve(strict=True)
        self.sha256 = file_sha256(self.model_path)
        self._processor = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.vocab_size = self._processor.vocab_size()
        for name in ("pad", "bos", "eos", "unk"):
            value = getattr(self._processor, f"{name}_id")()
            if value < 0:
                raise ValueError(f"Tokenizer has no {name} token: {self.model_path}")
            setattr(self, f"{name}_id", value)
        for role in ("user", "assistant"):
            piece = f"<{role}>"
            value = self._processor.piece_to_id(piece)
            if value == self.unk_id or self._processor.id_to_piece(value) != piece:
                raise ValueError(f"Tokenizer is missing the exact role token {piece!r}")
            setattr(self, f"{role}_id", value)
        ids = [getattr(self, f"{name}_id") for name in
               ("pad", "bos", "eos", "unk", "user", "assistant")]
        if len(set(ids)) != 6:
            raise ValueError("Tokenizer special token IDs must be distinct")

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("encode expects text as str")
        return self._processor.encode(text, out_type=int, add_bos=False, add_eos=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self._processor.decode([int(value) for value in ids])

    def validate_config(self, config: Any) -> None:
        """Reject mismatched model vocabularies, including any special token ID."""
        mismatches = []
        for name in ("vocab_size", "pad_id", "bos_id", "eos_id", "unk_id",
                     "user_id", "assistant_id"):
            actual = config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)
            expected = getattr(self, name)
            if actual != expected:
                mismatches.append(f"{name}: config={actual!r}, tokenizer={expected}")
        if mismatches:
            raise ValueError("Model/tokenizer mismatch: " + "; ".join(mismatches))


def clean_content(value: Any) -> str:
    """Validate nonblank Unicode text, preserving its exact whitespace/content."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Message content must be a nonblank string")
    # JSON can contain lone surrogates; SentencePiece cannot safely encode them.
    value.encode("utf-8")
    return value


def validate_messages(messages: Any, *, add_generation_prompt: bool = False) -> list[dict[str, str]]:
    """Return canonical messages with strict system?, (user, assistant)* ordering."""
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a nonempty list of role/content objects")
    result = []
    expected = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"Message {index} must be a role/content object")
        role = message.get("role")
        if index == 0 and role == "system":
            pass
        elif role != expected:
            raise ValueError(f"Message {index}: expected {expected!r}, got {role!r}")
        else:
            expected = "assistant" if role == "user" else "user"
        result.append({"role": role, "content": clean_content(message.get("content"))})
    final_role = "user" if add_generation_prompt else "assistant"
    if result[-1]["role"] != final_role:
        raise ValueError(f"Conversation must end with {final_role!r}")
    return result


def serialize(messages: Sequence[Mapping[str, str]], tokenizer: Tokenizer, *,
              add_generation_prompt: bool = False) -> tuple[list[int], list[int]]:
    """Serialize chat with UNshifted labels; only assistant content/EOS is trained.

    System text uses BOS + encode('System:\\n' + content + '\\n\\n'), not a new
    special token. BOS occurs once. Inference must use this same function with
    add_generation_prompt=True and a final user message; no system is implicit.
    Reserved structural tokens in content are rejected instead of allowing role
    injection or supervising a delimiter as ordinary assistant text.
    """
    canonical = validate_messages(messages, add_generation_prompt=add_generation_prompt)
    ids = [tokenizer.bos_id]
    labels = [IGNORE_INDEX]
    structural = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id,
                  tokenizer.user_id, tokenizer.assistant_id}
    for message in canonical:
        role, content = message["role"], message["content"]
        text = f"System:\n{content}\n\n" if role == "system" else content
        tokens = tokenizer.encode(text)
        if not tokens:
            raise ValueError(f"{role} content encoded to no tokens")
        if structural.intersection(tokens):
            raise ValueError(f"{role} content contains a reserved structural token")
        if role == "system":
            ids.extend(tokens)
            labels.extend([IGNORE_INDEX] * len(tokens))
            continue
        ids.append(tokenizer.user_id if role == "user" else tokenizer.assistant_id)
        labels.append(IGNORE_INDEX)
        ids.extend(tokens)
        ids.append(tokenizer.eos_id)
        labels.extend(tokens + [tokenizer.eos_id] if role == "assistant"
                      else [IGNORE_INDEX] * (len(tokens) + 1))
    if add_generation_prompt:
        ids.append(tokenizer.assistant_id)
        labels.append(IGNORE_INDEX)
    return ids, labels
