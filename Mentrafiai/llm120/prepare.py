"""Streaming local corpus preparation: python -m llm120.prepare --help.

Pretrain .txt files contain blank-line-separated documents; .jsonl files contain
one {"text": ...} object (or JSON string) per line. SFT inputs must be JSONL.
Only one document is tokenized at a time. Exact normalized hashes live in a
scratch SQLite database, not an in-memory corpus-sized set. No downloads occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sqlite3
import tempfile
import unicodedata
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from .data import DISK_DTYPE, TOKEN_DTYPE, _positive_int, _validated_sft, normalize_record
from .tokenization import TEMPLATE_VERSION, Tokenizer, clean_content, file_sha256

SPLIT_VERSION = "sha256-content-v1"


def _normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _split_for(fingerprint: str, seed: int, val_fraction: float) -> str:
    digest = hashlib.sha256(f"{SPLIT_VERSION}\0{seed}\0{fingerprint}".encode("ascii")).digest()
    return "val" if int.from_bytes(digest, "big") < int(val_fraction * (1 << 256)) else "train"


def _source_records(path: Path, metadata: dict[str, Any]) -> Iterator[Any]:
    """Hash the exact consumed source bytes, yielding invalid rows as errors."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        if path.suffix.lower() == ".txt":
            paragraph: list[str] = []
            invalid = False
            for raw in source:
                digest.update(raw)
                size += len(raw)
                try:
                    text = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    invalid = True
                    continue
                if text.strip():
                    paragraph.append(text)
                elif paragraph or invalid:
                    yield ValueError("Invalid UTF-8 document") if invalid else {"text": "".join(paragraph)}
                    paragraph, invalid = [], False
            if paragraph or invalid:
                yield ValueError("Invalid UTF-8 document") if invalid else {"text": "".join(paragraph)}
        else:
            for number, raw in enumerate(source, 1):
                digest.update(raw)
                size += len(raw)
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw.decode("utf-8-sig"))
                except (ValueError, UnicodeError) as exc:
                    record = ValueError(f"Line {number}: invalid UTF-8/JSON: {exc}")
                yield record
    metadata.update(sha256=digest.hexdigest(), size_bytes=size)


def _publish(stage: Path, output: Path, force: bool) -> None:
    """Publish only a finished dataset, restoring the old one if rename fails."""
    backup = None
    if output.exists():
        if not force:
            raise FileExistsError(f"Dataset already exists: {output}; use --force to replace it")
        backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent))
        backup.rmdir()
        output.rename(backup)
    try:
        stage.rename(output)
    except BaseException:
        if backup is not None:
            backup.rename(output)
        raise
    if backup is not None:
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            warnings.warn(f"Dataset published, but could not remove backup {backup}: {exc}",
                          UserWarning, stacklevel=2)


def prepare_dataset(inputs: Sequence[str | Path], output: str | Path,
                    tokenizer: Tokenizer | str | Path, *, mode: str,
                    val_fraction: float = 0.05, seed: int = 42,
                    context_length: int = 1024, force: bool = False) -> dict[str, Any]:
    """Prepare train/val files and return their manifest; fail without publishing.

    SFT split assignment hashes the FIRST user question (casefolded, NFKC and
    whitespace normalized). All response variants, system variants, and follow-up
    variations to that initial question stay together, conservatively avoiding
    initial-question leakage. Exact conversation dedup retains case and roles.
    Pretraining split assignment uses the normalized document fingerprint.
    Splits depend on content and seed, not input order or corpus size. Very small
    corpora may have an empty split; this is an error, never a forced reassignment.
    context_length applies to full SFT serialization, not pretraining documents.
    """
    if mode not in ("pretrain", "sft"):
        raise ValueError("mode must be pretrain or sft")
    if not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    _positive_int(context_length, "context_length")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(inputs, (str, Path)) or not inputs:
        raise ValueError("inputs must be a nonempty sequence of local file paths")
    paths = sorted({Path(path).resolve(strict=True) for path in inputs}, key=str)
    for path in paths:
        extensions = (".jsonl",) if mode == "sft" else (".txt", ".jsonl")
        if not path.is_file() or path.suffix.lower() not in extensions:
            raise ValueError(f"{mode} inputs must be local {extensions} files: {path}")
    output_arg = Path(output).absolute()
    if output_arg.is_symlink():
        raise ValueError("Output must not be a symbolic link")
    output = output_arg.resolve()
    if output == output.parent:
        raise ValueError("Output cannot be a filesystem root")
    if output.exists() and (not force or not output.is_dir()):
        raise FileExistsError(f"Dataset output exists: {output}; use --force for an existing directory")
    tok = tokenizer if isinstance(tokenizer, Tokenizer) else Tokenizer(tokenizer)
    if any(output == path or output in path.parents for path in paths + [tok.model_path.resolve()]):
        raise ValueError("Output cannot contain an input file or the tokenizer (unsafe --force)")
    if mode == "pretrain" and tok.vocab_size > 65536:
        raise ValueError("uint16 pretraining requires vocab_size <= 65536")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    counts = {"input_records": 0, "documents": 0, "tokens": 0, "duplicates": 0,
              "rejected": 0, "rejected_invalid": 0, "rejected_overlong": 0}
    suffix = "bin" if mode == "pretrain" else "jsonl"
    splits = {name: {"file": f"{name}.{suffix}", "documents": 0, "tokens": 0}
              for name in ("train", "val")}
    sources: list[dict[str, Any]] = []
    rejection_examples: list[str] = []
    try:
        with ExitStack() as stack:
            streams = {name: stack.enter_context((stage / info["file"]).open("wb"))
                       for name, info in splits.items()}
            connection = sqlite3.connect(stage / "dedup.sqlite3")
            stack.callback(connection.close)
            connection.execute("CREATE TABLE seen (hash TEXT PRIMARY KEY) WITHOUT ROWID")
            for path in paths:
                metadata: dict[str, Any] = {"path": str(path)}
                sources.append(metadata)
                for record in _source_records(path, metadata):
                    counts["input_records"] += 1
                    try:
                        if isinstance(record, Exception):
                            raise record
                        if mode == "pretrain":
                            text = clean_content(record.get("text") if isinstance(record, Mapping) else record)
                            ids = tok.encode(text)
                            if not ids:
                                raise ValueError("Document encoded to no tokens")
                            ids.append(tok.eos_id)
                            fingerprint = _fingerprint(_normalized_text(text))
                            group = fingerprint
                            normalized = None
                        else:
                            normalized = normalize_record(record)
                            ids, _ = _validated_sft(normalized, tok, context_length)
                            messages = normalized["messages"]
                            fingerprint = _fingerprint([
                                [msg["role"], _normalized_text(msg["content"])] for msg in messages])
                            question = next(msg["content"] for msg in messages if msg["role"] == "user")
                            group = _fingerprint(_normalized_text(question).casefold())
                    except (ValueError, UnicodeError, OverflowError) as exc:
                        counts["rejected"] += 1
                        key = "rejected_overlong" if isinstance(exc, OverflowError) else "rejected_invalid"
                        counts[key] += 1
                        if len(rejection_examples) < 3:
                            rejection_examples.append(f"{path.name}: {exc}")
                        continue
                    inserted = connection.execute("INSERT OR IGNORE INTO seen(hash) VALUES (?)", (fingerprint,))
                    if not inserted.rowcount:
                        counts["duplicates"] += 1
                        continue
                    split = _split_for(group, seed, val_fraction)
                    if mode == "pretrain":
                        streams[split].write(np.asarray(ids, dtype=DISK_DTYPE).tobytes())
                    else:
                        streams[split].write((json.dumps(normalized, ensure_ascii=False,
                                                        separators=(",", ":")) + "\n").encode("utf-8"))
                    splits[split]["documents"] += 1
                    splits[split]["tokens"] += len(ids)
                    counts["documents"] += 1
                    counts["tokens"] += len(ids)
                connection.commit()
        (stage / "dedup.sqlite3").unlink()
        if counts["rejected"]:
            warnings.warn(f"Rejected {counts['rejected']} records "
                          f"(invalid={counts['rejected_invalid']}, overlong={counts['rejected_overlong']}): "
                          + "; ".join(rejection_examples), UserWarning, stacklevel=2)
        for name, info in splits.items():
            if info["documents"] == 0:
                raise ValueError(f"{name} split is empty. Add unique documents/questions or adjust "
                                 f"--val-fraction/--seed; check --context-length for SFT. Counts: {counts}")
            info["size_bytes"] = (stage / info["file"]).stat().st_size
            info["sha256"] = file_sha256(stage / info["file"])
        manifest = {
            "manifest_version": 1, "mode": mode,
            "tokenizer_sha256": tok.sha256, "vocab_size": tok.vocab_size,
            "template_version": TEMPLATE_VERSION,
            "dtype": TOKEN_DTYPE if mode == "pretrain" else "jsonl",
            "seed": seed, "val_fraction": val_fraction, "split_version": SPLIT_VERSION,
            "split_key": "normalized_document" if mode == "pretrain" else "normalized_first_user_question",
            "dedup_normalization": "NFKC + collapsed whitespace; preserve case and conversation roles",
            "sources": sources, "counts": counts, "splits": splits,
        }
        if mode == "pretrain":
            manifest["byte_order"] = "little"
            manifest["text_document_policy"] = "blank-line-separated paragraphs"
        else:
            manifest["context_length"] = context_length
            manifest["overlong_policy"] = "reject_and_count_no_truncation"
        with (stage / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        _publish(stage, output, force)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("pretrain", "sft"))
    parser.add_argument("--input", required=True, action="append", nargs="+", type=Path,
                        help="Local files; repeat --input or list multiple files")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tokenizer_v2.model"))
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", "--context", type=int, default=1024,
                        help="Reject complete SFT examples longer than context+1 (default: 1024)")
    parser.add_argument("--force", action="store_true", help="Replace an existing dataset only after successful prep")
    args = parser.parse_args(argv)
    try:
        manifest = prepare_dataset([path for group in args.input for path in group], args.output,
                                   args.tokenizer, mode=args.mode, val_fraction=args.val_fraction,
                                   seed=args.seed, context_length=args.context_length, force=args.force)
    except (OSError, ValueError, OverflowError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
