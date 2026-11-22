"""Audit configured training artifacts, not universal factual correctness.

Run from Mentrafiai: python -m data_prep.audit_llm120 --full-pretrain
The full binary document scan reads about 2.4 GB and uses an in-memory hash set.
Only explicitly selected decoded samples are language/content-screened. JSONL
record/length/hash checks cover every configured SFT record. Old source files
are inventoried but are not implicitly included in the training mixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from data_prep.build_llm120_sft import HINGLISH
from data_prep.mutual_fund_curriculum import TOPICS
from llm120.data import DISK_DTYPE, _validated_sft
from llm120.prepare import _normalized_text
from llm120.tokenization import Tokenizer, file_sha256
from llm120.train import load_config, load_datasets

ROOT = Path(__file__).resolve().parent.parent


def binary_documents(path: Path, eos: int):
    """Yield whole uint16 documents, including EOS, with bounded block memory."""
    tokens = np.memmap(path, dtype=DISK_DTYPE, mode="r")
    carry = np.empty(0, dtype=DISK_DTYPE)
    for start in range(0, len(tokens), 4 * 1024 * 1024):
        block = np.concatenate((carry, tokens[start:start + 4 * 1024 * 1024]))
        previous = 0
        for end in np.flatnonzero(block == eos):
            yield block[previous:end + 1]
            previous = end + 1
        carry = block[previous:].copy()
    if len(carry):
        raise ValueError(f"{path}: final document has no EOS")


def fingerprint(tokens) -> bytes:
    return hashlib.sha256(tokens.tobytes()).digest()


def full_pretrain(root: Path, tok: Tokenizer) -> dict:
    result, validation = {}, set()
    for split in ("val", "train"):
        seen, histogram = set(), np.zeros(tok.vocab_size, dtype=np.int64)
        counts = Counter()
        for doc in binary_documents(root / f"{split}.bin", tok.eos_id):
            if int(doc.max()) >= tok.vocab_size:
                raise ValueError("Out-of-vocabulary token")
            key = fingerprint(doc)
            counts["documents"] += 1
            counts["tokens"] += len(doc)
            if key in seen:
                counts["duplicate_documents"] += 1
                counts["duplicate_tokens"] += len(doc)
            if split == "train" and key in validation:
                counts["exact_document_overlap_with_val"] += 1
            seen.add(key)
        # One blockwise pass is much faster than a histogram per document.
        data = np.memmap(root / f"{split}.bin", dtype=DISK_DTYPE, mode="r")
        for pos in range(0, len(data), 4 * 1024 * 1024):
            histogram += np.bincount(data[pos:pos + 4 * 1024 * 1024], minlength=tok.vocab_size)
        counts["unique_documents"] = len(seen)
        result[split] = dict(counts)
        result[split]["special_token_counts"] = {name: int(histogram[getattr(tok, f"{name}_id")])
                                               for name in ("pad", "bos", "eos", "unk", "user", "assistant")}
        if histogram[tok.pad_id] or histogram[tok.bos_id] or histogram[tok.unk_id]:
            raise ValueError(f"Unexpected PAD/BOS/UNK in {split} pretraining text")
        if split == "val":
            validation = seen
    if result["train"].get("exact_document_overlap_with_val", 0):
        raise ValueError("Exact pretraining document leakage")
    return result


def decoded_screen(root: Path, tok: Tokenizer) -> dict:
    """An interpretable sample screen, NOT English detection or fact certification."""
    result = {}
    for split in ("train", "val"):
        data = np.memmap(root / f"{split}.bin", dtype=DISK_DTYPE, mode="r")
        counts = Counter()
        for start in np.linspace(0, max(0, len(data) - 256), num=512, dtype=np.int64):
            text = tok.decode(data[int(start):int(start) + 256].tolist())
            letters = [ch for ch in text if ch.isalpha()]
            counts["windows_screened"] += 1
            if letters and sum(c.isascii() for c in letters) / len(letters) >= 0.9:
                counts["mostly_latin_letter_windows"] += 1
            if re.search(r"mutual fund|\bSIP\b|\bNAV\b|\bSEBI\b|\bAMFI\b", text, re.I):
                counts["mutual_fund_keyword_windows"] += 1
            if re.search(r"\b401\(?k|\bIRA\b|\bIRS\b|\bSEC\b", text):
                counts["foreign_finance_keyword_windows"] += 1
            if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text):
                counts["email_pattern_windows"] += 1
        result[split] = dict(counts)
    return result


def sft_audit(root: Path, tok: Tokenizer) -> dict:
    source_path = ROOT / "data/raw/sft/llm120_education.jsonl"
    source = [json.loads(line) for line in source_path.open(encoding="utf-8")]
    topic_by_question = {_normalized_text(next(m["content"] for m in r["messages"] if m["role"] == "user")).casefold():
                         r["topic"] for r in source}
    all_hashes, question_splits, coverage, results = set(), {}, {}, {}
    for split in ("train", "val"):
        stats, lengths, topics = Counter(), [], Counter()
        for line in (root / f"{split}.jsonl").open(encoding="utf-8"):
            record = json.loads(line)
            ids, _ = _validated_sft(record, tok, 1024)
            lengths.append(len(ids))
            messages = record["messages"]
            question = _normalized_text(next(m["content"] for m in messages if m["role"] == "user")).casefold()
            key = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if key in all_hashes:
                raise ValueError("Duplicate SFT conversation")
            all_hashes.add(key)
            if question in question_splits and question_splits[question] != split:
                raise ValueError("SFT first-question leakage")
            question_splits[question] = split
            if question not in topic_by_question:
                raise ValueError("Prepared SFT question missing from authored source")
            topics[topic_by_question[question]] += 1
            stats["records"] += 1
            if sum(m["role"] == "assistant" for m in messages) > 1:
                stats["multi_turn_records"] += 1
            for message in messages:
                if message["role"] == "system":
                    continue
                text = message["content"]
                if any(ch.isalpha() and not ch.isascii() for ch in text) or HINGLISH.search(text):
                    raise ValueError(f"Non-English-script/Hinglish marker in SFT {split}: {text[:80]}")
                if message["role"] == "assistant" and re.search(
                    r"Never stop your SIP|SEBI-compliant Indian mutual fund advisor|is far superior|100% sovereign-guaranteed", text, re.I
                ):
                    raise ValueError("Legacy unsafe answer phrase survived SFT rebuild")
        coverage[split] = topics
        results[split] = {**stats, "max_serialized_tokens": max(lengths), "topic_counts": dict(topics)}
    missing = sorted(set(TOPICS) - set(coverage["train"]) - set(coverage["val"]))
    if missing:
        raise ValueError(f"Missing curriculum topics: {missing}")
    results["coverage"] = {"defined_topics": len(TOPICS), "missing_from_combined_splits": missing,
                           "topics_only_in_validation": sorted(set(TOPICS) - set(coverage["train"])),
                           "note": "Topic presence is not evidence of factual correctness or trained recall."}
    return results


def audit(full: bool = False) -> dict:
    tok = Tokenizer(ROOT / "tokenizer/tokenizer_v2.model")
    report = {"technical_checks_passed": False, "all_facts_verified": False,
              "scope": "Configured artifacts only; full SFT validation and optional full pretrain exact-document scan",
              "tokenizer_sha256": tok.sha256, "datasets": {}}
    for stage in ("pretrain", "sft"):
        config = load_config(ROOT / f"configs/llm120_{stage}.yaml")
        tok.validate_config(config.model)
        train, val, identity = load_datasets(config, tok)
        root = Path(config.training.data_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        info = {"train_examples_or_windows": len(train), "val_examples_or_windows": len(val),
                "splits": manifest["splits"], "identity": identity}
        if stage == "sft":
            for source in manifest["sources"]:
                # Source may move between machines; basename is the recorded source artifact.
                local = ROOT / "data/raw/sft" / Path(source["path"].replace("\\", "/")).name
                if file_sha256(local) != source["sha256"]:
                    raise ValueError("SFT authored source hash mismatch")
            info["record_audit"] = sft_audit(root, tok)
        else:
            info["decoded_sample_screen"] = decoded_screen(root, tok)
            if full:
                info["full_document_audit"] = full_pretrain(root, tok)
                for name in ("train", "val"):
                    measured = info["full_document_audit"][name]
                    if measured["tokens"] != manifest["splits"][name]["tokens"] or measured["documents"] != manifest["splits"][name]["documents"]:
                        raise ValueError("Pretrain document/token counts differ from manifest")
            base = manifest.get("counts", {}).get("base_train_tokens")
            if base:
                total = manifest["splits"]["train"]["tokens"]
                info["identified_indian_mf_tail_tokens"] = total - base
                info["identified_indian_mf_tail_fraction"] = (total - base) / total
        report["datasets"][stage] = info
    report["source_inventory_not_automatically_used_for_training"] = [
        {"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size}
        for path in sorted((ROOT / "data").rglob("*.jsonl"))
        if "llm120_pretrain" not in path.parts and "llm120_sft" not in path.parts
    ]
    report["warnings"] = [
        "English/Latin script and keyword screens are heuristics, not language-quality certification.",
        "The inherited pretraining corpus includes general English and non-Indian finance; it is not exclusively mutual funds.",
        "Only the identified appended Indian-MF portion has source-level token counts; the base mix lacks complete surviving provenance.",
        "A full binary hash/overlap scan cannot verify economic, tax, regulatory, or arithmetic truth in every document.",
        "The small authored SFT set is a pilot. Expert review and an independent behavioral test set are still required.",
        "The tokenizer normalizes whitespace; do not expect faithful newline or code-indentation generation.",
        "Training samples with replacement; nominal token budgets are not completed epochs or exposure caps.",
        "Keep dated tax and fund data in retrieval context; do not treat memorized text as live investment advice.",
        "Existing serve.py loads the legacy 102M model, not a newly trained llm120 checkpoint.",
    ]
    report["technical_checks_passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-pretrain", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "data/llm120_audit.json")
    args = parser.parse_args()
    report = audit(args.full_pretrain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"technical_checks_passed": True, "all_facts_verified": False, "report": str(args.output),
               "sft": {s: report["datasets"]["sft"]["record_audit"][s]["records"] for s in ("train", "val")}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
