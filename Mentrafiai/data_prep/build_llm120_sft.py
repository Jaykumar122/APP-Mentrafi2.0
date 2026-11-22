"""Build an English educational SFT set; never reuse unreviewed legacy answers.

Legacy sources stay unchanged. Only recognized English planning questions are
retained: their unsafe scheme recommendations are replaced with educational
risk/horizon guidance. Old FAQs (some with blanket promises and stale tax facts)
are replaced by the explicit curriculum. Repetitive templates are capped per
risk/horizon bucket, rather than padded back to the old dataset size.

Run from Mentrafiai: python -m data_prep.build_llm120_sft --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from data_prep.build_grounded_sft import grounded_records
from data_prep.mutual_fund_curriculum import METADATA, SOURCES as REFERENCES, SYSTEM_PROMPT, TOPICS, curriculum_records
from llm120.data import _validated_sft
from llm120.prepare import _normalized_text, _publish, prepare_dataset
from llm120.tokenization import Tokenizer, file_sha256

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "data/processed/sft_train_v6_curated.jsonl",
           ROOT / "data/processed/sft_val_v6_curated.jsonl"]
HORIZON = re.compile(r"\*\*(\d+)-year horizon\*\*")
BUDGET = re.compile(r"\*\*₹([\d,]+)/month\*\*")
RISK = re.compile(r"\*\*(conservative|moderate|aggressive) risk profile\*\*")
HINGLISH = re.compile(r"\b(kya|hai|hain|karna|karo|ke liye|saal|hota|batao|mujhe|chahiye|kaise|samjhao)\b", re.I)


def english_question(text: str) -> bool:
    # Conservative known-template gate, not a general-purpose language detector.
    return (not any(ch.isalpha() and not ch.isascii() for ch in text)
            and not HINGLISH.search(text)
            and bool(re.match(r"(?:I am\b|My name is\b|Recommend\b|Suggest\b)", text)))


def planning_answer(budget: int, years: int, risk: str) -> str:
    if budget <= 0 or years <= 0 or risk not in {"conservative", "moderate", "aggressive"}:
        raise ValueError("Invalid planning inputs")
    if years <= 3:
        guidance = (
            "For a firm goal within three years, capital stability and access usually matter more than "
            "chasing returns. An equity-heavy portfolio, including an equity-heavy hybrid fund, can lose "
            "value near withdrawal. Compare suitable cash-management alternatives; even overnight, liquid, "
            "and short-duration debt funds carry risks and are not guaranteed deposits."
        )
    elif years <= 5:
        guidance = (
            "A three-to-five-year goal can still be exposed to substantial equity losses. Assess whether "
            "the deadline can move and how much loss the goal can absorb before considering market exposure. "
            "Compare credit quality, duration, liquidity, and costs rather than assuming a debt fund is safe."
        )
    else:
        guidance = (
            "A longer horizon may allow diversified equity exposure alongside a suitable stability component, "
            "but it does not guarantee positive returns. Allocation depends on loss capacity, existing holdings, "
            "and deadline flexibility, not just a risk label. Review overlap and reduce goal risk as withdrawal approaches."
        )
    if risk == "conservative" and years > 3:
        guidance += " A conservative investor should not be pushed into equity merely because the horizon is long."
    return (
        f"You described a monthly budget of Rs {budget:,}, a {years}-year horizon, and a {risk} risk preference. "
        f"A SIP means periodic purchases into a scheme; it is not a return guarantee. {guidance} "
        "Before discussing a portfolio, please clarify your goal, emergency savings, existing investments, "
        "and ability to tolerate losses. No verified scheme facts were supplied, so I cannot select named funds "
        "or invent rankings, NAVs, or expected returns. This is general education, not a personalized recommendation."
    )


def sip_value(payment: float, years: int, annual_rate: float, *, beginning: bool = False) -> float:
    """Illustrative monthly nominal compounding, NOT a CAGR-to-month conversion."""
    if payment < 0 or years <= 0 or not math.isfinite(annual_rate) or annual_rate < 0:
        raise ValueError("Invalid illustration inputs")
    n, rate = years * 12, annual_rate / 12
    factor = n if rate == 0 else math.expm1(n * math.log1p(rate)) / rate
    return payment * factor * (1 + rate if beginning else 1)


def calculation_records() -> list[dict]:
    records = []
    for years in (1, 3, 5, 10, 15):
        for payment in (500, 2000, 5000):
            for annual in (0.0, 0.08):
                final = sip_value(payment, years, annual)
                question = (f"For an illustration only, what is the future value of Rs {payment:,} paid at the "
                            f"end of each month for {years} years at a constant nominal annual rate of {annual:.0%}, "
                            "compounded monthly? Ignore fees and taxes.")
                answer = (f"Using your hypothetical assumptions, P = Rs {payment:,}, n = {years * 12} monthly payments, "
                          f"and i = {annual:.2f}/12. For end-of-month payments, FV = P * ((1+i)^n - 1)/i; "
                          "when i is zero, use FV = P*n. "
                          f"Total contributions are Rs {payment * years * 12:,}; the calculated future value is "
                          f"Rs {final:,.2f}. This assumes a constant nominal rate, not an effective annual return "
                          "or a scheme forecast. Real mutual fund returns vary and may be negative; charges, taxes, "
                          "and payment timing change outcomes. A beginning-of-month illustration multiplies FV by (1+i).")
                records.append({"topic": "sip_calculation", "illustration": {"payment": payment, "years": years,
                                "annual_nominal_rate": annual, "expected_value": round(final, 2)},
                                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                             {"role": "user", "content": question},
                                             {"role": "assistant", "content": answer}]})
    return records


GROUNDED_COUNT = 2400
# Leave room so a live prompt of this size still fits; inference uses the same limit.
GROUNDED_MAX_TOKENS = 1000


def grounded_sft_records(count: int = GROUNDED_COUNT) -> list[dict]:
    """Network 3 pipeline examples: verified context in, faithful explanation out."""
    tokenizer = Tokenizer(ROOT / "tokenizer/tokenizer_v2.model")

    def fits(messages) -> bool:
        try:
            ids, _ = _validated_sft({"messages": messages}, tokenizer, 1024)
        except OverflowError:
            return False
        return len(ids) <= GROUNDED_MAX_TOKENS

    return grounded_records(count, fits=fits)


def build_records(per_bucket: int = 8) -> tuple[list[dict], dict]:
    if type(per_bucket) is not int or per_bucket < 1:
        raise ValueError("per_bucket must be positive")
    stats = Counter()
    questions: dict[str, tuple[str, int, int, str]] = {}
    for source in SOURCES:
        with source.open(encoding="utf-8") as stream:
            for raw in stream:
                stats["legacy_rows_read"] += 1
                messages = json.loads(raw)["messages"]
                user = next(m["content"] for m in messages if m["role"] == "user")
                answer = messages[-1]["content"]
                horizon, budget, risk = HORIZON.search(answer), BUDGET.search(answer), RISK.search(answer)
                if not (horizon and budget and risk):
                    stats["legacy_answers_excluded_unreviewed"] += 1
                    continue
                if not english_question(user):
                    stats["questions_excluded_language_or_template"] += 1
                    continue
                key = _normalized_text(user).casefold()
                questions.setdefault(key, (user, int(budget.group(1).replace(",", "")),
                                           int(horizon.group(1)), risk.group(1)))
    records, buckets = curriculum_records() + calculation_records() + grounded_sft_records(), Counter()
    # Stable content ordering, not source order or hand-picked validation scores.
    for key in sorted(questions, key=lambda q: hashlib.sha256(q.encode()).hexdigest()):
        user, budget, years, risk = questions[key]
        bucket = (years, risk)
        if buckets[bucket] >= per_bucket:
            stats["repetitive_questions_capped"] += 1
            continue
        records.append({"topic": "risk_horizon_planning", "planning": {"budget": budget, "years": years, "risk": risk},
                        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                     {"role": "user", "content": user},
                                     {"role": "assistant", "content": planning_answer(budget, years, risk)}]})
        buckets[bucket] += 1
    stats["legacy_answers_reused"] = 0
    stats["planning_questions_retained"] = sum(buckets.values())
    stats["records_authored_or_corrected"] = len(records)
    return records, dict(stats)


def curriculum_training_split(stage: Path, records: list[dict], tok: Tokenizer, manifest: dict) -> None:
    """Teach all declared definitions; hold out only planning/calculation/dialogue questions.

    This is an explicit curriculum policy, not a held-out definition-knowledge
    benchmark. Never select curriculum membership based on observed model scores.
    """
    mandatory = {_normalized_text(r["messages"][1]["content"]).casefold()
                 for r in records if r["topic"] in TOPICS}
    rows = {name: [json.loads(line) for line in (stage / f"{name}.jsonl").open(encoding="utf-8")]
            for name in ("train", "val")}
    remaining = []
    for record in rows["val"]:
        question = _normalized_text(record["messages"][1]["content"]).casefold()
        if question in mandatory:
            rows["train"].append(record)
        else:
            remaining.append(record)
    rows["val"] = remaining
    if not remaining:
        raise ValueError("No held-out scenarios remain; add distinct scenario questions")
    for name, examples in rows.items():
        path = stage / f"{name}.jsonl"
        tokens = 0
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for record in examples:
                tokens += len(_validated_sft(record, tok, 1024)[0])
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest["splits"][name] = {"file": path.name, "documents": len(examples), "tokens": tokens,
                                    "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
    manifest["split_policy"] = "All declared definition questions train-only; other first-user-question groups content-hash split"
    manifest["validation_scope"] = "Held-out scenarios, calculations and dialogues, NOT held-out definition knowledge"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/llm120_sft")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer/tokenizer_v2.model")
    parser.add_argument("--per-bucket", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error("Output exists; use --force to rebuild")
    records, stats = build_records(args.per_bucket)
    tok = Tokenizer(args.tokenizer)
    for record in records:
        _validated_sft(record, tok, 1024)  # Fail, rather than silently dropping authored topics.
    source = ROOT / "data/raw/sft/llm120_education.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    temporary = source.with_suffix(".jsonl.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(source)
    finally:
        temporary.unlink(missing_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(prefix=".llm120-sft-", dir=args.output.parent))
    stage = staging_parent / "dataset"
    try:
        manifest = prepare_dataset([source], stage, tok, mode="sft", val_fraction=0.10, context_length=1024)
        curriculum_training_split(stage, records, tok, manifest)
        manifest["education_provenance"] = {
        **METADATA, "build_stats": stats, "references": REFERENCES,
        "builder_sha256": file_sha256(Path(__file__)),
        "curriculum_sha256": file_sha256(ROOT / "data_prep/mutual_fund_curriculum.py"),
        "legacy_question_sources": [{"path": str(p.relative_to(ROOT)), "sha256": file_sha256(p)} for p in SOURCES],
        "topic_counts": dict(Counter(r["topic"] for r in records)),
        "scope": "English educational seed set, not production advisory certification",
    }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _publish(stage, args.output, args.force)
    finally:
        shutil.rmtree(staging_parent)
    print(json.dumps({"build": stats, "counts": manifest["counts"], "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
