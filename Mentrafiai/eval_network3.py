"""Measure Network 3 (llm120 SFT model) on held-out grounded conversations.

    python eval_network3.py --checkpoint checkpoints/llm120_sft/best.pt [--limit 100] [--show 5]

Uses the validation split of data/llm120_sft (conversations the model never
trained on). Each prompt is the system context + question; the model answers
greedily. Automatic checks (all are proxies, not a human judgement):

  numbers_grounded   every number in the reply appears in the prompt
  key_figures        share of the reference answer's numbers that the reply also contains
  fund_names_valid   no scheme name in the reply that is absent from the context
  behaviour_ok       asks for missing details / says "no verified data" when the reference does
  finished           the reply ended with end-of-text, not by running out of tokens

"usable" = numbers_grounded AND fund_names_valid AND behaviour_ok AND finished.
Read the printed samples too: a high score on a small val set is not proof of quality.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from llm120.checkpoint import load_checkpoint
from llm120.generate import generate_ids
from llm120.grounding import CONTEXT_START, ungrounded_numbers, unlisted_scheme_mentions
from llm120.tokenization import Tokenizer, serialize
import serve  # reply_problems: the same checks the server applies

ROOT = Path(__file__).resolve().parent
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
FUND_NAME = re.compile(r"(?:[A-Z][\w&.']*\s){1,5}[\w ]*?- Direct Growth")


def numbers(text: str) -> set[str]:
    return {m.group(0).rstrip(",").replace(",", "") for m in NUMBER.finditer(text) if len(m.group(0).replace(".", "")) > 1}


def fund_names_ok(reply: str, context: str) -> bool:
    """Every scheme-like name in the reply must be a scheme listed in the context."""
    return not unlisted_scheme_mentions(reply, context)


def allocation_ok(reply: str, context: str) -> bool:
    """Each "<percent> ... Rs <amount>" pairing in the reply must match an allocation line in the context."""
    pairs = set(re.findall(r"(\d+)% (Rs [\d,]+)/month", context))
    if not pairs:
        return True
    percent_of = {amount: pct for pct, amount in pairs}
    for amount, pct in re.findall(r"(Rs [\d,]+) \((\d+)%\)", reply) + \
            [(a, p) for p, a in re.findall(r"(\d+)%, or (Rs [\d,]+) a month", reply)] + \
            [(a, p) for a, p in re.findall(r"(Rs [\d,]+) a month \((\d+)%\)", reply)]:
        if percent_of.get(amount) != pct:
            return False
    return True


def intent_of(system: str) -> str:
    match = re.search(r"Intent: ([A-Z_]+)\.", system)
    return match.group(1) if match else "UNKNOWN"


def behaviour_ok(reference: str, reply: str, context: str) -> bool:
    low = reply.lower()
    if "do not have verified data" in reference:
        return "verified" in low and ("cannot" in low or "do not have" in low or "no verified" in low)
    if re.search(r"Missing: (?!none)", context):
        return "please tell me" in low or "?" in reply or "need" in low
    if "no verified scheme" in reference:
        return "no verified" in low or "cannot name" in low
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/llm120_sft/best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data/llm120_sft/val.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer/tokenizer_v2.model")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N grounded examples")
    parser.add_argument("--show", type=int, default=4, help="Print this many sample replies")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, help="Write all results to this JSON file")
    args = parser.parse_args()

    tok = Tokenizer(args.tokenizer)
    model, payload = load_checkpoint(args.checkpoint, tok, device=args.device)
    if args.device == "cuda":
        model.half()
    stage = payload["metadata"].get("stage")
    print(f"checkpoint: {args.checkpoint} (stage={stage}, {model.num_parameters():,} parameters)")
    rows = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    rows = [r["messages"] for r in rows if r["messages"][0]["role"] == "system" and CONTEXT_START in r["messages"][0]["content"]]
    if args.limit:
        rows = rows[:args.limit]
    print(f"grounded validation conversations: {len(rows)}")

    results, by_intent = [], defaultdict(list)
    for index, messages in enumerate(rows):
        reference, prompt_messages = messages[-1]["content"], messages[:-1]
        ids, _ = serialize(prompt_messages, tok, add_generation_prompt=True)
        room = model.config.context_length - len(ids)
        if room < 64:
            continue
        tensor = torch.tensor([ids], device=args.device)
        result = generate_ids(model, tensor, max_new_tokens=min(450, room), temperature=0,
                              repetition_penalty=args.repetition_penalty)
        reply = tok.decode(result.token_ids).strip()
        prompt_text = " ".join(m["content"] for m in prompt_messages)
        context = prompt_messages[0]["content"]
        names_ok = fund_names_ok(reply, context)
        alloc_ok = allocation_ok(reply, context)
        problems = serve.reply_problems(reply, context)
        ref_numbers = numbers(reference)
        record = {
            "intent": intent_of(context),
            "numbers_grounded": not ungrounded_numbers(reply, prompt_text),
            "key_figures": (len(ref_numbers & numbers(reply)) / len(ref_numbers)) if ref_numbers else 1.0,
            "fund_names_valid": names_ok,
            "behaviour_ok": behaviour_ok(reference, reply, context),
            "allocation_ok": alloc_ok,
            "no_invented_content": not problems,
            "finished": result.finish_reason == "eos",
            "reply": reply, "reference": reference, "question": prompt_messages[-1]["content"],
        }
        record["usable"] = all(record[k] for k in ("numbers_grounded", "fund_names_valid", "behaviour_ok", "finished",
                                                   "allocation_ok", "no_invented_content"))
        results.append(record)
        by_intent[record["intent"]].append(record)
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(rows)}", flush=True)

    def summarize(items):
        n = len(items)
        return {"n": n,
                "usable": sum(r["usable"] for r in items) / n,
                "numbers_grounded": sum(r["numbers_grounded"] for r in items) / n,
                "key_figures": statistics.mean(r["key_figures"] for r in items),
                "fund_names_valid": sum(r["fund_names_valid"] for r in items) / n,
                "behaviour_ok": sum(r["behaviour_ok"] for r in items) / n,
                "allocation_ok": sum(r["allocation_ok"] for r in items) / n,
                "no_invented_content": sum(r["no_invented_content"] for r in items) / n,
                "finished": sum(r["finished"] for r in items) / n}

    overall = summarize(results)
    print("\n=== OVERALL ===")
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in overall.items()}))
    print("\n=== BY INTENT (usable / key_figures) ===")
    for intent, items in sorted(by_intent.items()):
        s = summarize(items)
        print(f"{intent:28s} n={s['n']:3d} usable={s['usable']:.2f} key_figures={s['key_figures']:.2f}")
    failures = [r for r in results if not r["usable"]]
    print(f"\n=== {min(args.show, len(failures))} FAILED SAMPLES ===")
    for r in failures[:args.show]:
        print(f"[{r['intent']}] Q: {r['question']}\n  REPLY: {r['reply'][:500]}\n  REF:   {r['reference'][:300]}\n")
    print(f"=== {args.show} RANDOM SAMPLES ===")
    for r in results[::max(1, len(results) // max(1, args.show))][:args.show]:
        print(f"[{r['intent']}] Q: {r['question']}\n  REPLY: {r['reply'][:500]}\n")
    if args.report:
        args.report.write_text(json.dumps({"overall": overall, "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
        print("report:", args.report)


if __name__ == "__main__":
    main()
