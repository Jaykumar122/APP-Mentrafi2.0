"""
SFT v3 augmentation — generation STABILITY + NAME FIDELITY focus.

Why v3 (see the sampling-diagnosis history): even with tuned sampling the 100M
model still drifts late in a generation and, most often, CORRUPTS THE FUND NAME
("Birlarath AXis", "Growid Aggressionive"). Two data-side levers this script
pulls, on top of the v2 faithful-by-construction approach:

  1. SHORTER, tighter-length responses. Longer generations correlate with more
     late-sequence drift, so every v3 recommendation answer is 2 sentences with a
     consistent structure (see REC_V3 / NAMECOPY_V3) — a much tighter token-length
     distribution than v2's 3-4 sentence templates.
  2. Heavier NAME-fidelity weighting. The fund name is the field failing most, so
     ~40% of the new examples repeat the pick's name VERBATIM at both the start
     and the end of the answer (NAMECOPY_V3), vs v2's ~18% copy-emphasis that was
     weighted toward numbers.

Winner selection reuses the EXACT serving-side score_fund() (via choose_winner),
so the training signal stays consistent with production ranking — same as v2.

READ-ONLY w.r.t. the live DB and all existing training files. Writes ONE preview
file for review; does NOT touch sft_train_v2/sft_val_v2 and does NOT train.

Run:  python data_prep/augment_sft_data_v3.py --out data/processed/sft_augment_v3_preview.jsonl
"""
import sys, os, json, random, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Reuse the v2 machinery verbatim so pool-sampling, candidate selection, winner
# scoring, and the prompt/candidate-block format stay identical to production.
from data_prep.augment_sft_data import (
    load_diverse_funds, pick_candidates, choose_winner, candidate_block,
    make_profile, RISK_WORDS, QUESTION_TEMPLATES,
)

# PLACEHOLDER_TEMPLATES

# ---------------------------------------------------------------------------
# Concise recommendation templates — exactly 2 sentences, consistent shape.
# The pick's name leads; only the winner's perf% and expense% appear as numbers
# (faithful-by-construction); risk word is the winner's own risk_level.
# ---------------------------------------------------------------------------
REC_V3 = [
    ("{name} is my pick for you — a {cat} fund at {risk} risk, which suits your "
     "{riskw} tolerance. It has returned {perf}% at a {er}% expense ratio, a solid "
     "fit for your {horizon}-year SIP."),
    ("For your {riskw}-risk, {horizon}-year goal, go with {name}. This {cat} fund "
     "carries {risk} risk and has delivered {perf}% returns at a {er}% expense ratio."),
    ("I'd recommend {name}: a {cat} fund with {risk} risk and {perf}% returns at a "
     "{er}% expense ratio. It matches your {riskw} tolerance and {horizon}-year "
     "horizon well."),
]

# ---------------------------------------------------------------------------
# Name-fidelity templates — the pick's name appears VERBATIM at the START and
# again at the END, so the model is trained to copy it exactly and to close on
# the same string it opened with (directly targets the name-corruption failure).
# ---------------------------------------------------------------------------
NAMECOPY_V3 = [
    ("{name}. That's my recommendation — a {cat} fund at {risk} risk with {perf}% "
     "returns and a {er}% expense ratio, matched to your {riskw} tolerance. In one "
     "line: go with {name}."),
    ("Go with {name}. It's a {cat} fund ({risk} risk, {perf}% return, {er}% expense "
     "ratio) that fits your {riskw} tolerance over {horizon} years. My pick: {name}."),
    ("My pick is {name} — {cat}, {risk} risk, {perf}% return, {er}% expense ratio, "
     "in line with your {riskw} tolerance and {horizon}-year plan. Again, that's {name}."),
]


def _build_user(profile, cands, rng):
    return (QUESTION_TEMPLATES[rng.randrange(len(QUESTION_TEMPLATES))].format(
                age=profile["age"], sip=profile["sip"],
                risk=RISK_WORDS[profile["risk"]], horizon=profile["horizon"])
            + "\n\n" + candidate_block(cands)
            + "\n\nBased on the above, give a recommendation with reasoning.")


def _gen(pool, rng, templates):
    profile = make_profile(rng)
    cands = pick_candidates(pool, profile["risk"], rng)
    if len(cands) < 2:
        return None
    win = choose_winner(cands, profile["risk"], profile["horizon"])
    user = _build_user(profile, cands, rng)
    asst = templates[rng.randrange(len(templates))].format(
        name=win["fund_name"], cat=win["category"], risk=win["risk_level"],
        riskw=RISK_WORDS[profile["risk"]], perf=win["performance"],
        er=win["expense_ratio"], horizon=profile["horizon"])
    return {"user": user, "assistant": asst}


def gen_recommendation_v3(pool, rng):
    return _gen(pool, rng, REC_V3)


def gen_namecopy_v3(pool, rng):
    return _gen(pool, rng, NAMECOPY_V3)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _length_report(rows):
    """Token-length distribution of the generated assistant answers, compared to
    the v2 recommendation answers, to confirm v3 is SHORTER and TIGHTER."""
    try:
        from tokenizer.tokenizer_utils import Tokenizer
        tok = Tokenizer("tokenizer/tokenizer.model")
    except Exception as e:  # tokenizer optional for the pure-data step
        print(f"\n(length report skipped: {e})")
        return

    def pct(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(round(p / 100.0 * (len(v) - 1))))] if v else 0

    def stats(label, texts):
        lens = [len(tok.encode(t)) for t in texts]
        if not lens:
            print(f"  {label}: (none)"); return
        spread = pct(lens, 90) - pct(lens, 10)
        print(f"  {label:22} n={len(lens):5}  p10={pct(lens,10):3}  p50={pct(lens,50):3}  "
              f"p90={pct(lens,90):3}  max={max(lens):3}  p10-p90 spread={spread}")

    print("\n  ASSISTANT length (tokens) — tighter spread = less late-drift risk:")
    stats("v3 preview (this run)", [ex["assistant"] for _, ex in rows])
    v2_path = "data/processed/sft_train_v2.jsonl"
    if os.path.exists(v2_path):
        import re
        v2_rec = []
        with open(v2_path, encoding="utf-8") as fh:
            for line in fh:
                ex = json.loads(line)
                # recommendation-shaped only, for a like-for-like comparison
                if "Candidate funds:" in ex.get("user", ""):
                    v2_rec.append(ex["assistant"])
        stats("v2 rec answers", v2_rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/sft_augment_v3_preview.jsonl")
    ap.add_argument("--n_rec", type=int, default=1500, help="concise recommendation examples")
    ap.add_argument("--n_namecopy", type=int, default=900,
                    help="name-fidelity examples (name verbatim at start AND end)")
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print("Loading diverse real-fund pool from PostgreSQL ...")
    pool = load_diverse_funds(limit=4000, seed=args.seed)
    print(f"  pool size (distinct fund families): {len(pool)}")

    rows, tagged = [], {"rec": 0, "namecopy": 0}
    for _ in range(args.n_rec):
        ex = gen_recommendation_v3(pool, rng)
        if ex:
            rows.append(("rec", ex)); tagged["rec"] += 1
    for _ in range(args.n_namecopy):
        ex = gen_namecopy_v3(pool, rng)
        if ex:
            rows.append(("namecopy", ex)); tagged["namecopy"] += 1

    rng.shuffle(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for _, ex in rows:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total = len(rows) or 1
    print(f"\n✓ wrote {len(rows)} preview examples -> {args.out}")
    print(f"  concise recommendation : {tagged['rec']}  ({100*tagged['rec']/total:.0f}%)")
    print(f"  name-fidelity copy     : {tagged['namecopy']}  ({100*tagged['namecopy']/total:.0f}%)")

    _length_report(rows)


if __name__ == "__main__":
    main()

