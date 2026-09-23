"""
SFT data augmentation for faithfulness (Parts 2-5 of the improvement plan).

Motivation (see audit_sft_faithfulness.py output): the current SFT set has only
~17 distinct fund names and ~22 distinct numeric values, so the model memorizes a
handful of names/numbers instead of learning to COPY whatever is in the prompt.
This script produces new, high-diversity examples whose assistant answers are
faithful-by-construction (every name/category/risk/number is copied verbatim from
the chosen candidate), plus copy-emphasis, communication-skill, and finance-
literacy examples.

READ-ONLY w.r.t. the live DB and the existing training files. It writes a single
preview file for review; it does NOT touch sft_train.jsonl / sft_val.jsonl and
does NOT start training.

Run:  python data_prep/augment_sft_data.py --out data/processed/sft_augment_preview.jsonl
"""
import sys, os, re, json, random, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from inference.mutual_fund_advisor import (
    _load_dotenv, _connect, _derive_risk_level, _is_single_asset,
    score_fund, UserProfile,
)

RISK_ORDER = ["Low", "Moderate", "High", "Very High"]

# The DB has no expense-ratio column; synthesize a plausible, category-consistent
# ratio so the candidate line and the assistant answer agree (internal consistency
# is what matters for teaching faithful copying).
ER_BY_CATEGORY = {
    "Debt": (0.15, 0.55), "Index": (0.10, 0.35), "Hybrid": (0.45, 0.95),
    "ELSS": (0.60, 1.05), "Gold": (0.40, 0.80), "Equity": (0.45, 1.15),
}


def synth_er(category, rng):
    lo, hi = ER_BY_CATEGORY.get(category, (0.30, 1.00))
    return round(rng.uniform(lo, hi), 2)


def _base_name(name):
    """Strip plan/option suffixes so share-class variants of one scheme collapse
    to a single family name — maximizes NAME diversity in the sampled pool."""
    n = re.split(r"\s*-\s*(direct|regular)\b", name, flags=re.I)[0]
    n = re.sub(r"\b(growth|idcw|income distribution.*|payout|reinvest\w*|"
               r"plan|option|dividend|regular|direct)\b", "", n, flags=re.I)
    n = re.sub(r"[\s\-]+", " ", n).strip(" -")
    # Clean up leftovers from the stripping above: drop any parenthetical remnant
    # ("Fund (Regular )" -> "Fund"), dangling connectors, and stray punctuation.
    n = re.sub(r"\([^)]*\)?", "", n)              # any (...) remnant, closed or not
    n = re.sub(r"\s+of\s*$", "", n, flags=re.I)   # trailing connector "... of"
    n = re.sub(r"\s{2,}", " ", n)
    return n.strip(" -()/,.:;&+")


def load_diverse_funds(limit=4000, seed=42):
    """Sample a broad, de-duplicated pool of REAL funds that have a usable 1-year
    return, deriving a risk level and synthesizing an expense ratio for each."""
    _load_dotenv()
    conn = _connect(); cur = conn.cursor()
    cur.execute(
        "SELECT name, category, subcategory, nav, one_year_return FROM funds "
        "WHERE one_year_return IS NOT NULL AND one_year_return BETWEEN -40 AND 120 "
        "AND nav IS NOT NULL AND nav > 1 "
        "ORDER BY random() LIMIT %s", (limit * 4,))
    rows = cur.fetchall(); conn.close()

    rng = random.Random(seed)
    seen, pool = set(), []
    for name, category, subcategory, nav, ret in rows:
        base = _base_name(name)
        key = base.lower()
        # Reject degenerate names: too short, or a bare single-token AMC stub
        # like "SBI" that would be a substring of other candidates in a panel
        # (the fuzzy matcher would then confuse them).
        if len(base) < 8 or len(base.split()) < 2 or key in seen:
            continue
        seen.add(key)
        risk, conflict = _derive_risk_level(category, subcategory, name)
        # Skip rows whose stored category conflicts with the fund name: their
        # derived risk is untrustworthy (the same reason serving down-ranks them),
        # and training on them would teach the model a wrong risk label. Excluding
        # them also stops penalized-vs-penalized panels where a commodity fund
        # wins by default on raw return.
        if conflict:
            continue
        pool.append({
            "fund_name": base,
            "category": category or "Equity",
            "nav": round(float(nav), 2),
            "expense_ratio": synth_er(category, rng),
            "performance": round(float(ret), 1),
            "risk_level": risk,
            "data_conflict": conflict,
        })
        if len(pool) >= limit:
            break
    return pool


def _risk_idx(r):
    r = (r or "").strip().lower()
    table = {"low": 0, "moderate": 1, "medium": 1, "medium-high": 2, "high": 2, "very high": 3}
    return table.get(r, 1)


def _family_key(name):
    """Coarser key than the display name: drop trailing series / roman-numeral /
    tenure tokens so two share-class or series variants of ONE scheme don't both
    land in the same candidate panel (which would let the fuzzy matcher confuse
    them). Used only for in-panel de-duplication."""
    k = name.lower()
    k = re.sub(r"\b(series|plan)\b.*$", "", k)
    k = re.sub(r"\b[ivxlcdm]{1,6}\b", "", k)          # roman numerals
    k = re.sub(r"\b\d{1,4}\s*(months?|days?|years?)?\b", "", k)
    # Share-class / housekeeping variants of one scheme ("... Fund Bonus",
    # "... Fund Quarterly", "... Unclaimed ...") must collapse to the same key,
    # otherwise two near-identical names land in one panel and the fund matcher
    # can't tell which one an answer refers to.
    k = re.sub(r"\b(fund of funds?|fof|etf|regular|direct|bonus|quarterly|monthly|"
               r"daily|weekly|annual|retail|institutional|super|unclaimed|"
               r"investor education|above|below)\b", "", k)
    return re.sub(r"[^a-z]+", "", k)


def pick_candidates(pool, target_risk, rng, k=4):
    """4 candidates biased toward the target risk with adjacent-risk contrast —
    mirrors what the real risk-ranked SQL retrieval would surface. De-duplicates
    on a coarse scheme family so no two variants of the same scheme co-occur."""
    tgt = _risk_idx(target_risk)
    ranked = sorted(pool, key=lambda f: (abs(_risk_idx(f["risk_level"]) - tgt), rng.random()))
    near = ranked[: max(k * 4, 16)]
    rng.shuffle(near)
    chosen, fams = [], set()
    for f in near:
        fam = _family_key(f["fund_name"])
        if fam in fams:
            continue
        # Also reject any name that is a substring of (or contains) one already
        # chosen — e.g. "Invesco India Low Duration Fund" vs the same name plus
        # " Discretionary". Two such names in one panel are inherently ambiguous:
        # a faithful answer naming the longer one still fuzzy-matches the shorter.
        low = f["fund_name"].lower()
        if any(low in c["fund_name"].lower() or c["fund_name"].lower() in low
               for c in chosen):
            continue
        chosen.append(f); fams.add(fam)
        if len(chosen) == k:
            break
    return chosen


def choose_winner(candidates, target_risk, horizon=None):
    """Pick the winner using the EXACT serving-side score_fund so the training
    signal matches production ranking: risk/category/horizon alignment plus the
    additive -40 down-ranks for single-asset/commodity and category/name-conflict
    funds (a one-off return spike alone can't win the #1 slot). Ties broken by
    higher performance."""
    profile = UserProfile(risk_level=target_risk, horizon_years=horizon)
    return max(
        candidates,
        key=lambda f: (score_fund(f, profile)[0], f["performance"]),
    )


def candidate_block(candidates):
    lines = ["Candidate funds:"]
    for f in candidates:
        lines.append(
            f"- {f['fund_name']} | {f['category']} | NAV: {f['nav']} | "
            f"Expense ratio: {f['expense_ratio']}% | Performance: {f['performance']}% | "
            f"Risk: {f['risk_level']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profiles & question phrasings
# ---------------------------------------------------------------------------
RISK_WORDS = {"Low": "low", "Moderate": "moderate", "High": "high", "Very High": "very high"}
HORIZONS = [1, 2, 3, 5, 7, 8, 10, 12, 15, 20, 25]
SIPS = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 8000, 10000, 12000, 15000, 20000, 25000]

QUESTION_TEMPLATES = [
    "I'm {age} years old and want to start a SIP of {sip}/month for {horizon} years. My risk tolerance is {risk}. Which fund is best?",
    "Recommend a mutual fund for me: age {age}, monthly SIP {sip}, risk tolerance {risk}, investment horizon {horizon} years.",
    "Which fund should I choose for a SIP of {sip} per month? I am {age} with {risk} risk tolerance and a {horizon} year horizon.",
    "Given my profile — age {age}, {risk} risk tolerance, {horizon} year goal, {sip} monthly investment — which fund should I pick?",
    "I am {age}, monthly SIP {sip}, risk tolerance {risk}, investment horizon {horizon} years. What do you suggest?",
]


def make_profile(rng):
    return {
        "age": rng.randint(19, 62),
        "sip": rng.choice(SIPS),
        "risk": rng.choice(RISK_ORDER),
        "horizon": rng.choice(HORIZONS),
    }


def _horizon_note(h):
    if h <= 3:
        return "With only a few years to invest, capital preservation matters more than chasing returns."
    if h <= 7:
        return f"A {h}-year horizon gives this pick time to work through short-term market swings."
    return f"With a {h}-year horizon, you have plenty of room to ride out volatility."


# ---------------------------------------------------------------------------
# Part 2 — diverse, faithful recommendation examples
# ---------------------------------------------------------------------------
REC_TEMPLATES = [
    ("At age {age} with a {horizon}-year horizon and {riskw} risk tolerance, I recommend "
     "{name} — a {cat} fund carrying {risk} risk. Reasoning: its risk level matches your "
     "stated tolerance, and it has delivered {perf}% returns with a {er}% expense ratio. "
     "{hnote} A monthly SIP of {sip} here fits your goals — review it periodically."),
    ("For your profile — age {age}, {riskw} risk tolerance, {horizon}-year goal — {name} "
     "stands out. It's a {cat} fund with {risk} risk, {perf}% trailing return, and a {er}% "
     "expense ratio. {hnote} Your {sip}/month SIP is well suited to this choice."),
    ("I'd go with {name} for you. As a {cat} fund with {risk} risk, it aligns with your "
     "{riskw} tolerance, posting {perf}% returns at a {er}% expense ratio. {hnote} "
     "A {sip} monthly SIP over {horizon} years fits your stated plan."),
]


def gen_recommendation(pool, rng):
    profile = make_profile(rng)
    cands = pick_candidates(pool, profile["risk"], rng)
    if len(cands) < 2:
        return None
    win = choose_winner(cands, profile["risk"], profile["horizon"])
    user = (QUESTION_TEMPLATES[rng.randrange(len(QUESTION_TEMPLATES))].format(
                age=profile["age"], sip=profile["sip"],
                risk=RISK_WORDS[profile["risk"]], horizon=profile["horizon"])
            + "\n\n" + candidate_block(cands)
            + "\n\nBased on the above, give a recommendation with reasoning.")
    asst = REC_TEMPLATES[rng.randrange(len(REC_TEMPLATES))].format(
        age=profile["age"], horizon=profile["horizon"], riskw=RISK_WORDS[profile["risk"]],
        name=win["fund_name"], cat=win["category"], risk=win["risk_level"],
        perf=win["performance"], er=win["expense_ratio"], sip=profile["sip"],
        hnote=_horizon_note(profile["horizon"]))
    return {"user": user, "assistant": asst}


# ---------------------------------------------------------------------------
# Part 3 — copy-emphasis: repeat the pick's name+numbers at START and END
# ---------------------------------------------------------------------------
def gen_copy_emphasis(pool, rng):
    profile = make_profile(rng)
    cands = pick_candidates(pool, profile["risk"], rng)
    if len(cands) < 2:
        return None
    win = choose_winner(cands, profile["risk"], profile["horizon"])
    user = (QUESTION_TEMPLATES[rng.randrange(len(QUESTION_TEMPLATES))].format(
                age=profile["age"], sip=profile["sip"],
                risk=RISK_WORDS[profile["risk"]], horizon=profile["horizon"])
            + "\n\n" + candidate_block(cands)
            + "\n\nBased on the above, give a recommendation with reasoning.")
    asst = (
        f"My recommendation is {win['fund_name']} ({win['category']}, {win['risk_level']} risk, "
        f"{win['performance']}% return, {win['expense_ratio']}% expense ratio).\n\n"
        f"Why it fits: your {RISK_WORDS[profile['risk']]} risk tolerance lines up with this "
        f"{win['category']} fund's {win['risk_level']} risk, and its {win['performance']}% return "
        f"comes at a low {win['expense_ratio']}% cost. {_horizon_note(profile['horizon'])} "
        f"A {profile['sip']}/month SIP suits it well.\n\n"
        f"In short: go with {win['fund_name']} — {win['category']}, {win['risk_level']} risk, "
        f"{win['performance']}% return, {win['expense_ratio']}% expense ratio."
    )
    return {"user": user, "assistant": asst}


# ---------------------------------------------------------------------------
# Part 4 — communication-skill examples (empathetic, plain-language, clarifying,
# concise). Consistent with the MentraFiAI persona; NO hedging/refusal patterns.
# ---------------------------------------------------------------------------
COMMUNICATION_EXAMPLES = [
    {"user": "I just had my first child and I'm honestly overwhelmed about money. Where do I even start with investing?",
     "assistant": "Congratulations — that's a big, wonderful change, and feeling overwhelmed is completely normal. Let's keep it simple. Start with three steps: (1) build an emergency fund of 3-6 months' expenses in a savings account or liquid fund, (2) once that's set, begin a small monthly SIP you won't miss, and (3) as your child grows, add a long-horizon equity fund for education goals. You don't have to do it all at once. Want to tell me your monthly income and any current savings so I can suggest a starting amount?"},
    {"user": "everyone keeps saying NAV but I don't get it, is a fund with NAV 300 more expensive than one with NAV 30?",
     "assistant": "Great question, and it trips up a lot of people. No — a higher NAV does not mean a fund is 'more expensive' or worse value. NAV is just the price of one unit today. If you invest 3,000, you get 10 units of the 300-NAV fund or 100 units of the 30-NAV fund — the same money either way. What actually matters is how much the NAV grows in percentage terms. So compare returns and expense ratios, not the NAV number itself."},
    {"user": "I want to invest but I might need the money back, not sure when. what should I do?",
     "assistant": "Thanks for being upfront about that — it's an important detail. When the timing is uncertain, liquidity and stability matter more than chasing high returns. A couple of clarifying questions so I can point you the right way: is there any chance you'll need this money within the next 12 months, or is it more of a 'maybe in a few years' situation? And roughly how much are you thinking of investing? With that, I can suggest whether a liquid/short-duration debt fund or a low-risk hybrid fund fits better."},
    {"user": "my friend doubled his money in a small cap fund. should I put all my savings there?",
     "assistant": "It's tempting when you hear a story like that — but putting all your savings into one small-cap fund is riskier than it sounds. Small caps can rise fast and fall just as hard, and your friend's past result doesn't guarantee yours. A steadier approach is to keep an emergency buffer safe, then invest the rest across a mix — say a large-cap or index fund as your core, with a smaller slice in small caps for growth. How much are you considering, and what's your time horizon? That'll help me suggest a sensible split."},
    {"user": "explain why I shouldn't just keep all my money in a savings account",
     "assistant": "Totally fair to want it safe — but here's the catch: a savings account usually earns around 3-4%, while inflation runs near 6%. That means your money slowly loses buying power every year even though the balance looks the same. Keeping 3-6 months of expenses there is smart for emergencies. Beyond that, investing even part of it in funds matched to your goals helps your money at least keep pace with, and ideally beat, inflation. Want me to walk through options based on how long you can stay invested?"},
    {"user": "I'm 55 and retiring in 5 years, is it too late to start investing?",
     "assistant": "Not at all — starting now still makes a real difference, and I'm glad you're thinking about it. With a 5-year runway to retirement, the priority shifts toward protecting what you build, so a larger share in lower-risk debt and conservative hybrid funds usually makes sense, with a modest equity portion for growth. The key is matching risk to your timeline. Could you share how much you can invest monthly and whether you have other retirement savings already? Then I can tailor a mix for you."},
]
# PLACEHOLDER_PART5


# ---------------------------------------------------------------------------
# Part 5 — general financial-literacy examples (same informational style as the
# serving-side FINANCIAL_GLOSSARY, but as trainable data so the model can
# explain concepts naturally beyond the fixed hardcoded list).
# ---------------------------------------------------------------------------
FINANCE_LITERACY = [
    ("What is diversification and why does it matter?",
     "Diversification means spreading your money across different investments — like large-cap, mid-cap, debt, and maybe gold — instead of putting it all in one. The idea is simple: when one part of your portfolio dips, another may hold steady or rise, smoothing out the bumps. It doesn't eliminate risk, but it reduces the damage any single bad bet can do. For most investors, a diversified mix of a few funds across asset types is safer than betting everything on one 'hot' fund."),
    ("How does compounding work in mutual funds?",
     "Compounding is when your returns start earning returns of their own. Say you invest and earn 12% in a year — the next year you earn returns not just on your original amount but also on that gain. Over long periods this snowballs: the bulk of wealth in a decades-long SIP often comes from compounding, not your contributions. That's why starting early, even with a small amount, beats waiting to invest a larger sum later."),
    ("What's the difference between a SIP and a lump sum investment?",
     "A SIP invests a fixed amount at regular intervals (usually monthly), while a lump sum puts a large amount in all at once. A SIP spreads your purchases over time, so you buy more units when prices are low and fewer when high — this 'rupee-cost averaging' reduces timing risk and suits regular earners. A lump sum can do better if markets rise steadily after you invest, but it carries more timing risk. Many investors use SIPs for monthly savings and lump sums for occasional windfalls."),
    ("How does inflation affect my savings?",
     "Inflation is the gradual rise in prices, which means each rupee buys a little less over time. If your savings earn 4% but inflation is 6%, you're effectively losing about 2% of buying power a year even though your balance grows. This is why simply parking everything in a savings account can quietly erode your wealth. Investing in assets that historically outpace inflation — like equity funds over the long term — helps your money grow in real terms, not just on paper."),
    ("What are the tax benefits of ELSS funds?",
     "ELSS (Equity Linked Savings Scheme) funds let you claim a deduction of up to 1.5 lakh under Section 80C, potentially lowering your taxable income. They have the shortest lock-in among 80C options — just 3 years. Because they're equity funds, returns are market-linked and carry equity risk, and gains above 1 lakh a year are taxed as long-term capital gains at 10%. They suit investors who want tax savings plus long-term growth and can stay invested through the lock-in."),
    ("How much does a high expense ratio really cost me over time?",
     "It adds up more than it looks. The expense ratio is charged every year on your whole invested amount, so a 1% higher ratio doesn't cost 1% once — it compounds against you annually. On a long-running SIP over 15-20 years, the difference between a 0.3% index fund and a 1.3% active fund can quietly eat a meaningful chunk of your final corpus. That's why, for similar strategies, a lower expense ratio is a real and lasting advantage."),
    ("What is the risk-return tradeoff?",
     "It's the basic rule that higher potential returns come with higher risk. Debt funds are steadier but grow slowly; equity funds swing more but tend to grow faster over long periods. There's no free lunch — an investment promising high returns with no risk is a red flag. The goal isn't to avoid risk entirely but to take on an amount you're comfortable with and that matches your time horizon: more equity when you have years to ride out dips, more debt when you'll need the money soon."),
    ("Should I stop my SIP when the market falls?",
     "Usually the opposite — a falling market is when a SIP works hardest for you. Because you invest a fixed amount, a lower price means your money buys more units, lowering your average cost. Investors who stop during dips often miss the recovery and lock in losses. Unless your goals or income have genuinely changed, staying consistent through ups and downs is what lets rupee-cost averaging and compounding do their job. Pausing should be a cash-flow decision, not a market-timing one."),
]

FL_PHRASINGS = ["{q}", "Can you explain: {q}", "{q} I keep hearing about this.", "Help me understand — {q}"]


def gen_finance_literacy(rng):
    q, a = FINANCE_LITERACY[rng.randrange(len(FINANCE_LITERACY))]
    user = FL_PHRASINGS[rng.randrange(len(FL_PHRASINGS))].format(q=q)
    return {"user": user, "assistant": a}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/sft_augment_preview.jsonl")
    ap.add_argument("--n_rec", type=int, default=900, help="Part 2 recommendation examples")
    ap.add_argument("--n_copy", type=int, default=250, help="Part 3 copy-emphasis (~12-15%)")
    ap.add_argument("--comm_repeat", type=int, default=6, help="times to cycle Part 4 set")
    ap.add_argument("--fl_repeat", type=int, default=10, help="times to cycle Part 5 set")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print("Loading diverse real-fund pool from PostgreSQL ...")
    pool = load_diverse_funds(limit=4000, seed=args.seed)
    print(f"  pool size (distinct fund families): {len(pool)}")

    rows, tagged = [], {"rec": 0, "copy": 0, "comm": 0, "lit": 0}

    for _ in range(args.n_rec):
        ex = gen_recommendation(pool, rng)
        if ex:
            rows.append(("rec", ex)); tagged["rec"] += 1
    for _ in range(args.n_copy):
        ex = gen_copy_emphasis(pool, rng)
        if ex:
            rows.append(("copy", ex)); tagged["copy"] += 1
    for _ in range(args.comm_repeat):
        for ex in COMMUNICATION_EXAMPLES:
            rows.append(("comm", ex)); tagged["comm"] += 1
    for _ in range(args.fl_repeat):
        for _ in range(len(FINANCE_LITERACY)):
            rows.append(("lit", gen_finance_literacy(rng))); tagged["lit"] += 1

    rng.shuffle(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for _, ex in rows:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total = len(rows)
    print(f"\n✓ wrote {total} preview examples -> {args.out}")
    print(f"  Part 2 recommendation : {tagged['rec']}  ({100*tagged['rec']/total:.0f}%)")
    print(f"  Part 3 copy-emphasis  : {tagged['copy']}  ({100*tagged['copy']/total:.0f}%)")
    print(f"  Part 4 communication  : {tagged['comm']}  ({100*tagged['comm']/total:.0f}%)")
    print(f"  Part 5 finance-lit    : {tagged['lit']}  ({100*tagged['lit']/total:.0f}%)")

    # Diversity of the generated recommendation set (name/number spread).
    names = {ex["assistant"].split(" — ")[0].split("(")[0].replace("My recommendation is", "")
             .replace("I'd go with", "").replace("For your profile", "").strip()
             for t, ex in rows if t in ("rec", "copy")}
    print(f"\n  distinct pick-name strings in generated rec/copy set (approx): {len(names)}")

    print("\n" + "=" * 70 + "\n  SAMPLE (10 examples across all parts)\n" + "=" * 70)
    picks = []
    for tag in ("rec", "copy", "comm", "lit"):
        picks += [(t, e) for t, e in rows if t == tag][:3]
    for t, ex in picks[:10]:
        u = ex["user"].split("\n")[0]
        print(f"\n[{t}] USER: {u[:110]}")
        print(f"      ASST: {ex['assistant'][:230]}")


if __name__ == "__main__":
    main()


