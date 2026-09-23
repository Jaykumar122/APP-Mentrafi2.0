"""
Part 1 (faithfulness audit) + Part 2 (diversity report) for the SFT data.

Faithfulness: for each example we parse the candidate-fund block out of the
`user` prompt, identify which candidate the `assistant` actually recommends
(reusing the serving-side fuzzy matcher), then check that every category / risk
/ performance% / expense% the assistant STATES matches that same candidate's
ground-truth attributes from the prompt. Mismatches mean the example is teaching
the model to fabricate — exactly the behavior we want to reduce.

Read-only. Writes nothing. Run:
    python data_prep/audit_sft_faithfulness.py
"""
import sys, os, json, re
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from inference.mutual_fund_advisor import fuzzy_match_fund, _normalize_name

DEFAULT_FILES = ["data/processed/sft_train.jsonl", "data/processed/sft_val.jsonl"]
FILES = DEFAULT_FILES

# Candidate line: "- Name | Category | NAV: n | Expense ratio: e% | Performance: p% | Risk: r"
CAND_RE = re.compile(
    r"-\s*(?P<name>.+?)\s*\|\s*(?P<category>.+?)\s*\|\s*NAV:\s*(?P<nav>[\d.]+)"
    r"\s*\|\s*Expense ratio:\s*(?P<er>[\d.]+)%\s*\|\s*Performance:\s*(?P<perf>[-\d.]+)%"
    r"\s*\|\s*Risk:\s*(?P<risk>[A-Za-z\- ]+)", re.I)


def parse_candidates(user_prompt):
    funds = []
    for m in CAND_RE.finditer(user_prompt):
        funds.append({
            "fund_name": m["name"].strip(), "category": m["category"].strip(),
            "nav": float(m["nav"]), "expense_ratio": float(m["er"]),
            "performance": float(m["perf"]), "risk_level": m["risk"].strip(),
        })
    return funds


def approx(a, b, tol=0.05):
    return abs(a - b) <= tol


def audit_example(ex):
    """Return (ok, list_of_problems). ok=False means the assistant response is
    unfaithful to its own prompt's candidate data."""
    user, asst = ex.get("user", ""), ex.get("assistant", "")
    cands = parse_candidates(user)
    problems = []
    if not cands:
        return None, ["no candidate block parseable"]  # skip from pass/fail denominator

    matched = fuzzy_match_fund(asst, cands)
    if matched is None:
        return False, ["assistant names no candidate fund (or an unknown one)"]

    norm_asst = _normalize_name(asst)

    # Does the assistant mention any OTHER candidate's full name as the pick?
    # (discussing a different fund than it recommended)
    # Category faithfulness: if the assistant states a cap/category word, it must
    # be the matched fund's category.
    cat = matched["category"].lower()
    for other_cat in ("large cap", "mid cap", "small cap", "index", "debt", "hybrid", "elss"):
        if other_cat in norm_asst and other_cat not in cat and cat not in other_cat:
            # allow generic "equity"/"fund" mentions; only flag explicit conflicting cap words
            problems.append(f"states category '{other_cat}' but pick is '{matched['category']}'")
            break

    # Risk faithfulness.
    risk = matched["risk_level"].lower()
    stated_risks = set(re.findall(r"\b(low|medium-high|medium|moderate|high|very high)\b", norm_asst))
    norm_risk = risk.replace("medium", "medium")
    if stated_risks:
        ok_risk = any(r in risk or risk in r or (r in ("medium", "moderate") and "medium" in risk)
                      for r in stated_risks)
        if not ok_risk:
            problems.append(f"states risk {sorted(stated_risks)} but pick risk is '{matched['risk_level']}'")

    # Performance% faithfulness: every X% the assistant cites should equal the
    # matched fund's performance or expense ratio (the only two legit numbers).
    pct_values = [float(x) for x in re.findall(r"([-\d.]+)\s*%", asst)]
    legit = {round(matched["performance"], 2), round(matched["expense_ratio"], 2)}
    for v in pct_values:
        if not any(approx(v, L) for L in legit):
            problems.append(f"cites {v}% not matching perf {matched['performance']}% / ER {matched['expense_ratio']}%")

    return (len(problems) == 0), problems


def main():
    import argparse
    global FILES
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                    help="jsonl files to audit (default: original sft_train/sft_val)")
    args = ap.parse_args()
    FILES = args.files

    total = ok = fail = skipped = 0
    fail_samples = []
    names = Counter(); cats = Counter()
    navs = []; perfs = []; ers = []
    per_file = {}

    for path in FILES:
        if not os.path.exists(path):
            print(f"MISSING: {path}"); continue
        f_ok = f_fail = f_total = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                # diversity: harvest candidate attributes + recommended name
                cands = parse_candidates(ex.get("user", ""))
                for c in cands:
                    cats[c["category"]] += 1
                    navs.append(c["nav"]); perfs.append(c["performance"]); ers.append(c["expense_ratio"])
                m = fuzzy_match_fund(ex.get("assistant", ""), cands) if cands else None
                if m:
                    names[m["fund_name"]] += 1

                verdict, probs = audit_example(ex)
                if verdict is None:
                    skipped += 1; continue
                total += 1; f_total += 1
                if verdict:
                    ok += 1; f_ok += 1
                else:
                    fail += 1; f_fail += 1
                    if len(fail_samples) < 12:
                        fail_samples.append((path, ex, probs))
        per_file[path] = (f_ok, f_fail, f_total)

    print("=" * 70)
    print("PART 1 — FAITHFULNESS AUDIT")
    print("=" * 70)
    for path, (o, x, t) in per_file.items():
        print(f"  {path}: {x}/{t} unfaithful ({100*x/max(t,1):.1f}%)")
    print(f"\n  TOTAL: {fail}/{total} unfaithful ({100*fail/max(total,1):.1f}%), {skipped} unparseable/skipped")

    print("\n  Sample failing examples (up to 12):")
    for path, ex, probs in fail_samples:
        pick = fuzzy_match_fund(ex["assistant"], parse_candidates(ex["user"]))
        print(f"\n  - [{os.path.basename(path)}] pick={pick['fund_name'] if pick else '?'}")
        for p in probs:
            print(f"      • {p}")
        print(f"      asst: {ex['assistant'][:180]}...")

    print("\n" + "=" * 70)
    print("PART 2 — DIVERSITY REPORT")
    print("=" * 70)
    print(f"  Distinct recommended fund names: {len(names)}")
    print(f"  Top 15 recommended names (count):")
    for n, c in names.most_common(15):
        print(f"      {c:5}  {n[:60]}")
    print(f"\n  Distinct candidate categories: {len(cats)}")
    for c, n in cats.most_common():
        print(f"      {n:6}  {c}")

    def rng(v, label):
        if v:
            v2 = sorted(v)
            print(f"  {label}: min {v2[0]}, p50 {v2[len(v2)//2]}, max {v2[-1]}, distinct {len(set(v))}")
    print()
    rng(navs, "NAV     "); rng(perfs, "Perf %  "); rng(ers, "Expense%")


if __name__ == "__main__":
    main()
