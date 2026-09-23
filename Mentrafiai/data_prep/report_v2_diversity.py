"""
Diversity + composition report for the FINAL v2 dataset (read-only).

Reports, per split and combined: example counts, distinct verbatim recommended
fund names (via the serving-side fuzzy matcher against each prompt's own
candidate block), the top-name concentration that caused the memorization
problem, and the distinct-value counts for the numeric fields.

Run:  python data_prep/report_v2_diversity.py
"""
import sys, os, json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from data_prep.audit_sft_faithfulness import parse_candidates, audit_example
from inference.mutual_fund_advisor import fuzzy_match_fund, _normalize_name

SPLITS = [
    ("train_v2", "data/processed/sft_train_v2.jsonl"),
    ("val_v2",   "data/processed/sft_val_v2.jsonl"),
]
BASELINE = [
    ("train (orig)", "data/processed/sft_train.jsonl"),
    ("val (orig)",   "data/processed/sft_val.jsonl"),
]


def scan(paths):
    names, cats = Counter(), Counter()
    navs, perfs, ers = set(), set(), set()
    n = n_rec = 0
    real_fail = raw_fail = 0
    for _, path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                n += 1
                cands = parse_candidates(ex.get("user", ""))
                if not cands:
                    continue
                n_rec += 1
                for c in cands:
                    cats[c["category"]] += 1
                    navs.add(c["nav"]); perfs.add(c["performance"]); ers.add(c["expense_ratio"])
                m = fuzzy_match_fund(ex.get("assistant", ""), cands)
                if m:
                    names[m["fund_name"]] += 1
                verdict, probs = audit_example(ex)
                if verdict is False:
                    raw_fail += 1
                    # Discount the known audit false positive: a cap keyword that
                    # sits inside the correctly-copied fund name is not a slip.
                    nm = _normalize_name(m["fund_name"]) if m else ""
                    real = [p for p in probs
                            if not (p.startswith("states category")
                                    and p.split("'")[1] in nm)]
                    if real:
                        real_fail += 1
    return dict(n=n, n_rec=n_rec, names=names, cats=cats, navs=navs,
                perfs=perfs, ers=ers, real_fail=real_fail, raw_fail=raw_fail)


def report(label, s):
    names = s["names"]
    tot = sum(names.values()) or 1
    top4 = sum(c for _, c in names.most_common(4))
    print(f"\n{label}")
    print(f"  examples                 : {s['n']}  (recommendation-shaped: {s['n_rec']})")
    print(f"  REAL faithfulness fails  : {s['real_fail']}   (raw audit flags incl. known FPs: {s['raw_fail']})")
    print(f"  distinct verbatim picks  : {len(names)}")
    print(f"  top-1 / top-4 share      : {100*names.most_common(1)[0][1]/tot:.1f}% / {100*top4/tot:.1f}%")
    print(f"  distinct NAV / perf / ER : {len(s['navs'])} / {len(s['perfs'])} / {len(s['ers'])}")
    print(f"  distinct categories      : {len(s['cats'])}")


def main():
    base = scan(BASELINE)
    v2_train = scan(SPLITS[:1])
    v2_val = scan(SPLITS[1:])
    v2_all = scan(SPLITS)

    print("=" * 74)
    print("V2 DATASET — COMPOSITION & DIVERSITY (final, post widened-sector rule)")
    print("=" * 74)
    report("BEFORE  original sft_train + sft_val", base)
    report("AFTER   sft_train_v2", v2_train)
    report("AFTER   sft_val_v2", v2_val)
    report("AFTER   v2 combined", v2_all)

    print("\n  v2 combined — top 8 recommended names:")
    tot = sum(v2_all["names"].values()) or 1
    for nm, c in v2_all["names"].most_common(8):
        print(f"      {c:5}  {100*c/tot:5.1f}%  {nm[:58]}")

    print("\n  v2 combined — candidate categories:")
    for c, k in v2_all["cats"].most_common(12):
        print(f"      {k:6}  {c}")


if __name__ == "__main__":
    main()
