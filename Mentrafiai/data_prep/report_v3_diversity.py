"""
Diversity + REAL-faithfulness report for the v3 augmentation preview (read-only).

Mirrors report_v2_diversity.py but points at the v3 preview and compares against
the v2 dataset. Crucially it discounts the ONE known audit false positive: a
cap keyword ("small cap", "index", ...) that sits inside the correctly-copied
fund name is NOT a fabrication — the raw audit flags it only because the DB
stores that fund's category as the generic "Equity".

Run:  python data_prep/report_v3_diversity.py
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

V3_PREVIEW = [("v3 preview", "data/processed/sft_augment_v3_preview.jsonl")]
V2 = [("train_v2", "data/processed/sft_train_v2.jsonl"),
      ("val_v2", "data/processed/sft_val_v2.jsonl")]


def scan(paths):
    names, cats = Counter(), Counter()
    navs, perfs, ers = set(), set(), set()
    n = n_rec = real_fail = raw_fail = 0
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
                    nm = _normalize_name(m["fund_name"]) if m else ""
                    # Drop cap-keyword-in-name false positives.
                    real = [p for p in probs
                            if not (p.startswith("states category") and p.split("'")[1] in nm)]
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
    print(f"  REAL faithfulness fails  : {s['real_fail']} ({100*s['real_fail']/max(s['n_rec'],1):.1f}%)"
          f"   (raw audit flags incl. known cap-in-name FPs: {s['raw_fail']})")
    print(f"  distinct verbatim picks  : {len(names)}")
    print(f"  top-1 / top-4 share      : {100*names.most_common(1)[0][1]/tot:.1f}% / {100*top4/tot:.1f}%")
    print(f"  distinct NAV / perf / ER : {len(s['navs'])} / {len(s['perfs'])} / {len(s['ers'])}")
    print(f"  distinct categories      : {len(s['cats'])}")


def main():
    v3 = scan(V3_PREVIEW)
    v2 = scan(V2)
    print("=" * 74)
    print("V3 PREVIEW — REAL FAITHFULNESS & DIVERSITY (vs v2 dataset)")
    print("=" * 74)
    report("V2   sft_train_v2 + sft_val_v2 (existing base)", v2)
    report("V3   sft_augment_v3_preview (new examples only)", v3)
    print("\n  v3 preview — top 8 recommended names:")
    tot = sum(v3["names"].values()) or 1
    for nm, c in v3["names"].most_common(8):
        print(f"      {c:5}  {100*c/tot:5.1f}%  {nm[:58]}")


if __name__ == "__main__":
    main()
