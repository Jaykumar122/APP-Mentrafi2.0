"""Snapshot real Direct-Growth schemes from the funds DB for grounded SFT examples.

Only NAV-derived fields are kept (name, NAV, NAV date, 1/3/5-year returns); the
expense ratio, AUM and riskometer columns were filled by formulas and are not used.
Output: data/raw/mf_corpus/fund_snapshot.json  {category_key: [fund, ...]}
Run: python -m data_prep.snapshot_funds
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from data_prep.gen_mf_pretrain import classify, load_funds
from llm120.grounding import clean_fund_name

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "mf_corpus" / "fund_snapshot.json"
KEYS = {"Liquid Fund": "liquid", "Arbitrage Fund": "arbitrage", "Short Duration Fund": "short_debt",
        "Corporate Bond Fund": "corp_bond", "Conservative Hybrid Fund": "cons_hybrid",
        "Balanced Advantage Fund": "baf", "Large Cap Fund": "large_cap", "Index Fund": "index",
        "Flexi Cap Fund": "flexi_cap", "Mid Cap Fund": "mid_cap", "Small Cap Fund": "small_cap",
        "ELSS Tax Saver Fund": "elss"}
PER_CATEGORY = 120


def main() -> None:
    rng = random.Random(7)
    groups: dict[str, list[dict]] = {key: [] for key in KEYS.values()}
    for f in load_funds():
        if (f.get("plan_type") or "").lower() != "direct" or "growth" not in (f.get("option_type") or "").lower():
            continue
        low = f["name"].lower()
        if (f.get("return_3y") is None or any(w in low for w in (
                "bonus", "idcw", "dividend", "distribution", "payout", "reinvest", "weekly", "monthly", "quarterly",
                "daily", "fortnight", "annual", "half yearly", "regular"))):
            continue
        category = classify(f["name"])
        key = KEYS.get(category[0]) if category else None
        name = clean_fund_name(f["name"])
        if key is None or len(name) > 85:
            continue
        groups[key].append({
            "name": name, "nav": round(float(f["nav"]), 2), "nav_date": f["nav_date"].isoformat(),
            "return_1y": None if f.get("return_1y") is None else round(float(f["return_1y"]), 1),
            "return_3y": round(float(f["return_3y"]), 1),
            "return_5y": None if f.get("return_5y") is None else round(float(f["return_5y"]), 1),
        })
    for key, items in groups.items():
        unique = {item["name"]: item for item in items}
        items = sorted(unique.values(), key=lambda item: item["name"])
        rng.shuffle(items)
        groups[key] = items[:PER_CATEGORY]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(groups, indent=1, ensure_ascii=False), encoding="utf-8")
    print({key: len(items) for key, items in groups.items()})


if __name__ == "__main__":
    main()
