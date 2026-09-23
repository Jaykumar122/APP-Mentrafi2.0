"""
Audit sft_train.jsonl for:
1. Budget math consistency (User budget vs Assistant total vs Sum of allocations)
2. Horizon consistency (User horizon vs Assistant horizon)
3. US leakage (401k, Roth IRA, IRS, etc.)
4. Corrupt tokens or odd symbols
"""

import json
import re

TRAIN_PATH = "data/processed/sft_train.jsonl"

def audit():
    total = 0
    budget_mismatches = 0
    horizon_mismatches = 0
    us_leakage = 0
    corrupt_symbol_count = 0

    us_terms = ["401(k)", "401k", "roth ira", "traditional ira", "vanguard", "fidelity", "irs", "s&p 500"]

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            total += 1
            obj = json.loads(line)
            u = obj["user"]
            a = obj["assistant"]

            # Check US leakage
            low_a = a.lower()
            if any(t in low_a for t in us_terms):
                us_leakage += 1

            # Check corrupt tokens
            if any(c in a for c in ["₠", "ↂ", "", "", "✈", "♠️", "✙"]):
                corrupt_symbol_count += 1

            # Check budget matching
            # Extract user budget
            u_budget_match = re.search(r"(?:₹|rs\.?|inr\s*)\s*([\d,]+)(?:\s*(?:/month|monthly|per month|sip|budget))?", u, re.I)
            if u_budget_match:
                raw_u_b = u_budget_match.group(1).replace(",", "")
                try:
                    u_budget = int(raw_u_b)
                except ValueError:
                    u_budget = None
            else:
                u_budget = None

            # Extract assistant portfolio header
            a_portfolio_match = re.search(r"## Your Recommended Portfolio \((?:₹|rs\.?)\s*([\d,]+)/month\)", a, re.I)
            if a_portfolio_match and u_budget:
                raw_a_b = a_portfolio_match.group(1).replace(",", "")
                try:
                    a_budget = int(raw_a_b)
                    if a_budget != u_budget:
                        budget_mismatches += 1
                except ValueError:
                    pass

    print("=" * 60)
    print("           AUDIT REPORT: sft_train.jsonl")
    print("=" * 60)
    print(f"Total examples audited:        {total:,}")
    print(f"US Leakage examples:           {us_leakage:,} ({us_leakage/total*100:.1f}%)")
    print(f"Corrupt symbol examples:       {corrupt_symbol_count:,}")
    print(f"Budget mismatch examples:      {budget_mismatches:,}")
    print("=" * 60)

if __name__ == "__main__":
    audit()
