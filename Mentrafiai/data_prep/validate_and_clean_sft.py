"""
MentraFiAI SFT Dataset Validator & Cleaner v2
==============================================
Runs 5 hard validation checks against sft_train_v3.jsonl / sft_val_v3.jsonl:

  1. BUDGET MATH: stated SIP budget == sum(per-fund allocations)
  2. NAME HANDLING: no hallucinated names; stated names reproduced exactly
  3. TOKEN CORRUPTION: no non-standard Unicode in assistant responses
  4. HORIZON ACCURACY: stated horizon years must match response
  5. US FINANCIAL LEAKAGE: zero 401(k), Roth IRA, USD references

Outputs per-check failure reports and cleaned datasets.
Also normalizes ₹ -> "Rs " because tokenizer_v2.model encodes ₹ as 4
byte-fallback tokens (root cause of garbled ₠/ↂ/₸ at inference).
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.system("")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "sft_train_v3": PROJECT_ROOT / "data/processed/sft_train_v3.jsonl",
    "sft_val_v3":   PROJECT_ROOT / "data/processed/sft_val_v3.jsonl",
}

OUT_TRAIN = PROJECT_ROOT / "data/processed/sft_train_clean.jsonl"
OUT_VAL   = PROJECT_ROOT / "data/processed/sft_val_clean.jsonl"

# ── Allowed Unicode characters ─────────────────────────────────
# Be generous: allow ASCII, Latin-1, Devanagari, common symbols, emojis
ALLOWED_RANGES = [
    (0x0009, 0x000A),   # tab, newline
    (0x000D, 0x000D),   # carriage return
    (0x0020, 0x007E),   # basic ASCII printable
    (0x00A0, 0x00FF),   # Latin-1 supplement
    (0x0900, 0x097F),   # Devanagari
    (0x20B9, 0x20B9),   # ₹ Indian Rupee sign
    (0x2013, 0x2014),   # en-dash, em-dash
    (0x2018, 0x201D),   # smart quotes
    (0x2022, 0x2022),   # bullet •
    (0x2026, 0x2026),   # ellipsis …
    (0x2190, 0x2199),   # arrows ← → etc.
    (0x2212, 0x2212),   # minus sign −
    (0x2264, 0x2265),   # ≤ ≥
    (0x25CF, 0x25CF),   # ● black circle
    (0x26A0, 0x26A0),   # ⚠ warning sign
    (0x2705, 0x2705),   # ✅
    (0x2714, 0x2714),   # ✔
    (0x2728, 0x2728),   # ✨
    (0x274C, 0x274C),   # ❌
    (0x2764, 0x2764),   # ❤
    (0xFE0E, 0xFE0F),   # Unicode variation selectors (invisible, after emojis)
    (0x1F389, 0x1F389), # 🎉
    (0x1F3AF, 0x1F3AF), # 🎯
    (0x1F44B, 0x1F44B), # 👋
    (0x1F44D, 0x1F44D), # 👍
    (0x1F4B0, 0x1F4B0), # 💰
    (0x1F4C8, 0x1F4C8), # 📈
    (0x1F4CA, 0x1F4CA), # 📊
    (0x1F60A, 0x1F60A), # 😊
    (0x1F64F, 0x1F64F), # 🙏
    (0x1F680, 0x1F680), # 🚀
]

def is_allowed_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ALLOWED_RANGES)


# ── US Financial Keywords ──────────────────────────────────────
US_TERMS = [
    "401k", "401(k)", "roth ira", "traditional ira",
    "vanguard", "fidelity", "charles schwab", "ameritrade",
    "s&p 500", "s&p500", "sp500",
    "irs ", "w-2", "1099",
    "dollars",
]

# For "$" we need to be careful to not match LaTeX math "$" delimiters
# Only flag real USD usage like "$100", "$X,XXX", "USD"
USD_RE = re.compile(r'(?<!\$)\$\d')  # $ followed by digit = real USD
USD_WORD_RE = re.compile(r'\bUSD\b', re.IGNORECASE)


# ── Common Indian names for hallucination detection ────────────
COMMON_INDIAN_NAMES = {
    "Rahul", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Rohit", "Kavya",
    "Suresh", "Meera", "Arjun", "Divya", "Kiran", "Pooja", "Raj", "Nisha",
    "Deepak", "Sunita", "Arun", "Ritu", "Sanjay", "Lalitha", "Mahesh", "Geetha",
    "Prakash", "Swati", "Ajay", "Rekha", "Naveen", "Shweta", "Harish", "Usha",
    "Aditya", "Tanvi", "Sameer", "Preeti", "Kunal", "Rohan", "Varun", "Isha",
    "Rajesh", "Neha", "Mohan", "Ravi", "Vijay", "Lakshmi", "Ganesh", "Sita",
    "Pankaj", "Ankita", "Manish", "Seema", "Sunil", "Jaya", "Vivek", "Ramesh",
    "Nitin", "Aarav", "Saurabh", "Ankit", "Harsh", "Tushar", "Sachin",
    "Shyam", "Gopal", "Abhi", "Simran", "Kritika", "Akhil", "Shreya",
}


# ── Regex Patterns ─────────────────────────────────────────────

# Match per-fund allocation lines: "₹XX,XXX/month (NN%)"
# This is specifically the per-fund line, NOT the "Total SIP" line
ALLOC_LINE_RE = re.compile(
    r'(?:₹|Rs\.?\s*)([0-9][0-9,]*)\s*/\s*month\s*\((\d+)%\)',
    re.IGNORECASE
)

# Match the stated budget in user message — specifically monthly SIP budget
# Must disambiguate from salary/income statements and age patterns
USER_BUDGET_PATTERNS = [
    # "₹20,000/month" or "₹20,000 monthly" or "₹20,000 per month"
    re.compile(r'(?:budget|invest|sip)[^.]*?(?:₹|Rs\.?\s*)([0-9][0-9,]*)\s*(?:/?\s*month|monthly|per\s*month)', re.IGNORECASE),
    # "₹20,000/month SIP budget"
    re.compile(r'(?:₹|Rs\.?\s*)([0-9][0-9,]*)\s*(?:/?\s*month|monthly|per\s*month)\s*(?:SIP|budget|invest)', re.IGNORECASE),
    # "SIP of ₹20,000"
    re.compile(r'SIP\s*(?:of\s*)?(?:₹|Rs\.?\s*)([0-9][0-9,]*)', re.IGNORECASE),
    # "invest ₹20,000 monthly" or "invest ₹20,000/month"
    re.compile(r'invest\s+(?:₹|Rs\.?\s*)([0-9][0-9,]*)\s*(?:/?\s*month|monthly|per\s*month)', re.IGNORECASE),
    # "Budget ₹20,000/month"
    re.compile(r'[Bb]udget\s*(?:₹|Rs\.?\s*)([0-9][0-9,]*)\s*(?:/?\s*month|monthly|per\s*month)', re.IGNORECASE),
    # "₹XX,XXX monthly budget"
    re.compile(r'(?:₹|Rs\.?\s*)([0-9][0-9,]*)\s*monthly\s*budget', re.IGNORECASE),
    # "₹XX,XXX/month budget"
    re.compile(r'(?:₹|Rs\.?\s*)([0-9][0-9,]*)/month\s*budget', re.IGNORECASE),
    # Standalone "₹XX,XXX/month" when there's no salary/income/earn context
    re.compile(r'(?:₹|Rs\.?\s*)([0-9][0-9,]*)\s*/\s*month(?!\s*(?:salary|income|earn))', re.IGNORECASE),
]

# Match salary/income in user message (to EXCLUDE from budget matching)
SALARY_RE = re.compile(
    r'(?:salary|income|earn|take\s*home)[^.]*?(?:₹|Rs\.?\s*)([0-9][0-9,]*)',
    re.IGNORECASE
)

# Match horizon in user message — must be "X-year horizon" or "for X years"
# NOT "X-year-old" which is age
USER_HORIZON_PATTERNS = [
    re.compile(r'(\d+)\s*[-–]?\s*year\s*horizon', re.IGNORECASE),
    re.compile(r'for\s+(\d+)\s+years', re.IGNORECASE),
    re.compile(r'[Hh]orizon[:\s]+(\d+)\s*years?', re.IGNORECASE),
    re.compile(r'[Hh]orizon\s+(\d+)\s*years?', re.IGNORECASE),
]

# Match horizon in assistant response
ASST_HORIZON_RE = re.compile(
    r'(\d+)\s*[-–]?\s*year\s*(?:horizon|period|term)',
    re.IGNORECASE
)

# Match user's name — "I am/I'm <Name>", "My name is <Name>", "Hi, I am <Name>"
USER_NAME_PATTERNS = [
    re.compile(r"(?:I(?:'m| am)\s+)([A-Z][a-z]{2,15})\b"),
    re.compile(r"[Mm]y name is\s+([A-Z][a-z]{2,15})\b"),
    re.compile(r"Hi,?\s+I(?:'m| am)\s+([A-Z][a-z]{2,15})\b"),
    re.compile(r"^([A-Z][a-z]{2,15}) here\b"),
    re.compile(r"Name:\s*([A-Z][a-z]{2,15})\b"),
]

# Words that look like names but aren't
NON_NAMES = {
    "Recommend", "Investor", "Income", "Monthly", "Budget", "Looking",
    "Explain", "Quick", "Please", "Hello", "Great", "Want", "Help",
    "Need", "Start", "Just", "Also", "Here", "Would", "Could",
    "Should", "What", "Which", "Where", "When", "How", "Will",
    "Have", "That", "This", "With", "From", "Been", "Being",
    "About", "Your", "They", "Their", "Some", "More", "Most",
    "Very", "Year", "Fund", "Plan", "Risk", "High", "Each",
    "Make", "Take", "Give", "Age", "Can", "Per", "Old",
    "Working", "Planning", "Investing", "Growing", "Saving",
}


def parse_amount(s: str) -> float:
    return float(s.replace(",", ""))


def extract_user_budget(user_text: str) -> float | None:
    """Extract stated SIP budget from user message (NOT salary/income)."""
    # First check if this is a salary/income statement
    salary_match = SALARY_RE.search(user_text)
    salary_amount = parse_amount(salary_match.group(1)) if salary_match else None

    for pat in USER_BUDGET_PATTERNS:
        m = pat.search(user_text)
        if m:
            amount = parse_amount(m.group(1))
            # Skip if this amount equals the stated salary (it's income, not budget)
            if salary_amount and abs(amount - salary_amount) < 1:
                continue
            return amount
    return None


def extract_user_horizon(user_text: str) -> int | None:
    """Extract horizon years from user message (NOT age)."""
    for pat in USER_HORIZON_PATTERNS:
        m = pat.search(user_text)
        if m:
            return int(m.group(1))
    return None


def extract_user_name(user_text: str) -> str | None:
    """Extract the user's stated name."""
    for pat in USER_NAME_PATTERNS:
        m = pat.search(user_text)
        if m:
            name = m.group(1)
            if name in NON_NAMES:
                return None
            return name
    return None


def extract_assistant_names(asst_text: str) -> set[str]:
    """Find known Indian names in the assistant response."""
    words = re.findall(r'\b([A-Z][a-z]{2,15})\b', asst_text)
    return {w for w in words if w in COMMON_INDIAN_NAMES}


def extract_alloc_amounts(asst_text: str) -> list[tuple[float, int]]:
    """Extract per-fund (amount, pct) from assistant text, excluding Total SIP line."""
    results = []
    for line in asst_text.split('\n'):
        # Skip the Total SIP summary line
        if 'total sip' in line.lower() or '100% allocated' in line.lower():
            continue
        m = ALLOC_LINE_RE.search(line)
        if m:
            amt = parse_amount(m.group(1))
            pct = int(m.group(2))
            results.append((amt, pct))
    return results


def extract_assistant_horizons(asst_text: str) -> list[int]:
    return [int(m) for m in ASST_HORIZON_RE.findall(asst_text)]


# ── Check Functions ────────────────────────────────────────────

def check_budget_math(user: str, assistant: str) -> tuple[bool, str]:
    """Check 1: Stated SIP budget == sum(per-fund allocations)."""
    budget = extract_user_budget(user)
    if budget is None:
        return True, "no_budget_stated"

    allocs = extract_alloc_amounts(assistant)
    if not allocs:
        return True, "no_alloc_lines"

    total_amt = sum(a[0] for a in allocs)
    total_pct = sum(a[1] for a in allocs)

    issues = []

    # Check amounts sum to budget (1% tolerance or ±1 rupee for tiny budgets)
    tolerance = max(budget * 0.01, 1.0)
    if abs(total_amt - budget) > tolerance:
        issues.append(f"amount_sum={total_amt} != budget={budget}")

    # Check percentages sum to 100
    if total_pct != 100:
        issues.append(f"pct_sum={total_pct} != 100")

    # Check each allocation's amount matches its stated percentage
    for amt, pct in allocs:
        expected = budget * pct / 100.0
        # Allow rounding to nearest 100 or ±1% of budget
        pct_tolerance = max(budget * 0.01, 100.0)
        if abs(amt - expected) > pct_tolerance:
            issues.append(f"fund Rs{amt:,.0f}({pct}%) should be Rs{expected:,.0f}")

    if issues:
        return False, f"MISMATCH: {'; '.join(issues)}"
    return True, f"ok: budget={budget}, sum={total_amt}"


def check_name_handling(user: str, assistant: str) -> tuple[bool, str]:
    """Check 2: No hallucinated names; stated names reproduced exactly."""
    user_name = extract_user_name(user)
    asst_names = extract_assistant_names(assistant)

    if user_name is None:
        if asst_names:
            # Check if the name appears in the user text at all (e.g., "Divya here")
            # These patterns weren't caught by our NAME_PATTERNS
            for name in asst_names:
                if name in user:
                    # The name IS in the user text, just not captured by our regex
                    return True, f"ok: name_in_user_text={name}"
            return False, f"HALLUCINATED: user gave no name, assistant used: {asst_names}"
        return True, "ok: no_name"
    else:
        if user_name not in assistant:
            truncated = [n for n in asst_names if user_name.startswith(n) or n.startswith(user_name)]
            if truncated:
                return False, f"TRUNCATED: user='{user_name}', asst has '{truncated}'"
            return False, f"NAME_MISSING: user='{user_name}', not in assistant"

        extra = asst_names - {user_name}
        if extra:
            return False, f"EXTRA_NAMES: user='{user_name}', extra: {extra}"

        return True, f"ok: name={user_name}"


def check_token_corruption(assistant: str) -> tuple[bool, str]:
    """Check 3: No non-standard Unicode characters."""
    bad_chars = []
    for i, ch in enumerate(assistant):
        if not is_allowed_char(ch):
            bad_chars.append((i, ch, hex(ord(ch))))

    if bad_chars:
        sample = bad_chars[:5]
        desc = ", ".join(f"pos={p} char={repr(c)} U+{h}" for p, c, h in sample)
        return False, f"BAD_CHARS({len(bad_chars)}): {desc}"
    return True, "ok: clean"


def check_horizon_accuracy(user: str, assistant: str) -> tuple[bool, str]:
    """Check 4: Stated horizon must match assistant response."""
    user_horizon = extract_user_horizon(user)
    if user_horizon is None:
        return True, "no_horizon_stated"

    asst_horizons = extract_assistant_horizons(assistant)
    if not asst_horizons:
        return True, "no_horizon_in_response"

    if user_horizon in asst_horizons:
        return True, f"ok: horizon={user_horizon}"
    else:
        return False, f"MISMATCH: user_horizon={user_horizon}, asst_horizons={set(asst_horizons)}"


def check_us_leakage(assistant: str) -> tuple[bool, str]:
    """Check 5: No US financial concepts."""
    lower = assistant.lower()
    found = []
    for term in US_TERMS:
        if term in lower:
            found.append(term)

    # Check for real USD usage (not LaTeX $ delimiters)
    if USD_RE.search(assistant):
        found.append("$<digit>")
    if USD_WORD_RE.search(assistant):
        found.append("USD")

    if found:
        return False, f"US_TERMS: {found}"
    return True, "ok: clean"


# ── INR Symbol Normalizer ──────────────────────────────────────

def normalize_inr_symbol(text: str) -> str:
    """Replace ₹ with 'Rs ' for tokenizer stability.
    ₹ = 4 byte-fallback tokens, 'Rs' = 1 clean token.
    """
    text = re.sub(r'₹\s*', 'Rs ', text)
    text = re.sub(r'Rs  +', 'Rs ', text)
    return text


# ── Main Pipeline ──────────────────────────────────────────────

def validate_dataset(filepath: str) -> tuple[list[dict], dict[str, list]]:
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                ex["_id"] = i
                examples.append(ex)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {i}: JSON parse error: {e}")

    print(f"  Loaded {len(examples):,} examples")

    failures = defaultdict(list)
    clean = []

    for ex in examples:
        idx = ex["_id"]
        user = ex.get("user", "")
        assistant = ex.get("assistant", "")
        failed = False

        for check_name, check_fn in [
            ("budget_math",      lambda: check_budget_math(user, assistant)),
            ("name_handling",    lambda: check_name_handling(user, assistant)),
            ("token_corruption", lambda: check_token_corruption(assistant)),
            ("horizon_accuracy", lambda: check_horizon_accuracy(user, assistant)),
            ("us_leakage",       lambda: check_us_leakage(assistant)),
        ]:
            ok, detail = check_fn()
            if not ok:
                failures[check_name].append({"id": idx, "detail": detail, "user": user[:100]})
                failed = True

        if not failed:
            clean.append(ex)

    return clean, failures


def print_report(name: str, total: int, clean: list, failures: dict):
    removed = total - len(clean)
    print(f"\n{'='*60}")
    print(f"  VALIDATION REPORT: {name}")
    print(f"{'='*60}")
    print(f"  Total examples:    {total:,}")
    print(f"  Passed all checks: {len(clean):,}")
    print(f"  Removed (any fail):{removed:,}")
    print()

    check_labels = {
        "budget_math":      "1. Budget Math",
        "name_handling":    "2. Name Handling",
        "token_corruption": "3. Token Corruption",
        "horizon_accuracy": "4. Horizon Accuracy",
        "us_leakage":       "5. US Financial Leakage",
    }

    for key, label in check_labels.items():
        fails = failures.get(key, [])
        status = "PASS" if not fails else "FAIL"
        print(f"  [{status}] {label}: {len(fails):,} failures")
        if fails:
            for f in fails[:3]:
                print(f"         ID {f['id']:>5}: {f['detail']}")
                print(f"                  user: {f['user'][:80]}...")
            if len(fails) > 3:
                print(f"         ... and {len(fails)-3} more")
        print()


def write_clean_dataset(examples: list[dict], outpath: Path, normalize_rupee: bool = True):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        for ex in examples:
            out = {"user": ex["user"], "assistant": ex["assistant"]}
            if normalize_rupee:
                out["user"] = normalize_inr_symbol(out["user"])
                out["assistant"] = normalize_inr_symbol(out["assistant"])
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"  Saved {len(examples):,} examples to {outpath}")


def main():
    print("=" * 60)
    print("  MentraFiAI SFT Dataset Validator & Cleaner v2")
    print("=" * 60)

    targets = [(name, path) for name, path in DATASETS.items() if path.exists()]
    if not targets:
        print("[ERROR] No dataset files found!")
        return

    print(f"\nFound {len(targets)} dataset(s):")
    for name, path in targets:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  - {name}: {path.name} ({size_mb:.1f} MB)")

    for name, path in targets:
        print(f"\n{'─'*60}")
        print(f"  Validating: {name}")
        print(f"{'─'*60}")

        with open(path, "r", encoding="utf-8") as f:
            total = sum(1 for line in f if line.strip())

        clean, failures = validate_dataset(str(path))
        print_report(name, total, clean, failures)

        if "train" in name:
            write_clean_dataset(clean, OUT_TRAIN, normalize_rupee=True)
        elif "val" in name:
            write_clean_dataset(clean, OUT_VAL, normalize_rupee=True)

    # Tokenizer diagnostic
    print(f"\n{'─'*60}")
    print("  TOKENIZER DIAGNOSTIC: Rs symbol")
    print(f"{'─'*60}")
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from tokenizer.tokenizer_utils import Tokenizer
        tok = Tokenizer(str(PROJECT_ROOT / "tokenizer/tokenizer_v2.model"))
        inr_ids = tok.encode("\u20B9")
        rs_ids = tok.encode("Rs")
        print(f"  U+20B9 (₹) encodes to {len(inr_ids)} tokens: {inr_ids}")
        print(f"  'Rs'     encodes to {len(rs_ids)} token(s): {rs_ids}")
        print(f"  'Rs 20,000' -> {tok.encode('Rs 20,000')}")
        if len(inr_ids) > 1:
            print(f"\n  [WARNING] ₹ uses {len(inr_ids)} byte-fallback tokens!")
            print(f"  All training data normalized: ₹ -> 'Rs ' for stable generation.")
    except Exception as e:
        print(f"  [SKIP] Could not load tokenizer: {e}")

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
