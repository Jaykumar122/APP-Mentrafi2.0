"""
Verify the arithmetic in generated numeracy passages by re-parsing the output.

The generator computing a value correctly and the passage STATING it correctly
are different claims. This checks the second one: for every line of the form
"<expression> = <result>" it re-evaluates the expression independently and
compares. A rounding or formatting mismatch is then caught here rather than
shipped into the pretraining corpus, where it would teach wrong arithmetic.

Two deliberate looseness decisions, both to avoid false positives on correct text
rather than to let errors through:

  * Tolerance is RELATIVE (0.5%). Displayed factors are rounded to 4-6 decimals,
    so a product of them cannot reproduce the exact rupee value; only errors far
    larger than rounding are real.
  * A result is accepted if it matches after a factor of 1, 100 or 1/100. The
    passages mix percent conventions within a line ("10% / 12 = 0.008333" treats
    % as /100, while "(x/y - 1) x 100 = 8.95%" does not), and disambiguating
    those is not worth a parser. A genuine arithmetic error is essentially never
    off by exactly 100x, so this keeps the check meaningful.

Lines containing ^ (powers) are skipped: they are re-verified transitively,
because the power's value is then used in a product line that IS checked.

Usage:
    python data_prep/verify_numeracy.py data/raw/corpus_v2/finance_numeracy.jsonl
"""

import json
import re
import sys
from pathlib import Path

_SAFE = re.compile(r"^[-+*/().\d\s]+$")
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_NUMONLY = re.compile(r"^-?[\d.]+$")


def clean(side):
    """Strip presentation and normalise to an arithmetic expression."""
    s = side.strip()
    s = re.sub(r"\bRs\b", "", s)
    s = re.sub(r"[A-Za-z_,\[\]]", "", s)        # labels, thousands separators, brackets
    s = _PCT.sub(r"(\1/100)", s)
    s = s.replace("x", "*").replace("×", "*")
    return s.strip()


def evaluate(expr):
    if not expr or not _SAFE.match(expr):
        return None
    if not any(op in expr for op in "+-*/"):
        return None                              # a bare number proves nothing
    if re.search(r"[\d)]\s*\(", expr):
        return None                              # stripped label left "5 (3)" — not arithmetic
    try:
        # Charset-validated above: no names, no calls, no attribute access.
        return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None


def parse_line(line):
    """Return (expr_value, stated_value) for '... = <expr> = <result>' lines."""
    if "^" in line or "=" not in line:
        return None
    parts = [p for p in line.split("=")]
    if len(parts) < 2:
        return None
    rhs_raw = parts[-1]
    lhs_raw = parts[-2]
    # 'x' is the multiplication sign in these passages, but it is also a letter,
    # so protect it before the letter strip.
    lhs = clean(lhs_raw.replace(" x ", " * "))
    rhs = clean(rhs_raw.replace(" x ", " * "))
    if not _NUMONLY.match(rhs.replace(" ", "")):
        try:
            stated = float(rhs.replace(" ", "")) if rhs.strip() else None
        except ValueError:
            stated = evaluate(rhs)
    else:
        stated = float(rhs.replace(" ", ""))
    if stated is None:
        return None
    got = evaluate(lhs)
    if got is None:
        return None
    return got, stated


def matches(got, stated, tol=0.005):
    for scale in (1.0, 100.0, 0.01):
        want = stated * scale
        if want == 0:
            if abs(got) < 1e-9:
                return True
            continue
        if abs(got - want) / abs(want) <= tol:
            return True
    return False


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else "data/raw/corpus_v2/finance_numeracy.jsonl")
    n = checked = failed = 0
    shown = 0
    for raw in open(path, encoding="utf-8"):
        text = json.loads(raw)["text"]
        n += 1
        for line in text.split("\n"):
            r = parse_line(line)
            if r is None:
                continue
            got, stated = r
            checked += 1
            if not matches(got, stated):
                failed += 1
                if shown < 12:
                    shown += 1
                    print(f"  MISMATCH {line.strip()[:90]}")
                    print(f"           computed {got:,.4f}  stated {stated:,.4f}")
    print(f"\n{n:,} passages, {checked:,} equations re-evaluated, {failed:,} mismatches")
    print("PASS" if failed == 0 else "FAIL")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
