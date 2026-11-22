"""
MentraFiAI deterministic financial calculator (the "symbolic math engine").

Every number shown to a user or placed in the LLM context comes from here, so
the language model never has to do arithmetic itself.

Conventions (used consistently by the Python server, the training-data
builders, and the React Native SIP calculator):

* SIP monthly rate      r = annual_rate / 12   (nominal annual rate, monthly
                                                compounding; the convention of
                                                common Indian SIP calculators)
* SIP installment timing: start of each month (annuity due)
                         FV = P * ((1 + r)^n - 1) / r * (1 + r)
                         with the r = 0 limit FV = P * n
* Lump sum              FV = P * (1 + annual_rate)^years   (annual CAGR)
* Inflation adjustment  real = nominal / (1 + inflation)^years

Tax helpers implement the capital-gains rules for transfers on or after
23 July 2024 (Finance (No. 2) Act 2024) and the Section 80C cap, for FY 2025-26.
Surcharge, the 87A rebate, grandfathering, and set-off of losses are NOT
modelled; results are estimates and the user must confirm with current law.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_INFLATION_PCT = 6.0
TAX_RULES_AS_OF = "FY 2025-26 rules (transfers on or after 23 July 2024)"
EQUITY_LTCG_RATE = 12.5          # Section 112A
EQUITY_LTCG_EXEMPTION = 125000.0  # per financial year, shared across all 112A gains
EQUITY_STCG_RATE = 20.0          # Section 111A
OTHER_LTCG_RATE = 12.5           # Section 112, no indexation
HEALTH_EDUCATION_CESS_PCT = 4.0
SECTION_80C_LIMIT = 150000.0
SCENARIO_RATES = (6.0, 8.0, 10.0, 12.0)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────
def _indian_grouping(integer_text: str) -> str:
    if len(integer_text) <= 3:
        return integer_text
    head, tail = integer_text[:-3], integer_text[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail


def format_inr_exact(val: float) -> str:
    """Whole-rupee amount with Indian digit grouping, e.g. ₹12,34,567."""
    if not math.isfinite(val):
        raise ValueError("Cannot format a non-finite amount")
    sign = "-" if val < 0 else ""
    return f"{sign}₹{_indian_grouping(str(int(round(abs(val)))))}"


def format_inr(val: float) -> str:
    """Readable rupee amount: ₹4,500 / ₹2.35 Lakh / ₹1.20 Crore."""
    if not math.isfinite(val):
        raise ValueError("Cannot format a non-finite amount")
    sign = "-" if val < 0 else ""
    abs_val = abs(val)
    # Round first so 99,99,999.6 shows as ₹1.00 Crore, not ₹100.00 Lakh.
    if round(abs_val / 100000.0, 2) >= 100:
        return f"{sign}₹{abs_val / 10000000.0:.2f} Crore"
    if round(abs_val) >= 100000:
        return f"{sign}₹{abs_val / 100000.0:.2f} Lakh"
    return f"{sign}{format_inr_exact(abs_val)}"


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _amount(value: float, name: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return float(value)


def _months(years: float) -> int:
    if isinstance(years, bool) or not isinstance(years, (int, float)) or not math.isfinite(years):
        raise ValueError("years must be a finite number")
    months = int(round(years * 12))
    if months < 1:
        raise ValueError("Investment period must be at least one month")
    if months > 100 * 12:
        raise ValueError("Investment period above 100 years is not supported")
    return months


def _rate(annual_pct: float, name: str = "annual rate") -> float:
    if isinstance(annual_pct, bool) or not isinstance(annual_pct, (int, float)) or not math.isfinite(annual_pct):
        raise ValueError(f"{name} must be a finite percentage")
    if annual_pct <= -100 or annual_pct > 100:
        raise ValueError(f"{name} must be between -100% and 100%")
    return float(annual_pct)


def sip_factor(months: int, monthly_rate: float, *, timing: str = "start") -> float:
    """Future value of ₹1 paid every month for `months` months."""
    if timing not in ("start", "end"):
        raise ValueError("timing must be 'start' or 'end'")
    if monthly_rate == 0:
        return float(months)
    factor = math.expm1(months * math.log1p(monthly_rate)) / monthly_rate
    return factor * (1.0 + monthly_rate) if timing == "start" else factor


# ─────────────────────────────────────────────────────────────────────────────
# Core calculations
# ─────────────────────────────────────────────────────────────────────────────
def real_value(nominal: float, years: float, inflation_pct: float = DEFAULT_INFLATION_PCT) -> float:
    """Today's purchasing power of a future amount."""
    _rate(inflation_pct, "inflation")
    return nominal / ((1.0 + inflation_pct / 100.0) ** years)


def calculate_sip(monthly_sip: float, years: float, annual_cagr: float = 12.0, *,
                  timing: str = "start", inflation_pct: float = DEFAULT_INFLATION_PCT) -> Dict[str, Any]:
    """Fixed monthly SIP projection (installment at the start of each month by default)."""
    monthly_sip = _amount(monthly_sip, "monthly_sip")
    annual_cagr = _rate(annual_cagr)
    n = _months(years)
    r = annual_cagr / 100.0 / 12.0
    invested = monthly_sip * n
    corpus = monthly_sip * sip_factor(n, r, timing=timing)
    real_corpus = real_value(corpus, n / 12.0, inflation_pct)
    return {
        "monthly_sip": monthly_sip,
        "years": n / 12.0,
        "months": n,
        "cagr": annual_cagr,
        "monthly_rate": r,
        "timing": timing,
        "invested": invested,
        "wealth_gain": corpus - invested,
        "corpus": corpus,
        "real_corpus": real_corpus,
        "inflation_pct": inflation_pct,
        "invested_fmt": format_inr(invested),
        "gain_fmt": format_inr(corpus - invested),
        "corpus_fmt": format_inr(corpus),
        "real_corpus_fmt": format_inr(real_corpus),
    }


def _step_up_projection(monthly_sip: float, months: int, monthly_rate: float,
                        step_up_pct: float, timing: str) -> Tuple[float, float]:
    corpus = invested = 0.0
    installment = monthly_sip
    for month in range(months):
        if month and month % 12 == 0:
            installment *= 1.0 + step_up_pct / 100.0
        invested += installment
        if timing == "start":
            corpus = (corpus + installment) * (1.0 + monthly_rate)
        else:
            corpus = corpus * (1.0 + monthly_rate) + installment
    return corpus, invested


def calculate_step_up_sip(monthly_sip: float, years: float, annual_cagr: float = 12.0,
                          step_up_pct: float = 10.0, *, timing: str = "start",
                          inflation_pct: float = DEFAULT_INFLATION_PCT) -> Dict[str, Any]:
    """SIP whose installment rises by step_up_pct after every 12 installments."""
    monthly_sip = _amount(monthly_sip, "monthly_sip")
    annual_cagr = _rate(annual_cagr)
    _amount(step_up_pct, "step_up_pct")
    if timing not in ("start", "end"):
        raise ValueError("timing must be 'start' or 'end'")
    n = _months(years)
    r = annual_cagr / 100.0 / 12.0
    corpus, invested = _step_up_projection(monthly_sip, n, r, step_up_pct, timing)
    last_installment = monthly_sip * (1.0 + step_up_pct / 100.0) ** ((n - 1) // 12)
    real_corpus = real_value(corpus, n / 12.0, inflation_pct)
    return {
        "initial_sip": monthly_sip,
        "final_sip": last_installment,
        "step_up_pct": step_up_pct,
        "years": n / 12.0,
        "months": n,
        "cagr": annual_cagr,
        "timing": timing,
        "invested": invested,
        "wealth_gain": corpus - invested,
        "corpus": corpus,
        "real_corpus": real_corpus,
        "invested_fmt": format_inr(invested),
        "gain_fmt": format_inr(corpus - invested),
        "corpus_fmt": format_inr(corpus),
        "real_corpus_fmt": format_inr(real_corpus),
        "final_sip_fmt": format_inr(last_installment),
    }


def calculate_goal_sip(target_corpus: float, years: float, annual_cagr: float = 12.0, *,
                       timing: str = "start") -> Dict[str, Any]:
    """Monthly SIP needed to reach target_corpus (rounded UP to the next rupee)."""
    target_corpus = _amount(target_corpus, "target_corpus", allow_zero=False)
    annual_cagr = _rate(annual_cagr)
    n = _months(years)
    r = annual_cagr / 100.0 / 12.0
    exact = target_corpus / sip_factor(n, r, timing=timing)
    required = float(math.ceil(exact - 1e-9))
    total_invested = required * n
    projected = required * sip_factor(n, r, timing=timing)
    return {
        "target_corpus": target_corpus,
        "years": n / 12.0,
        "months": n,
        "cagr": annual_cagr,
        "timing": timing,
        "required_sip_exact": exact,
        "required_sip": int(required),
        "total_invested": total_invested,
        "projected_corpus": projected,
        "wealth_gain": projected - total_invested,
        "required_sip_fmt": format_inr_exact(required),
        "invested_fmt": format_inr(total_invested),
        "gain_fmt": format_inr(projected - total_invested),
        "target_fmt": format_inr(target_corpus),
    }


def calculate_goal_step_up_sip(target_corpus: float, years: float, annual_cagr: float = 12.0,
                               step_up_pct: float = 10.0, *, timing: str = "start") -> Dict[str, Any]:
    """Starting SIP that reaches target_corpus with an annual step-up.

    The step-up corpus is linear in the starting installment, so the exact
    answer is target / corpus_of_one_rupee_start (no search needed).
    """
    target_corpus = _amount(target_corpus, "target_corpus", allow_zero=False)
    annual_cagr = _rate(annual_cagr)
    _amount(step_up_pct, "step_up_pct")
    n = _months(years)
    r = annual_cagr / 100.0 / 12.0
    unit_corpus, unit_invested = _step_up_projection(1.0, n, r, step_up_pct, timing)
    start = float(math.ceil(target_corpus / unit_corpus - 1e-9))
    projected, invested = _step_up_projection(start, n, r, step_up_pct, timing)
    return {
        "target_corpus": target_corpus,
        "years": n / 12.0,
        "months": n,
        "cagr": annual_cagr,
        "step_up_pct": step_up_pct,
        "starting_sip": int(start),
        "final_sip": start * (1.0 + step_up_pct / 100.0) ** ((n - 1) // 12),
        "total_invested": invested,
        "projected_corpus": projected,
        "starting_sip_fmt": format_inr_exact(start),
        "invested_fmt": format_inr(invested),
        "target_fmt": format_inr(target_corpus),
    }


def calculate_multi_scenario_sip(monthly_sip: float, years: float,
                                 rates: Tuple[float, ...] = SCENARIO_RATES) -> List[Dict[str, Any]]:
    """Same SIP across several illustrative return rates."""
    return [calculate_sip(monthly_sip, years, rate) for rate in rates]


def calculate_lumpsum(principal: float, years: float, annual_cagr: float = 12.0, *,
                      inflation_pct: float = DEFAULT_INFLATION_PCT) -> Dict[str, Any]:
    """One-time investment compounded annually at annual_cagr."""
    principal = _amount(principal, "principal")
    annual_cagr = _rate(annual_cagr)
    n = _months(years)
    years = n / 12.0
    corpus = principal * (1.0 + annual_cagr / 100.0) ** years
    real_corpus = real_value(corpus, years, inflation_pct)
    return {
        "principal": principal,
        "years": years,
        "cagr": annual_cagr,
        "invested": principal,
        "wealth_gain": corpus - principal,
        "corpus": corpus,
        "real_corpus": real_corpus,
        "invested_fmt": format_inr(principal),
        "gain_fmt": format_inr(corpus - principal),
        "corpus_fmt": format_inr(corpus),
        "real_corpus_fmt": format_inr(real_corpus),
    }


def calculate_cagr(start_value: float, end_value: float, years: float) -> float:
    """Compound annual growth rate in percent."""
    start_value = _amount(start_value, "start_value", allow_zero=False)
    end_value = _amount(end_value, "end_value")
    if isinstance(years, bool) or not isinstance(years, (int, float)) or not math.isfinite(years) or years <= 0:
        raise ValueError("years must be positive")
    return ((end_value / start_value) ** (1.0 / years) - 1.0) * 100.0


def real_return_pct(nominal_pct: float, inflation_pct: float = DEFAULT_INFLATION_PCT) -> float:
    """Inflation-adjusted return using the Fisher relation (not a simple subtraction)."""
    _rate(nominal_pct)
    _rate(inflation_pct, "inflation")
    return ((1.0 + nominal_pct / 100.0) / (1.0 + inflation_pct / 100.0) - 1.0) * 100.0


def calculate_swp(corpus: float, monthly_withdrawal: float, annual_return: float = 8.0,
                  max_years: int = 50) -> Dict[str, Any]:
    """Systematic withdrawal: growth is credited monthly, then the withdrawal is paid."""
    balance = _amount(corpus, "corpus", allow_zero=False)
    withdrawal = _amount(monthly_withdrawal, "monthly_withdrawal", allow_zero=False)
    r = _rate(annual_return) / 100.0 / 12.0
    months_paid = 0
    withdrawn = 0.0
    limit = int(max_years) * 12
    while months_paid < limit:
        balance *= 1.0 + r
        if balance < withdrawal:
            withdrawn += balance
            balance = 0.0
            break
        balance -= withdrawal
        withdrawn += withdrawal
        months_paid += 1
    lasts_full_period = months_paid >= limit
    return {
        "corpus": float(corpus),
        "monthly_withdrawal": withdrawal,
        "annual_return": annual_return,
        "months_fully_paid": months_paid,
        "years_fully_paid": months_paid / 12.0,
        "total_withdrawn": withdrawn,
        "remaining_corpus": balance,
        "lasts_full_period": lasts_full_period,
        "sustainable_forever": withdrawal <= corpus * r,
        "max_years": max_years,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tax engine (Section 111A / 112A / 112 / 50AA and Section 80C)
# ─────────────────────────────────────────────────────────────────────────────
FUND_TAX_CLASSES = {
    "equity": "Equity-oriented fund (at least 65% in domestic equity)",
    "debt": "Specified (debt-oriented) fund bought on or after 1 April 2023",
    "other": "Other fund (for example a hybrid with 35-65% equity, or debt bought before April 2023)",
}


def calculate_capital_gains_tax(gain: float, holding_months: float, fund_type: str = "equity", *,
                                other_equity_ltcg_this_fy: float = 0.0,
                                slab_rate_pct: Optional[float] = None,
                                include_cess: bool = True) -> Dict[str, Any]:
    """Estimate capital-gains tax on a mutual fund redemption.

    fund_type: 'equity', 'debt' (specified fund, Section 50AA) or 'other'.
    slab_rate_pct: the investor's marginal slab; required whenever gains are
    taxed at slab rate, otherwise the tax is reported as unknown (None).
    Surcharge, rebate, grandfathering and loss set-off are not modelled.
    """
    gain = float(gain)
    if not math.isfinite(gain):
        raise ValueError("gain must be finite")
    if fund_type not in FUND_TAX_CLASSES:
        raise ValueError("fund_type must be equity, debt or other")
    _amount(holding_months, "holding_months")
    _amount(other_equity_ltcg_this_fy, "other_equity_ltcg_this_fy")
    if slab_rate_pct is not None:
        _rate(slab_rate_pct, "slab_rate_pct")

    exemption_used = 0.0
    if gain <= 0:
        rate, taxable, term, section = 0.0, 0.0, "loss" if gain < 0 else "none", "No tax on a loss or zero gain"
    elif fund_type == "equity":
        if holding_months > 12:
            term, section, rate = "long-term", "Section 112A", EQUITY_LTCG_RATE
            remaining_exemption = max(0.0, EQUITY_LTCG_EXEMPTION - other_equity_ltcg_this_fy)
            exemption_used = min(gain, remaining_exemption)
            taxable = gain - exemption_used
        else:
            term, section, rate, taxable = "short-term", "Section 111A", EQUITY_STCG_RATE, gain
    elif fund_type == "debt":
        term, section, rate, taxable = "deemed short-term", "Section 50AA (slab rate)", slab_rate_pct, gain
    else:
        if holding_months > 24:
            term, section, rate, taxable = "long-term", "Section 112 (no indexation)", OTHER_LTCG_RATE, gain
        else:
            term, section, rate, taxable = "short-term", "Slab rate", slab_rate_pct, gain

    if rate is None:
        base_tax = cess = total_tax = None
    else:
        base_tax = taxable * rate / 100.0
        cess = base_tax * HEALTH_EDUCATION_CESS_PCT / 100.0 if include_cess else 0.0
        total_tax = base_tax + cess
    return {
        "gain": gain,
        "holding_months": holding_months,
        "fund_type": fund_type,
        "fund_class": FUND_TAX_CLASSES[fund_type],
        "term": term,
        "section": section,
        "rate_pct": rate,
        "exemption_used": exemption_used,
        "taxable_gain": taxable,
        "base_tax": base_tax,
        "cess": cess,
        "total_tax": total_tax,
        "post_tax_gain": None if total_tax is None else gain - total_tax,
        "needs_slab_rate": rate is None,
        "rules_as_of": TAX_RULES_AS_OF,
        "not_modelled": "surcharge, 87A rebate, grandfathering, loss set-off",
    }


def calculate_80c_saving(investment: float, slab_rate_pct: float, *, other_80c: float = 0.0,
                         regime: str = "old", include_cess: bool = True) -> Dict[str, Any]:
    """Tax saved by an 80C investment such as ELSS (old regime only)."""
    investment = _amount(investment, "investment")
    other_80c = _amount(other_80c, "other_80c")
    _rate(slab_rate_pct, "slab_rate_pct")
    if regime not in ("old", "new"):
        raise ValueError("regime must be old or new")
    if regime == "new":
        eligible = 0.0
    else:
        eligible = min(investment, max(0.0, SECTION_80C_LIMIT - other_80c))
    saving = eligible * slab_rate_pct / 100.0
    if include_cess:
        saving *= 1.0 + HEALTH_EDUCATION_CESS_PCT / 100.0
    return {
        "investment": investment,
        "other_80c": other_80c,
        "regime": regime,
        "limit": SECTION_80C_LIMIT,
        "eligible_deduction": eligible,
        "unused_limit": 0.0 if regime == "new" else max(0.0, SECTION_80C_LIMIT - other_80c - eligible),
        "slab_rate_pct": slab_rate_pct,
        "tax_saved": saving,
        "rules_as_of": TAX_RULES_AS_OF,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Goal helpers kept for API compatibility
# ─────────────────────────────────────────────────────────────────────────────
def calculate_retirement_sip(current_age: int, retirement_age: int = 60, target_corpus: Optional[float] = None,
                             monthly_sip: Optional[float] = None, annual_cagr: float = 12.0) -> Dict[str, Any]:
    if retirement_age <= current_age:
        raise ValueError("retirement_age must be greater than current_age")
    years = retirement_age - current_age
    if monthly_sip and not target_corpus:
        result = calculate_sip(monthly_sip, years, annual_cagr)
    else:
        result = calculate_goal_sip(target_corpus or 50000000.0, years, annual_cagr)
    result["current_age"] = current_age
    result["retirement_age"] = retirement_age
    return result


def calculate_education_inflation_sip(years: float, current_cost: float, education_inflation: float = 10.0,
                                      investment_cagr: float = 12.0) -> Dict[str, Any]:
    current_cost = _amount(current_cost, "current_cost", allow_zero=False)
    future_cost = current_cost * (1.0 + _rate(education_inflation, "education_inflation") / 100.0) ** years
    result = calculate_goal_sip(future_cost, years, investment_cagr)
    result.update({"current_cost": current_cost, "future_cost": future_cost,
                   "current_cost_fmt": format_inr(current_cost), "future_cost_fmt": format_inr(future_cost)})
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Natural-language number extraction
# ─────────────────────────────────────────────────────────────────────────────
_UNIT_MULTIPLIERS = (
    (r"(?:crores?|cr)", 10000000.0),
    (r"(?:lakhs?|lacs?|lac|l)", 100000.0),
    (r"(?:thousand|k)", 1000.0),
)
_NUMBER = r"(?P<num>\d+(?:,\d{2,3})*(?:\.\d+)?)"
_MONTHLY_CUE = re.compile(r"^\s*(?:/\s*-?\s*)?(?:(?:rs\.?|inr|rupees)\s*)?(?:per\s+month|a\s+month|every\s+month|monthly|/\s*(?:month|mo|m)\b|pm\b|p\.m\.)", re.I)
_MONTHLY_PREFIX = re.compile(r"(?:sip\s+(?:of|amount|is|:)?|monthly\s+(?:sip|investment|amount|budget)?\s*(?:of|is|:)?|invest(?:ing)?\s+(?:of\s+)?)\s*(?:₹|rs\.?|inr)?\s*$", re.I)


def extract_amounts(text: str) -> List[Dict[str, Any]]:
    """All rupee amounts with a monthly flag, in order of appearance.

    Years, ages, and percentages are not amounts: plain numbers below 100,
    numbers followed by %, years, months, or 'old', and four-digit calendar
    years such as 2030 without a rupee marker are ignored.
    """
    results: List[Dict[str, Any]] = []
    unit_pattern = "|".join(f"(?P<u{i}>{pattern})" for i, (pattern, _) in enumerate(_UNIT_MULTIPLIERS))
    pattern = re.compile(
        r"(?P<cur>₹|\brs\.?|\binr\b|\brupees\b)?\s*(?<![\d.,])" + _NUMBER +
        r"\s*(?:(?:" + unit_pattern + r")\b)?", re.I)
    for match in pattern.finditer(text):
        raw = match.group("num")
        value = float(raw.replace(",", ""))
        multiplier = 1.0
        for i, (_, unit_value) in enumerate(_UNIT_MULTIPLIERS):
            if match.group(f"u{i}"):
                multiplier = unit_value
        number_start, number_end = match.start("num"), match.end("num")
        after = text[match.end():match.end() + 12].lower()
        has_currency = bool(match.group("cur"))
        before = text[max(0, match.start() - 40):match.start()]
        monthly = bool(_MONTHLY_CUE.match(text[match.end():match.end() + 20]) or _MONTHLY_PREFIX.search(before))
        if multiplier == 1.0:
            # "Section 112A", "80C", "T4", "FY2025": part of a word, not money.
            if number_end < len(text) and text[number_end].isalpha():
                continue
            if not has_currency and number_start > 0 and text[number_start - 1].isalpha():
                continue
            if re.match(r"\s*(?:%|percent\b|years?\b|yrs?\b|months?\b|mos?\b|yo\b|year-old|-year|days?\b|old\b)", after):
                continue
            if not has_currency and value < 100:
                continue
            if (not has_currency and not monthly and re.fullmatch(r"(19|20)\d\d", raw)
                    and re.search(r"\b(in|by|until|till|before|after|from|since|year|fy|of)\s*$", before, re.I)):
                continue
        results.append({"amount": value * multiplier, "monthly": monthly, "start": match.start(),
                        "text": match.group(0).strip()})
    return results


def parse_currency_amount(text: str) -> Optional[float]:
    """First rupee amount in text: 5000, ₹10,000, 1,00,000, 10k, 1.5 lakh, 2 crore."""
    amounts = extract_amounts(text)
    return amounts[0]["amount"] if amounts else None


def _extract_years(clean_q: str) -> Optional[float]:
    patterns = [
        r"(?:investment\s+)?(?:duration|horizon|period|tenure)\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)\s*(years?|yrs?|months?)?",
        r"(?:for|in|over|after|within|next)\s+(\d+(?:\.\d+)?)\s*(years?|yrs?|months?)\b",
        r"\b(\d+(?:\.\d+)?)\s*[-–]?\s*(years?|yrs?)\s*(?:horizon|period|duration|time|sip)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean_q)
        if match:
            value = float(match.group(1))
            unit = match.group(2) or "years"
            return value / 12.0 if unit.startswith("month") else value
    no_age = re.sub(r"\bage\s*(?:is|of|:)?\s*\d+\s*(?:years?|yrs?|old)?\b", " ", clean_q)
    no_age = re.sub(r"\b\d+\s*(?:years?\s*old|yrs?\s*old|y/?o|-year-old)\b", " ", no_age)
    no_age = re.sub(r"\b(?:held|holding|hold)\s+(?:for\s+)?\d+\s*(?:years?|yrs?|months?)\b", " ", no_age)
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(years?|yrs?|months?)\b", no_age)
    if match:
        value = float(match.group(1))
        return value / 12.0 if match.group(2).startswith("month") else value
    return None


def _extract_rate(clean_q: str, default: float) -> float:
    """Return rate near return/CAGR/interest words; ignore step-up/inflation percentages."""
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", clean_q):
        window = clean_q[max(0, match.start() - 30):match.end() + 25]
        if re.search(r"step[\s-]?up|top[\s-]?up|increase|increment|inflation|slab|tax|bracket|ltcg|stcg", window):
            continue
        return float(match.group(1))
    return default


def _extract_step_up(clean_q: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:annual\s+|yearly\s+|every\s+year\s+)?"
                      r"(?:step[\s-]?up|top[\s-]?up|increase|increment)", clean_q)
    if not match:
        match = re.search(r"(?:step[\s-]?up|top[\s-]?up|increase|increment)\s*(?:sip\s*)?(?:of|by|at|:|rate)?\s*"
                          r"(\d+(?:\.\d+)?)\s*(?:%|percent)", clean_q)
    return float(match.group(1)) if match else None


def _extract_holding_months(clean_q: str) -> Optional[float]:
    match = re.search(r"(?:held|holding|hold|after|for)\s+(?:it\s+|them\s+|units\s+)?(?:for\s+)?(\d+(?:\.\d+)?)\s*(years?|yrs?|months?)", clean_q)
    if not match:
        return None
    value = float(match.group(1))
    return value if match.group(2).startswith("month") else value * 12.0


def _extract_slab(clean_q: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:tax\s+)?(?:slab|bracket)", clean_q) or \
        re.search(r"(?:slab|bracket)\D{0,15}(\d+(?:\.\d+)?)\s*%", clean_q)
    return float(match.group(1)) if match else None


def _fmt_years(years: float) -> str:
    if abs(years - round(years)) < 1e-9:
        value = int(round(years))
        return f"{value} year" if value == 1 else f"{value} years"
    months = int(round(years * 12))
    return f"{months} months"


DISCLAIMER = ("> *Returns are illustrative assumptions, not guarantees. Mutual fund returns vary and can be negative. "
              "Taxes, exit loads and expense ratios are not deducted unless stated.*")

TAX_NOTE = (
    "⚖️ **Taxation note (" + TAX_RULES_AS_OF + "):**\n"
    "• **Equity-oriented funds (≥65% domestic equity):** gains on units held more than 12 months are long-term and taxed at "
    "**12.5%** above the **₹1.25 Lakh** yearly exemption (Section 112A); units held 12 months or less are taxed at **20%** (Section 111A).\n"
    "• **Debt-oriented (specified) funds bought on or after 1 April 2023:** gains are taxed at your income-tax slab rate regardless of holding period.\n"
    "• **Other funds (for example hybrids with 35–65% equity):** 12.5% after 24 months, slab rate otherwise.\n"
    "• 4% cess applies; surcharge is not shown. Tax is due only on redemption and is **not deducted** from the corpus above. "
    "Confirm current rules before acting."
)


def _scenario_label(rate: float) -> str:
    return {6.0: "Debt / fixed income", 8.0: "Conservative hybrid", 10.0: "Balanced / hybrid",
            12.0: "Diversified equity"}.get(rate, "Illustrative")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown reports
# ─────────────────────────────────────────────────────────────────────────────
def _goal_report(target: float, years: float, cagr: float, monthly_budget: Optional[float]) -> str:
    res = calculate_goal_sip(target, years, cagr)
    step = calculate_goal_step_up_sip(target, years, cagr, 10.0)
    span = _fmt_years(res["years"])
    lines = [
        f"### 🎯 Goal Planning: {res['target_fmt']} in {span}",
        "",
        f"Assumed return: **{cagr:.1f}% a year** (illustrative), installments at the start of each month.",
        "",
        "💰 **Fixed monthly SIP needed:**",
        f"• **Required SIP:** **{res['required_sip_fmt']} / month** for {res['months']} months",
        f"• Total invested: **{res['invested_fmt']}**",
        f"• Projected corpus: **{format_inr(res['projected_corpus'])}** (estimated gain {res['gain_fmt']})",
        f"• Target in today's money (@ {DEFAULT_INFLATION_PCT:.0f}% inflation): **{format_inr(real_value(target, res['years']))}**",
        "",
        "🚀 **Alternative: 10% annual step-up SIP**",
        f"• Start at **{step['starting_sip_fmt']} / month** and raise it by 10% every 12 months "
        f"(last installment ≈ {format_inr_exact(step['final_sip'])}).",
        f"• Total invested: **{step['invested_fmt']}**, projected corpus **{format_inr(step['projected_corpus'])}**.",
    ]
    if monthly_budget:
        sip = calculate_sip(monthly_budget, years, cagr)
        gap = target - sip["corpus"]
        lines += ["", f"📌 **Your budget of {format_inr_exact(monthly_budget)}/month** projects to **{sip['corpus_fmt']}**, "
                  + (f"short of the target by **{format_inr(gap)}**." if gap > 0 else
                     f"which meets the target with **{format_inr(-gap)}** to spare.")]
    lines += ["", f"📋 **Required SIP at other return assumptions ({span}):**",
              "| Assumed return | Monthly SIP | Total invested |", "|:---|:---|:---|"]
    for rate in SCENARIO_RATES:
        alt = calculate_goal_sip(target, years, rate)
        lines.append(f"| {rate:.1f}% ({_scenario_label(rate)}) | {alt['required_sip_fmt']}/mo | {alt['invested_fmt']} |")
    lines += ["", DISCLAIMER, "", TAX_NOTE]
    return "\n".join(lines)


def _lumpsum_report(amount: float, years: float, cagr: float) -> str:
    res = calculate_lumpsum(amount, years, cagr)
    span = _fmt_years(res["years"])
    lines = [
        f"### 📈 Lump Sum Projection: {res['invested_fmt']} for {span}",
        "",
        f"Formula: **FV = P × (1 + {cagr:.1f}%)^{res['years']:g}**",
        "",
        f"• Amount invested: **{res['invested_fmt']}**",
        f"• Estimated gain: **{res['gain_fmt']}**",
        f"• **Projected value:** **{res['corpus_fmt']}**",
        f"• In today's money (@ {DEFAULT_INFLATION_PCT:.0f}% inflation): **{res['real_corpus_fmt']}**",
        "",
        "📋 **Other return assumptions:**",
        "| Assumed return | Projected value | Gain | Today's money |", "|:---|:---|:---|:---|",
    ]
    for rate in SCENARIO_RATES:
        alt = calculate_lumpsum(amount, years, rate)
        lines.append(f"| {rate:.1f}% ({_scenario_label(rate)}) | {alt['corpus_fmt']} | {alt['gain_fmt']} | {alt['real_corpus_fmt']} |")
    lines += ["", DISCLAIMER, "", TAX_NOTE]
    return "\n".join(lines)


def _sip_report(amount: float, years: float, cagr: float, step_up_pct: float, is_low_risk: bool) -> str:
    res = calculate_sip(amount, years, cagr)
    step = calculate_step_up_sip(amount, years, cagr, step_up_pct)
    span = _fmt_years(res["years"])
    lines = [f"### 📊 SIP Projection: {format_inr_exact(amount)}/month for {span}", ""]
    if is_low_risk:
        lines += [
            "🛡️ **Low-risk profile:** pure equity (mid/small/flexi cap) funds are usually unsuitable. "
            "Conservative hybrid and high-quality debt funds have historically been closer to 6–8% a year; "
            "the higher rates below are shown only for comparison.", ""]
    lines += [
        f"Formula (installment at the start of each month): **FV = P × [((1 + r)ⁿ − 1) / r] × (1 + r)**",
        f"*(P = {format_inr_exact(amount)}, r = {cagr:.1f}% ÷ 12 = {res['monthly_rate'] * 100:.4f}% a month, n = {res['months']})*",
        "",
        f"💰 **Fixed SIP at {cagr:.1f}% a year:**",
        f"• Total invested: **{res['invested_fmt']}** ({format_inr_exact(amount)} × {res['months']})",
        f"• Estimated gain: **{res['gain_fmt']}**",
        f"• **Projected corpus:** **{res['corpus_fmt']}**",
        f"• In today's money (@ {DEFAULT_INFLATION_PCT:.0f}% inflation): **{res['real_corpus_fmt']}**",
        "",
        f"🚀 **With a {step_up_pct:g}% annual step-up:**",
        f"• Total invested: **{step['invested_fmt']}** (last installment ≈ {step['final_sip_fmt']})",
        f"• **Projected corpus:** **{step['corpus_fmt']}** ({format_inr(step['corpus'] - res['corpus'])} more than the fixed SIP)",
        "",
        "📋 **Other return assumptions:**",
        "| Assumed return | Suits | Total invested | Projected corpus | Gain | Today's money |",
        "|:---|:---|:---|:---|:---|:---|",
    ]
    for scenario in calculate_multi_scenario_sip(amount, years):
        lines.append(f"| {scenario['cagr']:.1f}% | {_scenario_label(scenario['cagr'])} | {scenario['invested_fmt']} | "
                     f"**{scenario['corpus_fmt']}** | {scenario['gain_fmt']} | {scenario['real_corpus_fmt']} |")
    lines += ["", DISCLAIMER, "", TAX_NOTE]
    return "\n".join(lines)


def _tax_report(gain: float, holding_months: Optional[float], fund_type: str, slab: Optional[float]) -> str:
    given = holding_months is not None
    if not given:
        holding_months = 13.0 if fund_type == "equity" else 25.0
    res = calculate_capital_gains_tax(gain, holding_months, fund_type, slab_rate_pct=slab)
    if fund_type == "debt":
        holding_text = "any holding period"
    elif given:
        holding_text = f"{holding_months:g} months"
    else:
        holding_text = f"{holding_months:g} months (not stated; assumed long-term)"
    lines = [
        f"### ⚖️ Capital Gains Tax Estimate: {format_inr_exact(gain)} gain",
        "",
        f"• Fund type: **{res['fund_class']}**",
        f"• Holding period: **{holding_text}** → **{res['term']}**, {res['section']}",
    ]
    if res["needs_slab_rate"]:
        lines += ["• These gains are taxed at your **income-tax slab rate**. Tell me your slab (for example 20% or 30%) "
                  "and I will compute the exact amount."]
    else:
        if res["exemption_used"]:
            lines.append(f"• ₹1.25 Lakh yearly exemption used: **{format_inr_exact(res['exemption_used'])}**")
        lines += [
            f"• Taxable gain: **{format_inr_exact(res['taxable_gain'])}** at **{res['rate_pct']:g}%**",
            f"• Tax: {format_inr_exact(res['base_tax'])} + 4% cess {format_inr_exact(res['cess'])} = "
            f"**{format_inr_exact(res['total_tax'])}**",
            f"• Gain after tax: **{format_inr_exact(res['post_tax_gain'])}**",
        ]
    lines += ["", f"*Estimate under {TAX_RULES_AS_OF}. Surcharge, the 87A rebate, other equity gains in the same year, "
              "grandfathering and loss set-off are not included. Confirm with a tax professional.*"]
    return "\n".join(lines)


def _80c_report(investment: float, slab: Optional[float]) -> str:
    if slab is None:
        return ("### 🧾 Section 80C (ELSS) Tax Saving\n\n"
                f"Under the old tax regime, ELSS investments up to **{format_inr_exact(SECTION_80C_LIMIT)}** a year "
                "(shared with PPF, EPF, life insurance and other 80C items) reduce taxable income. "
                f"You mentioned {format_inr_exact(investment)}. Tell me your slab rate (5%, 20% or 30%) and other 80C "
                "investments, and I will compute the tax saved. The new tax regime gives no 80C deduction.")
    res = calculate_80c_saving(investment, slab)
    return "\n".join([
        "### 🧾 Section 80C (ELSS) Tax Saving",
        "",
        f"• Investment: **{format_inr_exact(investment)}**; deduction allowed: **{format_inr_exact(res['eligible_deduction'])}** "
        f"(80C limit {format_inr_exact(SECTION_80C_LIMIT)})",
        f"• Slab rate: **{slab:g}%** + 4% cess",
        f"• **Estimated tax saved (old regime):** **{format_inr_exact(res['tax_saved'])}**",
        "• New tax regime: **₹0** (80C is not available).",
        "",
        "*Assumes no other 80C investments this year. ELSS units have a 3-year lock-in per installment.*",
    ])


def detect_and_calculate(query: str, default_years: float = 10, default_cagr: float = 12.0,
                         risk_profile: Optional[str] = None) -> Optional[str]:
    """Return a markdown report for an explicit calculation query, else None."""
    clean_q = query.lower().strip()
    amounts = extract_amounts(query)
    if not amounts:
        return None
    years_found = _extract_years(clean_q)
    years = years_found if years_found and 1 / 12 <= years_found <= 60 else default_years
    cagr = _extract_rate(clean_q, default_cagr)
    slab = _extract_slab(clean_q)

    # Tax: capital gains on a stated profit, or Section 80C saving.
    if re.search(r"\b(80c|elss)\b", clean_q) and re.search(r"\b(save|saving|saved|deduction)\b", clean_q):
        return _80c_report(amounts[0]["amount"], slab)
    gain_context = re.search(
        r"\b(made|earned|making|profits?|sold|sell|selling|redeem\w*|withdr[ae]w\w*|book\w*|pay)\b"
        r"|\bgains?\s+(?:of|is|was|are|worth)\b|\btax\s+(?:on|for)\b", clean_q)
    if (re.search(r"\btax\b|\bltcg\b|\bstcg\b", clean_q) and gain_context
            and re.search(r"\b(gains?|profits?|ltcg|stcg|redeem\w*|sold|sell\w*)\b", clean_q)):
        fund_type = "equity"
        if re.search(r"\b(debt|liquid|gilt|bond|corporate bond|money market)\b", clean_q):
            fund_type = "debt"
        elif re.search(r"\b(hybrid|balanced|conservative hybrid|fund of funds|fof|gold|international)\b", clean_q):
            fund_type = "other"
        holding = _extract_holding_months(clean_q)
        if holding is None and "ltcg" in clean_q:
            holding = 13.0 if fund_type == "equity" else 25.0
        if holding is None and "stcg" in clean_q:
            holding = 6.0
        return _tax_report(amounts[0]["amount"], holding, fund_type, slab)

    calc_triggers = (
        "calculate", "compounding", "future value", "corpus", "how much will", "how much sip", "how much should",
        "how to get", "how to reach", "target", "become in", "grow to", "lumpsum", "lump sum", "sip of",
        "accumulate", "achieve", "reach", "what will", "maturity", "returns on", "how do i get",
        "how can i get", "projection", "one-time", "one time",
    )
    explicit_calc = re.search(r"\b(calculate|calculation|compute|future value|corpus|how much will|projection|"
                              r"maturity|grow to|become)\b", clean_q)
    wants_funds = re.search(r"\b(suggest|recommend|portfolio|which (?:mutual )?funds?|best (?:mutual )?funds?|"
                            r"where (?:should|can) i invest|what should i invest|build a plan|which scheme)\b", clean_q)
    if wants_funds and not explicit_calc:
        return None  # a portfolio request: the advisor route handles it (with fund cards)
    goal_words = ("target", "goal", "to get", "to reach", "reach", "accumulate", "need", "corpus of",
                  "achieve", "build", "want", "save up", "plan for", "how much sip", "how much should i invest",
                  "become a crorepati")
    is_goal = any(w in clean_q for w in goal_words)
    if years_found is None:
        # "I am 30 ... till 60": horizon from current and target age.
        age = re.search(r"\b(?:age\s*(?:is|of|:)?\s*|i\s*am\s*|i'm\s*)(\d{2})\b", clean_q)
        until = re.search(r"\b(?:till|until|by|retire\s+at|at\s+age)\s+(?:age\s+)?(\d{2})\b", clean_q)
        if age and until and int(until.group(1)) > int(age.group(1)):
            years_found = float(int(until.group(1)) - int(age.group(1)))
            years = years_found
    is_calc = any(t in clean_q for t in calc_triggers) or (
        ("sip" in clean_q or "lump" in clean_q or is_goal) and years_found is not None)
    if not is_calc:
        return None

    is_low_risk = bool(risk_profile and any(w in risk_profile.lower() for w in ("low", "conservative"))) or any(
        w in clean_q for w in ("low risk", "conservative", "safe", "capital protection", "capital preservation", "risk averse"))

    monthly = [a["amount"] for a in amounts if a["monthly"]]
    others = [a["amount"] for a in amounts if not a["monthly"]]
    is_lumpsum = any(w in clean_q for w in ("lumpsum", "lump sum", "one-time", "one time", "single investment"))

    if is_goal and others and not is_lumpsum:
        target = max(others)
        if target >= 10000:
            return _goal_report(target, years, cagr, monthly[0] if monthly else None)
    if is_lumpsum:
        return _lumpsum_report((others or monthly)[0], years, cagr)
    amount = monthly[0] if monthly else amounts[0]["amount"]
    step_up = _extract_step_up(clean_q) or 10.0
    return _sip_report(amount, years, cagr, step_up, is_low_risk)


def build_calculation_facts(intent: str, *, monthly_sip: Optional[float] = None, years: Optional[float] = None,
                            annual_cagr: float = 12.0, target: Optional[float] = None,
                            lumpsum: Optional[float] = None) -> Dict[str, str]:
    """Pre-formatted, verified numbers for the LLM context (see llm120.grounding)."""
    facts: Dict[str, str] = {}
    if years is None:
        return facts
    if target:
        goal = calculate_goal_sip(target, years, annual_cagr)
        facts.update({"target_corpus": goal["target_fmt"], "required_monthly_sip": goal["required_sip_fmt"],
                      "total_invested": goal["invested_fmt"], "assumed_return": f"{annual_cagr:g}% a year",
                      "horizon": _fmt_years(goal["years"])})
    elif lumpsum:
        res = calculate_lumpsum(lumpsum, years, annual_cagr)
        facts.update({"lumpsum": format_inr_exact(lumpsum), "projected_value": res["corpus_fmt"],
                      "estimated_gain": res["gain_fmt"], "value_in_todays_money": res["real_corpus_fmt"],
                      "assumed_return": f"{annual_cagr:g}% a year", "horizon": _fmt_years(res["years"])})
    elif monthly_sip:
        res = calculate_sip(monthly_sip, years, annual_cagr)
        step = calculate_step_up_sip(monthly_sip, years, annual_cagr, 10.0)
        facts.update({"monthly_sip": format_inr_exact(monthly_sip), "total_invested": res["invested_fmt"],
                      "projected_corpus": res["corpus_fmt"], "estimated_gain": res["gain_fmt"],
                      "value_in_todays_money": res["real_corpus_fmt"],
                      "step_up_10pct_corpus": step["corpus_fmt"],
                      "assumed_return": f"{annual_cagr:g}% a year", "horizon": _fmt_years(res["years"])})
    return facts
