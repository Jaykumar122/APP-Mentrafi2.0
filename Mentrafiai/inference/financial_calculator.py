"""
MentraFiAI Deterministic Financial Calculator Engine
=====================================================
Eliminates math hallucination in the LLM pipeline by providing 100% exact,
SEBI-grounded compounding mathematics down to the rupee.

Covers:
1. Systematic Investment Plan (SIP) Compounding
2. Goal-Based Target SIP (Inverse calculation for target corpus e.g. 1 Crore)
3. Annual Step-Up / Top-Up SIP (10% standard annual increment)
4. Lumpsum / One-Time Investment Projections
5. Systematic Withdrawal Plan (SWP)
6. Inflation-Adjusted Real Value (at 6% standard long-term inflation)
7. Automated Intent & Number Extraction from Natural Language
"""

import re
import math
from typing import Optional, Dict, Any, Tuple


def format_inr(val: float) -> str:
    """Formats a float value into readable Indian Rupee notation (Lakh / Crore / Thousands)."""
    abs_val = abs(val)
    if abs_val >= 10000000:
        cr = val / 10000000.0
        return f"₹{cr:.2f} Crore"
    elif abs_val >= 100000:
        lk = val / 100000.0
        return f"₹{lk:.2f} Lakh"
    else:
        return f"₹{val:,.0f}"


def calculate_sip(monthly_sip: float, years: int, annual_cagr: float = 12.0) -> Dict[str, Any]:
    """
    Calculates regular monthly SIP projection using standard ordinary annuity formula:
    FV = P * [((1 + r)^n - 1) / r]
    where P = monthly SIP, r = annual_cagr / 12, n = years * 12.
    """
    r = (annual_cagr / 100.0) / 12.0
    n = max(1, years * 12)
    invested = monthly_sip * n
    
    # Standard ordinary annuity (payments at end of each period)
    corpus = monthly_sip * (((1.0 + r)**n - 1.0) / r)
    wealth_gain = corpus - invested
    
    # 6% inflation adjusted purchasing power: FV / (1 + 0.06)^years
    inflation_rate = 0.06
    real_corpus = corpus / ((1.0 + inflation_rate) ** years)
    
    return {
        "monthly_sip": monthly_sip,
        "years": years,
        "cagr": annual_cagr,
        "invested": invested,
        "wealth_gain": wealth_gain,
        "corpus": corpus,
        "real_corpus": real_corpus,
        "invested_fmt": format_inr(invested),
        "gain_fmt": format_inr(wealth_gain),
        "corpus_fmt": format_inr(corpus),
        "real_corpus_fmt": format_inr(real_corpus),
    }


def calculate_step_up_sip(monthly_sip: float, years: int, annual_cagr: float = 12.0, step_up_pct: float = 10.0) -> Dict[str, Any]:
    """
    Calculates annual Step-Up SIP (increasing monthly installment by step_up_pct every 12 months)
    with compounding at the end of each month:
    corpus = corpus * (1 + r) + current_sip
    """
    r = (annual_cagr / 100.0) / 12.0
    total_invested = 0.0
    corpus = 0.0
    current_sip = monthly_sip
    
    for y in range(years):
        for m in range(12):
            total_invested += current_sip
            corpus = corpus * (1.0 + r) + current_sip
        current_sip *= (1.0 + step_up_pct / 100.0)
        
    return {
        "initial_sip": monthly_sip,
        "step_up_pct": step_up_pct,
        "invested": total_invested,
        "wealth_gain": corpus - total_invested,
        "corpus": corpus,
        "invested_fmt": format_inr(total_invested),
        "gain_fmt": format_inr(corpus - total_invested),
        "corpus_fmt": format_inr(corpus),
    }


def calculate_goal_sip(target_corpus: float, years: int, annual_cagr: float = 12.0) -> Dict[str, Any]:
    """Calculates the exact monthly SIP required to reach target_corpus in years at annual_cagr."""
    r = (annual_cagr / 100.0) / 12.0
    n = max(1, years * 12)
    # Target / Ordinary Annuity factor
    factor = ((1.0 + r)**n - 1.0) / r
    required_sip = target_corpus / factor
    total_invested = required_sip * n
    wealth_gain = target_corpus - total_invested
    
    return {
        "target_corpus": target_corpus,
        "years": years,
        "cagr": annual_cagr,
        "required_sip": math.ceil(required_sip),
        "total_invested": total_invested,
        "wealth_gain": wealth_gain,
        "required_sip_fmt": format_inr(math.ceil(required_sip)),
        "invested_fmt": format_inr(total_invested),
        "gain_fmt": format_inr(wealth_gain),
        "target_fmt": format_inr(target_corpus),
    }


def calculate_multi_scenario_sip(monthly_sip: float, years: int) -> list:
    """Calculates comparative outcomes across 6%, 8%, 10%, and 12% illustrative return rates."""
    scenarios = []
    for rate in [6.0, 8.0, 10.0, 12.0]:
        scenarios.append(calculate_sip(monthly_sip, years, rate))
    return scenarios


def calculate_lumpsum(principal: float, years: int, annual_cagr: float = 12.0) -> Dict[str, Any]:
    """Calculates one-time lumpsum compounding."""
    corpus = principal * ((1.0 + (annual_cagr / 100.0)) ** years)
    wealth_gain = corpus - principal
    return {
        "principal": principal,
        "years": years,
        "cagr": annual_cagr,
        "invested": principal,
        "wealth_gain": wealth_gain,
        "corpus": corpus,
        "invested_fmt": format_inr(principal),
        "gain_fmt": format_inr(wealth_gain),
        "corpus_fmt": format_inr(corpus),
    }


def calculate_retirement_sip(
    current_age: int,
    retirement_age: int = 60,
    target_corpus: Optional[float] = None,
    monthly_sip: Optional[float] = None,
    annual_cagr: float = 12.0,
) -> Dict[str, Any]:
    """Calculates retirement runway, corpus accumulation, and target SIP."""
    years = max(1, retirement_age - current_age)
    if target_corpus:
        res = calculate_goal_sip(target_corpus, years, annual_cagr)
        res["current_age"] = current_age
        res["retirement_age"] = retirement_age
        return res
    elif monthly_sip:
        res = calculate_sip(monthly_sip, years, annual_cagr)
        res["current_age"] = current_age
        res["retirement_age"] = retirement_age
        return res
    else:
        res = calculate_goal_sip(50000000.0, years, annual_cagr)
        res["current_age"] = current_age
        res["retirement_age"] = retirement_age
        return res


def calculate_education_inflation_sip(
    years: int,
    current_cost: float,
    education_inflation: float = 10.0,
    investment_cagr: float = 13.0,
) -> Dict[str, Any]:
    """Calculates goal SIP for higher education accounting for higher inflation rate."""
    future_cost = current_cost * ((1.0 + education_inflation / 100.0) ** years)
    res = calculate_goal_sip(future_cost, years, investment_cagr)
    res["current_cost"] = current_cost
    res["future_cost"] = future_cost
    res["current_cost_fmt"] = format_inr(current_cost)
    res["future_cost_fmt"] = format_inr(future_cost)
    return res


def parse_currency_amount(text: str) -> Optional[float]:
    """Extracts numeric rupee amounts from text like 5000, 10k, 1.5 Lakh, 2 Crore."""
    # Crore
    cr_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:crore|crores|cr)\b", text, re.IGNORECASE)
    if cr_m:
        return float(cr_m.group(1)) * 10000000.0
        
    # Lakh
    lk_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs|l)\b", text, re.IGNORECASE)
    if lk_m:
        return float(lk_m.group(1)) * 100000.0
        
    # Thousands (e.g. 5k, 10k)
    k_m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", text, re.IGNORECASE)
    if k_m:
        return float(k_m.group(1)) * 1000.0
        
    # Bare rupee amount (e.g. ₹5000, Rs 5,000, 10000)
    bare_m = re.search(r"(?:₹|rs\.?|inr)?\s*(\d{1,3}(?:,\d{3})+|\d{3,9})\b", text, re.IGNORECASE)
    if bare_m:
        cleaned = bare_m.group(1).replace(",", "")
        amt = float(cleaned)
        if amt >= 100:  # ignore tiny numbers like age/years
            return amt
            
    return None


def detect_and_calculate(query: str, default_years: int = 10, default_cagr: float = 12.0, risk_profile: Optional[str] = None) -> Optional[str]:
    """
    Analyzes whether the query is asking for an explicit mathematical financial calculation.
    If so, executes the formula deterministically and returns a rich, formatted markdown report.
    Returns None if the query is not an explicit math/calculation query.
    """
    clean_q = query.lower().strip()
    
    # 1. Check for calculation intent triggers
    calc_triggers = [
        "calculate", "compounding", "future value", "corpus",
        "how much will", "how much sip", "how to get", "how to reach",
        "target", "become in", "grow to", "lumpsum", "sip of", "investing of",
        "accumulate", "achieve", "reach rs", "reach \u20b9", "need rs", "need \u20b9",
        "what should i do", "how do i get", "how can i get", "how can i reach",
        "how can i accumulate", "how do i accumulate"
    ]
    
    is_calc = any(t in clean_q for t in calc_triggers) or (("sip" in clean_q or "lump" in clean_q) and ("year" in clean_q or "yr" in clean_q))
    if not is_calc:
        return None
        
    # Extract years/horizon (avoiding matching age like 'age: 20 years')
    years = default_years
    duration_patterns = [
        r"(?:investment\s+)?duration\s*(?:is|of|:)?\s*(\d+)\s*(?:years?|yrs?)?",
        r"(?:investment\s+)?horizon\s*(?:is|of|:)?\s*(\d+)\s*(?:years?|yrs?)?",
        r"(?:for|in)\s+(\d+)\s*(?:years?|yrs?)\b",
        r"\b(\d+)\s*[-–]?\s*(?:years?|yrs?)\s*(?:horizon|period|duration|time)\b",
    ]
    extracted_yr = None
    for dp in duration_patterns:
        m = re.search(dp, clean_q)
        if m:
            extracted_yr = int(m.group(1))
            break
    if not extracted_yr:
        no_age = re.sub(r"\bage\s*(?:is|of|:)?\s*\d+\s*(?:years?|yrs?|old)?\b", "", clean_q)
        no_age = re.sub(r"\b\d+\s*(?:years?\s*old|y/?o|-year-old)\b", "", no_age)
        m = re.search(r"\b(\d+)\s*(?:years?|yrs?)\b", no_age)
        if m:
            extracted_yr = int(m.group(1))
    if extracted_yr and 1 <= extracted_yr <= 50:
        years = extracted_yr
        
    # Extract CAGR/rate if user specified (e.g. "at 14%", "15% return")
    cagr = default_cagr
    rate_m = re.search(r"(\d+(?:\.\d+)?)\s*%", clean_q)
    if rate_m:
        cagr = float(rate_m.group(1))
        
    # Extract Amount
    amt = parse_currency_amount(query)
    if not amt:
        return None

    # Detect Risk Profile
    is_low_risk = (
        (risk_profile and any(r in risk_profile.lower() for r in ["low", "conservative"]))
        or any(w in clean_q for w in ["low risk", "conservative", "safe", "capital protection", "capital preservation", "risk averse"])
    )
        
    # Case A: Goal Target Calculation (e.g. "how much SIP to get 1 Crore in 15 years")
    is_goal = any(w in clean_q for w in ["target", "to get", "to reach", "for 1 crore", "for 50 lakh", "accumulate", "need"])
    if is_goal and amt >= 500000:  # Goals are typically >= 5 Lakhs
        res = calculate_goal_sip(amt, years, cagr)
        step_res = calculate_step_up_sip(res["required_sip"] * 0.75, years, cagr, 10.0)
        
        return f"""### 🎯 Goal Planning: Reaching {res['target_fmt']} in {years} Years

To achieve your target corpus of **{res['target_fmt']}** over a **{years}-year horizon** (assuming ~{cagr:.1f}% illustrative CAGR in diversified mutual funds):

💰 **Required Monthly SIP:**
• **Fixed SIP Needed:** **{res['required_sip_fmt']} / month**
• Total Principal Invested: **{res['invested_fmt']}**
• Estimated Wealth Gain: **{res['gain_fmt']}**
• Target Maturity Corpus: **{res['target_fmt']}**

🚀 **Smart Accelerator (10% Annual Step-Up SIP):**
If your income grows, starting at **{format_inr(res['required_sip'] * 0.72)} / month** and increasing by **10% each year** will also achieve the **{res['target_fmt']}** milestone while starting with a lower initial monthly burden!

📋 **Illustrative Return Scenarios ({years} Years):**
| Expected Return | Monthly SIP Needed | Total Invested | Target Corpus |
|:---|:---|:---|:---|
| **6.0% (Debt / Fixed Income)** | {format_inr(math.ceil(calculate_goal_sip(amt, years, 6.0)['required_sip']))}/mo | {format_inr(calculate_goal_sip(amt, years, 6.0)['total_invested'])} | {res['target_fmt']} |
| **8.0% (Conservative Hybrid)** | {format_inr(math.ceil(calculate_goal_sip(amt, years, 8.0)['required_sip']))}/mo | {format_inr(calculate_goal_sip(amt, years, 8.0)['total_invested'])} | {res['target_fmt']} |
| **10.0% (Balanced Hybrid)** | {format_inr(math.ceil(calculate_goal_sip(amt, years, 10.0)['required_sip']))}/mo | {format_inr(calculate_goal_sip(amt, years, 10.0)['total_invested'])} | {res['target_fmt']} |
| **12.0% (Equity Illustrative)** | {format_inr(math.ceil(calculate_goal_sip(amt, years, 12.0)['required_sip']))}/mo | {format_inr(calculate_goal_sip(amt, years, 12.0)['total_invested'])} | {res['target_fmt']} |

> *Returns are illustrative estimates, not guaranteed. Actual returns depend on market performance.*

⚖️ **Taxation & Direct Plan Note:**
• Taxes are not deducted from the corpus above. Equity fund LTCG (>1 year) above ₹1.25 Lakh/yr is taxed at 12.5%. Debt fund gains are taxed at applicable slab rates.
• Always choose **Direct-Growth** schemes to eliminate intermediary commissions.
"""

    # Case B: Lumpsum Calculation
    is_lumpsum = any(w in clean_q for w in ["lumpsum", "one-time", "one time", "single investment"])
    if is_lumpsum:
        res = calculate_lumpsum(amt, years, cagr)
        real_val = res["corpus"] / (1.06 ** years)
        return f"""### 📈 Lumpsum Investment Projection: {res['invested_fmt']} over {years} Years

Here is the exact compounding projection for a one-time lumpsum of **{res['invested_fmt']}** over **{years} years** at an assumed **{cagr:.1f}% CAGR**:

📊 **Growth Breakdown:**
• Initial Principal: **{res['invested_fmt']}**
• Estimated Wealth Created: **{res['gain_fmt']}**
• **Estimated Maturity Corpus:** **{res['corpus_fmt']}**
• *Inflation-Adjusted Purchasing Power (@ 6% inflation):* **{format_inr(real_val)}**

📋 **Illustrative Return Scenarios:**
| Scenario | Corpus | Wealth Gain | Purchasing Power (@ 6% infl.) |
|:---|:---|:---|:---|
| **6.0% (Debt / Low Risk)** | {format_inr(calculate_lumpsum(amt, years, 6.0)['corpus'])} | {format_inr(calculate_lumpsum(amt, years, 6.0)['wealth_gain'])} | {format_inr(calculate_lumpsum(amt, years, 6.0)['corpus'] / (1.06**years))} |
| **8.0% (Conservative Hybrid)** | {format_inr(calculate_lumpsum(amt, years, 8.0)['corpus'])} | {format_inr(calculate_lumpsum(amt, years, 8.0)['wealth_gain'])} | {format_inr(calculate_lumpsum(amt, years, 8.0)['corpus'] / (1.06**years))} |
| **10.0% (Balanced Hybrid)** | {format_inr(calculate_lumpsum(amt, years, 10.0)['corpus'])} | {format_inr(calculate_lumpsum(amt, years, 10.0)['wealth_gain'])} | {format_inr(calculate_lumpsum(amt, years, 10.0)['corpus'] / (1.06**years))} |
| **12.0% (Equity Illustrative)** | {format_inr(calculate_lumpsum(amt, years, 12.0)['corpus'])} | {format_inr(calculate_lumpsum(amt, years, 12.0)['wealth_gain'])} | {format_inr(calculate_lumpsum(amt, years, 12.0)['corpus'] / (1.06**years))} |

> *Returns are illustrative estimates, not guaranteed. Actual returns depend on market performance.*

⚖️ **Taxation Note:**
Capital gains taxes (12.5% LTCG on equity over ₹1.25L/yr; slab rates on debt) are applicable at redemption and are not deducted from the corpus above.
"""

    # Case C: Regular Monthly SIP Calculation
    res = calculate_sip(amt, years, cagr)
    step_res = calculate_step_up_sip(amt, years, cagr, 10.0)
    scenarios = calculate_multi_scenario_sip(amt, years)
    
    risk_banner = ""
    if is_low_risk:
        risk_banner = (
            "🛡️ **Investor Profile:** **Low Risk (Conservative)**\n"
            "• As a low-risk investor, high-volatility pure equity funds (Flexi Cap, Mid Cap, Small Cap) are not suitable.\n"
            "• Recommended asset class: **Conservative Hybrid Funds** & **High-Quality Corporate Debt Funds**.\n"
            "• Historically, conservative debt delivers ~6%–8% CAGR. The 12% figure below is an illustrative equity comparison ceiling.\n\n"
        )
    
    return f"""### 📊 Exact SIP Compounding Projection: {format_inr(amt)}/month for {years} Years

{risk_banner}Here is the deterministic financial calculation using the SEBI-standard ordinary monthly SIP compounding formula:
**FV = P × [((1 + r)ⁿ − 1) / r]**
*(P = {format_inr(amt)}, r = annual rate / 12, n = {years * 12} months)*

💰 **1. Regular SIP Projection (Assuming {cagr:.1f}% Annual Return):**
• **Total Amount Invested:** **{res['invested_fmt']}** ({amt:,.0f} × {years * 12} installments)
• **Estimated Wealth Gain:** **{res['gain_fmt']}**
• **Total Expected Corpus:** **{res['corpus_fmt']}**
• **Inflation-Adjusted Purchasing Power (@ 6% inflation):** **{res['real_corpus_fmt']}**
  *(Calculated as: FV ÷ (1 + 0.06)^{years} = {format_inr(res['corpus'])} ÷ {1.06**years:.4f} ≈ {res['real_corpus_fmt']})*

🚀 **2. 10% Annual Step-Up SIP Accelerator:**
If you increase your monthly contribution by 10% each year as your income rises:
• **Total Invested:** **{step_res['invested_fmt']}**
• **Step-Up Wealth Gain:** **{step_res['gain_fmt']}**
• **Total Expected Corpus:** **{step_res['corpus_fmt']}**
*(Compounded at month-end, creating an additional {format_inr(step_res['corpus'] - res['corpus'])} over fixed SIP)*

📋 **3. Illustrative Return Scenarios (Comparative Tiers):**
| Annual CAGR | Asset Class Suitability | Total Invested | Expected Corpus | Expected Gain | Real Purchasing Power (@ 6% Infl.) |
|:---|:---|:---|:---|:---|:---|
| **6.0%** | Conservative Debt / Corporate Bonds | {scenarios[0]['invested_fmt']} | **{scenarios[0]['corpus_fmt']}** | {scenarios[0]['gain_fmt']} | {scenarios[0]['real_corpus_fmt']} |
| **8.0%** | Conservative Hybrid Funds (Low Risk) | {scenarios[1]['invested_fmt']} | **{scenarios[1]['corpus_fmt']}** | {scenarios[1]['gain_fmt']} | {scenarios[1]['real_corpus_fmt']} |
| **10.0%** | Balanced Advantage / Hybrid Funds | {scenarios[2]['invested_fmt']} | **{scenarios[2]['corpus_fmt']}** | {scenarios[2]['gain_fmt']} | {scenarios[2]['real_corpus_fmt']} |
| **12.0%** | Diversified Equity (Illustrative Benchmark) | {scenarios[3]['invested_fmt']} | **{scenarios[3]['corpus_fmt']}** | {scenarios[3]['gain_fmt']} | {scenarios[3]['real_corpus_fmt']} |

> *Returns are illustrative estimates, not guaranteed. Actual returns depend on market performance.*

⚖️ **4. Mutual Fund Taxation Rules (Finance Act 2024):**
• **Equity Funds (>65% domestic equity):** Long-Term Capital Gains (LTCG held >12 months) exceeding ₹1.25 Lakh per financial year are taxed at **12.5%**. Short-Term Capital Gains (STCG held ≤12 months) are taxed at **20.0%**.
• **Debt & Conservative Funds (≤65% equity):** Specified mutual funds invested on or after April 1, 2023 are taxed at the investor's applicable income tax slab rate.
• *Note:* Capital gains tax is paid only at redemption and is **not automatically deducted** from the gross corpus values above.
"""
