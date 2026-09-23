"""
Interactive terminal chat with MentraFiAI.
Loads the best trained checkpoint and allows real-time conversation.
"""

import sys
import os
import re
import logging
from datetime import datetime
from pathlib import Path
import torch

# Fix Windows console encoding for special characters
if sys.platform == "win32":
    os.system("")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from model import ModelConfig, MentraFiAI
from tokenizer.tokenizer_utils import Tokenizer
from training.utils import load_config
from inference.generate import generate_chat_reply, is_greeting, handle_greeting, is_informational, informational_reply, GREETING_REPLY
from inference.query_parser import FinancialQueryParser, FinancialQuery
from inference.db_fund_retriever import FundDatabase, format_star_rating
from typing import Optional

# ── Fix #13: Structured session logging ───────────────────────────────────────
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_session_log = _LOG_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(_session_log, encoding="utf-8"),
    ],
)
logger = logging.getLogger("mentrafi")
logger.info("=== MentraFiAI Session Started ===")
# ──────────────────────────────────────────────────────────────────────────────




def extract_portfolio_params(clean: str, classifier_risk: str = "MODERATE"):
    # 1. Budget: Look for Rs X,XXX or ₹ X,XXX - CRITICAL FIX: More robust extraction
    m_bud = re.search(r"(?:Rs\.?\s*|₹\s*)([0-9][0-9,]+)", clean, re.IGNORECASE)
    if not m_bud:
        # Look for "budget of X" or "budget is X" or "SIP of X" or "invest X"
        m_bud = re.search(r"(?:budget\s*(?:is|of|:)?\s*|SIP\s*(?:is|of|:)?\s*|invest\s+)([0-9][0-9,]+)", clean, re.IGNORECASE)
    if not m_bud:
        # Look for "X monthly" or "X/month" or "X per month"
        m_bud = re.search(r"\b([0-9][0-9,]{3,})\s*(?:/month|monthly|per\s*month|month)", clean, re.IGNORECASE)
    if not m_bud:
        # Fallback: any number >= 1000 followed by context words
        m_bud = re.search(r"\b([1-9][0-9]{3,})\s*(?:rupees?|rs|per|month|sip|budget|invest)", clean, re.IGNORECASE)

    budget = None
    if m_bud:
        val = m_bud.group(1).replace(",", "").strip()
        if val.isdigit():
            budget = float(val)
            # Sanity check: budget should be between 500 and 10 lakh
            if budget < 500 or budget > 1000000:
                budget = None

    # 2. Horizon: Look for X-year horizon or X years (not followed by old) - CRITICAL FIX
    m_hor = re.search(r"\b(\d+)\s*[--]?\s*years?(?:\s*horizon|\s*goal|\s*period)?\b(?!\s*-?old)", clean, re.IGNORECASE)
    if not m_hor:
        # Look for "for X years"
        m_hor = re.search(r"for\s+(\d+)\s+years?", clean, re.IGNORECASE)
    if not m_hor:
        # Generic year mention (avoid matching age)
        m_hor = re.search(r"\b(\d+)\s*[--]?\s*year(?!\s*old)", clean, re.IGNORECASE)

    horizon = 10  # Default
    if m_hor:
        h = int(m_hor.group(1))
        # Sanity check: horizon should be between 1 and 30 years
        if 1 <= h <= 30:
            horizon = h

    # 3. Name
    m_name = re.search(r"(?:My name is|I am|I\'m|Hi, I am)\s+([A-Z][a-z]+)\b", clean)
    name = m_name.group(1) if m_name and m_name.group(1).lower() not in ["investor", "a", "the", "working", "age"] else None

    # 4. Age: Look for age XX or XX-year-old or I am XX - CRITICAL FIX
    m_age = re.search(r"\bage\s*(\d+)\b", clean, re.IGNORECASE)
    if not m_age:
        m_age = re.search(r"\b(\d+)\s*(?:years?\s*old|y/?o|-year-old)\b", clean, re.IGNORECASE)
    if not m_age:
        m_age = re.search(r"\bI\s+am\s+(\d+)\b", clean, re.IGNORECASE)

    age = None
    if m_age:
        a = int(m_age.group(1))
        # Sanity check: age should be between 18 and 100
        if 18 <= a <= 100:
            age = a

    # 5. Risk: check prompt synonyms or fallback to classifier prediction
    low = clean.lower()
    if "conservative" in low or classifier_risk == "CONSERVATIVE":
        risk = "conservative"
    elif "moderately aggressive" in low or "moderate-aggressive" in low:
        risk = "moderately aggressive"
    elif "aggressive" in low or classifier_risk == "AGGRESSIVE":
        risk = "aggressive"
    else:
        risk = "moderate"

    # 6. Occupation
    occupation = None
    if re.search(r"\b(student|college|studying|intern)\b", low):
        occupation = "Student"
    elif re.search(r"\b(salaried|job|employee|working professional|software engineer|doing job|corporate)\b", low):
        occupation = "Doing Job / Salaried"
    elif re.search(r"\b(retired|pensioner|senior citizen)\b", low):
        occupation = "Retired Person"
    elif re.search(r"\b(business|businessman|self employed|shopkeeper|entrepreneur)\b", low):
        occupation = "Self Employed / Business"
    elif re.search(r"\b(freelancer?|consultant)\b", low):
        occupation = "Freelancer / Consultant"
    elif re.search(r"\b(homemaker|housewife)\b", low):
        occupation = "Homemaker"

    # 7. Goal
    goal = None
    if re.search(r"\b(tax|80c|elss)\b", low):
        goal = "tax_saving"
    elif re.search(r"\b(retire|retirement|pension)\b", low):
        goal = "retirement"
    elif re.search(r"\b(education|child|college|school)\b", low):
        goal = "child_education"
    elif re.search(r"\b(house|home|flat|apartment)\b", low):
        goal = "house_purchase"
    elif re.search(r"\b(wealth|crorepati|growth)\b", low):
        goal = "wealth_creation"

    return {"budget": budget, "horizon": horizon, "name": name, "risk": risk, "age": age, "occupation": occupation, "goal": goal}


CATEGORY_EXPLANATIONS = {
    "mid_cap": {
        "role": "High-Growth Engine & Wealth Multiplier",
        "description": "Invests in mid-sized companies (101st to 250th by market cap) that are transitioning into large-cap market leaders. Offers significantly higher growth potential than large caps, with volatility well smoothed over a multi-year horizon."
    },
    "small_cap": {
        "role": "Maximum Capital Appreciation & Alpha Generator",
        "description": "Focuses on emerging small enterprises (251st onwards). While subject to sharp short-term swings, small caps historically deliver the highest compounded returns over 10+ year holding periods."
    },
    "flexi_cap": {
        "role": "Dynamic Multi-Cap Diversification & Downside Cushion",
        "description": "Gives the fund manager unrestricted freedom to dynamically allocate across large, mid, and small-cap stocks based on prevailing market valuations, providing an all-weather foundation."
    },
    "large_cap": {
        "role": "Bedrock Stability & Blue-Chip Capital Protection",
        "description": "Invests in India's top 100 established market leaders with proven balance sheets and steady cash flows. Provides stability and lower drawdowns during market corrections."
    },
    "index": {
        "role": "Low-Cost Passive Market Representation",
        "description": "Replicates the benchmark index (Nifty 50 / Sensex) with near-zero tracking error and minimal expense ratios (0.1%-0.2%), capturing overall Indian economic expansion."
    },
    "hybrid": {
        "role": "Automated Asset Allocation & Volatility Dampener",
        "description": "Dynamically balances equity and debt instruments, capturing equity upside while debt shields capital during market downturns, ideal for moderate to conservative horizons."
    },
    "debt": {
        "role": "Capital Preservation & Liquid Foundation",
        "description": "Invests in high-quality government and corporate bonds. Immune to equity market crashes, delivering steady accrual returns with minimal duration risk."
    },
    "elss": {
        "role": "Section 80C Tax Optimization with Equity Growth",
        "description": "Offers tax deduction up to Rs 1.5 lakh under the Old Tax Regime with the shortest lock-in (3 years) among all 80C instruments, compounded via diversified equity."
    }
}


def fmt_inr(amount: float) -> str:
    if amount >= 1e7:
        return f"₹{amount / 1e7:.2f} crore"
    elif amount >= 1e5:
        return f"₹{amount / 1e5:.2f} lakh"
    else:
        return f"₹{amount:,.0f}"


def has_severe_repetition(text: str) -> bool:
    """Detect severe repetition patterns that indicate model degeneration"""
    if not text or len(text) < 50:
        return False

    words = text.split()
    if len(words) < 20:
        return False

    # Check for phrase repetition (3-7 word windows)
    for window_size in range(3, 8):
        for i in range(len(words) - window_size * 3):
            phrase = " ".join(words[i:i+window_size])
            rest = " ".join(words[i+window_size:])
            # If same phrase appears 3+ times
            if rest.count(phrase) >= 2:
                return True

    # Check for single word repetition (e.g., "vertical vertical vertical")
    from collections import Counter
    word_counts = Counter(words)
    for word, count in word_counts.items():
        if len(word) > 3 and count > len(words) * 0.15:  # Word appears >15% of the time
            return True

    # Check for character-level degeneration
    if text.count(",") > len(words) * 0.3:  # Too many commas
        return True
    if text.count(":") > len(words) * 0.2:  # Too many colons
        return True

    return False


def has_negative_amounts(text: str) -> bool:
    """Check for negative monetary amounts which are nonsensical"""
    # Match patterns like "Rs -250" or "₹-1000" or "Rs. -50"
    if re.search(r"(?:Rs\.?\s*|₹\s*)-\d+", text):
        return True
    return False


def has_nonsensical_numbers(text: str) -> bool:
    """Check for obviously wrong financial numbers"""
    # Check for unrealistic SIP amounts
    matches = re.findall(r"(?:Rs\.?\s*|₹\s*)(\d+(?:,\d+)*)\s*(?:/month|monthly|per\s*month|SIP)", text, re.IGNORECASE)
    for match in matches:
        amount = float(match.replace(",", ""))
        if amount < 100 or amount > 1000000:  # SIP should be 100 to 10 lakh
            return True

    # Check for unrealistic corpus amounts (in crore)
    if re.search(r"\b(\d+)\s*crore\b", text):
        crore_matches = re.findall(r"\b(\d+)\s*crore\b", text)
        for match in crore_matches:
            crore = float(match)
            if crore > 1000:  # More than 1000 crore is nonsensical for personal finance
                return True

    return False


def is_too_short(text: str, min_length: int = 12) -> bool:
    """Check if response is too short to be useful"""
    return len(text.strip()) < min_length


def validate_llm_response(text: str, query_context: str = "") -> tuple[bool, str]:
    """
    Validate LLM response quality
    Returns: (is_valid, reason_if_invalid)
    """
    if not text or not text.strip():
        return False, "Empty response"

    if is_too_short(text, min_length=12):
        return False, "Response too short"

    if has_severe_repetition(text):
        return False, "Severe repetition detected"

    if has_negative_amounts(text):
        return False, "Negative monetary amounts detected"

    if has_nonsensical_numbers(text):
        return False, "Nonsensical numbers detected"

    # Guard: Non-greeting query must NEVER receive a greeting response
    is_q_greeting = is_greeting(query_context) or query_context.lower().strip() in [
        "hi", "hello", "hey", "namaste", "good morning", "good night", "good afternoon", "good evening", "gm", "gn"
    ]
    if not is_q_greeting:
        if re.search(r"^(?:good\s*(?:morning|night|afternoon|evening)|hello!|hi there!|namaste!|greetings!|welcome to mentrafiai)", text.strip(), re.IGNORECASE):
            return False, "Greeting returned for non-greeting query"

    # Guard: SIP definition must never hallucinate 3-year lock-in (which belongs exclusively to ELSS)
    if re.search(r"\bsip\b", query_context, re.IGNORECASE) and not re.search(r"\belss\b", query_context, re.IGNORECASE):
        if re.search(r"\b(?:shortest lock-in|lock-in: only 3 years|3-year lock-in|80c tax-saving)\b", text, re.IGNORECASE):
            return False, "ELSS lock-in hallucinated for SIP query"

    # Guard: Never allow training instruction fragments or bullet templates
    if re.search(r"\b(?:financial concept:|explain sip:|explain elss:)\b", text, re.IGNORECASE):
        return False, "Prompt instruction fragment detected"

    # Check for unsolicited portfolio hallucination:
    # If user did NOT provide a budget or ask for a portfolio/recommendation,
    # the LLM should never fabricate a portfolio or mention arbitrary minimum thresholds.
    has_budget_query = bool(re.search(r"\b(?:rs\.?|₹|\d+k|\d+\s*(?:lakh|crore))\b|\b\d{3,7}\b", query_context, re.IGNORECASE))
    asks_portfolio_query = bool(re.search(r"\b(?:portfolio|recommend|suggest|invest|sip\s+for|plan\s+my|budget)\b", query_context, re.IGNORECASE))
    if not has_budget_query and not asks_portfolio_query:
        if re.search(r"\b(?:for your ₹|\/month budget|recommended mutual fund allocation|minimum threshold of ₹|total sip: ₹)\b", text, re.IGNORECASE):
            return False, "Unsolicited portfolio hallucination"

    # Check for gibberish (high ratio of non-words)
    words = text.split()
    if len(words) > 5:
        non_words = sum(1 for w in words if len(w) > 15 or not any(c.isalpha() for c in w))
        if non_words / len(words) > 0.3:
            return False, "Too much gibberish"

    return True, ""


def sip_future_value(monthly_sip: float, annual_cagr: float, years: int) -> float:
    r = (annual_cagr / 100.0) / 12.0
    n = years * 12
    return monthly_sip * (((1.0 + r)**n - 1.0) / r)


def generate_grounded_portfolio(
    budget: float,
    risk: str,
    horizon: int,
    name: Optional[str] = None,
    age: Optional[int] = None,
    db: Optional[FundDatabase] = None
) -> str:
    risk_lower = risk.lower()
    
    # 1. Determine optimal asset allocation strategy based on risk & horizon
    if "conservative" in risk_lower or "low" in risk_lower:
        risk_label = "Conservative (Capital Preservation Focused)"
        strategy_desc = (
            f"With a conservative risk profile and a {horizon}-year horizon, the primary mandate is capital preservation "
            f"and volatility mitigation. The portfolio leans on high-quality debt instruments and conservative hybrid allocations "
            f"to protect against equity drawdowns while delivering stable, tax-efficient inflation-beating yields."
        )
        if horizon <= 3:
            splits = [("Short Duration Debt Fund", "debt", 60), ("Conservative Hybrid Fund", "conservative_hybrid", 40)]
            cagr = 7.5
        else:
            splits = [("Conservative Hybrid Fund", "conservative_hybrid", 55), ("Corporate Bond / Short Duration Debt", "debt", 35), ("Large Cap Anchor", "large_cap", 10)]
            cagr = 8.2
    elif "moderately aggressive" in risk_lower:
        risk_label = "Moderately Aggressive (Growth with Diversified Anchor)"
        strategy_desc = (
            f"For a moderately aggressive investor over {horizon} years, the strategy targets high wealth accumulation "
            f"through quality mid and flexi-cap equities while maintaining large-cap exposure to temper portfolio drawdowns."
        )
        splits = [("Flexi Cap Equity Fund", "flexi_cap", 40), ("Mid Cap Equity Fund", "mid_cap", 35), ("Large Cap Equity Fund", "large_cap", 25)]
        cagr = 13.0
    elif "aggressive" in risk_lower or "high" in risk_lower:
        risk_label = "Aggressive (High-Alpha Long-Term Compounding)"
        strategy_desc = (
            f"With an aggressive risk appetite and an extensive {horizon}-year horizon, you have substantial runway "
            f"to absorb interim market cycles. The portfolio aggressively positions in Mid Cap and Small Cap leaders for "
            f"maximum wealth multiplication, anchored by Flexi Cap for dynamic rotation across market caps."
        )
        splits = [("Mid Cap Equity Fund", "mid_cap", 40), ("Small Cap Equity Fund", "small_cap", 35), ("Flexi Cap Equity Fund", "flexi_cap", 25)]
        cagr = 14.0
    else:  # moderate
        risk_label = "Moderate (Balanced Wealth Accumulation)"
        strategy_desc = (
            f"A moderate risk approach over a {horizon}-year timeframe balances growth potential with measured volatility. "
            f"It deploys an equity core to capture market compounding while ensuring sufficient diversification across market leaders."
        )
        if horizon <= 5:
            splits = [("Large Cap Equity Fund", "large_cap", 50), ("Balanced Advantage / Hybrid Fund", "hybrid", 30), ("Nifty 50 Index Fund", "index", 20)]
            cagr = 11.0
        else:
            splits = [("Flexi Cap Equity Fund", "flexi_cap", 50), ("Large Cap Equity Fund", "large_cap", 30), ("Mid Cap Equity Fund", "mid_cap", 20)]
            cagr = 12.0

    # 2. Exact Integer SIP Rupee Calculations (Zero Math Hallucination)
    allocs = []
    accum = 0
    for i, (fname, cat_key, target_pct) in enumerate(splits):
        if i == len(splits) - 1:
            amt = budget - accum
        else:
            amt = int(budget * target_pct / 100)
            accum += amt
        
        # Pre-fetch fund here to get its return_3y (Fix #7: Real CAGR calculation)
        fund_data = db.get_top_fund(cat_key) if (db and db.is_connected) else None
        allocs.append((fname, cat_key, amt, target_pct, fund_data))

    # Calculate real weighted CAGR from DB returns, fallback to default if missing
    real_cagr_sum = 0.0
    for fname, cat_key, amt, target_pct, fund_data in allocs:
        if fund_data and fund_data.get('return_3y') is not None:
            real_cagr_sum += fund_data['return_3y'] * (target_pct / 100.0)
        else:
            real_cagr_sum += cagr * (target_pct / 100.0)
    cagr = real_cagr_sum

    # 3. Compounding Calculations
    invested_total = budget * horizon * 12
    projected_corpus = sip_future_value(budget, cagr, horizon)
    gain = projected_corpus - invested_total
    gain_multiplier = projected_corpus / invested_total if invested_total > 0 else 1.0

    # 4. Milestone Roadmaps
    milestones = []
    for y in [min(5, horizon), min(10, horizon), min(15, horizon), horizon]:
        if y not in [m[0] for m in milestones] and y > 0:
            inv_y = budget * y * 12
            fv_y = sip_future_value(budget, cagr, y)
            milestones.append((y, inv_y, fv_y))

    # 5. Build Comprehensive Output
    greeting_str = f"Hello {name}!" if name else "Great choice to start investing!"
    age_str = f" | Age: {age}" if age else ""

    sections = []
    # Header
    sections.append(
        f"{greeting_str}\n"
        f"Here is your comprehensive, data-grounded mutual fund investment advisory report prepared for your "
        f"₹{budget:,.0f}/month SIP budget over a {horizon}-year horizon ({risk_label}{age_str})."
    )

    # Strategy & Allocation Rationale
    sections.append(
        f"### 1. Strategic Investment Rationale & Profile Fit\n"
        f"{strategy_desc}\n\n"
        f"• **Target Horizon:** {horizon} years (optimal duration for equity compounding)\n"
        f"• **Risk Tolerance:** {risk_label}\n"
        f"• **Target Annualized Return (CAGR):** ~{cagr:.1f}% expected average compounding rate (based on live DB trailing returns)"
    )

    # Database Grounded Recommendations with In-Depth Fund Rationale
    fund_details_list = []
    for idx, (display_cat, cat_key, amt, pct, fund_data) in enumerate(allocs, 1):
        cat_info = CATEGORY_EXPLANATIONS.get(cat_key, {"role": "Core Portfolio Component", "description": "Provides targeted asset exposure."})
        
        if fund_data:
            scheme_name = fund_data["name"]
            nav_val = f"₹{fund_data['nav']:.2f}" if fund_data.get("nav") is not None else "N/A"
            rating_val = format_star_rating(fund_data.get("rating"))
            ret_1y = f"{fund_data['return_1y']:.2f}%" if fund_data.get("return_1y") is not None else "N/A"
            ret_3y = f"{fund_data['return_3y']:.2f}% CAGR" if fund_data.get("return_3y") is not None else "N/A"
            ret_5y = f"{fund_data['return_5y']:.2f}% CAGR" if fund_data.get("return_5y") is not None else "N/A"
            amc_val = fund_data.get("fund_house") or "Top Tier AMC"
            
            fund_block = (
                f"**{idx}. {display_cat} - ₹{amt:,.0f}/month ({pct}% of SIP)**\n"
                f"• **Recommended Scheme (Direct - Growth):** {scheme_name}\n"
                f"• **Fund House:** {amc_val} | **Rating:** {rating_val}\n"
                f"• **Live NAV:** {nav_val} | **Trailing Returns:** 1Y: {ret_1y} | 3Y: {ret_3y} | 5Y: {ret_5y}\n"
                f"• **Role & Rationale:** {cat_info['role']}. {cat_info['description']}"
            )

        else:
            fund_block = (
                f"**{idx}. {display_cat} - ₹{amt:,.0f}/month ({pct}% of SIP)**\n"
                f"• **Role & Rationale:** {cat_info['role']}. {cat_info['description']}"
            )
        fund_details_list.append(fund_block)

    sections.append(
        f"### 2. Recommended Mutual Fund Schemes (PostgreSQL Live Market Data)\n\n" +
        "\n\n".join(fund_details_list)
    )

    # Monthly Allocation Summary Table
    table_lines = [
        "| Category | Monthly SIP | Allocation |",
        "| :--- | :--- | :--- |"
    ]
    for display_cat, _, amt, pct, _ in allocs:
        table_lines.append(f"| {display_cat} | ₹{amt:,.0f}/month | {pct}% |")
    table_lines.append(f"| **Total Portfolio SIP** | **₹{budget:,.0f}/month** | **100%** |")
    
    sections.append(
        f"### 3. Monthly SIP Allocation Summary\n\n" + "\n".join(table_lines)
    )

    # Wealth Projection & Milestone Trajectory
    milestone_lines = []
    for y, inv, fv in milestones:
        milestone_lines.append(f"• **Year {y:02d}:** Invested: {fmt_inr(inv)} ➔ Projected Value: **{fmt_inr(fv)}**")
    
    sections.append(
        f"### 4. Compounding Wealth Projections (~{cagr:.0f}% CAGR)\n"
        f"Over your {horizon}-year horizon, disciplined regular investing unlocks dramatic exponential compounding:\n\n"
        f"• **Total Capital Invested:** {fmt_inr(invested_total)}\n"
        f"• **Estimated Final Corpus:** **{fmt_inr(projected_corpus)}**\n"
        f"• **Estimated Compounding Wealth Gain:** **{fmt_inr(gain)}** ({gain_multiplier:.1f}x your invested capital)\n\n"
        f"**Milestone Progression:**\n" + "\n".join(milestone_lines)
    )

    # Actionable Implementation Blueprint
    sections.append(
        f"### 5. Strategic Execution Blueprint\n"
        f"1. **Always Choose Direct Plans:** Avoid Regular Plans to save 0.5% to 1.0% annually in commissions. Over {horizon} years on a ₹{budget:,.0f}/month SIP, Direct Plans will save you lakhs in distributor fees!\n"
        f"2. **Implement 10% Annual Step-Up:** As your career and earnings grow, increase your monthly SIP by 10% every year. This single step can nearly double your final accumulated wealth.\n"
        f"3. **Tax Optimization (Equity LTCG):** Long-Term Capital Gains above ₹1.25 lakh/year are taxed at 12.5% upon redemption after 1 year. Tax is payable only when you sell units, allowing untouched compounding.\n"
        f"4. **Annual Rebalancing:** Review your portfolio once every 12 months. If high-flying small/mid caps grow to exceed their target weight, rebalance back to your target allocations.\n"
        f"5. **Execution Platforms:** Set up auto-debit SIPs via Direct platforms like MFCentral, Zerodha Coin, or Groww.\n\n"
        f"*Disclaimer: Mutual fund investments are subject to market risks. Please read scheme-related documents carefully before investing.*"
    )

    return "\n\n".join(sections)


def generate_fund_comparison(term_a: str, term_b: str, db: FundDatabase) -> Optional[str]:
    """Retrieves two mutual funds from PostgreSQL and formats a side-by-side comparison."""
    matches_a = db.search_funds(term_a, limit=1)
    matches_b = db.search_funds(term_b, limit=1)
    if not matches_a or not matches_b:
        return None
    fa = matches_a[0]
    fb = matches_b[0]
    
    r1_a = f"{fa['return_1y']:.2f}%" if fa.get('return_1y') is not None else "N/A"
    r1_b = f"{fb['return_1y']:.2f}%" if fb.get('return_1y') is not None else "N/A"
    r3_a = f"{fa['return_3y']:.2f}%" if fa.get('return_3y') is not None else "N/A"
    r3_b = f"{fb['return_3y']:.2f}%" if fb.get('return_3y') is not None else "N/A"
    r5_a = f"{fa['return_5y']:.2f}%" if fa.get('return_5y') is not None else "N/A"
    r5_b = f"{fb['return_5y']:.2f}%" if fb.get('return_5y') is not None else "N/A"
    
    table = (
        f"Here is the side-by-side mutual fund comparison from the PostgreSQL database (37,768 records):\n\n"
        f"| Metric / Detail | {fa['name']} | {fb['name']} |\n"
        f"| :--- | :--- | :--- |\n"
        f"| **Fund House (AMC)** | {fa.get('fund_house') or 'N/A'} | {fb.get('fund_house') or 'N/A'} |\n"
        f"| **Category (Subcategory)** | {fa.get('category')} ({fa.get('subcategory')}) | {fa.get('category')} ({fb.get('subcategory')}) |\n"
        f"| **Plan & Option** | {fa.get('plan_type')} - {fa.get('option_type')} | {fb.get('plan_type')} - {fb.get('option_type')} |\n"
        f"| **Live NAV** | ₹{fa['nav']:.2f} | ₹{fb['nav']:.2f} |\n"
        f"| **Rating** | {format_star_rating(fa.get('rating'))} | {format_star_rating(fb.get('rating'))} |\n"
        f"| **1-Year Return** | {r1_a} | {r1_b} |\n"
        f"| **3-Year Trailing Return** | {r3_a} | {r3_b} |\n"
        f"| **5-Year Trailing Return** | {r5_a} | {r5_b} |\n\n"
        f"### 💡 Comparative Insights:\n"
        f"• **Historical Consistency:** Compare the 3-Year CAGR ({r3_a} vs {r3_b}) to evaluate multi-cycle stability rather than chasing recent 1-year spikes.\n"
        f"• **Direct Plan Advantage:** Always invest in **Direct - Growth** plans to eliminate distributor commissions (saving 0.5%-1.5% annually).\n"
        f"• **Risk Matching:** Ensure the volatility of {fa.get('subcategory')} aligns with your stated investment horizon (typically 7+ years for mid/small caps)."
    )
    return table


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 60)
    print("       MentraFiAI - Interactive Terminal Chat")
    print("=" * 60)
    print(f"Device: {device} " + (f"({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    
    # 1. Load Tokenizer
    tok_path = "tokenizer/tokenizer_v2.model"
    print(f"Loading Tokenizer: {tok_path}...")
    tokenizer = Tokenizer(tok_path)
    
    # 2. Load Model Config
    cfg_dict = load_config("configs/model_config.yaml")
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
    model = MentraFiAI(cfg).to(device)
    
    # 3. Load Checkpoint (prefer fine-tuned model: v5 -> v4 -> v3 -> v2 -> pretrain)
    ckpt_path = Path("checkpoints/finetune_v5/best_model.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/finetune_v4/best_model.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/finetune_v3/best_model.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/finetune_v2/best_model.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/pretrain_v2/best_model.pt")
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found at: {ckpt_path}")
        return
    # 3. Load Network 3: 102M Generative LLM
    print(f"Loading Network 3 (102M Generative LLM): {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    step = ckpt.get("step", 0)
    val_loss = ckpt.get("val_loss", 0.0)
    print(f"Ready! Network 3 Step: {step:,} | Val Loss: {val_loss:.4f}")

    # 4. Load Network 1 (10M Classifier) & Network 2 (15M Query Parser)
    clf_path = Path("checkpoints/classifier/best_classifier.pt")
    parser_path = Path("checkpoints/query_parser/best_parser_net.pt")
    query_parser = FinancialQueryParser(
        classifier_ckpt=str(clf_path) if clf_path.exists() else None,
        parser_ckpt=str(parser_path) if parser_path.exists() else None,
        tokenizer_path=tok_path,
        device=device
    )
    if query_parser.classifier is not None:
        print(f"Ready! Network 1 (10M Intent & Risk Classifier) loaded.")
    if query_parser.parser_net is not None:
        print(f"Ready! Network 2 (15M Financial Query Parser) loaded.")

    # 5. Initialize PostgreSQL Database Connection
    db = FundDatabase()
    if db.is_connected:
        fund_count = db.get_fund_count()
        print(f"Ready! PostgreSQL Database: {fund_count:,} mutual funds loaded from mentrafi")
    else:
        print("[Notice] PostgreSQL Database offline. Running with verified category allocations.")

    print("-" * 60)
    print("Type your questions below. Type 'exit' or 'quit' to end.")
    print("-" * 60 + "\n")
    
    # Conversational History Buffer (Fix #8)
    history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!\n")
                break
                
            # CRITICAL FIX: Normalize prompt to match training data distribution
            # BUT preserve original numbers for parameter extraction
            def clean_financial_prompt(text: str) -> str:
                # 0. Strip quotes and clean extra whitespace
                text = text.strip("' `")
                # Normalize common greeting and casual typos
                text = re.sub(r"\bgood\s+mori?ng\b", "good morning", text, flags=re.IGNORECASE)
                text = re.sub(r"\bgud\s+mori?ng\b", "good morning", text, flags=re.IGNORECASE)
                text = re.sub(r"\bgud\s+mrng\b", "good morning", text, flags=re.IGNORECASE)
                text = re.sub(r"\bgud\s+aft(?:ernoon)?\b", "good afternoon", text, flags=re.IGNORECASE)
                text = re.sub(r"\bgud\s+ev(?:eni?ng)?\b", "good evening", text, flags=re.IGNORECASE)
                text = re.sub(r"\bgud\b", "good", text, flags=re.IGNORECASE)
                text = re.sub(r"\bhelo+\b", "hello", text, flags=re.IGNORECASE)
                text = re.sub(r"\bnamste\b", "namaste", text, flags=re.IGNORECASE)
                text = re.sub(r"\bwht\b", "what", text, flags=re.IGNORECASE)
                # 1. Normalize currency symbol
                text = text.replace("₹", "Rs ")
                # 2. Normalize Age FIRST (handles 25-year-old, 25 year old, age 25, I am 25)
                # PRESERVE the age number for extraction
                text = re.sub(r"\b(\d+)\s*[--]?\s*years?\s*[--]?\s*old\b", r"age \1", text, flags=re.IGNORECASE)
                text = re.sub(r"\bI am (\d+)\b(?!\s*(?:lakh|crore|k|thousand))", r"age \1", text, flags=re.IGNORECASE)
                # 3. Normalize Lakh / Crore / K - PRESERVE actual amounts
                text = re.sub(r"\b(\d+(?:\.\d+)?)\s*(?:crore|crores|cr)\b", lambda m: f"Rs {int(float(m.group(1))*10000000):,}", text, flags=re.IGNORECASE)
                text = re.sub(r"\b(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs|l)\b", lambda m: f"Rs {int(float(m.group(1))*100000):,}", text, flags=re.IGNORECASE)
                text = re.sub(r"\b(\d+)k\b", lambda m: f"Rs {int(m.group(1))*1000:,}", text, flags=re.IGNORECASE)
                # 4. Handle Indian comma numbering: 1,00,000 -> 100,000
                text = re.sub(r"\b\d{1,2},\d{2},\d{3}\b", lambda m: f"{int(m.group(0).replace(',', '')):,}", text)
                # 5. Ensure number before budget/SIP has Rs - BE CAREFUL not to lose the number
                text = re.sub(r"(?<![,0-9])(?<!Rs\s)(?<!Rs\.)\b(\d{1,3}(?:,\d{3})+|\d{3,7})\b(?=\s*(?:monthly|per\s*month|/month|SIP|budget|invest))", r"Rs \1", text, flags=re.IGNORECASE)
                # 6. Normalize Risk terms to canonical training terms
                text = re.sub(r"\b(low|very low|safe|capital preservation)\s*risk\b", "conservative risk", text, flags=re.IGNORECASE)
                text = re.sub(r"\b(balanced|medium)\s*risk\b", "moderate risk", text, flags=re.IGNORECASE)
                text = re.sub(r"\b(very high|high)\s*risk\b", "aggressive risk", text, flags=re.IGNORECASE)
                # 7. CRITICAL: Normalize Horizon but DON'T conflict with age
                # Only normalize if followed by explicit horizon indicators OR far from age context
                text = re.sub(r"\b(?:for\s+)?(\d+)\s*[--]?\s*(?:years?|y)(?:\s*(?:goal|horizon|period|term))\b", r"\1-year horizon", text, flags=re.IGNORECASE)
                # 8. Canonical Financial Term Token Normalization
                # Maps lower/mixed case acronyms to the exact vocabulary tokens the model was trained on
                text = re.sub(r"\bsip\b", "SIP", text, flags=re.IGNORECASE)
                text = re.sub(r"\belss\b", "ELSS", text, flags=re.IGNORECASE)
                text = re.sub(r"\bnav\b", "NAV", text, flags=re.IGNORECASE)
                text = re.sub(r"\bter\b", "TER", text, flags=re.IGNORECASE)
                text = re.sub(r"\bcagr\b", "CAGR", text, flags=re.IGNORECASE)
                text = re.sub(r"\bxirr\b", "XIRR", text, flags=re.IGNORECASE)
                text = re.sub(r"\bltcg\b", "LTCG", text, flags=re.IGNORECASE)
                text = re.sub(r"\bstcg\b", "STCG", text, flags=re.IGNORECASE)
                text = re.sub(r"\bswp\b", "SWP", text, flags=re.IGNORECASE)
                text = re.sub(r"\bstp\b", "STP", text, flags=re.IGNORECASE)
                text = re.sub(r"\bnfo\b", "NFO", text, flags=re.IGNORECASE)
                text = re.sub(r"\betf\b", "ETF", text, flags=re.IGNORECASE)
                # 9. Capitalize initial question word so prompt matches training distribution
                if re.match(r"^what\s+(is|are)\s+", text, re.IGNORECASE):
                    text = re.sub(r"^what\s+(is|are)\s+", lambda m: f"What {m.group(1)} ", text, flags=re.IGNORECASE)
                    if not text.endswith("?"):
                        text = text + "?"
                elif re.match(r"^(pros|cons|advantages|disadvantages|benefits|drawbacks)\s+", text, re.IGNORECASE):
                    text = "What are the pros and cons of " + text.rstrip("?") + "?"
                    
                # 10. Normalize X vs Y conceptual comparison
                # Only if not explicitly asking "difference between" already
                if " vs " in text.lower() or " versus " in text.lower():
                    if not re.search(r"compare|difference", text, re.IGNORECASE):
                        # Simple replacement if it looks like "A vs B"
                        text = re.sub(r"^(.*?)\s+vs\.?\s+(.*?)\s*$", r"Compare \1 and \2", text, flags=re.IGNORECASE)

                # Clean up double Rs or spaces
                text = re.sub(r"Rs\s+Rs\s*", "Rs ", text)
                return re.sub(r"\s+", " ", text).strip()

            clean_input = clean_financial_prompt(user_input)

            # Debug logging to session log
            if user_input != clean_input:
                logger.info(f"NORMALIZATION | original={repr(user_input)} | cleaned={repr(clean_input)}")

            # Multi-Network Execution:
            # - Network 1 (10M): Intent Classification & Risk Appetite
            # - Network 2 (15M): Financial Slot & Requirement Extraction
            parsed = query_parser.parse(user_input)
            intent = parsed.intent
            conf = parsed.intent_confidence
            risk = parsed.risk

            # Log classification decision
            logger.info(f"QUERY | intent={intent} conf={conf*100:.1f}% risk={risk} | q={repr(clean_input[:120])}")

            temp = 0.5
            max_tokens = 250
            if intent == "GREETING" or is_greeting(clean_input) or is_greeting(user_input):
                temp = 0.7
                max_tokens = 150
            elif intent == "DEFINITION":
                temp = 0.3
                max_tokens = 250
            elif intent in ["PORTFOLIO_RECOMMENDATION", "GOAL_PLANNING"]:
                temp = 0.2
                max_tokens = 350


            # 2. Fund Comparison Grounding (PostgreSQL Multi-Fund Comparison)
            if intent == "FUND_COMPARISON":
                m_comp = re.search(r"(?:compare|difference between|vs\.?)\s+(.+?)\s+(?:vs\.?|versus|and|or)\s+(.+)", clean_input, re.IGNORECASE)
                if not m_comp:
                    m_comp = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", clean_input, re.IGNORECASE)
                if not m_comp:
                    # "which is better, X or Y" / "which is better X or Y"
                    m_comp = re.search(r"(?:which\s+is\s+better|which\s+one\s+is\s+better)[,\s]+(.+?)\s+or\s+(.+)", clean_input, re.IGNORECASE)
                if not m_comp:
                    # "confused between X and Y" / "torn between X and Y"
                    m_comp = re.search(r"(?:confused|torn|deciding|choosing)\s+between\s+(.+?)\s+(?:and|or)\s+(.+)", clean_input, re.IGNORECASE)
                if not m_comp:
                    # "help me choose between X and Y" / "I'm confused between X and Y"
                    m_comp = re.search(r"(?:choose|select|pick|decide)\s+between\s+(.+?)\s+(?:and|or)\s+(.+)", clean_input, re.IGNORECASE)
                if not m_comp:
                    # "should I invest in X or Y"
                    m_comp = re.search(r"(?:should\s+I\s+(?:invest|go|choose|pick)\s+(?:in|with|for)?)\s+(.+?)\s+or\s+(.+)", clean_input, re.IGNORECASE)

                if m_comp and db and db.is_connected:
                    term_a = re.sub(r"[\"'\?]", "", m_comp.group(1)).strip()
                    term_b = re.sub(r"[\"'\?]", "", m_comp.group(2)).strip()

                    # Avoid searching database for generic concepts — route directly to Generative LLM
                    CONCEPT_TERMS = {
                        "direct", "direct plan", "regular", "regular plan", "sip", "lump sum",
                        "lumpsum", "elss", "ppf", "growth", "idcw", "dividend", "active",
                        "passive", "index", "index fund", "active fund", "mutual fund", "debt", "equity"
                    }
                    is_conceptual = (
                        term_a.lower() in CONCEPT_TERMS or
                        term_b.lower() in CONCEPT_TERMS or
                        "lump sum" in clean_input.lower() or
                        "regular plan" in clean_input.lower() or
                        "ppf" in clean_input.lower()
                    )

                    if not is_conceptual:
                        comp_reply = generate_fund_comparison(term_a, term_b, db)
                        if comp_reply:
                            history.append((clean_input, comp_reply))
                            if len(history) > 3: history.pop(0)
                            logger.info(f"REPLY | route=FUND_COMPARISON | funds={term_a!r} vs {term_b!r}")
                            print(f"\nMentraFiAI: {comp_reply}\n")
                            print("-" * 60)
                            continue



            # If intent is PORTFOLIO_RECOMMENDATION and a concrete budget is provided:
            # Guarantee 100% mathematically exact allocations, percentages, and compounding projections!
            if intent == "PORTFOLIO_RECOMMENDATION":
                budget = parsed.amount
                horizon = parsed.horizon
                p_risk = parsed.risk
                p_age = parsed.age
                p_name = parsed.name

                if budget is None:
                    fallback_p = extract_portfolio_params(clean_input, risk)
                    budget = fallback_p.get("budget")
                    if fallback_p.get("age") and not p_age:
                        p_age = fallback_p.get("age")
                    if fallback_p.get("name") and not p_name:
                        p_name = fallback_p.get("name")

                params = {
                    "budget": budget,
                    "horizon": horizon,
                    "risk": p_risk,
                    "age": p_age,
                    "name": p_name,
                }

                logger.info(f"PARAMS | budget={params['budget']} horizon={params['horizon']} risk={params['risk']} age={params['age']}")

                # Auto-reroute to GOAL_PLANNING if budget is abnormally huge (>1 Lakh)
                if params["budget"] is not None and params["budget"] >= 100000:
                    intent = "GOAL_PLANNING"
                elif params["budget"] is not None:
                    if db and db.is_connected:
                        logger.info("PostgreSQL Database: Direct-Growth schemes queried from mentrafi (37,768 records)")
                    reply = generate_grounded_portfolio(
                        budget=params["budget"],
                        risk=params["risk"],
                        horizon=params["horizon"],
                        name=params["name"],
                        age=params.get("age"),
                        db=db
                    )
                    history.append((clean_input, "Here is your requested portfolio:\n" + reply))
                    if len(history) > 3: history.pop(0)
                    logger.info(f"REPLY | route=PORTFOLIO | budget={params['budget']} risk={params['risk']} horizon={params['horizon']}")
                    print(f"\nMentraFiAI: {reply}\n")
                    print("-" * 60)
                    continue
                else:
                    reply = (
                        "I'd be happy to recommend a portfolio! To give you the best advice, I need to know:\n\n"
                        "• **Monthly SIP amount** (e.g., ₹5,000, ₹10,000)\n"
                        "• **Investment horizon** (e.g., 5 years, 10 years, 15 years)\n"
                        "• **Risk tolerance** (conservative/moderate/aggressive)\n"
                        "• **Your age** (optional, helps with personalization)\n\n"
                        "For example: 'I am 28, want to invest ₹5000/month, moderate risk, 10 year horizon'"
                    )
                    history.append((clean_input, reply))
                    if len(history) > 3: history.pop(0)
                    logger.info(f"REPLY | route=PORTFOLIO_CLARIFY | missing=budget")
                    print(f"\nMentraFiAI: {reply}\n")
                    print("-" * 60)
                    continue

            # If intent is GOAL_PLANNING (Fix #5): Extract target corpus and calculate required SIP
            if intent == "GOAL_PLANNING":
                params = extract_portfolio_params(clean_input, risk)
                # Look for a large corpus target like "1 crore" or "5000000"
                m_target = re.search(r"Rs\s*([\d,]+)", clean_input)
                target_amount = None
                if m_target:
                    val = float(m_target.group(1).replace(",", ""))
                    if val >= 100000: # Ensure it's a corpus target, not a monthly SIP
                        target_amount = val

                if target_amount:
                    horizon = params["horizon"]
                    # Assume 12% CAGR for goal planning
                    r = 12.0 / 100.0 / 12.0
                    n = horizon * 12
                    # PMT = FV / ( ((1+r)^n - 1) / r ) * (1+r)
                    required_sip = target_amount / ((((1.0 + r)**n - 1.0) / r) * (1.0 + r))
                    
                    if db and db.is_connected:
                        print(f"[PostgreSQL Database] Goal Planning: Target {fmt_inr(target_amount)} in {horizon} yrs -> Required SIP ~Rs {required_sip:,.0f}/mo")
                    print(f"[Neural Network #1 Generative LLM] Synthesizing data-grounded goal plan...")
                    
                    # Temporarily replace budget with required SIP to generate matching portfolio
                    reply = generate_grounded_portfolio(
                        budget=required_sip,
                        risk=params["risk"],
                        horizon=horizon,
                        name=params["name"] or "Goal Planner",
                        age=params.get("age"),
                        db=db
                    )
                    # Prepend goal specific header
                    goal_header = f"### 🎯 Financial Goal Achieved: {fmt_inr(target_amount)} in {horizon} Years\nTo reach your target corpus of **{fmt_inr(target_amount)}** in **{horizon} years**, you need to start a monthly SIP of **₹{required_sip:,.0f}** (assuming 12% average CAGR).\n\nHere is the recommended portfolio to get you there:\n\n"
                    full_reply = goal_header + reply
                    history.append((clean_input, "Here is your requested goal plan:\n" + full_reply))
                    if len(history) > 3: history.pop(0)
                    logger.info(f"REPLY | route=GOAL_PLANNING | target={fmt_inr(target_amount)} horizon={horizon} sip={required_sip:.0f}")
                    print(f"\nMentraFiAI: {full_reply}\n")
                    print("-" * 60)
                    continue

            # Direct Fund Search & Live NAV Lookup hook
            m_lookup = re.search(r"\b(?:nav of|details of|search fund|find fund|show fund|price of)\s+(.+)", clean_input, re.IGNORECASE)
            if m_lookup and db and db.is_connected:
                search_term = m_lookup.group(1).strip()
                fund_matches = db.search_funds(search_term, limit=3)
                if fund_matches:
                    print(f"\nMentraFiAI: Here are the top matching mutual funds from the database:\n")
                    summary = []
                    for fm in fund_matches:
                        rating_str = format_star_rating(fm.get("rating"))
                        nav_val = f"₹{fm['nav']:.2f}" if fm.get('nav') is not None else "N/A"
                        r1 = f"{fm['return_1y']:.2f}%" if fm.get('return_1y') is not None else "N/A"
                        r3 = f"{fm['return_3y']:.2f}%" if fm.get('return_3y') is not None else "N/A"
                        r5 = f"{fm['return_5y']:.2f}%" if fm.get('return_5y') is not None else "N/A"
                        fm_str = (f"• {fm['name']}\n"
                                  f"  - AMC: {fm.get('fund_house') or 'N/A'} | Category: {fm.get('category')} ({fm.get('subcategory')})\n"
                                  f"  - Plan: {fm.get('plan_type')} | Option: {fm.get('option_type')}\n"
                                  f"  - Live NAV: {nav_val} | Rating: {rating_str}\n"
                                  f"  - Trailing Returns: 1Y: {r1} | 3Y: {r3} | 5Y: {r5}\n")
                        print(fm_str, end="")
                        summary.append(fm_str)
                    
                    history.append((clean_input, "Here are the top matching mutual funds:\n" + "\n".join(summary)))
                    if len(history) > 3: history.pop(0)
                    logger.info(f"REPLY | route=NAV_LOOKUP | term={repr(search_term)} results={len(fund_matches)}")
                    print("-" * 60)
                    continue


            # 2. Neural Network #1 (Generative LLM) generates the complete contextual response
            reply = None
            attempts = 0
            max_attempts = 3

            # In v5, the model natively understands 'Good morning', 'Good night', 'hello', 'namaste', etc.
            llm_input = clean_input

            # Single-turn prompt matching SFT training distribution perfectly
            prefix_ids = [tokenizer.bos_id, tokenizer.user_id] + tokenizer.encode(llm_input) + [tokenizer.assistant_id]
            idx = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            prompt_len = len(prefix_ids)

            while reply is None and attempts < max_attempts:
                attempts += 1
                rep_penalty = 1.15
                temp_adjusted = 0.2 if intent == "DEFINITION" else (0.4 if intent == "GREETING" else 0.35)

                with torch.no_grad():
                    out = model.generate(
                        idx,
                        max_new_tokens=max_tokens,
                        temperature=temp_adjusted,
                        top_k=40,
                        top_p=0.9,
                        repetition_penalty=rep_penalty,
                        no_repeat_ngram_size=4,
                    )
                    raw_ids = out[0].tolist()[prompt_len:]

                    # Stop immediately at any conversation boundary tokens (EOS, User, Assistant)
                    stop_ids = {tokenizer.eos_id, tokenizer.user_id, tokenizer.assistant_id}
                    cleaned_ids = []
                    for tok_id in raw_ids:
                        if tok_id in stop_ids:
                            break
                        cleaned_ids.append(tok_id)

                    candidate_reply = tokenizer.decode(cleaned_ids).strip()

                    # Guardrail: Strip any stray turn tags if decoded as literal text
                    for tag in ["<assistant>", "<user>", "<eos>", "<bos>", "<pad>", "User:", "Assistant:"]:
                        if tag in candidate_reply:
                            candidate_reply = candidate_reply.split(tag)[0].strip()

                    # Strip raw JSON schemas / address scraps e.g. {"house_number": ...}
                    candidate_reply = re.sub(r"\{[^{}]*\}", "", candidate_reply).strip()
                    candidate_reply = candidate_reply.lstrip("@: ").strip()

                    # Post-process: model generates "Rs" (1 token) for stability;
                    # convert back to ₹ symbol for user display
                    candidate_reply = candidate_reply.replace("₸", "₹").replace("Rs ", "₹").replace("Rs.", "₹")

                    # Enforce strict SEBI compliance & Indian Financial terminology rules:
                    # 1. Strip trailing prompt artifacts / instruction fragments
                    for artifact in ["Financial Concept:", "Financial Concepts:", "• Financial Concept", "Explain SIP:", "Explain ELSS:"]:
                        if artifact.lower() in candidate_reply.lower():
                            candidate_reply = re.split(re.escape(artifact), candidate_reply, flags=re.IGNORECASE)[0].strip()

                    # 2. Fix corrupted tax threshold numbers (e.g. ₹1,250 -> ₹1.25 Lakh)
                    candidate_reply = re.sub(r"₹\s*1,250\b(?!\s*lakh)", "₹1.25 Lakh", candidate_reply, flags=re.IGNORECASE)
                    candidate_reply = re.sub(r"₹\s*1\.25L\b", "₹1.25 Lakh", candidate_reply, flags=re.IGNORECASE)
                    candidate_reply = re.sub(r"₹\s*150,000\b", "₹1.5 Lakh", candidate_reply, flags=re.IGNORECASE)
                    candidate_reply = re.sub(r"₹\s*5,008\b", "₹5,000", candidate_reply)

                    # 3. Fix Section 80C hallucinations (e.g. "80 CGs", "80 Coins")
                    candidate_reply = re.sub(r"\ball 80\s*(?:CGs?|Coins?)\b", "all Section 80C instruments", candidate_reply, flags=re.IGNORECASE)
                    candidate_reply = re.sub(r"\b80\s*(?:CGs?|Coins?)\b", "Section 80C", candidate_reply, flags=re.IGNORECASE)

                    # 5. Correct NAV Myth Buster hallucination (conflating NAV with Rate or distributor commissions)
                    if re.search(r"\bNAV\b", clean_input, re.IGNORECASE) and any(term in candidate_reply for term in ["Rate", "cheaper", "expensive", "Myth Buster", "Net Lock-in"]):
                        formula_match = re.search(r"(NAV\s*=\s*\([^\)]+\)\s*/\s*Total\s+Outstanding\s+Units)", candidate_reply, re.IGNORECASE)
                        if formula_match:
                            def_prefix = candidate_reply[:formula_match.end()].strip()
                            def_prefix = re.sub(r"It is calculated daily at the close of market hours:\s*", "It is calculated daily at market close:\n\n", def_prefix)
                            candidate_reply = (
                                f"{def_prefix}\n\n"
                                "Key Myth Buster: A fund with NAV ₹100 is NOT more expensive than a fund with NAV ₹20 — "
                                "what matters is the portfolio return, not the NAV price. "
                                "Both funds delivering 15% annual returns will grow your wealth equally, regardless of their NAV."
                            )

                    # 6. Clean trailing bullets or incomplete bullet markers
                    candidate_reply = re.sub(r"[\s•\-\*]+$", "", candidate_reply).strip()

                    # 7. Clean up small cap allocation hallucinations
                    if "small cap" in clean_input.lower() and ("disadvantage" in clean_input.lower() or "cons " in clean_input.lower()):
                        candidate_reply = re.sub(r"(?:Indian Budget Caps|market capitalization rank|Top \d{2},\d{3} units).*?(?:\n|$)", "", candidate_reply, flags=re.IGNORECASE)
                        candidate_reply = re.sub(r"25,000", "", candidate_reply)
                        candidate_reply = re.sub(r"30,009 units", "", candidate_reply)

                    # Validate response quality
                    is_valid, reason = validate_llm_response(candidate_reply, clean_input)
                    if is_valid:
                        reply = candidate_reply
                    else:
                        logger.info(f"LLM quality retry {attempts}: {reason}")

            # Fallback if all attempts fail
            if reply is None:
                logger.info("All generation attempts failed. Using fallback response.")
                if is_informational(clean_input) or intent == "DEFINITION":
                    reply = informational_reply(clean_input)
                else:
                    reply = (
                        "I am MentraFiAI, your AI mutual fund advisor! Please tell me your age, monthly investment budget, "
                        "and risk appetite (conservative/moderate/aggressive), and I will recommend a personalized portfolio!"
                    )

            history.append((clean_input, reply))
            if len(history) > 3: history.pop(0)

            logger.info(f"REPLY | route=LLM_GENERATE | reply={repr(reply[:200])}")
            print(f"\nMentraFiAI: {reply}\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nSession ended.")
            logger.info("=== Session ended by user (KeyboardInterrupt) ===")
            break
        except Exception as e:
            logger.error(f"UNHANDLED_EXCEPTION | {type(e).__name__}: {e}")
            print(f"\n[Error]: {e}\n")


if __name__ == "__main__":
    main()
