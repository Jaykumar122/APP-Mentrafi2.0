"""
MentraFiAI SFT Dataset Generator v2
Generates a large, high-quality SFT dataset for a mutual fund advisor AI.
Target: ~3000 examples across 8 categories.
"""

import json
import math
import os
import random

random.seed(42)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sft", "mentrafiai_sft_v2.jsonl")

# ---------------------------------------------------------------------------
# SIP math utility
# ---------------------------------------------------------------------------

def sip_future_value(monthly_sip: float, annual_rate: float, years: int) -> float:
    """FV = P × [((1+r)^n - 1) / r] × (1+r)"""
    r = annual_rate / 12
    n = years * 12
    if r == 0:
        return monthly_sip * n
    fv = monthly_sip * (((1 + r) ** n - 1) / r) * (1 + r)
    return fv


def fmt_inr(amount: float) -> str:
    """Format amount in Indian notation (lakhs / crores)."""
    if amount >= 1e7:
        return f"₹{amount / 1e7:.2f} crore"
    elif amount >= 1e5:
        return f"₹{amount / 1e5:.2f} lakh"
    else:
        return f"₹{amount:,.0f}"


def sip_math_line(monthly_sip: float, annual_rate: float, years: int) -> str:
    fv = sip_future_value(monthly_sip, annual_rate, years)
    invested = monthly_sip * years * 12
    gain = fv - invested
    rate_label = f"{int(annual_rate * 100)}%"
    return (
        f"₹{monthly_sip:,.0f}/month × {years} years @ {rate_label} CAGR "
        f"→ **{fmt_inr(fv)}** (invested {fmt_inr(invested)}, gains {fmt_inr(gain)})"
    )


# ---------------------------------------------------------------------------
# Shared pools
# ---------------------------------------------------------------------------

NAMES = [
    "Rahul", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Rohit", "Kavya",
    "Suresh", "Meera", "Arjun", "Divya", "Kiran", "Pooja", "Raj", "Nisha",
    "Deepak", "Sunita", "Arun", "Ritu", "Sanjay", "Lalitha", "Mahesh", "Geetha",
    "Prakash", "Swati", "Ajay", "Rekha", "Naveen", "Shweta", "Harish", "Usha",
    "Rajan", "Padma", "Vinod", "Sarla", "Mohan", "Parvati", "Bhaskar", "Anita",
    "Chetan", "Lata", "Tarun", "Mamta", "Girish", "Sushma", "Ashok", "Radha",
    "Nitin", "Vandana",
]

CITIES = [
    "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata",
    "Ahmedabad", "Jaipur", "Lucknow", "Kochi", "Chandigarh", "Coimbatore",
    "Nagpur", "Bhopal", "Indore", "Surat", "Vadodara", "Visakhapatnam", "Patna",
]

PROFESSIONS = [
    "software engineer", "doctor", "teacher", "government employee", "business owner",
    "chartered accountant", "marketing manager", "nurse", "freelancer", "banker",
    "architect", "lawyer", "HR professional", "data analyst", "sales executive",
    "civil engineer", "pharmacist", "professor", "entrepreneur", "retail manager",
]

RISK_LEVELS = ["conservative", "moderate", "moderately aggressive", "aggressive"]

FUND_CATEGORIES = {
    "large_cap": {
        "name": "Large Cap Equity funds",
        "desc": "invest in India's top 100 companies by market cap — blue-chip, relatively stable, lower volatility",
        "return": 11,
        "risk": "moderate",
        "horizon": 5,
    },
    "mid_cap": {
        "name": "Mid Cap Equity funds",
        "desc": "target companies ranked 101–250 by market cap — higher growth potential with higher short-term volatility",
        "return": 13,
        "risk": "moderately high",
        "horizon": 7,
    },
    "small_cap": {
        "name": "Small Cap Equity funds",
        "desc": "invest in companies beyond the top 250 — highest long-term growth potential, significant volatility, best for 7+ year horizons",
        "return": 15,
        "risk": "high",
        "horizon": 7,
    },
    "flexi_cap": {
        "name": "Flexi Cap / Multi Cap funds",
        "desc": "fund manager allocates dynamically across large, mid, and small caps — good diversification in a single fund",
        "return": 12,
        "risk": "moderate-high",
        "horizon": 5,
    },
    "elss": {
        "name": "ELSS (Equity Linked Savings Scheme) funds",
        "desc": "equity funds with a mandatory 3-year lock-in that qualify for Section 80C deduction up to ₹1.5 lakh per year",
        "return": 12,
        "risk": "moderate-high",
        "horizon": 3,
    },
    "index": {
        "name": "Index funds (Nifty 50 / Nifty Next 50)",
        "desc": "passively track a market index with very low expense ratios (0.1–0.2%) — no fund manager risk, true market returns",
        "return": 11,
        "risk": "moderate",
        "horizon": 5,
    },
    "debt": {
        "name": "Debt funds (liquid / short-duration / corporate bond)",
        "desc": "invest in government securities and corporate bonds — capital preservation with modest returns, suitable for short horizons",
        "return": 7,
        "risk": "low",
        "horizon": 1,
    },
    "hybrid": {
        "name": "Hybrid / Balanced Advantage funds",
        "desc": "automatically rebalance between equity and debt based on market valuations — smoother ride for moderate-risk investors",
        "return": 10,
        "risk": "moderate",
        "horizon": 3,
    },
    "international": {
        "name": "International / US equity funds",
        "desc": "provide geographic diversification by investing in global markets, especially US tech — also hedges against INR depreciation",
        "return": 12,
        "risk": "moderate-high",
        "horizon": 5,
    },
}

HEDGING_PHRASES = [
    "Please note that past performance does not guarantee future returns — these projections are indicative.",
    "Mutual fund investments are subject to market risk; please read all scheme-related documents carefully.",
    "These are estimates based on historical CAGR — actual returns will vary with market conditions.",
    "Past returns are not a guarantee of future performance. Please review the fund's SID before investing.",
    "Remember: markets can be volatile in the short term. Stay disciplined with your SIP.",
    "As with all equity investments, returns are not guaranteed and may fluctuate. Consult a SEBI-registered advisor for personalised advice.",
    "Equity funds carry market risk; however, systematic investing over long horizons has historically rewarded patient investors.",
]

WARM_OPENERS = [
    "Great question!", "That's a smart move!", "Happy to help you plan this out!",
    "You're thinking about this the right way!", "Absolutely, let's work through this together!",
    "Love the initiative to start early!", "This is exactly the kind of planning that builds wealth!",
    "Smart thinking!", "You're on the right track!", "Let's dive right in!",
    "Excellent — this is a great step towards financial security!",
    "Wonderful — let me put together a solid plan for you!",
]

CLOSERS = [
    "Feel free to ask if you'd like to tweak the allocation or explore any fund category further!",
    "Let me know if you want me to model a different SIP amount or horizon!",
    "Happy to answer any follow-up questions about any of these funds!",
    "If your risk appetite changes, I can easily re-adjust the recommendations — just say the word!",
    "Would you like me to explain any of these fund categories in more detail?",
    "Start small and stay consistent — wealth is built one SIP at a time!",
    "Remember: the best time to start a SIP is today. Good luck on your investing journey! 🎯",
    "Reach out any time if you have more questions — I'm here to help!",
]


def rnd_name():
    return random.choice(NAMES)


def rnd_city():
    return random.choice(CITIES)


def rnd_profession():
    return random.choice(PROFESSIONS)


def rnd_hedge():
    return random.choice(HEDGING_PHRASES)


def rnd_opener():
    return random.choice(WARM_OPENERS)


def rnd_closer():
    return random.choice(CLOSERS)


def pick_funds(n=2, exclude=None):
    keys = list(FUND_CATEGORIES.keys())
    if exclude:
        keys = [k for k in keys if k not in exclude]
    return random.sample(keys, min(n, len(keys)))


# ---------------------------------------------------------------------------
# Category 1: Full Profile Recommendations (1000 examples)
# ---------------------------------------------------------------------------

def gen_full_profile(idx: int) -> dict:
    name = rnd_name()
    age = random.randint(22, 58)
    city = rnd_city()
    profession = rnd_profession()
    monthly_sip = random.choice([1000, 2000, 3000, 5000, 7000, 8000, 10000, 12000, 15000, 20000, 25000, 30000])
    risk = random.choice(RISK_LEVELS)
    horizon = random.randint(3, 25)

    # pick fund combo based on risk + horizon
    if risk == "conservative":
        if horizon <= 3:
            primary, secondary = "debt", "hybrid"
            tertiary = None
        else:
            primary, secondary = "hybrid", "large_cap"
            tertiary = "index"
    elif risk == "moderate":
        if horizon <= 5:
            primary, secondary = "large_cap", "hybrid"
            tertiary = "index"
        else:
            primary, secondary = "flexi_cap", "large_cap"
            tertiary = "index"
    elif risk == "moderately aggressive":
        if horizon >= 7:
            primary, secondary = "mid_cap", "flexi_cap"
            tertiary = "large_cap"
        else:
            primary, secondary = "flexi_cap", "large_cap"
            tertiary = "hybrid"
    else:  # aggressive
        if horizon >= 7:
            primary, secondary = "small_cap", "mid_cap"
            tertiary = "flexi_cap" if horizon > 10 else "large_cap"
        else:
            primary, secondary = "mid_cap", "flexi_cap"
            tertiary = "large_cap"

    funds_used = [primary, secondary] + ([tertiary] if tertiary else [])

    # split SIP
    if tertiary:
        splits = [0.40, 0.35, 0.25]
    else:
        splits = [0.60, 0.40]

    alloc = [(funds_used[i], int(monthly_sip * splits[i])) for i in range(len(funds_used))]

    # projections
    equity_rate = 0.12
    debt_rate = 0.07
    is_debt_heavy = risk == "conservative" and horizon <= 3
    proj_rate = debt_rate if is_debt_heavy else equity_rate
    fv = sip_future_value(monthly_sip, proj_rate, horizon)
    invested = monthly_sip * horizon * 12

    user_variants = [
        (f"Hi, I'm {name}, a {age}-year-old {profession} from {city}. I want to invest ₹{monthly_sip:,}/month in mutual funds. My risk tolerance is {risk} and my horizon is {horizon} years. What should I invest in?", True),
        (f"I'm {age} years old, work as a {profession} in {city}, and want to start a SIP of ₹{monthly_sip:,} per month. I'm a {risk} investor with a {horizon}-year goal. Can you recommend some funds?", False),
        (f"Name: {name}. Age: {age}. Profession: {profession}. City: {city}. Monthly SIP budget: ₹{monthly_sip:,}. Risk profile: {risk}. Investment horizon: {horizon} years. Please suggest a mutual fund portfolio.", True),
        (f"Hey MentraFiAI! I'm a {age}-year-old {profession} from {city}. I can invest ₹{monthly_sip:,} every month. I'm {risk} when it comes to risk and I'm looking at a {horizon}-year horizon. What do you suggest?", False),
        (f"Looking for mutual fund advice. Age {age}, {profession} in {city}, ₹{monthly_sip:,}/month budget, {risk} risk tolerance, {horizon}-year horizon.", False),
    ]
    user, has_name = random.choice(user_variants)

    # build assistant response
    fund_blocks = []
    for fkey, amt in alloc:
        f = FUND_CATEGORIES[fkey]
        pct = int(amt / monthly_sip * 100)
        fund_blocks.append(
            f"**{f['name']}** — ₹{amt:,}/month ({pct}%)\n"
            f"  - {f['desc'].capitalize()}.\n"
            f"  - Risk level: {f['risk']}. Recommended minimum horizon: {f['horizon']}+ years.\n"
            f"  - {sip_math_line(amt, proj_rate if fkey in ('debt', 'hybrid') else equity_rate, horizon)}"
        )

    fund_text = "\n\n".join(fund_blocks)

    total_math = sip_math_line(monthly_sip, proj_rate, horizon)
    gain = fv - invested
    gain_pct = (gain / invested) * 100

    age_note = ""
    if age < 30:
        age_note = f"Starting at {age} gives you a massive compounding advantage — time is your biggest asset right now!"
    elif age > 50:
        age_note = f"At {age}, capital protection becomes increasingly important as you near retirement. The {risk}-leaning allocation balances growth with safety."
    else:
        age_note = f"At {age} with a {horizon}-year horizon, you're well-positioned for the equity cycle to work in your favour."

    greeting = f"{rnd_opener()} {name}," if has_name else f"{rnd_opener()}"

    assistant = (
        f"{greeting} I've put together a personalised portfolio for you based on your {risk} risk profile and {horizon}-year horizon.\n\n"
        f"## Your Recommended Portfolio (₹{monthly_sip:,}/month)\n\n"
        f"{fund_text}\n\n"
        f"## Overall Projection\n\n"
        f"{total_math}\n"
        f"That's a **{gain_pct:.0f}% return on your invested capital** over {horizon} years — thanks to the power of compounding.\n\n"
        f"## Why This Allocation?\n\n"
        f"{age_note} The {FUND_CATEGORIES[primary]['name']} forms the core of your portfolio because it aligns well with your {risk} risk appetite. "
        f"The {FUND_CATEGORIES[secondary]['name']} adds {'stability' if risk in ('conservative','moderate') else 'growth potential'} to the mix."
        + (f" The smaller allocation to {FUND_CATEGORIES[tertiary]['name']} {'provides a safety cushion' if risk in ('conservative','moderate') else 'boosts the overall return potential'}." if tertiary else "")
        + f"\n\n**Pro tip:** Set up an auto-debit SIP on the 1st or 5th of every month so you never miss an instalment. Review your portfolio annually and rebalance if needed.\n\n"
        f"⚠️ *{rnd_hedge()}*\n\n"
        f"{rnd_closer()}"
    )

    return {"user": user, "assistant": assistant}


# ---------------------------------------------------------------------------
# Category 2: Income-aware recommendations (400 examples)
# ---------------------------------------------------------------------------

def gen_income_aware(idx: int) -> dict:
    name = rnd_name()
    age = random.randint(22, 55)
    city = rnd_city()
    profession = rnd_profession()

    income_options = list(range(20000, 310000, 5000))
    income = random.choice(income_options)

    # SIP = 15-30% of income
    sip_pct = random.uniform(0.15, 0.30)
    monthly_sip = round((income * sip_pct) / 500) * 500  # round to nearest 500
    monthly_sip = max(500, monthly_sip)

    risk = random.choice(RISK_LEVELS)
    horizon = random.randint(5, 20)

    # fund selection
    if income < 40000:
        primary, secondary = "index", "elss"
        rationale = "Index funds have the lowest expense ratio, making every rupee count at your income level. ELSS also saves you tax under Section 80C."
    elif income < 80000:
        primary, secondary = "flexi_cap", "elss"
        tertiary = "large_cap"
        rationale = "Flexi Cap gives you diversified equity exposure. ELSS saves tax. A sprinkle of Large Cap adds stability."
    elif risk == "aggressive":
        primary, secondary, tertiary = "mid_cap", "small_cap", "flexi_cap"
        rationale = "With your income and risk appetite, mid and small caps are your wealth-creation engines over this horizon."
    else:
        primary, secondary = "flexi_cap", "large_cap"
        tertiary = "elss" if income > 100000 else None
        rationale = "Flexi Cap + Large Cap is a time-tested combination for steady long-term compounding."

    funds_list = [primary, secondary] + ([tertiary] if "tertiary" in dir() and tertiary else [])

    splits = [0.50, 0.30, 0.20] if len(funds_list) == 3 else [0.60, 0.40]
    alloc = [(funds_list[i], int(monthly_sip * splits[i] / 100) * 100 or int(monthly_sip * splits[i])) for i in range(len(funds_list))]
    # recalc cleanly
    alloc = []
    for i, fk in enumerate(funds_list):
        amt = round(monthly_sip * splits[i] / 500) * 500 or 500
        alloc.append((fk, amt))

    fv = sip_future_value(monthly_sip, 0.12, horizon)
    invested = monthly_sip * horizon * 12

    user_variants = [
        (f"Hi, I earn ₹{income:,}/month as a {profession} in {city}. I'm {age} years old. How much should I invest and where?", False),
        (f"My monthly salary is ₹{income:,}. I'm a {age}-year-old {profession}. Can you suggest a good SIP plan for me?", False),
        (f"I take home ₹{income:,} every month (I'm a {profession} from {city}, age {age}). What mutual funds should I invest in and how much should my SIP be?", False),
        (f"Name: {name}. Monthly income: ₹{income:,}. Age: {age}. Profession: {profession}. Please tell me how much to invest and suggest funds.", True),
        (f"Earning ₹{income:,}/month in {city}. Age {age}, work as {profession}. Want to start mutual fund SIPs. How much and which ones?", False),
    ]
    user, has_name = random.choice(user_variants)

    fund_blocks = []
    for fkey, amt in alloc:
        f = FUND_CATEGORIES[fkey]
        fund_blocks.append(f"• **{f['name']}**: ₹{amt:,}/month — {f['desc']}")

    rationale_var = "rationale" in dir() and rationale or "This allocation balances growth and stability."

    greeting = f"{rnd_opener()} {name}!" if has_name else f"{rnd_opener()}"

    assistant = (
        f"{greeting} Based on your monthly income of ₹{income:,}, I'd recommend keeping your SIP at around **{int(sip_pct*100)}% of take-home pay** — which comes to roughly **₹{monthly_sip:,}/month**. "
        f"This is sustainable, leaves room for EMIs/expenses, and is enough to build meaningful wealth.\n\n"
        f"## Recommended SIP Breakdown\n\n"
        + "\n".join(fund_blocks)
        + f"\n\n## Why This Mix?\n\n{rationale_var}\n\n"
        f"## Wealth Projection\n\n"
        f"{sip_math_line(monthly_sip, 0.12, horizon)}\n\n"
        f"At ₹{income:,}/month income, a SIP of ₹{monthly_sip:,} means you're investing **{int(sip_pct*100)}%** of your income — the golden range recommended by most financial planners is 20–30%. "
        f"As your salary grows, increase your SIP by 10% every year (called a 'Step-Up SIP') for exponential compounding.\n\n"
        f"**Tax tip:** If you haven't maxed out your Section 80C limit (₹1.5 lakh/year), consider routing ₹{min(monthly_sip, 12500):,}/month into ELSS — you'll save up to ₹46,800 in taxes annually!\n\n"
        f"⚠️ *{rnd_hedge()}*\n\n"
        f"{rnd_closer()}"
    )

    return {"user": user, "assistant": assistant}


# ---------------------------------------------------------------------------
# Category 3: Goal-based (400 examples)
# ---------------------------------------------------------------------------

GOALS = {
    "child_education": {
        "label": "child's higher education",
        "horizons": [10, 12, 15, 18],
        "amounts": [2000000, 3000000, 5000000, 10000000],
        "notes": "Education inflation in India runs at 8–10% per year, so the real cost will be significantly higher than today's fees. Starting early is critical.",
        "funds": ["flexi_cap", "mid_cap", "large_cap", "index"],
    },
    "retirement": {
        "label": "retirement corpus",
        "horizons": [15, 20, 25, 30],
        "amounts": [10000000, 20000000, 30000000, 50000000],
        "notes": "A retirement corpus should aim to cover 25–30 years of post-retirement expenses. Equity dominance early, shifting to debt as you approach retirement.",
        "funds": ["flexi_cap", "large_cap", "elss", "index", "mid_cap"],
    },
    "home_downpayment": {
        "label": "home down payment",
        "horizons": [3, 5, 7],
        "amounts": [500000, 1000000, 2000000, 3000000],
        "notes": "For a goal within 3–5 years, keep equity exposure moderate to protect capital from short-term market swings.",
        "funds": ["hybrid", "large_cap", "index", "debt"],
    },
    "emergency_fund": {
        "label": "emergency fund",
        "horizons": [1, 2],
        "amounts": [100000, 200000, 300000, 500000],
        "notes": "An emergency fund should be liquid and capital-safe. Equity is NOT suitable here.",
        "funds": ["debt"],
    },
    "wedding": {
        "label": "wedding fund",
        "horizons": [2, 3, 5, 7],
        "amounts": [500000, 1000000, 2000000, 5000000],
        "notes": "Wedding expenses are non-negotiable and have a fixed deadline — keep risk moderate.",
        "funds": ["hybrid", "large_cap", "debt"],
    },
    "car": {
        "label": "car purchase",
        "horizons": [2, 3, 5],
        "amounts": [500000, 800000, 1200000, 2000000],
        "notes": "For a fixed short-term goal, avoid high-volatility equity. Hybrid or large-cap with a debt buffer is ideal.",
        "funds": ["hybrid", "large_cap", "debt"],
    },
    "vacation": {
        "label": "international vacation",
        "horizons": [1, 2, 3],
        "amounts": [100000, 200000, 500000],
        "notes": "Short-horizon goal — keep it in liquid or short-duration debt funds to protect the corpus.",
        "funds": ["debt", "hybrid"],
    },
    "business_capital": {
        "label": "business seed capital",
        "horizons": [3, 5, 7],
        "amounts": [500000, 1000000, 2000000, 5000000],
        "notes": "Business capital has a defined need date — balance growth with capital protection.",
        "funds": ["flexi_cap", "hybrid", "large_cap"],
    },
}

def gen_goal_based(idx: int) -> dict:
    name = rnd_name()
    age = random.randint(22, 55)
    city = rnd_city()
    goal_key = random.choice(list(GOALS.keys()))
    g = GOALS[goal_key]

    horizon = random.choice(g["horizons"])
    target = random.choice(g["amounts"])
    fv_needed = target

    # back-calculate required SIP
    r = 0.12 / 12
    n = horizon * 12
    req_sip = fv_needed * r / (((1 + r) ** n - 1) * (1 + r))
    req_sip_rounded = max(500, round(req_sip / 500) * 500)

    # pick 2 funds
    fund_keys = random.sample(g["funds"], min(2, len(g["funds"])))
    primary = fund_keys[0]
    secondary = fund_keys[1] if len(fund_keys) > 1 else fund_keys[0]

    rate = 0.07 if goal_key in ("emergency_fund",) else 0.10 if goal_key in ("home_downpayment", "wedding", "car", "vacation") else 0.12
    actual_fv = sip_future_value(req_sip_rounded, rate, horizon)

    user_variants = [
        (f"Hi, I'm {name} from {city}, age {age}. I want to save for my {g['label']}. I need roughly {fmt_inr(target)} in {horizon} years. How do I plan this with mutual funds?", True),
        (f"My goal is to build {fmt_inr(target)} for my {g['label']} in {horizon} years. I'm {age} years old. What SIP should I start?", False),
        (f"I'm {age} years old from {city}. Goal: {g['label']}. Target: {fmt_inr(target)}. Time: {horizon} years. What mutual fund plan do you suggest?", False),
        (f"Can you help me plan a SIP for {g['label']}? I want {fmt_inr(target)} ready in {horizon} years. I'm {age}.", False),
        (f"{name} here, {age} yo from {city}. Planning for {g['label']} — need {fmt_inr(target)} in about {horizon} years. What funds and how much per month?", True),
    ]
    user, has_name = random.choice(user_variants)

    f1 = FUND_CATEGORIES[primary]
    f2 = FUND_CATEGORIES[secondary]
    split1 = int(req_sip_rounded * 0.60)
    split2 = req_sip_rounded - split1

    greeting = f"{rnd_opener()} {name}!" if has_name else f"{rnd_opener()}"

    assistant = (
        f"{greeting} A {g['label']} worth {fmt_inr(target)} in {horizon} years is a well-defined goal — and mutual fund SIPs are one of the best ways to get there.\n\n"
        f"## Required Monthly SIP\n\n"
        f"To accumulate **{fmt_inr(fv_needed)}** in {horizon} years at ~{int(rate*100)}% CAGR, you need to invest approximately **₹{req_sip_rounded:,}/month**.\n\n"
        f"Let's verify: {sip_math_line(req_sip_rounded, rate, horizon)} ✅\n\n"
        f"## Recommended Fund Allocation\n\n"
        f"**{f1['name']}** — ₹{split1:,}/month (60%)\n"
        f"  - {f1['desc'].capitalize()}.\n"
        f"  - Risk: {f1['risk']} | Min horizon: {f1['horizon']}+ years\n\n"
        f"**{f2['name']}** — ₹{split2:,}/month (40%)\n"
        f"  - {f2['desc'].capitalize()}.\n"
        f"  - Risk: {f2['risk']} | Min horizon: {f2['horizon']}+ years\n\n"
        f"## Why These Funds for This Goal?\n\n"
        f"{g['notes']}\n\n"
        f"For a **{horizon}-year horizon**, {'equity funds have enough time to ride out market cycles and deliver strong real returns' if horizon >= 5 else 'a mix of hybrid/debt provides capital safety while still beating inflation'}.\n\n"
        f"**Action plan:**\n"
        f"1. Start SIP of ₹{split1:,} in {f1['name']}\n"
        f"2. Start SIP of ₹{split2:,} in {f2['name']}\n"
        f"3. Review annually; if income grows, step up SIP by 10%\n"
        f"4. As you approach the goal by 1–2 years, gradually shift to debt to lock in gains\n\n"
        f"⚠️ *{rnd_hedge()}*\n\n"
        f"{rnd_closer()}"
    )

    return {"user": user, "assistant": assistant}


# ---------------------------------------------------------------------------
# Category 4: Age-based life stage (300 examples)
# ---------------------------------------------------------------------------

def gen_age_based(idx: int) -> dict:
    name = rnd_name()
    city = rnd_city()
    profession = rnd_profession()

    age = random.randint(22, 65)
    monthly_sip = random.choice([1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000])

    if age <= 25:
        stage = "early career"
        equity_pct = 90
        fund1, fund2, fund3 = "small_cap", "mid_cap", "index"
        stage_advice = (
            f"At {age}, time is your most powerful asset. You can afford to take higher equity risk because you have 35+ years for compounding to work. "
            f"Small Cap and Mid Cap funds might be volatile short-term, but historically they've delivered 14–16% CAGR over long periods. "
            f"Don't panic during market dips — they are buying opportunities at your age!"
        )
        horizon = random.randint(20, 35)
    elif age <= 30:
        stage = "young professional"
        equity_pct = 85
        fund1, fund2, fund3 = "flexi_cap", "mid_cap", "elss"
        stage_advice = (
            f"In your late 20s, you're likely earning well and have fewer financial obligations than you will in your 30s. "
            f"This is the golden window to build wealth aggressively. ELSS also gives you a ₹46,800 annual tax saving — a no-brainer. "
            f"Aim to invest at least 20–25% of your take-home."
        )
        horizon = random.randint(15, 30)
    elif age <= 35:
        stage = "family building"
        equity_pct = 80
        fund1, fund2, fund3 = "flexi_cap", "large_cap", "elss"
        stage_advice = (
            f"Your 30s often come with responsibilities — EMIs, family expenses, life insurance. "
            f"Flexi Cap keeps the growth engine running while Large Cap provides stability. "
            f"Make sure you also have 6 months of expenses in a liquid fund as emergency buffer before aggressively investing."
        )
        horizon = random.randint(10, 25)
    elif age <= 45:
        stage = "peak earning"
        equity_pct = 70
        fund1, fund2, fund3 = "large_cap", "flexi_cap", "hybrid"
        stage_advice = (
            f"Peak earning years mean you can invest more, but retirement is now 15–20 years away. "
            f"Start moderating risk gradually. Large Cap and Flexi Cap give good equity growth while Hybrid provides a buffer. "
            f"Also start thinking about NPS to complement your mutual fund portfolio for retirement."
        )
        horizon = random.randint(10, 20)
    elif age <= 55:
        stage = "pre-retirement"
        equity_pct = 55
        fund1, fund2, fund3 = "hybrid", "large_cap", "debt"
        stage_advice = (
            f"At {age}, you're 10–15 years from retirement. Start the shift from aggressive to balanced. "
            f"Hybrid funds automatically manage this balance. Begin building a debt ladder for the first 5 years of post-retirement expenses. "
            f"Avoid small/mid cap now unless you have a specific long-horizon goal."
        )
        horizon = random.randint(5, 15)
    else:
        stage = "retirement planning / retired"
        equity_pct = 30
        fund1, fund2, fund3 = "debt", "hybrid", "large_cap"
        stage_advice = (
            f"At {age}+, capital preservation is paramount. Shift the bulk of your portfolio to debt and hybrid funds. "
            f"A small equity allocation (20–30%) in Large Cap or Index funds can still beat inflation over 5–7 years. "
            f"Consider a Systematic Withdrawal Plan (SWP) from a hybrid fund for regular monthly income."
        )
        horizon = random.randint(3, 10)

    sip1 = int(monthly_sip * 0.40)
    sip2 = int(monthly_sip * 0.35)
    sip3 = monthly_sip - sip1 - sip2

    fv = sip_future_value(monthly_sip, 0.12, horizon)
    invested = monthly_sip * horizon * 12

    f1 = FUND_CATEGORIES[fund1]
    f2 = FUND_CATEGORIES[fund2]
    f3 = FUND_CATEGORIES[fund3]

    user_variants = [
        (f"I'm {name}, {age} years old, working as a {profession} in {city}. I want to start investing ₹{monthly_sip:,}/month in mutual funds. What's right for my age?", True),
        (f"Age {age}, {profession} from {city}. Monthly investment budget: ₹{monthly_sip:,}. What mutual funds suit my life stage?", False),
        (f"Hi, I'm {age} years old and want to start SIPs of ₹{monthly_sip:,}/month. As a {profession} in {city}, what's a good fund mix for my age?", False),
        (f"I'm a {age}-year-old {profession}. Just getting into mutual funds. Can invest ₹{monthly_sip:,}/month. What do you suggest given my age?", False),
        (f"I'm {name}, {age} from {city}. What mutual fund strategy is best for a {age}-year-old {profession} investing ₹{monthly_sip:,}/month?", True),
    ]
    user, has_name = random.choice(user_variants)

    greeting = f"{rnd_opener()} {name}!" if has_name else f"{rnd_opener()}"

    assistant = (
        f"{greeting} At {age}, you're in the **{stage}** phase of your investing journey — and I've tailored this plan specifically for your life stage.\n\n"
        f"## Life Stage Analysis: {stage.title()} ({age} years)\n\n"
        f"{stage_advice}\n\n"
        f"**Recommended equity/debt split:** {equity_pct}% equity / {100 - equity_pct}% debt\n\n"
        f"## Your Portfolio (₹{monthly_sip:,}/month)\n\n"
        f"**1. {f1['name']}** — ₹{sip1:,}/month (40%)\n"
        f"   {f1['desc'].capitalize()}.\n\n"
        f"**2. {f2['name']}** — ₹{sip2:,}/month (35%)\n"
        f"   {f2['desc'].capitalize()}.\n\n"
        f"**3. {f3['name']}** — ₹{sip3:,}/month (25%)\n"
        f"   {f3['desc'].capitalize()}.\n\n"
        f"## Wealth Projection ({horizon} years)\n\n"
        f"{sip_math_line(monthly_sip, 0.12, horizon)}\n\n"
        f"**Invested:** {fmt_inr(invested)} | **Estimated corpus:** {fmt_inr(fv)}\n\n"
        f"## Key Action Steps\n\n"
        f"1. ✅ Start all three SIPs via auto-debit on a fixed date\n"
        f"2. ✅ Review allocation every year — as you age, gradually reduce equity %\n"
        f"3. ✅ {'Max out Section 80C with ELSS before March 31 every year' if age < 55 else 'Consider SWP from hybrid fund for tax-efficient income post-retirement'}\n"
        f"4. ✅ Don't stop SIPs during market corrections — stay the course\n\n"
        f"⚠️ *{rnd_hedge()}*\n\n"
        f"{rnd_closer()}"
    )

    return {"user": user, "assistant": assistant}


# ---------------------------------------------------------------------------
# Category 5: Conversational follow-ups (300 examples)
# ---------------------------------------------------------------------------

FOLLOWUP_TEMPLATES = [
    {
        "user_templates": [
            "What if I increase my SIP to ₹{amount}?",
            "If I increase my monthly investment to ₹{amount}, how much more will I get?",
            "I'm thinking of bumping up my SIP to ₹{amount}/month. What difference does it make?",
            "Can you show me the projection if I invest ₹{amount} instead?",
        ],
        "gen": lambda: {
            "amount": random.choice([5000, 8000, 10000, 15000, 20000, 25000, 30000]),
            "horizon": random.choice([5, 7, 10, 15, 20]),
            "old_amount": random.choice([2000, 3000, 5000, 7000]),
        },
        "resp": lambda d: (
            f"Great initiative! Increasing your SIP makes a massive difference thanks to compounding. Let me show you the numbers:\n\n"
            f"**Current SIP (₹{d['old_amount']:,}/month for {d['horizon']} years):**\n"
            f"{sip_math_line(d['old_amount'], 0.12, d['horizon'])}\n\n"
            f"**Increased SIP (₹{d['amount']:,}/month for {d['horizon']} years):**\n"
            f"{sip_math_line(d['amount'], 0.12, d['horizon'])}\n\n"
            f"**Additional wealth created:** {fmt_inr(sip_future_value(d['amount'] - d['old_amount'], 0.12, d['horizon']))}\n\n"
            f"That extra ₹{d['amount'] - d['old_amount']:,}/month creates significantly more wealth over {d['horizon']} years — this is the compounding effect at work. "
            f"Even a small SIP increase early in your journey has an outsized impact over the long term.\n\n"
            f"I'd strongly recommend this increase if your budget allows. You can also opt for a **Step-Up SIP** (increase by 10% every year automatically) rather than a one-time bump — it's even more powerful.\n\n"
            f"⚠️ *{rnd_hedge()}*"
        ),
    },
    {
        "user_templates": [
            "Can you explain ELSS funds more? Are they good for tax saving?",
            "I've heard of ELSS for tax saving. Can you explain how it works?",
            "What exactly is ELSS and how does it save tax?",
            "Tell me more about ELSS funds and Section 80C.",
        ],
        "gen": lambda: {"income": random.choice([50000, 75000, 100000, 150000])},
        "resp": lambda d: (
            f"Absolutely! ELSS (Equity Linked Savings Scheme) is one of the **best tax-saving instruments** under Section 80C of the Income Tax Act — here's why:\n\n"
            f"## What is ELSS?\n\n"
            f"ELSS funds are equity mutual funds with a **mandatory 3-year lock-in period**. In exchange, investments up to ₹1.5 lakh per financial year qualify for deduction under Section 80C.\n\n"
            f"## Tax Benefit Calculation\n\n"
            f"If you're in the 30% tax bracket (income > ₹10 lakh/year):\n"
            f"- ELSS investment: ₹1,50,000/year (₹12,500/month)\n"
            f"- Tax saved: ₹1,50,000 × 30% = **₹45,000/year** (plus 4% cess = ₹46,800 total)\n\n"
            f"## ELSS vs Other 80C Options\n\n"
            f"| Option | Lock-in | Returns | Tax on maturity |\n"
            f"|--------|---------|---------|------------------|\n"
            f"| ELSS | 3 years | 12%+ (market) | LTCG 10% above ₹1L |\n"
            f"| PPF | 15 years | ~7.1% | Tax-free |\n"
            f"| FD (5-yr) | 5 years | ~6.5% | Fully taxable |\n"
            f"| NSC | 5 years | ~7.7% | Taxable |\n\n"
            f"ELSS has the **shortest lock-in** among all 80C options and the **highest return potential**. The 3-year lock-in also enforces investing discipline.\n\n"
            f"## Is ELSS Good For You?\n\n"
            f"Yes, if:\n"
            f"- You haven't maxed your 80C limit yet\n"
            f"- You have a 3+ year horizon\n"
            f"- You're in the 20% or 30% tax bracket\n\n"
            f"Pro tip: Start an ELSS SIP early in the financial year (April) rather than dumping a lumpsum in March — you get SIP averaging benefits too.\n\n"
            f"⚠️ *{rnd_hedge()}*"
        ),
    },
    {
        "user_templates": [
            "Is this safe? I'm worried about losing money.",
            "What if the market crashes? Will I lose all my money?",
            "I'm a bit nervous about mutual funds. Are they risky?",
            "My friend lost money in mutual funds. How safe are these?",
        ],
        "gen": lambda: {"horizon": random.choice([5, 7, 10, 15])},
        "resp": lambda d: (
            f"Your concern is completely valid — and it's a sign of a thoughtful investor! Let me address this honestly.\n\n"
            f"## Can You Lose Money in Mutual Funds?\n\n"
            f"**Short answer: Yes, in the short term. Rarely over long periods.**\n\n"
            f"Equity mutual funds can fall significantly in a market crash — for example, during COVID-19 (March 2020), markets fell ~38% in weeks. However:\n\n"
            f"- **Nifty 50 one year later (March 2021):** Up ~75% from the bottom\n"
            f"- **Investors who stayed invested:** Recovered fully + made profits\n"
            f"- **Investors who panicked and sold:** Locked in losses permanently\n\n"
            f"## Historical Data (India)\n\n"
            f"- Any rolling **5-year SIP** in Nifty 50 since 2000 has delivered positive returns ~96% of the time\n"
            f"- Any rolling **7-year SIP** has delivered positive returns ~99% of the time\n"
            f"- Over 10+ years, equity SIPs have *never* delivered negative returns historically\n\n"
            f"## How SIPs Actually Help\n\n"
            f"When markets fall, your SIP buys **more units** at a lower price (Rupee Cost Averaging). When markets recover, those cheap units give you amplified returns. This is why systematic investing beats trying to time the market.\n\n"
            f"## Safety Measures in Place\n\n"
            f"- Funds are regulated by **SEBI** (India's securities regulator)\n"
            f"- Your money is held in a **separate trust** — the fund house cannot misuse it\n"
            f"- **AMFI** oversees all fund houses for compliance\n\n"
            f"**Bottom line:** Over your {d['horizon']}-year horizon, staying invested in a disciplined SIP is historically one of the safest ways to beat inflation and build wealth. The risk is real but manageable with time.\n\n"
            f"⚠️ *That said, equity markets are volatile short-term. Only invest money you won't need for {d['horizon']} years.*"
        ),
    },
    {
        "user_templates": [
            "What does NAV mean?",
            "Can you explain what NAV is in mutual funds?",
            "What is Net Asset Value? How does it affect my investment?",
        ],
        "gen": lambda: {},
        "resp": lambda d: (
            f"Great question! NAV (Net Asset Value) is essentially the **price per unit** of a mutual fund on any given day.\n\n"
            f"## How NAV is Calculated\n\n"
            f"NAV = (Total Assets of Fund – Liabilities) ÷ Number of Units Outstanding\n\n"
            f"For example, if a fund has ₹100 crore in assets, ₹1 crore in liabilities, and 1 crore units outstanding:\n"
            f"NAV = (100 – 1) ÷ 1 = **₹99 per unit**\n\n"
            f"## Does a Lower NAV Mean Cheaper?\n\n"
            f"**No — this is a very common misconception!** A fund with NAV ₹10 is NOT cheaper or better than one with NAV ₹500. What matters is the **percentage growth** of the NAV over time, not the absolute value.\n\n"
            f"Example:\n"
            f"- Fund A: NAV ₹10, grows to ₹12 → 20% return\n"
            f"- Fund B: NAV ₹500, grows to ₹600 → 20% return\n"
            f"Both give the same return — NAV level doesn't matter.\n\n"
            f"## NAV and Your SIP\n\n"
            f"When you invest ₹5,000 via SIP:\n"
            f"- If NAV = ₹50, you get 100 units\n"
            f"- If NAV = ₹100, you get 50 units\n\n"
            f"Your returns depend on how much the NAV grows from your purchase price — not on the NAV level itself. This is why SIPs work — you buy fewer units when NAV is high and more when it's low (Rupee Cost Averaging).\n\n"
            f"Hope that clears it up! Any other questions?"
        ),
    },
    {
        "user_templates": [
            "What is the difference between SIP and lumpsum?",
            "Should I do SIP or invest a lumpsum?",
            "I have ₹{amount} to invest. Should I put it all at once or spread it out?",
            "Lumpsum vs SIP — which is better?",
        ],
        "gen": lambda: {"amount": random.choice([50000, 100000, 200000, 500000])},
        "resp": lambda d: (
            f"Great timing question! Both SIP and lumpsum have their place. Here's an honest comparison:\n\n"
            f"## SIP (Systematic Investment Plan)\n\n"
            f"- Invest a **fixed amount every month** (e.g., ₹5,000/month)\n"
            f"- Benefits from **Rupee Cost Averaging** — buys more units when markets are low\n"
            f"- Reduces timing risk — you don't need to predict market peaks/bottoms\n"
            f"- Builds investing discipline and habit\n"
            f"- Ideal for salaried individuals with regular income\n\n"
            f"## Lumpsum Investment\n\n"
            f"- Invest a **large amount all at once** (e.g., ₹{d['amount']:,} today)\n"
            f"- Works best when markets are **near a bottom or correction**\n"
            f"- Gives the entire corpus more time in the market (higher upside if timed right)\n"
            f"- Risk: If markets fall right after you invest, you face paper losses immediately\n"
            f"- Ideal for bonus/windfall money\n\n"
            f"## What the Data Says\n\n"
            f"Studies show that in **strongly trending bull markets**, lumpsum outperforms SIP. But since we can't predict markets, SIP wins for most regular investors because:\n"
            f"1. It eliminates timing risk completely\n"
            f"2. Emotionally easier — no 'should I invest now or wait?' anxiety\n"
            f"3. Ideal for regular monthly income\n\n"
            f"## My Recommendation for You\n\n"
            f"If you have ₹{d['amount']:,} as a lumpsum right now:\n"
            f"- **Option A (safest):** Split into 6–12 monthly STPs (Systematic Transfer Plan) from a liquid fund into equity\n"
            f"- **Option B (if markets have corrected 15%+):** Invest 50% as lumpsum now, 50% over 6 months\n"
            f"- **Option C (ongoing):** Keep the lumpsum as emergency fund and start a fresh monthly SIP\n\n"
            f"⚠️ *{rnd_hedge()}*"
        ),
    },
    {
        "user_templates": [
            "Can I withdraw my money any time from mutual funds?",
            "How liquid are mutual funds? Can I get my money back quickly?",
            "What if I need money urgently — can I sell my mutual fund units?",
        ],
        "gen": lambda: {},
        "resp": lambda d: (
            f"Absolutely — this is a really important question that most new investors don't think about upfront!\n\n"
            f"## Liquidity of Mutual Funds\n\n"
            f"Most open-ended mutual funds are **highly liquid**. Here's how it works:\n\n"
            f"**Step 1:** Submit a redemption request on your app/platform (Zerodha, Groww, MFCentral, etc.)\n"
            f"**Step 2:** The redemption is processed at end-of-day NAV\n"
            f"**Step 3:** Money is credited to your bank account\n\n"
            f"## Redemption Timelines by Fund Type\n\n"
            f"| Fund Type | Money in Bank |\n"
            f"|-----------|---------------|\n"
            f"| Liquid funds | Same day or T+1 |\n"
            f"| Overnight funds | T+1 |\n"
            f"| Debt funds | T+2 to T+3 |\n"
            f"| Equity funds | T+3 |\n"
            f"| ELSS funds | After 3-year lock-in only |\n\n"
            f"## Exit Loads\n\n"
            f"Some funds charge an **exit load** if you withdraw within 1 year (typically 1% of redemption amount). After 1 year, most equity funds have zero exit load.\n\n"
            f"## Tax on Withdrawal\n\n"
            f"- **STCG (held < 1 year):** 20% tax on gains\n"
            f"- **LTCG (held > 1 year):** 12.5% tax on gains above ₹1.25 lakh/year\n"
            f"- **Debt funds:** Gains added to income, taxed at your slab rate\n\n"
            f"**Tip:** Keep 3–6 months of expenses in a liquid fund or savings account as your emergency fund, so you don't have to disturb your equity investments during a crisis.\n\n"
            f"Hope that helps clarify! Any other questions?"
        ),
    },
    {
        "user_templates": [
            "What is expense ratio and does it matter?",
            "I saw expense ratio mentioned. How important is it?",
            "Should I look at expense ratio when choosing a fund?",
        ],
        "gen": lambda: {"amount": random.choice([3000, 5000, 10000])},
        "resp": lambda d: (
            f"Yes, expense ratio matters more than most people realise — especially over long horizons! Let me break it down.\n\n"
            f"## What is Expense Ratio?\n\n"
            f"The expense ratio is the **annual fee** a mutual fund charges to manage your money, expressed as a percentage of your invested amount. It's deducted daily from the fund's NAV (so you never write a cheque — it's already factored into returns).\n\n"
            f"## Typical Ranges\n\n"
            f"| Fund Type | Typical Expense Ratio |\n"
            f"|-----------|----------------------|\n"
            f"| Index funds | 0.10% – 0.25% |\n"
            f"| Large Cap active | 0.50% – 1.5% |\n"
            f"| Mid/Small Cap | 0.60% – 1.8% |\n"
            f"| ELSS | 0.60% – 1.5% |\n"
            f"| Debt funds | 0.10% – 0.75% |\n\n"
            f"## Why It Compounds Against You\n\n"
            f"Let's say you invest ₹{d['amount']:,}/month for 20 years:\n\n"
            f"- At 12% gross return, **0.5% expense ratio** → net ~11.5% → corpus: {fmt_inr(sip_future_value(d['amount'], 0.115, 20))}\n"
            f"- At 12% gross return, **1.5% expense ratio** → net ~10.5% → corpus: {fmt_inr(sip_future_value(d['amount'], 0.105, 20))}\n\n"
            f"That **1% difference in expense ratio** can cost you lakhs over 20 years!\n\n"
            f"## Practical Advice\n\n"
            f"- For **Large Cap** investing: prefer Index funds (Nifty 50/Next 50) over active large-cap funds — most active large-cap funds underperform index over 10+ years\n"
            f"- For **Mid/Small Cap**: active management adds more value, so slightly higher expense ratio is acceptable\n"
            f"- Always compare the **Direct Plan** vs Regular Plan — Direct plans have 0.5–1% lower expense ratio (huge difference over time)\n\n"
            f"Always invest in **Direct Plans** through platforms like Zerodha Coin, Groww, or MFCentral to save on distributor commissions.\n\n"
            f"⚠️ *{rnd_hedge()}*"
        ),
    },
    {
        "user_templates": [
            "What happens to my mutual fund if the fund house shuts down?",
            "Is my money safe if the AMC goes bankrupt?",
            "What if the mutual fund company closes?",
        ],
        "gen": lambda: {},
        "resp": lambda d: (
            f"This is an excellent question and one that shows you're thinking carefully — your money is actually **very well protected**. Here's why:\n\n"
            f"## Your Money is Held in a Separate Trust\n\n"
            f"Under SEBI regulations, mutual fund assets are held in a **trust structure** completely separate from the AMC (Asset Management Company). "
            f"The AMC is just a service provider that manages the fund — it doesn't own your money.\n\n"
            f"If an AMC shuts down:\n"
            f"1. SEBI steps in and appoints a new AMC to manage the fund, OR\n"
            f"2. The fund is wound up and assets are liquidated and returned to investors at NAV\n\n"
            f"In either case, **you do not lose your money** due to the AMC going bankrupt.\n\n"
            f"## Regulatory Safeguards\n\n"
            f"- **SEBI** regulates all mutual funds in India — strict compliance requirements\n"
            f"- **Trustees** independently oversee each fund's assets on behalf of investors\n"
            f"- **Custodian banks** (separate from AMC) hold the actual securities\n"
            f"- **AMFI** (Association of Mutual Funds in India) maintains standards\n\n"
            f"## Historical Precedent\n\n"
            f"When Franklin Templeton wound up 6 debt schemes in 2020, investors received their money back over 2 years as the portfolio matured — nobody lost their principal permanently.\n\n"
            f"**Bottom line:** Mutual funds are one of the most regulated financial products in India. Your principal is not at risk due to AMC failure — only market risk applies to returns.\n\n"
            f"Any other concerns? Happy to address them!"
        ),
    },
]

def gen_conversational_followup(idx: int) -> dict:
    template = random.choice(FOLLOWUP_TEMPLATES)
    data = template["gen"]()
    user_tmpl = random.choice(template["user_templates"])
    try:
        user = user_tmpl.format(**data)
    except KeyError:
        user = user_tmpl
    assistant = template["resp"](data)
    return {"user": user, "assistant": assistant}


# ---------------------------------------------------------------------------
# Category 6: Normal chat / greetings (200 examples)
# ---------------------------------------------------------------------------

GREET_TEMPLATES = [
    {
        "user": ["Hi", "Hello", "Hey", "Hi!", "Hello!", "Hey there!", "Hi MentraFiAI!", "Hello, anyone there?", "Good morning!",
                 "Good evening!", "Good afternoon!", "Hey!", "Namaste!", "Hi, I'm new here.", "Hii", "Hello MentraFiAI", "Hey buddy"],
        "assistant": [
            "Namaste! 👋 Welcome to MentraFiAI — your personal mutual fund advisor! I'm here to help you plan your investments, understand mutual funds, calculate SIP projections, and build a portfolio tailored to your goals.\n\nTo get started, you can tell me:\n- Your age, monthly income, and risk appetite\n- A specific financial goal (retirement, child education, home, etc.)\n- Or just ask me anything about mutual funds!\n\nHow can I help you today? 😊",
            "Hello! Welcome aboard! I'm MentraFiAI, your AI-powered mutual fund advisor. I can help you:\n\n✅ Choose the right funds for your risk profile\n✅ Calculate SIP amounts to reach your goals\n✅ Explain complex concepts in simple language\n✅ Compare investment options\n\nWhat would you like to explore today?",
            "Hey there! Great to have you here! I'm MentraFiAI — think of me as your always-available financial advisor who specialises in mutual funds and SIP planning for Indian investors.\n\nWhat's on your mind? Whether it's your first SIP or you're looking to optimise an existing portfolio, I've got you! 🚀",
            "Hi! Always happy to help! Whether you are looking to start your very first ₹500 SIP, save tax with ELSS, or plan a long-term retirement corpus, just let me know what you need!",
        ],
    },
    {
        "user": ["Who are you?", "What is your name?", "Introduce yourself", "Tell me about yourself", "Are you an AI?", "Who made you?"],
        "assistant": [
            "I'm MentraFiAI, an AI-powered mutual fund and personal finance advisor designed specifically for Indian investors! 🇮🇳\n\nMy purpose is to guide you through mutual fund selection, asset allocation, SIP planning, and goal-based investing without any distributor bias or commission pressure. Feel free to ask me anything about funds, SIP calculations, or financial planning!",
            "I am MentraFiAI — your dedicated mutual fund expert and investment co-pilot. I can evaluate your risk profile, design personalized fund portfolios, calculate SIP growth projections, and explain complex financial concepts in plain English. How can I assist your financial journey today?",
        ],
    },
    {
        "user": ["How are you?", "How are you doing?", "Are you doing well?", "Hope you are doing good"],
        "assistant": [
            "I'm doing fantastic, thank you for asking! Ready to help you build wealth and navigate the world of mutual funds. How are your investment plans shaping up today? 😊",
            "I'm doing great! Watching the power of compounding at work always keeps me energized! How can I help you with your investments today?",
        ],
    },
    {
        "user": ["Goodbye", "Bye", "Bye bye", "See you later", "Have a great day", "Good night"],
        "assistant": [
            "Goodbye! Wishing you happy and disciplined investing. Remember: stay invested, keep your SIPs active, and let compounding do the heavy lifting. Reach out whenever you need guidance! 👋",
            "Take care! Remember that the secret to wealth is staying consistent with your investments. See you next time! 🌟",
        ],
    },
    {
        "user": ["How do I start investing?", "I am a complete beginner, where do I begin?", "How to start mutual funds?", "Step by step guide to invest in mutual funds"],
        "assistant": (
            "Welcome to the investing journey! Starting is much easier than most people think. Here is your simple 4-step roadmap:\n\n"
            "1. **Complete your KYC online** (takes 5 minutes using your Aadhaar & PAN on SEBI-registered apps like Zerodha Coin, Groww, or MFCentral).\n"
            "2. **Build an Emergency Fund first** (keep 3-6 months of expenses in a liquid fund or savings account).\n"
            "3. **Determine your goal and horizon** (Short term < 3 yrs: Debt/Hybrid; Long term 5+ yrs: Equity/Index).\n"
            "4. **Start a small monthly SIP** (even ₹500 or ₹1,000/month in a Nifty 50 Index Fund or Flexi Cap Fund via auto-debit).\n\n"
            "If you share your age, monthly budget, and investment horizon, I can recommend your exact first fund portfolio right now! 😊"
        ),
    },
    {
        "user": ["Thank you!", "Thanks a lot!", "This was very helpful!", "Thank you so much!", "Great help!",
                 "Thanks, this was exactly what I needed!", "Appreciate it!", "You've been very helpful!"],
        "assistant": [
            "You're very welcome! 😊 It's my pleasure to help you on your wealth-building journey. Remember: the best investment is one you start today and stay committed to.\n\nFeel free to come back any time — whether it's for a portfolio review, a new goal, or just to check your SIP projections. All the best! 🎯",
            "Happy to help! 🙏 Your financial future is important, and asking the right questions is the first step. Good luck with your investments — and remember, consistency with your SIP beats trying to time the market every time!\n\nCome back whenever you need guidance! 💪",
            "Glad I could help! Starting your investment journey (or improving an existing one) is one of the best decisions you can make. Stay disciplined, don't panic during market dips, and let compounding do its magic.\n\nWishing you all the best! Feel free to ask more questions anytime! 🌟",
        ],
    },
    {
        "user": ["What can you do?", "What are your capabilities?", "How can you help me?",
                 "What kind of questions can I ask you?", "What do you know about?"],
        "assistant": [
            "Great question! Here's what I can help you with as your mutual fund advisor:\n\n## 📊 Portfolio Planning\n- Recommend funds based on your age, income, risk tolerance, and goals\n- Create a personalised asset allocation\n- Suggest SIP amounts that fit your budget\n\n## 🧮 SIP Calculations\n- Calculate how much your SIP will grow to over any period\n- Show you how much SIP you need to reach a specific goal amount\n- Model step-up SIP scenarios\n\n## 📚 Education\n- Explain any mutual fund concept: NAV, expense ratio, ELSS, SWP, etc.\n- Compare fund categories (large cap vs small cap, SIP vs lumpsum, ELSS vs PPF)\n- Help you understand risk and returns\n\n## 🎯 Goal-Based Planning\n- Child education fund\n- Retirement planning\n- Home down payment\n- Emergency fund\n- Wedding, car, vacation — any goal!\n\nWhat would you like to start with?",
        ],
    },
    {
        "user": ["Can you buy mutual funds for me?", "Can you execute trades?", "Can you invest on my behalf?",
                 "Do you buy and sell funds?"],
        "assistant": (
            "I appreciate the trust, but I should be transparent: I'm an AI advisor, not a trading platform — I can't directly buy or sell funds on your behalf.\n\n"
            "What I *can* do is give you expert, personalised advice on **which funds to buy, how much to invest, and when to review**.\n\n"
            "To actually execute the SIPs I recommend, you can use these SEBI-registered platforms:\n\n"
            "- **Zerodha Coin** — direct plans, low cost\n"
            "- **Groww** — beginner-friendly\n"
            "- **MFCentral** — official AMFI portal, all AMCs\n"
            "- **Paytm Money** — easy KYC\n"
            "- **ET Money** — great for tracking\n\n"
            "All are free to use and SEBI-regulated. I'll give you the roadmap — they'll execute it! Shall I start by recommending a portfolio for you? 😊"
        ),
    },
    {
        "user": ["What is a mutual fund?", "Explain mutual funds to me", "I'm new to investing. What is a mutual fund?",
                 "Can you explain what mutual funds are?"],
        "assistant": (
            "Of course! Let me explain mutual funds in simple terms:\n\n"
            "## What is a Mutual Fund?\n\n"
            "A mutual fund is a **pool of money collected from many investors** and managed by a professional fund manager. "
            "This pooled money is invested in stocks, bonds, or both — depending on the fund type.\n\n"
            "**Simple analogy:** Imagine 1,000 people each put ₹5,000 into a common pot (₹50 lakh total). "
            "A professional manager uses this ₹50 lakh to buy shares of 30–50 companies. "
            "Each investor owns a proportional share of this diversified portfolio. That's a mutual fund!\n\n"
            "## Why Mutual Funds?\n\n"
            "✅ **Diversification** — your ₹5,000 is spread across 50+ stocks, not just one\n"
            "✅ **Professional management** — experts research and select stocks for you\n"
            "✅ **Low entry barrier** — start with just ₹100/month via SIP\n"
            "✅ **Regulated** — SEBI ensures transparency and investor protection\n"
            "✅ **Liquidity** — redeem any time (except ELSS lock-in)\n\n"
            "## Types of Mutual Funds\n\n"
            "- **Equity funds** — invest in stocks (higher risk, higher potential return, 12–15% long-term)\n"
            "- **Debt funds** — invest in bonds (lower risk, 6–8% returns, short-term goals)\n"
            "- **Hybrid funds** — mix of equity + debt (moderate risk, ~10% returns)\n\n"
            "Want me to help you pick the right type for your goals? Just tell me your age, budget, and what you're saving for! 😊"
        ),
    },
    {
        "user": ["What is SIP?", "Explain SIP to me", "How does SIP work?", "I've heard of SIP. What is it?"],
        "assistant": (
            "SIP stands for **Systematic Investment Plan** — and it's probably the single most powerful tool for building wealth as a salaried Indian investor.\n\n"
            "## How SIP Works\n\n"
            "A SIP is simply a **standing instruction to invest a fixed amount every month** into a mutual fund. "
            "On your chosen date (say, the 5th of every month), your bank automatically transfers ₹5,000 to your chosen fund. That's it!\n\n"
            "## Why SIP is Powerful\n\n"
            "**1. Rupee Cost Averaging:**\nWhen markets are high, your ₹5,000 buys fewer units. When markets fall, it buys more units. Over time, your average cost per unit is lower than the average market price.\n\n"
            "**2. Compounding:**\nYour returns generate their own returns. ₹5,000/month for 20 years at 12% CAGR = " + fmt_inr(sip_future_value(5000, 0.12, 20)) + " (invested: " + fmt_inr(5000*20*12) + "). The difference? Compounding!\n\n"
            "**3. Discipline:**\nRemoves the temptation to 'time the market'. Set it, forget it, and review annually.\n\n"
            "**4. Flexibility:**\nYou can pause, stop, or increase your SIP at any time with no penalty (except ELSS lock-in).\n\n"
            "## How to Start\n\n"
            "1. Complete KYC on any app (Groww, Zerodha, Paytm Money)\n"
            "2. Choose a fund\n"
            "3. Set SIP amount + date + bank auto-debit\n"
            "4. Done! Takes 15 minutes.\n\n"
            "Want me to suggest which funds to set up a SIP in? 😊"
        ),
    },
    {
        "user": ["Tell me something about the stock market", "What's happening in the market today?",
                 "Should I invest in stocks directly?", "Stock market vs mutual funds?"],
        "assistant": (
            "Happy to share some context! While I'm specialised in mutual fund advisory, let me address this well:\n\n"
            "## Direct Stocks vs Mutual Funds\n\n"
            "| Factor | Direct Stocks | Mutual Funds |\n"
            "|--------|--------------|---------------|\n"
            "| Expertise needed | High | Low |\n"
            "| Diversification | Low (unless large portfolio) | High (50+ stocks) |\n"
            "| Time commitment | High (research, monitoring) | Low |\n"
            "| Minimum investment | ~₹1 per share | ₹100/month |\n"
            "| Risk | High (single stock can go to zero) | Moderate (diversified) |\n"
            "| Returns potential | Very high (if you pick right) | Market-matching |\n\n"
            "## My Take\n\n"
            "For most salaried investors without the time to research companies deeply, **mutual funds are the smarter choice**. "
            "You get professional management, diversification, and regulatory oversight.\n\n"
            "If you want some exposure to direct stocks, a good framework is:\n"
            "- 70-80% in mutual fund SIPs (core, steady)\n"
            "- 20-30% in direct stocks (only in companies you understand well)\n\n"
            "As for market news — I'd recommend not letting daily market movements affect your SIP decisions. Stay the course!\n\n"
            "Want me to help you build a mutual fund portfolio? 😊"
        ),
    },
    {
        "user": ["Are you better than a human advisor?", "Why should I trust an AI for financial advice?",
                 "Can I trust your recommendations?"],
        "assistant": (
            "Honest answer: I'm a different kind of tool, not a replacement for a SEBI-registered human advisor for complex situations. Here's my take:\n\n"
            "## What I'm Good At\n\n"
            "✅ Available 24/7 — no appointment needed\n"
            "✅ No commission bias — I don't earn from recommending any specific fund\n"
            "✅ Instant SIP calculations and projections\n"
            "✅ Clear, jargon-free explanations\n"
            "✅ Consistent, data-backed recommendations\n"
            "✅ Great for straightforward SIP planning and fund selection\n\n"
            "## Where a Human Advisor Adds More Value\n\n"
            "- Complex tax situations (HUF, business income, etc.)\n"
            "- Estate planning and wills\n"
            "- Insurance + investment integration\n"
            "- Behavioural coaching (talking you off the ledge during market crashes)\n"
            "- SEBI-registered advisors are legally accountable\n\n"
            "## My Recommendation\n\n"
            "Use me for education, quick analysis, and building your initial portfolio framework. For large amounts (₹50L+) or complex family financial planning, consult a SEBI-registered investment advisor (RIA) for a comprehensive plan.\n\n"
            "All my suggestions come with the standard disclaimer: **mutual fund investments are subject to market risk** — I provide guidance, not guarantees. Sound good? Let's get started! 😊"
        ),
    },
]

def gen_normal_chat(idx: int) -> dict:
    template = random.choice(GREET_TEMPLATES)
    user_list = template["user"]
    assistant_data = template["assistant"]

    user = random.choice(user_list)
    if isinstance(assistant_data, list):
        assistant = random.choice(assistant_data)
    else:
        assistant = assistant_data

    return {"user": user, "assistant": assistant}


# ---------------------------------------------------------------------------
# Category 7: Risk-specific (200 examples)
# ---------------------------------------------------------------------------

RISK_PROFILES = {
    "very_conservative": {
        "labels": ["very conservative", "extremely risk averse", "capital preservation focused", "I can't afford to lose any money"],
        "funds": [("debt", 0.60), ("hybrid", 0.30), ("large_cap", 0.10)],
        "rate": 0.07,
        "desc": "Your safety-first approach is completely valid — especially if you're near retirement or have a short investment horizon. For very conservative investors, capital preservation takes priority over high returns.",
        "caveats": "Note: At these allocations, your real return (after 6% inflation) may be very modest. Over long periods, staying too conservative can also hurt — inflation silently erodes purchasing power.",
    },
    "conservative": {
        "labels": ["conservative", "low risk tolerance", "safety-first investor", "don't want too much risk"],
        "funds": [("hybrid", 0.50), ("large_cap", 0.30), ("debt", 0.20)],
        "rate": 0.09,
        "desc": "A conservative approach with a tilt towards stability. Hybrid funds give you equity growth with automatic rebalancing. Large Cap adds blue-chip equity exposure with lower volatility than mid/small caps.",
        "caveats": "Over 7+ years, this allocation should comfortably beat inflation and FD returns while keeping volatility manageable.",
    },
    "moderate": {
        "labels": ["moderate risk", "balanced investor", "okay with some ups and downs", "moderate"],
        "funds": [("flexi_cap", 0.40), ("large_cap", 0.30), ("hybrid", 0.20), ("index", 0.10)],
        "rate": 0.11,
        "desc": "A classic balanced portfolio. Flexi Cap + Large Cap gives the core equity exposure, Hybrid adds stability, and Index fund provides low-cost market returns.",
        "caveats": "This is one of the most time-tested allocations for long-term wealth creation with reasonable stability.",
    },
    "moderately_aggressive": {
        "labels": ["moderately aggressive", "above average risk tolerance", "growth oriented", "willing to take some risk for higher returns"],
        "funds": [("flexi_cap", 0.35), ("mid_cap", 0.30), ("large_cap", 0.25), ("international", 0.10)],
        "rate": 0.12,
        "desc": "A growth-focused portfolio. Mid Cap adds meaningful return potential. The international allocation provides USD diversification — particularly useful when INR depreciates.",
        "caveats": "Expect higher short-term volatility compared to a conservative portfolio. Stay invested through market dips — don't panic.",
    },
    "aggressive": {
        "labels": ["aggressive", "high risk tolerance", "maximum returns", "I can handle 30-40% drawdowns"],
        "funds": [("mid_cap", 0.35), ("small_cap", 0.30), ("flexi_cap", 0.25), ("international", 0.10)],
        "rate": 0.14,
        "desc": "A full-throttle growth portfolio. Small Cap and Mid Cap are your wealth-creation engines. Historically, a diversified portfolio of mid and small caps has delivered 15–18% CAGR over 10-year periods.",
        "caveats": "⚠️ This portfolio can fall 40-50% in severe bear markets (2008, 2020). You MUST have a 7+ year horizon and strong emotional discipline not to sell during crashes.",
    },
    "very_aggressive": {
        "labels": ["very aggressive", "extremely high risk tolerance", "all in on small cap", "want maximum possible returns, okay with losses short term"],
        "funds": [("small_cap", 0.50), ("mid_cap", 0.30), ("international", 0.20)],
        "rate": 0.15,
        "desc": "Maximum growth mode. This is a pure-equity, high-risk portfolio dominated by small and mid caps with international diversification.",
        "caveats": "⚠️ CRITICAL: This portfolio is NOT suitable for horizons under 10 years. Small Cap funds can drop 60-70% in crashes. Only invest money you absolutely do not need for 10+ years. This is for serious long-term wealth builders only.",
    },
}

def gen_risk_specific(idx: int) -> dict:
    name = rnd_name()
    age = random.randint(22, 60)
    city = rnd_city()
    risk_key = random.choice(list(RISK_PROFILES.keys()))
    profile = RISK_PROFILES[risk_key]
    risk_label = random.choice(profile["labels"])
    monthly_sip = random.choice([2000, 3000, 5000, 7000, 10000, 15000, 20000, 25000])
    horizon = random.randint(5, 20)

    fund_alloc = []
    for fk, pct in profile["funds"]:
        amt = round(monthly_sip * pct / 500) * 500 or 500
        fund_alloc.append((fk, amt))

    fund_blocks = []
    for fk, amt in fund_alloc:
        f = FUND_CATEGORIES[fk]
        pct_display = int(amt / monthly_sip * 100)
        fund_blocks.append(
            f"**{f['name']}** — ₹{amt:,}/month (~{pct_display}%)\n"
            f"   {f['desc'].capitalize()}."
        )

    fv = sip_future_value(monthly_sip, profile["rate"], horizon)
    invested = monthly_sip * horizon * 12

    user_variants = [
        (f"I'm {name}, {age} years old from {city}. I am a {risk_label} investor. I can invest ₹{monthly_sip:,}/month for {horizon} years. What funds do you suggest?", True),
        (f"My risk profile is {risk_label}. I'm {age} and want to start a SIP of ₹{monthly_sip:,}/month. What's a good portfolio for someone like me?", False),
        (f"Hi, I'm {age} years old with {risk_label} risk tolerance. Monthly SIP budget: ₹{monthly_sip:,}. {horizon}-year horizon. Best fund allocation?", False),
        (f"{name} here, {age}yo from {city}. I describe myself as a {risk_label} investor. ₹{monthly_sip:,}/month available. Suggest a portfolio please.", True),
        (f"I want a mutual fund portfolio suited for a {risk_label} investor. I'm {age}, can invest ₹{monthly_sip:,}/month, horizon is {horizon} years.", False),
    ]
    user, has_name = random.choice(user_variants)

    greeting = f"{rnd_opener()} {name}!" if has_name else f"{rnd_opener()}"

    assistant = (
        f"{greeting} As a **{risk_label}** investor, I've designed a portfolio that matches your risk appetite perfectly.\n\n"
        f"## Risk Profile: {risk_label.title()}\n\n"
        f"{profile['desc']}\n\n"
        f"## Your Portfolio (₹{monthly_sip:,}/month)\n\n"
        + "\n\n".join(fund_blocks)
        + f"\n\n## Projection ({horizon} years @ {int(profile['rate']*100)}% blended CAGR)\n\n"
        f"{sip_math_line(monthly_sip, profile['rate'], horizon)}\n\n"
        f"**Invested:** {fmt_inr(invested)} | **Estimated corpus:** {fmt_inr(fv)}\n\n"
        f"## Important Note\n\n"
        f"{profile['caveats']}\n\n"
        f"⚠️ *{rnd_hedge()}*\n\n"
        f"{rnd_closer()}"
    )

    return {"user": user, "assistant": assistant}


# ---------------------------------------------------------------------------
# Category 8: Comparison questions (200 examples)
# ---------------------------------------------------------------------------

COMPARISONS = [
    {
        "users": [
            "What's the difference between large cap and small cap funds?",
            "Should I invest in large cap or small cap?",
            "Large cap vs small cap — which is better?",
            "What's better for long term: large cap or small cap mutual funds?",
        ],
        "assistant": lambda: (
            f"Great comparison question! The choice depends on your **risk tolerance and investment horizon**. Here's a comprehensive breakdown:\n\n"
            f"## Large Cap vs Small Cap Funds\n\n"
            f"| Factor | Large Cap | Small Cap |\n"
            f"|--------|-----------|----------|\n"
            f"| Companies | Top 100 by market cap | Beyond top 250 |\n"
            f"| Examples | Reliance, TCS, HDFC Bank | Emerging businesses |\n"
            f"| Risk | Moderate | High |\n"
            f"| Volatility | Lower | Very high |\n"
            f"| Return potential | 10–12% CAGR | 14–18% CAGR |\n"
            f"| Min horizon | 5 years | 7+ years |\n"
            f"| Best for | Conservative/moderate investors | Aggressive, long-term investors |\n\n"
            f"## SIP Math Comparison\n\n"
            f"₹10,000/month for 15 years:\n"
            f"- Large Cap (11% CAGR): {fmt_inr(sip_future_value(10000, 0.11, 15))}\n"
            f"- Small Cap (15% CAGR): {fmt_inr(sip_future_value(10000, 0.15, 15))}\n\n"
            f"That's a significant difference — but small cap also comes with significantly more **drawdown risk** (can fall 50-60% in crashes).\n\n"
            f"## My Recommendation\n\n"
            f"Don't choose one over the other — **combine them!**\n\n"
            f"- Young investor (22-35), 10+ year horizon: 30% Large Cap + 30% Mid Cap + 20% Small Cap + 20% Flexi Cap\n"
            f"- Moderate investor: 50% Large Cap + 30% Flexi Cap + 20% Mid Cap (skip small cap)\n"
            f"- Conservative investor: 70% Large Cap / Index + 30% Hybrid\n\n"
            f"⚠️ *{rnd_hedge()}*"
        ),
    },
    {
        "users": [
            "SIP vs lumpsum — which is better for me?",
            "Should I invest my bonus as lumpsum or spread it via SIP?",
            "What's the difference between SIP and one-time investment?",
            "Lumpsum or SIP for long term investing?",
        ],
        "assistant": lambda: (
            f"Both have their place! Here's an honest, data-backed comparison:\n\n"
            f"## SIP (Systematic Investment Plan)\n\n"
            f"**Best for:** Regular monthly investing from salary/income\n\n"
            f"✅ Eliminates timing risk through Rupee Cost Averaging\n"
            f"✅ Builds investing discipline — automatic, emotionless\n"
            f"✅ Lower psychological stress — you invest regardless of market level\n"
            f"✅ Ideal for salaried individuals\n"
            f"❌ Slightly lower returns in strongly trending bull markets\n\n"
            f"**Example:** ₹10,000/month SIP for 10 years @ 12%: {fmt_inr(sip_future_value(10000, 0.12, 10))}\n\n"
            f"## Lumpsum Investment\n\n"
            f"**Best for:** Windfall money — bonus, inheritance, matured FD\n\n"
            f"✅ Full corpus in market immediately — maximises compounding time\n"
            f"✅ Outperforms SIP in sustained bull markets\n"
            f"✅ No ongoing commitment needed\n"
            f"❌ High timing risk — investing at a market peak can lead to years of waiting to break even\n"
            f"❌ Psychologically hard if markets fall 20% right after you invest\n\n"
            f"**Example:** ₹12,00,000 lumpsum for 10 years @ 12%: {fmt_inr(1200000 * (1.12**10))}\n\n"
            f"## The Smart Hybrid Approach\n\n"
            f"If you have a lumpsum (bonus/windfall):\n"
            f"1. Park it in a **liquid fund** immediately (earning ~7%)\n"
            f"2. Set up an **STP (Systematic Transfer Plan)** from the liquid fund into equity over 6–12 months\n"
            f"3. This way you get the benefits of both — immediate deployment + cost averaging\n\n"
            f"⚠️ *{rnd_hedge()}*"
        ),
    },
    {
        "users": [
            "ELSS vs PPF — which is better for tax saving?",
            "Should I invest in ELSS or PPF for 80C?",
            "Compare ELSS and PPF for me",
            "PPF vs ELSS — which gives better returns and tax benefits?",
        ],
        "assistant": lambda: (
            f"This is one of the most common tax-planning dilemmas! Here's a thorough comparison:\n\n"
            f"## ELSS vs PPF Comparison\n\n"
            f"| Factor | ELSS | PPF |\n"
            f"|--------|------|-----|\n"
            f"| Full form | Equity Linked Savings Scheme | Public Provident Fund |\n"
            f"| Lock-in | 3 years | 15 years |\n"
            f"| Returns | 12–15% (market-linked) | ~7.1% (government-set) |\n"
            f"| Risk | Market risk (equity) | Risk-free (sovereign) |\n"
            f"| Tax on returns | LTCG 12.5% above ₹1.25L/year | Fully tax-free |\n"
            f"| Section 80C | Up to ₹1.5L deduction | Up to ₹1.5L deduction |\n"
            f"| Partial withdrawal | After 3 years | After 7 years (partial) |\n"
            f"| Who manages | Fund manager (SEBI) | Government (Post Office/Bank) |\n\n"
            f"## Wealth Creation Comparison\n\n"
            f"₹12,500/month (₹1.5L/year) for 15 years:\n"
            f"- ELSS (12% CAGR): {fmt_inr(sip_future_value(12500, 0.12, 15))} (after 12.5% LTCG on gains)\n"
            f"- PPF (7.1%): {fmt_inr(sip_future_value(12500, 0.071, 15))} (fully tax-free)\n\n"
            f"Even accounting for LTCG tax on ELSS gains, ELSS typically creates more wealth over 10+ years.\n\n"
            f"## My Recommendation\n\n"
            f"**Both — split your ₹1.5L/year allocation!**\n\n"
            f"- ₹1,00,000/year in ELSS (higher growth, 3-year liquidity)\n"
            f"- ₹50,000/year in PPF (guaranteed, risk-free, 15-year retirement anchor)\n\n"
            f"This gives you growth potential from ELSS + guaranteed tax-free returns from PPF.\n\n"
            f"**Who should prefer PPF?** Very conservative investors, those near retirement, or anyone who can't tolerate market volatility.\n"
            f"**Who should prefer ELSS?** Younger investors (25-45) with 5+ year horizon who want to maximise wealth.\n\n"
            f"⚠️ *{rnd_hedge()}*"
        ),
    },
    {
        "users": [
            "Active vs passive funds — which is better?",
            "Should I invest in index funds or actively managed funds?",
            "Index fund vs active fund — what do you recommend?",
            "What's better: Nifty 50 index fund or an active large cap fund?",
        ],
        "assistant": lambda: (
            f"This is a debate that's divided the investment world! Here's the honest truth backed by data:\n\n"
            f"## Active vs Passive (Index) Funds\n\n"
            f"| Factor | Active Fund | Index Fund |\n"
            f"|--------|-------------|------------|\n"
            f"| Management | Fund manager picks stocks | Tracks an index (e.g., Nifty 50) |\n"
            f"| Expense ratio | 0.5% – 1.8% | 0.1% – 0.25% |\n"
            f"| Aim | Beat the market | Match the market |\n"
            f"| Risk | Manager risk + market risk | Only market risk |\n"
            f"| Transparency | Holdings disclosed monthly | Always replicates index |\n\n"
            f"## The Data on Large Cap Active Funds\n\n"
            f"According to SPIVA India reports:\n"
            f"- Over 5 years: ~60–70% of active Large Cap funds **underperform** Nifty 50 (after expenses)\n"
            f"- Over 10 years: ~75–80% underperform\n\n"
            f"This is why legendary investor Warren Buffett recommends index funds for most investors!\n\n"
            f"## Where Active Funds Still Win\n\n"
            f"- **Mid Cap funds:** Active managers have more room to add alpha in the less-efficient mid-cap space\n"
            f"- **Small Cap funds:** Very inefficient market — skilled managers can outperform\n"
            f"- **Flexi Cap / Multi Cap:** Dynamic allocation decisions can beat static index\n\n"
            f"## My Recommendation: Core + Satellite\n\n"
            f"**Core (60-70%):** Nifty 50 + Nifty Next 50 Index funds — low cost, reliable market returns\n"
            f"**Satellite (30-40%):** Active Mid Cap + Small Cap funds for alpha generation\n\n"
            f"This combines the best of both worlds — low cost for large caps, active management where it adds value.\n\n"
            f"**Expense ratio impact:** ₹10,000/month for 20 years:\n"
            f"- Index fund (0.2% expense): ~{fmt_inr(sip_future_value(10000, 0.118, 20))}\n"
            f"- Active fund (1.5% expense): ~{fmt_inr(sip_future_value(10000, 0.105, 20))}\n\n"
            f"That difference is the cost of active management — worth it for mid/small cap, less so for large cap.\n\n"
            f"⚠️ *{rnd_hedge()}*"
        ),
    },
    {
        "users": [
            "Mutual funds vs FD — which is better?",
            "Should I put money in FD or mutual funds?",
            "Fixed deposit vs mutual fund — what's the difference?",
            "I've been putting money in FD. Should I switch to mutual funds?",
        ],
        "assistant": lambda: (
            f"This is a very common question — especially for first-time investors who grew up seeing their parents use FDs! Let me give you an honest, complete picture.\n\n"
            f"## Mutual Funds vs Fixed Deposits\n\n"
            f"| Factor | Mutual Funds (Equity) | Fixed Deposit |\n"
            f"|--------|----------------------|---------------|\n"
            f"| Returns | 10–15% CAGR (market-linked) | 6–7.5% (fixed) |\n"
            f"| Risk | Market risk | Virtually zero (up to ₹5L DICGC insured) |\n"
            f"| Inflation beating? | Yes, strongly over 7+ years | Barely (FD rate ≈ inflation rate) |\n"
            f"| Liquidity | T+3 days | Penalty for early withdrawal |\n"
            f"| Tax | LTCG 12.5% on equity gains | Taxed at your income slab |\n"
            f"| Lock-in | None (except ELSS) | Usually 1-5 years |\n"
            f"| Suited for | Long-term wealth building | Short-term, guaranteed returns |\n\n"
            f"## The Inflation Problem with FDs\n\n"
            f"India's average inflation is ~5-6%. FD rates are ~6.5-7.5%. Your **real return** (after inflation and tax) from an FD is often close to ZERO or even negative!\n\n"
            f"₹10,000/month for 15 years:\n"
            f"- FD equivalent (7%): {fmt_inr(sip_future_value(10000, 0.07, 15))}\n"
            f"- Equity Mutual Fund (12%): {fmt_inr(sip_future_value(10000, 0.12, 15))}\n\n"
            f"The difference is enormous — and that's the cost of playing it 'safe' with FDs over the long term.\n\n"
            f"## My Recommendation\n\n"
            f"FDs are NOT bad — they have their place:\n"
            f"- ✅ Emergency fund (3-6 months expenses) — use FD or liquid fund\n"
            f"- ✅ Goals within 1-2 years — FD or debt mutual fund\n"
            f"- ✅ Senior citizens needing fixed income — FD is fine\n\n"
            f"But for **any goal 5+ years away**, equity mutual funds will almost certainly deliver far better real returns.\n\n"
            f"A balanced approach: FD for emergency fund + liquidity, mutual fund SIPs for long-term wealth building.\n\n"
            f"⚠️ *{rnd_hedge()}*"
        ),
    },
    {
        "users": [
            "What is the difference between direct and regular mutual fund plans?",
            "Should I invest in direct or regular plan?",
            "Direct vs regular plan — which is better?",
        ],
        "assistant": lambda: (
            f"This is one of the most impactful decisions you'll make — and most investors don't even know it exists!\n\n"
            f"## Direct vs Regular Plan\n\n"
            f"Every mutual fund scheme in India has two versions:\n\n"
            f"| Factor | Direct Plan | Regular Plan |\n"
            f"|--------|-------------|---------------|\n"
            f"| Distributor commission | None | 0.5% – 1.5% per year |\n"
            f"| Expense ratio | Lower | Higher |\n"
            f"| NAV | Higher (same fund, better returns) | Lower |\n"
            f"| Returns | ~0.5–1% higher per year | Lower |\n"
            f"| Where to buy | MFCentral, Zerodha Coin, Groww | Banks, agents, distributors |\n\n"
            f"## Why This Matters — The Maths\n\n"
            f"₹10,000/month for 20 years:\n"
            f"- Direct Plan (12% CAGR): {fmt_inr(sip_future_value(10000, 0.12, 20))}\n"
            f"- Regular Plan (11% CAGR, 1% lower): {fmt_inr(sip_future_value(10000, 0.11, 20))}\n\n"
            f"That **1% difference** — which goes to the distributor — costs you lakhs over 20 years. This money comes from YOUR corpus.\n\n"
            f"## Should You ALWAYS Choose Direct?\n\n"
            f"**Yes, for most investors.** The exception is if you genuinely need handholding from a SEBI-registered advisor who charges a transparent fee (not commission). In that case, their advice might be worth more than the cost.\n\n"
            f"## How to Invest in Direct Plans\n\n"
            f"- **Zerodha Coin** — direct plans, ₹0 commission\n"
            f"- **MFCentral** — AMFI's official platform\n"
            f"- **Groww** — easy to use, direct plans\n"
            f"- **Paytm Money** — direct plans\n"
            f"- Each AMC's own website (HDFC MF, SBI MF, etc.)\n\n"
            f"**Pro tip:** If you're currently in Regular Plans, you can switch to Direct Plans — it's a redemption + reinvestment (check for exit loads/LTCG implications before switching).\n\n"
            f"⚠️ *{rnd_hedge()}*"
        ),
    },
    {
        "users": [
            "Debt fund vs FD — which is better for short term?",
            "Should I use debt funds or FDs for 2-year goals?",
            "How are debt mutual funds different from fixed deposits?",
        ],
        "assistant": lambda: (
            f"Great question for short-term goal planning! Here's the comparison:\n\n"
            f"## Debt Funds vs FD\n\n"
            f"| Factor | Debt Mutual Fund | Fixed Deposit |\n"
            f"|--------|-----------------|---------------|\n"
            f"| Returns | 6.5–8.5% | 6–7.5% |\n"
            f"| Risk | Low (credit + interest rate risk) | Negligible (DICGC insured up to ₹5L) |\n"
            f"| Liquidity | T+1 to T+3 | Penalty on early exit |\n"
            f"| Taxation | As per income slab (indexed from FY25) | As per income slab |\n"
            f"| TDS | None | 10% TDS if interest > ₹40,000/year |\n"
            f"| Minimum amount | ₹100 | Varies (usually ₹1,000) |\n\n"
            f"## When Debt Funds Win\n\n"
            f"- ✅ Better liquidity — no lock-in, no penalty\n"
            f"- ✅ No TDS deduction (you pay tax only at withdrawal)\n"
            f"- ✅ Can earn slightly higher returns with the right fund\n"
            f"- ✅ Better for parking large amounts (above ₹5L DICGC limit)\n\n"
            f"## When FDs Win\n\n"
            f"- ✅ Absolutely guaranteed returns — no surprises\n"
            f"- ✅ Simpler — no fund selection needed\n"
            f"- ✅ DICGC insured up to ₹5 lakh per bank\n"
            f"- ✅ Senior citizens get higher FD rates (0.25–0.50% extra)\n\n"
            f"## Recommended Debt Funds by Horizon\n\n"
            f"- **< 3 months:** Liquid funds or overnight funds\n"
            f"- **3-12 months:** Ultra short-duration / low-duration debt funds\n"
            f"- **1-3 years:** Short-duration / corporate bond funds\n"
            f"- **3+ years:** Banking & PSU debt / dynamic bond funds\n\n"
            f"For most short-term goals, a combination of FD (for guaranteed portion) + debt fund (for flexible portion) works well.\n\n"
            f"⚠️ *{rnd_hedge()}*"
        ),
    },
    {
        "users": [
            "Growth vs dividend option in mutual funds — which to choose?",
            "Should I pick growth plan or dividend plan?",
            "What's better: growth option or IDCW in mutual funds?",
        ],
        "assistant": lambda: (
            f"Another important decision! After SEBI's 2021 renaming (Dividend → IDCW: Income Distribution cum Capital Withdrawal), let me clarify this:\n\n"
            f"## Growth Option\n\n"
            f"- All profits **stay in the fund** and get reinvested\n"
            f"- NAV grows over time\n"
            f"- Tax only when you **redeem** (sell) — LTCG/STCG applies at redemption\n"
            f"- **Best for wealth accumulation** — compounding is maximised\n\n"
            f"## IDCW (Income Distribution cum Capital Withdrawal)\n\n"
            f"- Fund periodically distributes **part of NAV** as 'dividend'\n"
            f"- This is NOT extra income — it's your own money being returned!\n"
            f"- Taxed at your income slab rate in the year you receive it (TDS 10% above ₹5,000)\n"
            f"- NAV falls by the dividend amount after distribution\n\n"
            f"## The Hidden Truth About IDCW\n\n"
            f"Many investors think IDCW is free money — it's not. When a fund 'pays dividend' of ₹2/unit, the NAV drops by ₹2. "
            f"You're essentially getting your own money back and paying tax on it!\n\n"
            f"## Which to Choose?\n\n"
            f"- **Growth option** — for almost all investors building wealth (under 60)\n"
            f"- **IDCW** — only if you genuinely need regular cash flow (retired, no other income) AND your tax bracket is low (< 20%)\n\n"
            f"For a ₹10,000/month SIP over 15 years, always pick **Growth** — it maximises your corpus by keeping all gains compounding inside the fund.\n\n"
            f"⚠️ *{rnd_hedge()}*"
        ),
    },
]

def gen_comparison(idx: int) -> dict:
    template = random.choice(COMPARISONS)
    user = random.choice(template["users"])
    assistant = template["assistant"]()
    return {"user": user, "assistant": assistant}


# ---------------------------------------------------------------------------
# Master generation loop
# ---------------------------------------------------------------------------

def generate_all():
    examples = []

    print("Generating Category 1: Full Profile Recommendations (2500)...")
    for i in range(2500):
        examples.append(gen_full_profile(i))

    print("Generating Category 2: Income-aware Recommendations (1000)...")
    for i in range(1000):
        examples.append(gen_income_aware(i))

    print("Generating Category 3: Goal-based (1000)...")
    for i in range(1000):
        examples.append(gen_goal_based(i))

    print("Generating Category 4: Age-based Life Stage (800)...")
    for i in range(800):
        examples.append(gen_age_based(i))

    print("Generating Category 5: Conversational Follow-ups (800)...")
    for i in range(800):
        examples.append(gen_conversational_followup(i))

    print("Generating Category 6: Normal Chat / Greetings (800)...")
    for i in range(800):
        examples.append(gen_normal_chat(i))

    print("Generating Category 7: Risk-specific (600)...")
    for i in range(600):
        examples.append(gen_risk_specific(i))

    print("Generating Category 8: Comparison Questions (600)...")
    for i in range(600):
        examples.append(gen_comparison(i))

    print("Generating Category 9: Basic Glossary QA (1000)...")
    
    GLOSSARY_QA = [
        {
            "terms": ["expense ratio", "fund expenses"],
            "users": ["What is an expense ratio?", "Explain expense ratio", "What do you mean by expense ratio in mutual funds?", "How does expense ratio affect my returns?"],
            "assistant": "The expense ratio is the annual fee a mutual fund charges as a percentage of your invested amount. It covers the fund manager's fee, administrative costs, and marketing expenses. A lower expense ratio means more of the returns stay with you. For example, if you invest ₹1 lakh and the expense ratio is 1%, you pay ₹1,000 per year. Over 10-20 years, even a 0.5% difference in expense ratio can result in a massive difference in your final corpus. This is why Index Funds and Direct Plans are highly recommended — they have the lowest expense ratios."
        },
        {
            "terms": ["exit load", "withdrawal fee"],
            "users": ["What is an exit load?", "Do mutual funds charge a fee when I withdraw?", "Explain exit load to me.", "What happens if I sell my mutual funds early?"],
            "assistant": "An exit load is a fee charged by the mutual fund company when you redeem (sell) your units before a specified period, usually 1 year for equity funds. It is designed to discourage short-term trading and encourage long-term investing. Typically, the exit load is 1% of the redemption amount if withdrawn within 365 days. After 1 year, the exit load becomes zero for most equity funds. Always check a fund's exit load period before investing if you think you might need the money soon."
        },
        {
            "terms": ["sip", "systematic investment plan"],
            "users": ["What is a SIP?", "Can you explain how SIP works?", "What does SIP stand for?", "Why should I start a SIP?"],
            "assistant": "SIP stands for Systematic Investment Plan. It is a method of investing a fixed amount of money into a mutual fund at regular intervals, typically every month. It works exactly like a recurring deposit (RD), but the money goes into mutual funds.\n\nSIPs are incredibly powerful because they enforce investing discipline and give you the benefit of 'Rupee Cost Averaging' — you automatically buy more units when the market is down and fewer when the market is up. This removes the need to time the market and is the best wealth-creation strategy for salaried investors."
        },
        {
            "terms": ["nav", "net asset value"],
            "users": ["What is NAV?", "Explain Net Asset Value", "Does a lower NAV mean the fund is cheaper?", "How is NAV calculated?"],
            "assistant": "NAV stands for Net Asset Value. It is the per-unit price of a mutual fund on any given day. It is calculated by taking the total value of all the assets the fund holds, subtracting its liabilities, and dividing by the total number of units issued to investors.\n\n**A very common myth:** A lower NAV does NOT mean the fund is 'cheaper' or better. If you invest ₹10,000 in a fund with an NAV of ₹10, you get 1000 units. If the NAV is ₹100, you get 100 units. If both funds grow by 10%, your ₹10,000 becomes ₹11,000 in both cases. Focus on the fund's percentage growth, not its absolute NAV."
        },
        {
            "terms": ["elss", "tax saving fund"],
            "users": ["What is an ELSS fund?", "How can I save tax with mutual funds?", "Explain ELSS", "What are tax-saving mutual funds?"],
            "assistant": "ELSS stands for Equity Linked Savings Scheme. It is a category of equity mutual funds specifically designed to help you save tax under Section 80C of the Income Tax Act.\n\nWhen you invest in an ELSS fund, you can claim a tax deduction of up to ₹1.5 lakh per financial year. ELSS funds have a mandatory lock-in period of 3 years, which is actually the shortest lock-in among all 80C options (like PPF, NSC, or Tax-Saver FDs). Since they invest heavily in the stock market, they offer the highest potential returns for tax-saving investments over the long term."
        },
        {
            "terms": ["index fund", "passive investing"],
            "users": ["What is an index fund?", "Explain index mutual funds", "Should I invest in index funds?", "What is passive investing?"],
            "assistant": "An index fund is a type of mutual fund that passively tracks a specific stock market index, like the Nifty 50 or Sensex. Instead of a fund manager actively picking stocks to beat the market, the index fund simply buys the exact same 50 companies in the exact same proportions as the Nifty 50 index.\n\nBecause there is no active stock-picking, index funds have incredibly low expense ratios (often around 0.1% to 0.2%). Over a 10-15 year period, most actively managed large-cap funds fail to beat the index, making index funds one of the safest and most reliable ways to invest in large companies."
        },
        {
            "terms": ["compounding", "compound interest"],
            "users": ["What is compounding?", "How does compounding work in mutual funds?", "Explain the power of compounding", "Why is compounding important?"],
            "assistant": "Compounding is the financial equivalent of a snowball rolling down a hill. It happens when the returns you earn on your investment start earning returns of their own. \n\nFor example, if you invest ₹100 and it grows by 10%, you have ₹110. In year two, you earn 10% not just on your original ₹100, but on the new ₹110 — making it ₹121. Over 10, 15, or 20 years, this effect becomes exponential. In mutual funds, this is why starting early is the most important factor in wealth creation. A 25-year-old investing ₹5,000 a month will retire with significantly more money than a 35-year-old investing ₹10,000 a month."
        },
        {
            "terms": ["etf", "exchange traded fund"],
            "users": ["What is an ETF?", "Explain Exchange Traded Funds", "ETF vs Mutual Fund", "Should I buy ETFs?"],
            "assistant": "ETF stands for Exchange-Traded Fund. Like an index fund, an ETF tracks a market index (like the Nifty 50) or a commodity (like Gold). However, unlike regular mutual funds which are bought/sold at the end-of-day NAV, ETFs trade directly on the stock exchange exactly like individual shares. You can buy or sell them at any time during market hours.\n\nTo invest in an ETF, you MUST have a Demat account. They generally have slightly lower expense ratios than index funds, but you may incur brokerage charges when buying/selling."
        },

        {
            "terms": ["stp", "systematic transfer plan"],
            "users": ["What is an STP?", "Explain Systematic Transfer Plan", "How does STP work?", "When should I use STP instead of SIP?"],
            "assistant": "An STP (Systematic Transfer Plan) allows you to transfer a fixed amount of money systematically from one mutual fund scheme (usually a low-risk Liquid or Debt Fund) to another scheme (usually an Equity Fund) at regular intervals.\n\n**When to use STP:** If you receive a large lumpsum (such as an annual bonus, property sale, or inheritance), dumping it all into equity on a single day carries high timing risk. By placing the lumpsum in a liquid fund earning ~6.5-7% and setting up an STP into equity over 6-12 months, your money earns decent returns while benefiting from Rupee Cost Averaging!"
        },
        {
            "terms": ["swp", "systematic withdrawal plan"],
            "users": ["What is an SWP?", "Explain Systematic Withdrawal Plan", "How to get monthly income from mutual funds?", "Is SWP better than FD interest?"],
            "assistant": "An SWP (Systematic Withdrawal Plan) allows you to withdraw a fixed predetermined amount from your mutual fund investments at regular intervals (monthly, quarterly, or yearly) directly into your bank account, while the remaining balance stays invested and continues to compound.\n\n**Key Benefits:**\n1. **Regular Cash Flow:** Excellent for retirees seeking pension-like monthly income.\n2. **High Tax Efficiency:** Unlike FD interest where the entire interest payout is taxed at your income tax slab, with SWP only the capital gains component of each withdrawal is taxed, resulting in dramatically lower taxes!"
        },
        {
            "terms": ["direct vs regular", "regular plan"],
            "users": ["What is the difference between Direct and Regular mutual funds?", "Why are Direct plans better?", "Should I switch to Direct plans?", "What are distributor commissions in regular plans?"],
            "assistant": "Every mutual fund scheme comes in two variants: **Direct Plan** and **Regular Plan**.\n\n• **Regular Plan:** You invest through an intermediary, broker, or bank. The AMC pays them an annual distributor commission (around 0.5% to 1.5% every year) deducted continuously from your NAV.\n• **Direct Plan:** You invest directly with the fund house (via platforms like Zerodha Coin, Groww, or MFCentral) with zero distributor commission.\n\n**The Impact:** Over 15-20 years, that 1% saved in a Direct Plan compounds to lakhs of rupees of extra wealth in your pocket for the exact same portfolio. Always choose Direct Plans!"
        },
        {
            "terms": ["rupee cost averaging", "market timing"],
            "users": ["What is Rupee Cost Averaging?", "How does SIP average my cost?", "Explain rupee cost averaging in simple terms"],
            "assistant": "Rupee Cost Averaging (RCA) is the mathematical advantage built into every SIP. Because you invest a fixed amount of rupees every month (e.g. ₹5,000):\n\n• When market prices fall, the fund's NAV is low, so your ₹5,000 buys **more units**.\n• When market prices rise, the fund's NAV is high, so your ₹5,000 buys **fewer units**.\n\nOver time, your average purchase price per unit is significantly lower than the average market peak, completely eliminating the need to guess or time market highs and lows."
        },
        {
            "terms": ["large cap vs mid cap vs small cap"],
            "users": ["What is market cap in mutual funds?", "Explain large cap, mid cap, and small cap", "Difference between large, mid, and small cap funds"],
            "assistant": "SEBI categorizes Indian companies strictly based on their market capitalization:\n\n1. **Large Cap (Top 1-100 companies):** Blue-chip giants like Reliance, TCS, and HDFC Bank. Stable, resilient during crashes, modest 11-13% expected CAGR.\n2. **Mid Cap (Ranked 101-250):** Fast-growing emerging leaders. Moderately volatile with 13-16% potential CAGR.\n3. **Small Cap (Beyond 250):** High-growth young companies. Extremely volatile with heavy drawdown potential, but highest long-term wealth creation (15-18%+ CAGR) for 7+ year horizons."
        },
        {
            "terms": ["mutual fund taxation", "ltcg", "stcg"],
            "users": ["How are mutual funds taxed in India?", "Explain LTCG and STCG tax on mutual funds", "What is the tax on equity mutual funds?"],
            "assistant": "Mutual fund taxation depends on asset class and holding period:\n\n• **Equity Funds (Holding < 1 year - STCG):** Taxed at 20% flat on gains.\n• **Equity Funds (Holding > 1 year - LTCG):** Profits up to ₹1.25 Lakh in a financial year are **100% Tax-Free**. Gains above ₹1.25 Lakh are taxed at a flat 12.5%.\n• **Debt Funds:** Gains are added to your annual income and taxed at your applicable income tax slab rate."
        },

    ]

    for i in range(1000):
        item = random.choice(GLOSSARY_QA)
        user = random.choice(item["users"])
        # Add random greetings or filler to make it varied
        prefixes = ["", "Hi MentraFiAI, ", "Could you please tell me: ", "I have a question: ", "Hey, ", "Quick question: "]
        user = random.choice(prefixes) + user
        examples.append({"user": user, "assistant": item["assistant"]})

    # shuffle for good measure
    random.shuffle(examples)

    return examples


def main():
    output_path = os.path.abspath(OUTPUT_PATH)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n{'='*60}")
    print("MentraFiAI SFT Dataset Generator v2")
    print(f"{'='*60}\n")

    examples = generate_all()

    print(f"\nSaving {len(examples)} examples to:\n  {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Count per category (we know the order before shuffle)
    counts = {
        "Full Profile Recommendations": 2500,
        "Income-aware Recommendations": 1000,
        "Goal-based": 1000,
        "Age-based Life Stage": 800,
        "Conversational Follow-ups": 800,
        "Normal Chat / Greetings": 800,
        "Risk-specific": 600,
        "Comparison Questions": 600,
        "Basic Glossary QA": 1000,
    }

    print(f"\n{'='*60}")
    print("CATEGORY COUNTS")
    print(f"{'='*60}")
    for cat, count in counts.items():
        print(f"  {cat:<35} {count:>5} examples")
    print(f"  {'-'*42}")
    print(f"  {'TOTAL':<35} {sum(counts.values()):>5} examples")
    print(f"{'='*60}\n")

    # Validate a sample
    print("Sample validation — first example:")
    print(f"  User: {examples[0]['user'][:80]}...")
    print(f"  Asst: {examples[0]['assistant'][:120]}...")
    print()

    # Check avg response length
    avg_len = sum(len(e["assistant"].split()) for e in examples) / len(examples)
    print(f"Average assistant response length: {avg_len:.0f} words")

    min_len = min(len(e["assistant"].split()) for e in examples)
    print(f"Minimum assistant response length: {min_len} words")

    print(f"\n[SUCCESS] Done! Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
