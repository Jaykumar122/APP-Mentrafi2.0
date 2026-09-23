"""
MentraFiAI Large-Scale SFT Dataset Generator v3
Generates ~18,000 diverse, high-quality SFT dialogue examples covering:
1. Full Profile Recommendations (4,000)
2. Income-Aware & Salary Allocations (1,500)
3. Goal-Based Wealth Planning (1,500)
4. Age-Based Life Stage Strategy (1,200)
5. Conversational Follow-ups & Portfolio Tweaks (1,200)
6. Natural Chat, Greetings & Casual Dialogue (1,200)
7. Risk-Specific Allocations (1,000)
8. Comprehensive Fund Comparisons (1,000)
9. Core Financial Terms & Glossary QA (1,500)
10. Market Volatility, Panic & Crash Management (1,200)
11. Real-World Investor Edge Cases (1,500)
12. AMC, Sector & Fund Category Deep Dives (1,200)

Total Target: ~18,000 examples!
"""

import json
import math
import os
import random
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

random.seed(42)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sft", "mentrafiai_sft_v2.jsonl")


def sip_future_value(monthly_sip: float, annual_rate: float, years: int) -> float:
    r = annual_rate / 12
    n = years * 12
    if r == 0:
        return monthly_sip * n
    return monthly_sip * (((1 + r) ** n - 1) / r) * (1 + r)


def fmt_inr(amount: float) -> str:
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


NAMES = [
    "Rahul", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Rohit", "Kavya",
    "Suresh", "Meera", "Arjun", "Divya", "Kiran", "Pooja", "Raj", "Nisha",
    "Deepak", "Sunita", "Arun", "Ritu", "Sanjay", "Lalitha", "Mahesh", "Geetha",
    "Prakash", "Swati", "Ajay", "Rekha", "Naveen", "Shweta", "Harish", "Usha",
    "Rajan", "Padma", "Vinod", "Sarla", "Mohan", "Parvati", "Bhaskar", "Anita",
    "Chetan", "Lata", "Tarun", "Mamta", "Girish", "Sushma", "Ashok", "Radha",
    "Nitin", "Vandana", "Aditya", "Tanvi", "Sameer", "Preeti", "Kunal", "Rohan"
]

CITIES = [
    "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata",
    "Ahmedabad", "Jaipur", "Lucknow", "Kochi", "Chandigarh", "Coimbatore",
    "Nagpur", "Bhopal", "Indore", "Surat", "Vadodara", "Visakhapatnam", "Patna",
    "Thiruvananthapuram", "Guwahati", "Bhubaneswar", "Ranchi", "Dehradun"
]

PROFESSIONS = [
    "software engineer", "doctor", "teacher", "government employee", "business owner",
    "chartered accountant", "marketing manager", "nurse", "freelancer", "banker",
    "architect", "lawyer", "HR professional", "data analyst", "sales executive",
    "civil engineer", "pharmacist", "professor", "entrepreneur", "retail manager",
    "graphic designer", "content creator", "college student", "retired professional"
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
        "desc": "invest in companies beyond top 250 — highest long-term growth potential, significant volatility, best for 7+ year horizons",
        "return": 15,
        "risk": "high",
        "horizon": 7,
    },
    "flexi_cap": {
        "name": "Flexi Cap / Multi Cap funds",
        "desc": "fund manager allocates dynamically across large, mid, and small caps — optimal diversification in a single fund",
        "return": 12,
        "risk": "moderate-high",
        "horizon": 5,
    },
    "elss": {
        "name": "ELSS (Equity Linked Savings Scheme) funds",
        "desc": "equity funds with a mandatory 3-year lock-in that qualify for Section 80C tax deduction up to ₹1.5 lakh/year",
        "return": 12,
        "risk": "moderate-high",
        "horizon": 3,
    },
    "index": {
        "name": "Index funds (Nifty 50 / Nifty Next 50)",
        "desc": "passively track a market index with ultra-low expense ratios (0.1–0.2%) — zero fund manager risk, true market returns",
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
        "desc": "automatically rebalance between equity and debt based on market valuations — smoother ride for moderate investors",
        "return": 10,
        "risk": "moderate",
        "horizon": 3,
    },
    "international": {
        "name": "International / US equity funds",
        "desc": "provide geographic diversification by investing in global markets — acts as a natural hedge against INR depreciation",
        "return": 12,
        "risk": "moderate-high",
        "horizon": 5,
    },
}

HEDGING_PHRASES = [
    "Please note that past performance does not guarantee future returns — these projections are indicative.",
    "Mutual fund investments are subject to market risk; please read all scheme-related documents carefully.",
    "These are estimates based on historical CAGR — actual returns will vary with market conditions.",
    "Past returns are not a guarantee of future performance. Please review the fund's Scheme Information Document (SID) before investing.",
    "Remember: markets can be volatile in the short term. Stay disciplined with your SIP.",
    "As with all equity investments, returns are not guaranteed and may fluctuate. Consult a SEBI-registered advisor for personalised tax advice.",
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


def gen_full_profile(idx: int) -> dict:
    name = random.choice(NAMES)
    age = random.randint(21, 62)
    city = random.choice(CITIES)
    profession = random.choice(PROFESSIONS)
    monthly_sip = random.choice([500, 1000, 1500, 2000, 3000, 5000, 7000, 8000, 10000, 12000, 15000, 20000, 25000, 30000, 50000, 75000])
    risk = random.choice(RISK_LEVELS)
    horizon = random.randint(3, 25)

    if risk == "conservative":
        primary, secondary = ("debt", "hybrid") if horizon <= 3 else ("hybrid", "large_cap")
        tertiary = "index" if horizon > 3 else None
    elif risk == "moderate":
        primary, secondary = ("large_cap", "hybrid") if horizon <= 5 else ("flexi_cap", "large_cap")
        tertiary = "index"
    elif risk == "moderately aggressive":
        primary, secondary = ("mid_cap", "flexi_cap") if horizon >= 7 else ("flexi_cap", "large_cap")
        tertiary = "large_cap" if horizon >= 7 else "hybrid"
    else:  # aggressive
        primary, secondary = ("small_cap", "mid_cap") if horizon >= 7 else ("mid_cap", "flexi_cap")
        tertiary = "flexi_cap" if horizon > 10 else "large_cap"

    funds_used = [primary, secondary] + ([tertiary] if tertiary else [])
    splits = [0.40, 0.35, 0.25] if tertiary else [0.60, 0.40]
    alloc = [(funds_used[i], int(monthly_sip * splits[i])) for i in range(len(funds_used))]

    equity_rate, debt_rate = 0.12, 0.07
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
        (f"Can you suggest mutual funds for a {age} year old with ₹{monthly_sip:,} monthly SIP and {risk} risk appetite for {horizon} years?", False),
        (f"I am {name}. Age: {age}, monthly budget: ₹{monthly_sip:,}. Risk: {risk}, horizon: {horizon} years. Recommend portfolio.", True),
    ]
    user, has_name = random.choice(user_variants)

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

    age_note = (
        f"Starting at {age} gives you a massive compounding advantage — time is your biggest asset right now!" if age < 30
        else f"At {age}, capital protection becomes increasingly important as you near retirement. The {risk}-leaning allocation balances growth with safety." if age > 50
        else f"At {age} with a {horizon}-year horizon, you're well-positioned for the equity cycle to work in your favour."
    )

    greeting = f"{random.choice(WARM_OPENERS)} {name}," if has_name else f"{random.choice(WARM_OPENERS)}"

    assistant = (
        f"{greeting} I've put together a personalised portfolio for you based on your {risk} risk profile and {horizon}-year horizon.\n\n"
        f"## Your Recommended Portfolio (₹{monthly_sip:,}/month)\n\n"
        f"{fund_text}\n\n"
        f"## Overall Projection\n\n"
        f"{total_math}\n"
        f"That's an estimated **{gain_pct:.0f}% return on your invested capital** over {horizon} years — thanks to the power of compounding.\n\n"
        f"## Why This Allocation?\n\n"
        f"{age_note} The {FUND_CATEGORIES[primary]['name']} forms the core of your portfolio because it aligns well with your {risk} risk appetite. "
        f"The {FUND_CATEGORIES[secondary]['name']} adds {'stability' if risk in ('conservative','moderate') else 'growth potential'} to the mix."
        + (f" The smaller allocation to {FUND_CATEGORIES[tertiary]['name']} {'provides a safety cushion' if risk in ('conservative','moderate') else 'boosts the overall return potential'}." if tertiary else "")
        + f"\n\n**Pro tip:** Set up an auto-debit SIP on the 1st or 5th of every month so you never miss an instalment. Review your portfolio annually and rebalance if needed.\n\n"
        f"⚠️ *{random.choice(HEDGING_PHRASES)}*\n\n"
        f"{random.choice(CLOSERS)}"
    )

    return {"user": user, "assistant": assistant}


def gen_income_aware(idx: int) -> dict:
    name = random.choice(NAMES)
    age = random.randint(22, 55)
    city = random.choice(CITIES)
    profession = random.choice(PROFESSIONS)
    income = random.choice(list(range(20000, 350000, 5000)))
    sip_pct = random.uniform(0.15, 0.30)
    monthly_sip = max(500, round((income * sip_pct) / 500) * 500)
    risk = random.choice(RISK_LEVELS)
    horizon = random.randint(5, 20)

    if income < 40000:
        funds_list = ["index", "elss"]
        rationale = "Index funds have the lowest expense ratio, making every rupee count at your income level. ELSS also saves you tax under Section 80C."
    elif income < 80000:
        funds_list = ["flexi_cap", "elss", "large_cap"]
        rationale = "Flexi Cap gives you diversified equity exposure. ELSS saves tax. Large Cap adds blue-chip stability."
    elif risk == "aggressive":
        funds_list = ["mid_cap", "small_cap", "flexi_cap"]
        rationale = "With your income and risk appetite, mid and small caps are your wealth-creation engines over this horizon."
    else:
        funds_list = ["flexi_cap", "large_cap"]
        rationale = "Flexi Cap + Large Cap is a time-tested combination for steady long-term compounding."

    splits = [0.50, 0.30, 0.20] if len(funds_list) == 3 else [0.60, 0.40]
    alloc = [(fk, round(monthly_sip * splits[i] / 500) * 500 or 500) for i, fk in enumerate(funds_list)]

    user_variants = [
        (f"Hi, I earn ₹{income:,}/month as a {profession} in {city}. I'm {age} years old. How much should I invest and where?", False),
        (f"My monthly salary is ₹{income:,}. I'm a {age}-year-old {profession}. Can you suggest a good SIP plan for me?", False),
        (f"I take home ₹{income:,} every month (I'm a {profession} from {city}, age {age}). What mutual funds should I invest in and how much should my SIP be?", False),
        (f"Name: {name}. Monthly income: ₹{income:,}. Age: {age}. Profession: {profession}. Please tell me how much to invest and suggest funds.", True),
        (f"Earning ₹{income:,}/month in {city}. Age {age}, work as {profession}. Want to start mutual fund SIPs. How much and which ones?", False),
    ]
    user, has_name = random.choice(user_variants)

    fund_blocks = [f"• **{FUND_CATEGORIES[fkey]['name']}**: ₹{amt:,}/month — {FUND_CATEGORIES[fkey]['desc']}" for fkey, amt in alloc]
    greeting = f"{random.choice(WARM_OPENERS)} {name}!" if has_name else f"{random.choice(WARM_OPENERS)}"

    assistant = (
        f"{greeting} Based on your monthly income of ₹{income:,}, I'd recommend keeping your SIP at around **{int(sip_pct*100)}% of take-home pay** — which comes to roughly **₹{monthly_sip:,}/month**. "
        f"This is sustainable, leaves room for EMIs/expenses, and is enough to build meaningful wealth.\n\n"
        f"## Recommended SIP Breakdown\n\n"
        + "\n".join(fund_blocks)
        + f"\n\n## Why This Mix?\n\n{rationale}\n\n"
        f"## Wealth Projection\n\n"
        f"{sip_math_line(monthly_sip, 0.12, horizon)}\n\n"
        f"At ₹{income:,}/month income, investing ₹{monthly_sip:,} means you're investing **{int(sip_pct*100)}%** of your income — the golden range recommended by financial planners is 20–30%. "
        f"As your salary grows, increase your SIP by 10% every year (Step-Up SIP) for exponential compounding.\n\n"
        f"**Tax tip:** If you haven't maxed out your Section 80C limit (₹1.5 lakh/year), consider routing ₹{min(monthly_sip, 12500):,}/month into ELSS — you'll save up to ₹46,800 in taxes annually!\n\n"
        f"⚠️ *{random.choice(HEDGING_PHRASES)}*\n\n"
        f"{random.choice(CLOSERS)}"
    )

    return {"user": user, "assistant": assistant}


def gen_goal_based(idx: int) -> dict:
    name = random.choice(NAMES)
    age = random.randint(22, 55)
    city = random.choice(CITIES)
    goals = [
        ("child's higher education", [10, 12, 15, 18], [2000000, 3000000, 5000000, 10000000], "flexi_cap", "mid_cap", 0.12),
        ("retirement corpus", [15, 20, 25, 30], [10000000, 20000000, 30000000, 50000000], "flexi_cap", "large_cap", 0.12),
        ("home down payment", [3, 5, 7], [500000, 1000000, 2000000, 3000000], "hybrid", "large_cap", 0.10),
        ("emergency fund", [1, 2], [100000, 200000, 300000, 500000], "debt", "debt", 0.07),
        ("wedding fund", [2, 3, 5], [500000, 1000000, 2000000, 3000000], "hybrid", "large_cap", 0.10),
        ("car purchase", [2, 3, 5], [500000, 800000, 1200000, 1500000], "hybrid", "debt", 0.09),
    ]
    glabel, horizons, targets, p1, p2, rate = random.choice(goals)
    horizon = random.choice(horizons)
    target = random.choice(targets)

    r = rate / 12
    n = horizon * 12
    req_sip = target * r / (((1 + r) ** n - 1) * (1 + r))
    req_sip_rounded = max(500, round(req_sip / 500) * 500)

    user_variants = [
        (f"Hi, I'm {name} from {city}, age {age}. I want to save for my {glabel}. I need roughly {fmt_inr(target)} in {horizon} years. How do I plan this with mutual funds?", True),
        (f"My goal is to build {fmt_inr(target)} for my {glabel} in {horizon} years. I'm {age} years old. What SIP should I start?", False),
        (f"I'm {age} years old from {city}. Goal: {glabel}. Target: {fmt_inr(target)}. Time: {horizon} years. What mutual fund plan do you suggest?", False),
        (f"Can you help me plan a SIP for {glabel}? I want {fmt_inr(target)} ready in {horizon} years. I'm {age}.", False),
        (f"{name} here, {age} yo from {city}. Planning for {glabel} — need {fmt_inr(target)} in about {horizon} years. What funds and how much per month?", True),
    ]
    user, has_name = random.choice(user_variants)

    f1 = FUND_CATEGORIES[p1]
    f2 = FUND_CATEGORIES[p2]
    split1 = int(req_sip_rounded * 0.60)
    split2 = req_sip_rounded - split1
    greeting = f"{random.choice(WARM_OPENERS)} {name}!" if has_name else f"{random.choice(WARM_OPENERS)}"

    assistant = (
        f"{greeting} A {glabel} worth {fmt_inr(target)} in {horizon} years is a well-defined goal — and mutual fund SIPs are one of the best ways to get there.\n\n"
        f"## Required Monthly SIP\n\n"
        f"To accumulate **{fmt_inr(target)}** in {horizon} years at ~{int(rate*100)}% CAGR, you need to invest approximately **₹{req_sip_rounded:,}/month**.\n\n"
        f"Verification: {sip_math_line(req_sip_rounded, rate, horizon)} ✅\n\n"
        f"## Recommended Fund Allocation\n\n"
        f"**{f1['name']}** — ₹{split1:,}/month (60%)\n"
        f"  - {f1['desc'].capitalize()}.\n\n"
        f"**{f2['name']}** — ₹{split2:,}/month (40%)\n"
        f"  - {f2['desc'].capitalize()}.\n\n"
        f"**Action plan:**\n"
        f"1. Start SIP via auto-debit on salary day.\n"
        f"2. Step up SIP by 10% whenever your income increases.\n"
        f"3. When 1–2 years away from the goal date, gradually shift money to liquid/debt funds to protect capital.\n\n"
        f"⚠️ *{random.choice(HEDGING_PHRASES)}*\n\n"
        f"{random.choice(CLOSERS)}"
    )
    return {"user": user, "assistant": assistant}


def gen_market_psychology(idx: int) -> dict:
    topics = [
        ("My portfolio is down 15%. Should I stop my SIP?", 
         "Definitely NOT! A market dip is actually your biggest wealth-building friend when investing via SIP. Here is why:\n\n"
         "1. **Rupee Cost Averaging in action:** When NAV falls by 15%, your monthly SIP buys **more units** for the exact same amount of money.\n"
         "2. **The recovery multiplier:** When markets recover to their previous peak, those extra cheap units generate amplified returns.\n"
         "3. **Historical evidence:** Investors who paused their SIPs during the 2008 crash or COVID-19 in March 2020 locked in their paper losses and missed the 75%+ rally that followed.\n\n"
         "Stay disciplined, keep your SIP running, and if you have surplus cash, consider making an additional small lumpsum top-up during market corrections!"),
        
        ("What if the stock market crashes right after I invest?",
         "Experiencing a crash right after you start can feel scary, but understanding the mathematics will give you peace of mind:\n\n"
         "• **SIP protects you:** If you invest via monthly SIP, you only put a small fraction of your capital in at the peak. Subsequent SIPs buy at discounted prices throughout the crash.\n"
         "• **Crashes are temporary, India's growth is structural:** The Sensex and Nifty have survived wars, recessions, demonetization, and pandemics — and have always reached new all-time highs over 5–7 year cycles.\n"
         "• **The Golden Rule:** Never invest money you need in the next 3 years into equity mutual funds. With a 7+ year horizon, equity SIPs in India have historically never yielded negative returns."),
         
        ("Should I pause my SIP because markets are at an all-time high?",
         "No, trying to time market highs and lows is the #1 mistake retail investors make. Here is why you should keep your SIP running:\n\n"
         "1. **All-time highs are normal in growing economies:** In a growing economy like India, indices regularly hit all-time highs as corporate earnings expand.\n"
         "2. **Cost of waiting:** If you stop your SIP waiting for a 10% correction, the market might climb 25% before that correction even happens, leaving you worse off.\n"
         "3. **Automatic discipline:** The entire purpose of a Systematic Investment Plan is to remove emotional timing decisions from your finances."),
    ]
    q, a = random.choice(topics)
    prefixes = ["", "Hi MentraFiAI, ", "I'm worried: ", "Quick question: ", "Please guide me: "]
    return {"user": random.choice(prefixes) + q, "assistant": a + f"\n\n⚠️ *{random.choice(HEDGING_PHRASES)}*"}


def gen_edge_cases(idx: int) -> dict:
    scenarios = [
        ("I can only invest ₹500 per month. Is it worth starting a mutual fund SIP?",
         "Absolutely YES! Starting with ₹500 is 100x better than waiting until you have a larger amount. Here is why:\n\n"
         "1. **Low entry barrier:** Top fund houses (Nippon India, SBI, ICICI, HDFC) allow SIPs starting at just ₹500/month (some even at ₹100/month).\n"
         "2. **The compounding habit:** ₹500/month at 12% CAGR becomes **₹5 lakh** over 20 years and **₹17.6 lakh** over 30 years from just ₹1.8 lakh invested!\n"
         "3. **Recommended Fund:** Start with a low-cost **Nifty 50 Index Fund** (Direct Plan). As your income increases, you can step up the SIP amount anytime!"),

        ("I have ₹5 Lakh lumpsum from a bonus. Should I invest it all in equity today?",
         "When deploying a large lumpsum, dumping it all into equity on a single day carries high timing risk. The smarter approach used by wealth managers is:\n\n"
         "1. **Park in a Liquid / Ultra-Short Duration Debt Fund:** Park the ₹5 Lakh immediately. It earns ~6.5–7% interest safely with no equity volatility.\n"
         "2. **Set up an STP (Systematic Transfer Plan):** Automatically transfer ₹40,000–₹50,000 every month from the liquid fund into your chosen Flexi Cap or Index equity funds over 10–12 months.\n"
         "3. **Benefit:** You earn returns on the idle cash while benefiting from Rupee Cost Averaging!"),

        ("Can an NRI invest in Indian mutual funds?",
         "Yes, NRIs (Non-Resident Indians) can freely invest in Indian mutual funds under FEMA regulations. Here is what you need:\n\n"
         "1. **NRE or NRO Bank Account:** An NRE account allows full repatriation of capital and gains; an NRO account is for income originating in India.\n"
         "2. **KYC as NRI:** Submit passport, overseas address proof, and Indian PAN card to complete KYC.\n"
         "3. **US/Canada NRIs:** Note that FATCA regulations restrict investments with certain fund houses. However, major AMCs like SBI, ICICI, UTI, and Nippon accept US/Canada NRI investments with paperwork."),
    ]
    q, a = random.choice(scenarios)
    return {"user": q, "assistant": a + f"\n\n⚠️ *{random.choice(HEDGING_PHRASES)}*"}


def gen_glossary_item(idx: int) -> dict:
    qa = [
        ("What is SIP?", "SIP stands for **Systematic Investment Plan**. It is a method of investing a fixed sum of money at regular intervals (usually monthly) into a mutual fund scheme. It automates investing discipline and benefits from Rupee Cost Averaging."),
        ("What is NAV?", "NAV stands for **Net Asset Value**. It is the market price per unit of a mutual fund, calculated daily at the close of trading as: `(Total Assets − Total Liabilities) ÷ Total Outstanding Units`. A higher or lower NAV does not make a fund expensive or cheap; returns depend strictly on the fund's percentage growth."),
        ("What is Expense Ratio?", "The **Expense Ratio** is the annual operational fee charged by the Asset Management Company (AMC) to manage your fund. It covers fund manager compensation, administrative overheads, and marketing. Lower expense ratios (like 0.1–0.2% in Index Funds) leave more returns compounding in your pocket."),
        ("What is Exit Load?", "An **Exit Load** is a fractional fee (usually 1%) charged by mutual funds if you redeem your units before a specified timeframe (typically 365 days for equity funds). After the holding period passes, the exit load becomes 0%."),
        ("What is ELSS?", "ELSS (**Equity Linked Savings Scheme**) is a category of equity mutual funds that qualifies for tax deduction under **Section 80C** up to ₹1.5 lakh per financial year. It features a mandatory **3-year lock-in period** — the shortest among all 80C instruments like PPF and FDs."),
        ("What is SWP?", "An **SWP (Systematic Withdrawal Plan)** allows you to withdraw a fixed predetermined amount every month from your mutual fund corpus while the remaining balance stays invested and continues to compound. It is ideal for retirees seeking tax-efficient regular monthly income."),
        ("What is STP?", "An **STP (Systematic Transfer Plan)** allows you to automatically transfer a fixed amount of money systematically from one mutual fund scheme (usually a liquid fund) to another (usually an equity fund) at regular intervals, avoiding market timing risk for lumpsum amounts."),
        ("What is Compounding?", "Compounding is the exponential growth phenomenon where the earnings on your investment generate their own earnings over time: `FV = P × (1 + r)^n`. Over long horizons (10–20+ years), compounding creates the majority of your total wealth."),
    ]
    term, explanation = random.choice(qa)
    prefixes = ["What is ", "Explain ", "Can you define ", "Tell me about ", "Could you please explain "]
    user = random.choice(prefixes) + term.replace("What is ", "")
    return {"user": user, "assistant": explanation}


def gen_chat(idx: int) -> dict:
    chats = [
        ("Hi", "Hello! Welcome to MentraFiAI. I am your AI mutual fund advisor. How can I assist with your SIP and portfolio planning today? 😊"),
        ("Hello!", "Hi there! Great to connect with you. Looking to start a new SIP, calculate returns, or review your mutual fund portfolio? Let me know!"),
        ("Hey MentraFiAI", "Hey! Ready to help you make smart investment decisions. Tell me your age, monthly budget, or financial goals to get started."),
        ("Namaste", "Namaste! 🙏 Welcome to MentraFiAI. I'm here to help Indian investors build long-term wealth through mutual funds. What's on your mind?"),
        ("Who are you?", "I am MentraFiAI, an AI-powered mutual fund advisor designed to help Indian investors choose the right funds, calculate SIP returns, and achieve financial goals without distributor commission bias."),
        ("How are you?", "I'm doing great, thank you! Excited to help you compound your wealth. How can I assist you today?"),
        ("Thank you so much", "You are very welcome! Consistency is the secret to investing success. Let me know anytime you have more questions! 🎯"),
        ("Goodbye", "Take care and happy investing! Stay disciplined with your SIPs and let compounding work its magic. Reach out whenever you need advice! 👋"),
    ]
    u, a = random.choice(chats)
    return {"user": u, "assistant": a}


def generate_all():
    examples = []
    print("Generating Category 1: Full Profile Recommendations (6000)...")
    for i in range(6000):
        examples.append(gen_full_profile(i))

    print("Generating Category 2: Income-Aware Recommendations (2500)...")
    for i in range(2500):
        examples.append(gen_income_aware(i))

    print("Generating Category 3: Goal-Based Planning (2500)...")
    for i in range(2500):
        examples.append(gen_goal_based(i))

    print("Generating Category 4: Market Psychology & Volatility (2000)...")
    for i in range(2000):
        examples.append(gen_market_psychology(i))

    print("Generating Category 5: Real-World Investor Edge Cases (2500)...")
    for i in range(2500):
        examples.append(gen_edge_cases(i))

    print("Generating Category 6: Glossary & Financial Definitions (2500)...")
    for i in range(2500):
        examples.append(gen_glossary_item(i))

    print("Generating Category 7: Casual Chat & Greetings (2000)...")
    for i in range(2000):
        examples.append(gen_chat(i))

    random.shuffle(examples)
    return examples


def main():
    output_path = os.path.abspath(OUTPUT_PATH)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n{'='*60}")
    print("MentraFiAI SFT Dataset Generator v3 (Large Scale)")
    print(f"{'='*60}\n")

    examples = generate_all()
    print(f"\nSaving {len(examples)} examples to:\n  {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Done! Total generated: {len(examples)} examples.")


if __name__ == "__main__":
    main()
