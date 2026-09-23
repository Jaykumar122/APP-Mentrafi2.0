"""
MentraFiAI SFT v5 Master Dataset Builder
Solves:
1. Conversational Graces & Time-Aware Greetings:
   - "Good morning" -> morning salutation + financial offer
   - "Good evening" -> evening salutation
   - "Good night" -> night closing, compounding while you sleep
   - "Hello", "Hi", "Hey", "Namaste", "Pranam", "Kem cho", "Vanakkam"
   - "Thank you", "Thanks a lot", "Bye", "Goodbye", "Take care"
   - "Who are you?", "What can you do?", "Help me"
2. Conceptual & Definitional Integrity (Eliminates Hallucinations):
   - "What is ELSS?" -> Equity Linked Savings Scheme (80C, 3y lock-in, NOT EDGAR!)
   - "What is an index fund?" -> Passive Nifty/Sensex tracking (NOT Unified Solution Fund!)
   - "Direct vs Regular" -> 0.5-1.5% TER savings, zero broker commissions
   - "What is NAV, TER, Exit Load, CAGR, XIRR, Rule of 72"
3. Budget 2024/2026 Taxation (Exact Math):
   - STCG: 20% (< 1 year)
   - LTCG: 12.5% (> 1 year, profits > 1.25 Lakh tax-free)
   - Debt Funds: Taxed at slab rate
4. Strict Portfolio Allocation Math (Balanced distribution so it doesn't drown out chat)
5. Zero US Financial Leakage (No 401k, Roth IRA, Vanguard, IRS)
"""

import os, sys, json, random, re
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

random.seed(42)

BASE_DIR = Path(r"D:\Mentrafiai")
OUT_TRAIN = BASE_DIR / "data" / "processed" / "sft_train_v5.jsonl"
OUT_VAL = BASE_DIR / "data" / "processed" / "sft_val_v5.jsonl"

def sip_future_value(monthly_sip: float, annual_rate: float, years: int) -> float:
    r = annual_rate / 12.0
    n = years * 12
    if r == 0:
        return monthly_sip * n
    return monthly_sip * (((1 + r) ** n - 1) / r) * (1 + r)

def fmt_inr(amount: float) -> str:
    if amount >= 1e7:
        return f"Rs {amount / 1e7:.2f} crore"
    elif amount >= 1e5:
        return f"Rs {amount / 1e5:.2f} lakh"
    else:
        return f"Rs {amount:,.0f}"

NAMES = [
    "Rahul", "Priya", "Amit", "Sneha", "Vikram", "Ananya", "Rohit", "Kavya",
    "Suresh", "Meera", "Arjun", "Divya", "Kiran", "Pooja", "Raj", "Nisha",
    "Deepak", "Sunita", "Arun", "Ritu", "Sanjay", "Lalitha", "Mahesh", "Geetha",
    "Prakash", "Swati", "Ajay", "Rekha", "Naveen", "Shweta", "Harish", "Usha",
    "Aditya", "Tanvi", "Sameer", "Preeti", "Kunal", "Rohan", "Varun", "Isha"
]

CITIES = [
    "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata",
    "Ahmedabad", "Jaipur", "Lucknow", "Kochi", "Chandigarh", "Indore", "Surat"
]

PROFESSIONS = [
    "software engineer", "doctor", "teacher", "business owner", "chartered accountant",
    "marketing manager", "banker", "civil engineer", "data analyst", "sales manager"
]

# ---------------------------------------------------------------------------
# Category 1: Conversational Graces, Greetings & Closings (Pillar 0)
# ---------------------------------------------------------------------------
CONVERSATIONAL_BANK = [
    # Good morning
    ("Good morning", "Good morning! 🌅 Hope you have a wonderful and productive day ahead. How can I assist with your mutual funds, SIPs, or investment goals today?"),
    ("good morning", "Good morning! 🌅 I hope your day is off to a great start. What financial questions or mutual fund plans can I help you explore today?"),
    ("Good Morning MentraFiAI", "Good morning! 🌅 Welcome back. Ready to review your mutual funds, calculate a SIP goal, or explore top-performing funds?"),
    ("good mrng", "Good morning! 🌅 Hope you're having a great day. Tell me your age, monthly investment budget, and risk appetite, and I can suggest a personalized mutual fund portfolio!"),
    ("Good morning! How are you?", "Good morning! 🌅 I am doing great and excited to help you on your wealth creation journey. How can I assist you with Indian mutual funds today?"),

    # Good evening & Afternoon
    ("Good afternoon", "Good afternoon! ☀️ Hope your day is going well. How can I assist your wealth creation journey in Indian mutual funds today?"),
    ("good afternoon", "Good afternoon! What financial goals, SIP queries, or mutual fund comparisons would you like to discuss today?"),
    ("Good evening", "Good evening! 🌆 I hope you had a productive day. Ready to plan your investments or review your mutual fund portfolio?"),
    ("good evening", "Good evening! 🌆 Welcome to MentraFiAI. How can I assist you with mutual fund planning or market insights today?"),

    # Good night
    ("Good night", "Good night! 🌙 Rest well. The beauty of disciplined SIP investing is that compounding works for you 24/7, even while you sleep. Have a peaceful and restful night!"),
    ("good night", "Good night! 🌙 Sleep well. Whenever you are ready to continue building your wealth, I will be right here to assist. Sweet dreams!"),
    ("Good Night MentraFiAI", "Good night! 🌙 Wishing you a relaxing rest. Tomorrow is another opportunity to stay disciplined with your savings and financial freedom journey. Good night!"),
    ("gn", "Good night! 🌙 Rest well and take care. See you next time!"),
    ("good night, see you tomorrow", "Good night! 🌙 Looking forward to continuing our discussion tomorrow. Rest well and happy investing!"),

    # Hello, Hi, Hey, Greetings
    ("hi", "Hello! Welcome to MentraFiAI, your intelligent Indian mutual fund advisor. I can help you find the right mutual funds, calculate SIP projections, plan financial goals, or explain SEBI rules. How can I assist you today?"),
    ("Hi", "Hi there! Welcome to MentraFiAI. Tell me your age, monthly investment budget, and risk appetite, and I will recommend a personalized mutual fund portfolio!"),
    ("hello", "Hello! 👋 Welcome to MentraFiAI, your AI financial co-pilot for Indian mutual funds. What financial goal or investment question would you like to explore today?"),
    ("Hello", "Hello! I am MentraFiAI, here to provide data-backed, fiduciary mutual fund advice. What investment question can I answer for you today?"),
    ("hey", "Hey there! Welcome to MentraFiAI. Looking to start a new SIP, compare funds, or check tax rules? Let me know how I can help!"),
    ("Hey MentraFiAI", "Hey! Great to connect with you. What mutual fund or investment topic can I assist you with today?"),
    ("Namaste", "Namaste! 🙏 Welcome to MentraFiAI. How can I assist your wealth creation journey in Indian mutual funds today?"),
    ("namaste", "Namaste! 🙏 I am your AI mutual fund advisor. Tell me your monthly budget, horizon, and risk tolerance to get started!"),
    ("Pranam", "Pranam! 🙏 Welcome to MentraFiAI. I am here to assist you with unbiased, data-grounded mutual fund guidance."),
    ("Kem cho", "Kem cho! Welcome to MentraFiAI. How can I help you grow your investments and build long-term wealth today?"),
    ("Vanakkam", "Vanakkam! Welcome to MentraFiAI. How can I assist your mutual fund investment journey today?"),

    # Thank you & Closings
    ("Thank you", "You are very welcome! Consistency and discipline with your monthly SIPs are the true secrets to wealth creation. Feel free to ask anytime you need financial guidance!"),
    ("thank you", "You're most welcome! Glad I could help. Let me know whenever you need more insights into mutual funds or SIP planning."),
    ("thanks", "Happy to help! Keep investing consistently, and feel free to reach out whenever you have questions."),
    ("thanks a lot", "You're very welcome! Stay disciplined with your financial goals, and let compounding do the heavy lifting."),
    ("thank you so much", "My pleasure! Wishing you great success on your wealth creation journey. Reach out anytime!"),

    # Goodbyes
    ("bye", "Goodbye! Take care and happy investing! Remember: time in the market beats timing the market. See you soon!"),
    ("goodbye", "Goodbye! Have a great day ahead. I'll be here whenever you're ready to plan your next financial milestone."),
    ("see you later", "See you later! Stay disciplined with your investments and have a wonderful day!"),
    ("take care", "Thank you, you take care as well! Happy investing and talk to you soon."),

    # Persona & Capability Questions
    ("Who are you?", "I am MentraFiAI, an AI mutual fund and wealth advisory assistant built specifically for Indian retail investors. I help you construct personalized SIP portfolios, evaluate fund performance, calculate compounding projections, and navigate SEBI regulations without distributor bias."),
    ("who are you", "I am MentraFiAI, your intelligent AI co-pilot for Indian mutual funds. You can share your age, monthly budget, and risk tolerance, and I will recommend tailored Direct-Growth mutual fund portfolios."),
    ("What can you do?", "Here is what I can do for you:\n1. Personalized Portfolios: Tailored mutual fund allocations based on your age, risk profile, and monthly SIP budget.\n2. Goal Planning: Exact monthly SIP required for retirement, child education, or home buying.\n3. Fund Comparison: Compare expense ratios, rolling returns, and Sharpe ratios across Indian schemes.\n4. Taxation Rules: Current Indian capital gains tax guidance (12.5% LTCG, 20% STCG).\n5. Financial Concepts: Explain SIP, NAV, ELSS, expense ratios, index funds, and rupee-cost averaging."),
    ("how can you help me", "I can help you build wealth through Indian mutual funds! Tell me how much you can invest monthly (e.g. Rs 5,000 or Rs 10,000), your investment horizon (e.g. 5, 10, or 20 years), and your risk tolerance (conservative, moderate, or aggressive), and I will create a personalized investment blueprint for you."),
    ("Are you an advisor?", "I am MentraFiAI, an AI financial advisory co-pilot. I provide data-grounded, fiduciary-aligned mutual fund recommendations and mathematical projections. Mutual fund investments are subject to market risks, and you should always read scheme documents carefully before investing."),
]

def build_conversational_example():
    q, a = random.choice(CONVERSATIONAL_BANK)
    return {"user": q, "assistant": a}


# ---------------------------------------------------------------------------
# Category 2: Crystal-Clear Financial Definitions (Pillar 1 - Zero Hallucination)
# ---------------------------------------------------------------------------
DEFINITIONS_BANK = [
    # ELSS (Must be 100% accurate, NOT EDGAR!)
    ("What is ELSS?",
     "ELSS (Equity Linked Savings Scheme) is a diversified equity mutual fund that qualifies for tax deduction under Section 80C of the Income Tax Act up to Rs 1.5 Lakh per financial year.\n\n"
     "Key features:\n"
     "• Shortest Lock-in: Only 3 years, which is the lowest lock-in among all 80C tax-saving instruments (PPF is 15 years, Tax-Saver FD is 5 years).\n"
     "• Equity Growth: Invests at least 80% in equity shares, offering high wealth compounding potential over the long term.\n"
     "• Taxation: Returns are taxed as equity LTCG at 12.5% for gains exceeding Rs 1.25 Lakh in a financial year."),

    ("Explain ELSS mutual fund and its benefits.",
     "ELSS stands for Equity Linked Savings Scheme. It is an equity-oriented mutual fund designed for dual benefits: tax savings and capital growth.\n\n"
     "1. Tax Deduction: Invest up to Rs 1.50 Lakh annually to save tax under Section 80C (Old Tax Regime).\n"
     "2. 3-Year Lock-in: Each SIP installment is locked for exactly 36 months from the allotment date.\n"
     "3. Wealth Accumulation: Unlike fixed-income tax savers (like PPF or NSC), ELSS participates in equity market growth (typically 12-14% CAGR historically).\n"
     "4. Mandatory Direct Plan: Always invest in Direct-Growth plans to eliminate distributor commissions."),

    # Index Fund (Must be accurate, NOT MSF/Unified Solution!)
    ("What is an index fund and how does it work?",
     "An index fund is a passive mutual fund that replicates a specific market benchmark, such as the Nifty 50 or BSE Sensex, by holding the exact same stocks in the exact same proportions.\n\n"
     "Key advantages:\n"
     "• Low Expense Ratio: Because there is no active stock-picking by a fund manager, index fund TER is ultra-low (typically 0.10% to 0.25%).\n"
     "• Eliminates Manager Bias: Removes the risk of a human fund manager underperforming the broader market.\n"
     "• Long-Term Compounding: Over 10-15 year horizons, broad-market index funds historically beat the majority of actively managed large-cap funds.\n"
     "• Ideal for Beginners: A simple Nifty 50 Index Fund is the best starting point for first-time equity investors."),

    ("What is an index fund?",
     "An index fund is a low-cost passive mutual fund that tracks a market index like the Nifty 50 or BSE Sensex. It buys the same companies in the same proportion as the index, aiming to match market returns rather than beat them. It features ultra-low expense ratios and zero fund manager risk."),

    # Direct vs Regular Plans
    ("What is the difference between direct and regular mutual fund plans?",
     "Every mutual fund scheme in India is offered in two variants:\n\n"
     "1. Direct Plan: You invest directly with the AMC (via platforms like Zerodha Coin, Groww, MFCentral, or AMC websites). Zero intermediary commissions are paid. The Total Expense Ratio (TER) is 0.5% to 1.5% lower every year.\n"
     "2. Regular Plan: You invest through a broker, bank, or distributor. The AMC deducts an ongoing annual commission from your fund's NAV to pay the distributor for the lifetime of your investment.\n\n"
     "Wealth Impact: Over a 20-year horizon with a Rs 10,000 monthly SIP, that 1% commission in a Regular plan costs you over Rs 25 Lakh in lost wealth! Always choose Direct - Growth plans."),

    ("Why should I choose Direct plans over Regular plans?",
     "You should always choose Direct plans because they have a 0.5% to 1.5% lower annual Total Expense Ratio (TER) compared to Regular plans. In Regular plans, this fee is deducted daily from your NAV to pay broker commissions. Over a 15-20 year horizon, choosing Direct plans results in 15% to 25% higher total accumulated wealth!"),

    # NAV
    ("What is NAV in mutual funds?",
     "NAV (Net Asset Value) represents the per-unit market price of a mutual fund scheme. It is calculated daily at the close of market hours:\n\n"
     "NAV = (Total Market Value of Fund Assets - Liabilities) / Total Outstanding Units\n\n"
     "Important Myth Buster: A fund with NAV Rs 20 is NOT cheaper than a fund with NAV Rs 200. Both will deliver the exact same return if their underlying portfolio grows by 15%."),

    # Expense Ratio (TER)
    ("What is Expense Ratio (TER) in mutual funds?",
     "The Total Expense Ratio (TER) is the annual percentage fee charged by the Asset Management Company (AMC) to cover fund management fees, administrative costs, custodial fees, and marketing. It is deducted daily from the fund's NAV. A lower expense ratio means more of your money compounds in your portfolio."),

    # Exit Load
    ("What is exit load in mutual funds?",
     "Exit load is a penalty fee (typically 1%) charged by mutual funds if you redeem (sell) your units before a specified minimum holding period—usually 365 days for equity funds. If you hold your units for more than 1 year, the exit load drops to 0%. Debt and liquid funds usually have exit loads lasting only 7 to 30 days or zero exit load."),

    # Systematic Investment Plan (SIP)
    ("What is SIP and how does it work?",
     "A SIP (Systematic Investment Plan) is a disciplined method of investing a fixed sum of money (e.g. Rs 1,000 or Rs 5,000) into a mutual fund at regular intervals (typically monthly).\n\n"
     "How it works:\n"
     "1. Automated auto-debit from your bank account every month.\n"
     "2. Rupee-Cost Averaging: When markets fall, fund NAV drops, so your fixed amount buys MORE units. When markets rise, it buys fewer units.\n"
     "3. Power of Compounding: Small regular investments grow exponentially over 10-20 years into multi-crore wealth without needing to time the market."),

    ("SIP kya hota hai aur kaise kaam karta hai?",
     "SIP (Systematic Investment Plan) ek aisa disciplined tareeka hai jisme aap har mahine ek fixed amount (jaise Rs 1,000 ya Rs 5,000) apne bank account se mutual fund me invest karte hain.\n"
     "Fayde:\n"
     "1. Rupee-Cost Averaging: Market girne par zyada units milti hain, aur market chadne par kam units, jisse average purchase price kam ho jata hai.\n"
     "2. Compounding: Chhoti-chhoti bachat lambe samay me crore rupaye ka corpus bana sakti hai.\n"
     "3. Market time karne ki koi zaroorat nahi hoti."),

    # SWP (Systematic Withdrawal Plan)
    ("What is SWP and how does it work for retirement?",
     "A SWP (Systematic Withdrawal Plan) allows you to withdraw a fixed sum of money from your mutual fund investments at regular intervals (typically monthly), while the remaining balance continues to stay invested and compound.\n\n"
     "Key benefits for retirees:\n"
     "• Predictable Monthly Income: Acts like a self-generated pension without selling off your entire corpus.\n"
     "• High Tax Efficiency: Unlike Fixed Deposit interest which is taxed at your full slab rate, only the capital gains component of each SWP installment is subject to capital gains tax (12.5% LTCG above Rs 1.25L exemption).\n"
     "• Longevity of Capital: Keeping the core corpus in a Balanced Advantage or Hybrid Fund allows it to outpace inflation."),

    # STP (Systematic Transfer Plan)
    ("What is STP in mutual funds?",
     "An STP (Systematic Transfer Plan) allows you to automatically transfer a fixed amount of money periodically from one mutual fund scheme (usually a low-risk Liquid or Short-Duration Debt fund) to an Equity mutual fund within the same AMC.\n\n"
     "Ideal Use Case: If you receive a large lump sum (e.g. bonus, property sale, inheritance), you park it in a Liquid Fund and run an STP into an equity fund over 6 to 12 months to avoid market timing risk."),

    # Large vs Mid vs Small Cap (SEBI Definition)
    ("What is the difference between Large Cap, Mid Cap, and Small Cap funds under SEBI?",
     "Under SEBI categorization rules, Indian equity mutual funds are classified by market capitalization rank:\n\n"
     "1. Large Cap Funds: Top 100 companies by market capitalization (e.g., Reliance, TCS, HDFC Bank). High stability, lowest equity volatility.\n"
     "2. Mid Cap Funds: 101st to 250th companies by market capitalization. Fast-growing emerging leaders with higher growth potential and moderate volatility.\n"
     "3. Small Cap Funds: 251st company onwards. High growth potential, but high short-term price swings; best suited for 7+ to 10+ year horizons."),

    # Flexi Cap vs Multi Cap
    ("What is the difference between Flexi Cap and Multi Cap funds?",
     "Both invest across large, mid, and small-cap stocks, but have different SEBI allocation mandates:\n\n"
     "• Flexi Cap Fund: The fund manager has 100% dynamic freedom to allocate any percentage across large, mid, and small caps based on prevailing market valuations (min 65% total equity).\n"
     "• Multi Cap Fund: SEBI mandates a strict minimum 25% allocation in Large Caps, 25% in Mid Caps, and 25% in Small Caps at all times, regardless of market conditions."),

    # Rule of 72 & Compounding
    ("What is the Rule of 72 in mutual funds?",
     "The Rule of 72 is a quick mental math formula used by investors to estimate how many years it takes for an investment to double in value:\n\n"
     "Years to Double = 72 / Expected Annual Return Rate (%)\n\n"
     "Examples:\n"
     "• At 12% CAGR (typical Equity/Flexi Cap): 72 / 12 = 6 years to double\n"
     "• At 8% CAGR (Debt/Hybrid Fund): 72 / 8 = 9 years to double\n"
     "• At 14% CAGR (aggressive Mid/Small Cap): 72 / 14 = ~5.1 years to double"),

    # CAGR vs XIRR
    ("What is the difference between CAGR and XIRR in mutual funds?",
     "• CAGR (Compound Annual Growth Rate) measures the annualized smoothed return of a single lump sum investment over multiple years.\n"
     "• XIRR (Extended Internal Rate of Return) is used for SIPs, calculating the true annualized return across multiple cash inflows occurring on different dates throughout your investment tenure."),

    # Market Crash Advice
    ("The stock market crashed today! Should I stop my SIP and sell my mutual funds?",
     "No, absolutely do NOT stop your SIP or panic-sell your mutual funds! Here is why market corrections build long-term wealth:\n\n"
     "1. Rupee-Cost Averaging: When the market crashes, fund NAVs drop. Your fixed monthly SIP buys MORE units at bargain prices.\n"
     "2. Bear markets create future wealth: Bull markets make you feel good, but disciplined SIP investing in bear markets is what creates life-changing wealth when markets recover.\n"
     "3. Selling locks in paper losses: If you redeem during a correction, you turn temporary paper drawdowns into permanent financial losses.\n\n"
     "Stay disciplined, maintain your long-term investment horizon, and let compounding do its job."),

    # NFO Caution
    ("Is it good to invest in a New Fund Offer (NFO) at Rs 10 NAV?",
     "In almost all cases, NO. NFOs should generally be avoided in favor of established existing mutual funds:\n\n"
     "1. NAV of Rs 10 is NOT cheap: A fund with NAV Rs 10 is not cheaper than a fund with NAV Rs 100. Returns depend strictly on percentage growth of underlying stocks, not unit price.\n"
     "2. No Track Record: An NFO has zero audited history. You have no way of knowing how the fund manager handles market crashes or volatility.\n"
     "3. Marketing Hype: AMCs launch NFOs during market peaks to gather AUM through distributor commissions.\n\n"
     "Prefer existing funds with at least 5 to 7 years of proven performance across bull and bear cycles."),
]

def build_definitions_example():
    q, a = random.choice(DEFINITIONS_BANK)
    prefixes = ["", "Hi MentraFiAI, ", "Please explain: ", "Can you define: "]
    return {"user": random.choice(prefixes) + q, "assistant": a}


# ---------------------------------------------------------------------------
# Category 3: Budget 2024 / 2026 Indian Taxation (Pillar 2 - Exact Rules)
# ---------------------------------------------------------------------------
TAX_BANK = [
    ("How is LTCG taxed on mutual funds?",
     "Under the revised Indian Budget tax rules, Long-Term Capital Gains (LTCG) on equity mutual funds are taxed as follows:\n\n"
     "• Holding Period: Units must be held for more than 12 months (1 year).\n"
     "• Annual Exemption Limit: Profits up to Rs 1.25 Lakh per financial year are 100% Tax-Free.\n"
     "• Tax Rate: Profits exceeding Rs 1.25 Lakh are taxed at a flat rate of 12.5%.\n"
     "• Note: Indexation benefit is not available for equity mutual funds."),

    ("What is the STCG tax rate on equity mutual funds?",
     "Under current Indian Budget rules, Short-Term Capital Gains (STCG) on equity mutual funds (units held for 12 months or less) are taxed at a flat rate of 20% (plus applicable cess and surcharge). There is no basic exemption limit for STCG."),

    ("How are debt mutual funds taxed in India?",
     "For debt mutual funds purchased on or after April 1, 2023, capital gains are treated as short-term capital gains and added directly to your taxable income, taxed according to your applicable personal Income Tax Slab rate, regardless of how long you hold the units."),

    ("What is the capital gains tax on mutual funds under Budget 2024 / 2026?",
     "Here is the complete summary of Indian mutual fund taxation:\n\n"
     "1. Equity Mutual Funds:\n"
     "   - Short-Term (< 1 year): Flat 20% STCG.\n"
     "   - Long-Term (> 1 year): First Rs 1.25 Lakh profit per financial year is 100% Tax-Free. Profits above Rs 1.25 Lakh are taxed at a flat 12.5% LTCG.\n"
     "2. Debt Mutual Funds:\n"
     "   - Taxed as per your personal income tax slab rate.\n"
     "3. Hybrid Funds (>65% Equity):\n"
     "   - Taxed identically to Equity Mutual Funds (12.5% LTCG / 20% STCG)."),
]

def build_tax_example():
    q, a = random.choice(TAX_BANK)
    prefixes = ["", "Hi MentraFiAI, ", "Tax question: ", "Can you explain: "]
    return {"user": random.choice(prefixes) + q, "assistant": a}


# ---------------------------------------------------------------------------
# Category 4: Portfolio Recommendations (Balanced Math)
# ---------------------------------------------------------------------------
FUND_NAMES = {
    "large_cap": "Large Cap Equity Fund",
    "mid_cap": "Mid Cap Equity Fund",
    "small_cap": "Small Cap Equity Fund",
    "flexi_cap": "Flexi Cap Equity Fund",
    "elss": "ELSS Tax Saver Fund",
    "index": "Nifty 50 Index Fund",
    "debt": "Short Duration Debt Fund",
    "hybrid": "Balanced Advantage / Hybrid Fund",
}

RISK_LEVELS = ["conservative", "moderate", "moderately aggressive", "aggressive"]

def build_portfolio_example():
    name = random.choice(NAMES)
    age = random.randint(22, 55)
    city = random.choice(CITIES)
    profession = random.choice(PROFESSIONS)
    
    monthly_sip = random.choice([
        1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 30000, 50000
    ])
    risk = random.choice(RISK_LEVELS)
    horizon = random.choice([3, 5, 7, 10, 15, 20])

    if risk == "conservative":
        splits = [("hybrid", 50), ("large_cap", 30), ("debt", 20)]
        cagr = 9
    elif risk == "moderate":
        splits = [("flexi_cap", 50), ("large_cap", 30), ("mid_cap", 20)]
        cagr = 12
    elif risk == "moderately aggressive":
        splits = [("flexi_cap", 40), ("mid_cap", 35), ("large_cap", 25)]
        cagr = 13
    else:  # aggressive
        splits = [("mid_cap", 40), ("small_cap", 35), ("flexi_cap", 25)]
        cagr = 14

    allocs = []
    accum = 0
    for i, (fkey, target_pct) in enumerate(splits):
        if i == len(splits) - 1:
            amt = monthly_sip - accum
        else:
            amt = int(monthly_sip * target_pct / 100)
            accum += amt
        actual_pct = round(amt / monthly_sip * 100)
        allocs.append((FUND_NAMES[fkey], amt, actual_pct))

    user_prompts = [
        f"I am {age}, {profession} in {city}, Rs {monthly_sip:,} monthly budget, {risk} risk, {horizon}-year horizon.",
        f"I am {age} years old. Monthly investment budget: Rs {monthly_sip:,}. Risk tolerance: {risk}. Horizon: {horizon} years.",
        f"Recommend mutual funds for Rs {monthly_sip:,}/month SIP budget, {risk} risk appetite, {horizon} years.",
        f"Hi, I am {name}. I want to start a Rs {monthly_sip:,} monthly SIP for {horizon} years. My risk level is {risk}.",
        f"Please suggest an Indian mutual fund portfolio: budget Rs {monthly_sip:,}/month, horizon {horizon} years, risk {risk}."
    ]
    user = random.choice(user_prompts)
    greeting = f"Hello {name}!" if name in user else "Great choice to start investing!"
    
    alloc_lines = "\n".join([f"- {fname}: Rs {amt:,}/month ({pct}%)" for fname, amt, pct in allocs])
    fv = sip_future_value(monthly_sip, cagr / 100.0, horizon)
    invested = monthly_sip * horizon * 12

    assistant = (
        f"{greeting} For your Rs {monthly_sip:,}/month budget over a {horizon}-year horizon ({risk} risk), "
        f"here is your recommended mutual fund allocation:\n\n"
        f"{alloc_lines}\n\n"
        f"Total SIP: Rs {monthly_sip:,}/month (100% allocated)\n"
        f"Estimated Corpus: At ~{cagr}% expected CAGR, your total investment of {fmt_inr(invested)} "
        f"is projected to grow to approximately {fmt_inr(fv)}.\n\n"
        f"Always invest via Direct Plans on platforms like Zerodha Coin, Groww, or MFCentral to save on distributor commissions. "
        f"Review your portfolio annually.\n\n"
        f"Mutual fund investments are subject to market risks. Please read scheme documents carefully."
    )
    return {"user": user, "assistant": assistant}


# ---------------------------------------------------------------------------
# Master Builder: Generates 32,000+ Clean, High-Fidelity SFT v5 Examples
# ---------------------------------------------------------------------------
def build_master_dataset():
    print("=" * 75)
    print("      BUILDING SFT v5 DATASET: BALANCED CONVERSATIONAL & FINANCIAL AGENT")
    print("=" * 75)

    categories = [
        ("Conversational Graces & Time Greetings", build_conversational_example, 5000),
        ("Financial Definitions (ELSS, Index, NAV, Direct)", build_definitions_example, 6000),
        ("Indian Taxation (12.5% LTCG, 20% STCG)", build_tax_example, 5000),
        ("Portfolio Allocations (Strict Math)", build_portfolio_example, 10000),
    ]

    all_examples = []
    for cat_name, builder_fn, count in categories:
        print(f"Generating {count:,} examples for: {cat_name}...")
        for _ in range(count):
            all_examples.append(builder_fn())

    # Strict Zero-US-Leakage Filter
    us_terms = ["401k", "401(k)", "roth ira", "traditional ira", "vanguard", "fidelity", "irs", "s&p 500", "dollars", "tfsa", "403b", "529 plan", "social security"]
    clean_examples = []
    for ex in all_examples:
        low = ex["assistant"].lower()
        if not any(t in low for t in us_terms):
            clean_examples.append(ex)

    print(f"\nTotal clean examples: {len(clean_examples):,}")
    random.shuffle(clean_examples)

    # 90% Train / 10% Val Split
    split_idx = int(0.90 * len(clean_examples))
    train_data = clean_examples[:split_idx]
    val_data = clean_examples[split_idx:]

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Train dataset: {len(train_data):,} examples -> {OUT_TRAIN}")
    print(f"[SUCCESS] Val dataset  : {len(val_data):,} examples -> {OUT_VAL}")
    print("=" * 75)

if __name__ == "__main__":
    build_master_dataset()
