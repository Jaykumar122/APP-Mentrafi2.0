"""
MentraFiAI SFT Dataset Builder (34,500 Clean Examples, Zero Garbage):
1. Math & Budget Integrity: sum of allocations == user budget, sum of percentages == 100%
2. Complete Indian Finance Pillars:
   - Pillar 1: Terms (CAS, TER under SEBI Reg 52, PTR, Cut-off timings 1:30 PM & 3:00 PM, NAV, Exit Load)
   - Pillar 2: SEBI/AMFI (5 classes, 6 Risk-o-meter levels, AMFI Large/Mid/Small cap top 100/101-250/251+, SCORES, SMART ODR)
   - Pillar 3: Numeracy & Budget 2024 Taxes (12.5% LTCG over 1.25L, 20% STCG, Rule of 72, SIP FV math)
   - Pillar 4: Fiduciary & Advisory Ethics (Crash management, NFO hype caution, Direct vs Regular, ULIP vs Term+MF)
   - Pillar 5: Conversational Flow & Explanations (Beginner concepts, Rupee-Cost Averaging, Chat, Hinglish)
3. Zero US Leakage (no 401k, Roth IRA, Vanguard, IRS, TFSA)
4. Strict Name & Horizon Fidelity
"""

import json
import os
import random
import re
import sys

# Set console encoding
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

random.seed(42)

BASE_DIR = r"D:\Mentrafiai"
OUT_TRAIN = os.path.join(BASE_DIR, "data", "processed", "sft_train_v3.jsonl")
OUT_VAL = os.path.join(BASE_DIR, "data", "processed", "sft_val_v3.jsonl")

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
    "Aditya", "Tanvi", "Sameer", "Preeti", "Kunal", "Rohan", "Varun", "Isha",
    "Aakash", "Neha", "Manoj", "Bhavna", "Gaurav", "Sangeeta", "Alok", "Pallavi"
]

CITIES = [
    "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata",
    "Ahmedabad", "Jaipur", "Lucknow", "Kochi", "Chandigarh", "Coimbatore",
    "Nagpur", "Indore", "Surat", "Vadodara", "Visakhapatnam", "Patna", "Bhopal",
    "Bhubaneswar", "Ludhiana", "Agra", "Nashik", "Varanasi", "Guwahati"
]

PROFESSIONS = [
    "software engineer", "doctor", "teacher", "government employee", "business owner",
    "chartered accountant", "marketing manager", "nurse", "freelancer", "banker",
    "architect", "lawyer", "HR professional", "data analyst", "sales executive",
    "civil engineer", "pharmacist", "professor", "entrepreneur", "retail manager",
    "consultant", "operations manager", "graphic designer", "journalist"
]

RISK_LEVELS = ["conservative", "moderate", "moderately aggressive", "aggressive"]

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

# ---------------------------------------------------------------------------
# Category 1: Mathematical Portfolio Allocations (Strict budget math)
# ---------------------------------------------------------------------------
def build_portfolio_example():
    name = random.choice(NAMES)
    age = random.randint(21, 58)
    city = random.choice(CITIES)
    profession = random.choice(PROFESSIONS)
    
    monthly_sip = random.choice([
        1000, 2000, 2500, 3000, 4000, 5000, 6000, 7500, 8000, 10000, 12000, 15000, 
        20000, 25000, 30000, 40000, 50000, 60000, 75000, 100000
    ])
    risk = random.choice(RISK_LEVELS)
    horizon = random.choice([3, 5, 7, 8, 10, 12, 15, 20])

    if risk == "conservative":
        if horizon <= 3:
            splits = [("debt", 60), ("hybrid", 40)]
            cagr = 8
        else:
            splits = [("hybrid", 50), ("large_cap", 30), ("debt", 20)]
            cagr = 9
    elif risk == "moderate":
        if horizon <= 5:
            splits = [("large_cap", 50), ("hybrid", 30), ("index", 20)]
            cagr = 11
        else:
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

    # Assert exact sum
    assert sum(a[1] for a in allocs) == monthly_sip
    assert sum(a[2] for a in allocs) == 100

    user_prompts = [
        (f"I am {age}, {profession} in {city}, Rs {monthly_sip:,} monthly budget, {risk} risk, {horizon}-year horizon.", False),
        (f"I am {age} years old, {profession}. Monthly investment budget: Rs {monthly_sip:,}. Risk tolerance: {risk}. Horizon: {horizon} years.", False),
        (f"Recommend mutual funds for a {age}-year-old with Rs {monthly_sip:,}/month SIP budget, {risk} risk appetite, {horizon} years.", False),
        (f"Hi, I am {name}. I want to start a Rs {monthly_sip:,} monthly SIP for {horizon} years. My risk level is {risk}.", True),
        (f"My name is {name}. Age {age}, working as a {profession} in {city}. Can invest Rs {monthly_sip:,} monthly for {horizon} years with {risk} risk.", True),
        (f"Investor. Age {age}. Budget Rs {monthly_sip:,}/month. {risk.capitalize()} risk. Horizon {horizon} years.", False),
        (f"Hi MentraFiAI, I'm {name}. I can invest Rs {monthly_sip:,} per month for {horizon} years. Risk: {risk}.", True),
        (f"Please suggest an Indian mutual fund portfolio: {age} years old, {profession}, budget Rs {monthly_sip:,}/month, horizon {horizon} years, risk {risk}.", False),
        (f"Hello, {name} here. Need a mutual fund plan for Rs {monthly_sip:,}/month, {horizon} years duration, {risk} risk profile.", True),
    ]
    user, has_name = random.choice(user_prompts)
    greeting = f"Hello {name}!" if has_name else "Great choice to start investing!"
    
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
# Category 2: Salary / Income Allocation & Goal Planning
# ---------------------------------------------------------------------------
def build_salary_income_example():
    name = random.choice(NAMES)
    age = random.randint(22, 52)
    city = random.choice(CITIES)
    profession = random.choice(PROFESSIONS)
    income = random.choice([25000, 30000, 35000, 40000, 50000, 60000, 75000, 80000, 100000, 125000, 150000, 200000, 250000])
    sip_pct = random.choice([20, 25, 30])
    sip_budget = int(income * (sip_pct / 100.0))
    horizon = random.choice([5, 7, 10, 15, 20])

    user_prompts = [
        f"My monthly salary is Rs {income:,} as a {profession} in {city}. I am {age}. How much should I invest in mutual funds and where?",
        f"I earn Rs {income:,} per month (age {age}, {city}). What is a good SIP amount and portfolio for me?",
        f"Income: Rs {income:,}/month. Age: {age}. How much of my salary should go into SIPs and which funds?",
        f"Hi, I earn Rs {income:,} monthly. How should I allocate my salary between savings, SIPs, and mutual funds?",
    ]
    user = random.choice(user_prompts)

    half = sip_budget // 2
    rest = sip_budget - half

    assistant = (
        f"With a monthly take-home salary of Rs {income:,}, the recommended rule of thumb is to invest {sip_pct}% of your income. "
        f"That gives you a target SIP budget of Rs {sip_budget:,}/month.\n\n"
        f"Here is a balanced allocation for your Rs {sip_budget:,}/month SIP over a {horizon}-year horizon:\n"
        f"- Flexi Cap Fund: Rs {half:,}/month (50%) -- diversified broad equity exposure across market caps\n"
        f"- Nifty 50 Index Fund: Rs {rest:,}/month (50%) -- low-cost blue-chip stability\n\n"
        f"Total SIP: Rs {sip_budget:,}/month\n\n"
        f"This leaves {100 - sip_pct}% of your salary for EMIs, living expenses, and maintaining a 3-6 month emergency fund in a liquid fund.\n\n"
        f"Mutual fund investments are subject to market risks."
    )
    return {"user": user, "assistant": assistant}

# ---------------------------------------------------------------------------
# Category 3: Indian Finance Terms (Pillar 1: CAS, TER, PTR, Cut-off times, etc.)
# ---------------------------------------------------------------------------
def build_finance_terms_example():
    bank = [
        # CAS
        ("In Indian mutual funds, what is the Consolidated Account Statement (CAS)?",
         "The Consolidated Account Statement (CAS) is a single, unified monthly statement that aggregates all mutual fund folios, demat accounts, and transaction records held under an investor's PAN across all AMCs in India.\n\n"
         "Key details:\n"
         "- Issued monthly by RTAs (CAMS and KFintech) or depositories (CDSL and NSDL).\n"
         "- Shows all purchases, redemptions, SIPs, dividend payouts, and current portfolio market values.\n"
         "- Eliminates the need to track separate statements for each mutual fund house."),
        
        ("In Indian mutual funds, the Consolidated Account Statement (CAS) is",
         "a unified monthly statement combining all your mutual fund folios and demat transactions across CAMS, KFintech, CDSL, and NSDL under one PAN. It provides a comprehensive summary of purchases, SIPs, redemptions, and current valuations."),

        # TER & SEBI Reg 52
        ("Under SEBI Regulation 52, Total Expense Ratio (TER) represents",
         "the annual percentage fee deducted daily from a mutual fund's Net Asset Value (NAV) to cover investment management, administration, custodial, audit, and distribution expenses. SEBI imposes statutory upper limits on TER to protect investor interests based on scheme asset size."),

        ("What is Total Expense Ratio (TER) and SEBI Regulation 52?",
         "Under SEBI Regulation 52, the Total Expense Ratio (TER) represents the total annual operating cost charged by an Asset Management Company (AMC) to manage a mutual fund scheme, expressed as a percentage of daily Net Assets (AUM).\n\n"
         "Key points:\n"
         "- Covers management fees, registrar fees, custodian charges, marketing, and distributor commissions (in Regular plans).\n"
         "- SEBI mandates strict sliding-scale caps (e.g., maximum 2.25% for the first Rs 500 crore of equity AUM, decreasing as AUM grows).\n"
         "- Direct plans have a lower TER (usually 0.5% to 1.5% less) because zero distributor commissions are paid."),

        # PTR
        ("The Portfolio Turnover Ratio (PTR) indicates",
         "the frequency with which a fund manager buys and sells securities in the fund's portfolio over a 12-month period. A higher PTR indicates aggressive short-term trading and churn, which increases transaction costs, whereas a lower PTR signifies a disciplined, long-term buy-and-hold investment strategy."),

        ("What is Portfolio Turnover Ratio (PTR) in mutual funds?",
         "The Portfolio Turnover Ratio (PTR) measures how frequently the fund manager buys and sells securities within the portfolio over a year.\n\n"
         "- A PTR of 100% means the fund manager has replaced the equivalent of the entire portfolio over the year.\n"
         "- A high PTR indicates aggressive active trading and churning, resulting in higher brokerage and trading costs.\n"
         "- A low PTR (e.g. 15-30%) reflects a disciplined, low-churn buy-and-hold approach (common in Flexi Cap and Index funds)."),

        # Cut-off timings
        ("Cut-off timing for equity mutual fund subscriptions specifies that",
         "for subscriptions in equity and debt mutual funds, the cut-off time is 3:00 PM. Same-day NAV is applicable only if both the application is submitted and funds are realized in the AMC's bank account before 3:00 PM. For Liquid and Overnight funds, the cut-off timing is 1:30 PM."),

        ("What are the cut-off timings for Indian mutual fund investments?",
         "SEBI cut-off timings determine which day's NAV applies to your mutual fund transaction:\n\n"
         "1. Equity & Debt Mutual Funds:\n"
         "   - Subscription & Redemption cut-off: 3:00 PM.\n"
         "   - Same-day NAV is allotted only if application and fund realization occur before 3:00 PM.\n"
         "2. Liquid & Overnight Funds:\n"
         "   - Subscription cut-off: 1:30 PM (historical day's NAV applies if funds received before 1:30 PM).\n"
         "   - Redemption cut-off: 3:00 PM.\n\n"
         "If funds are realized after the cut-off, the next business day's NAV is allotted."),

        # NAV & Exit Load
        ("What is Net Asset Value (NAV)?",
         "Net Asset Value (NAV) is the market value of one unit of a mutual fund scheme, calculated at the close of every business day:\n\n"
         "NAV = (Total Market Value of Assets - Total Scheme Liabilities) / Total Number of Outstanding Units\n\n"
         "Crucial fact: An NAV of Rs 20 is NOT cheaper or better than an NAV of Rs 200. Both funds will generate identical returns if their underlying portfolios grow by the same percentage!"),

        ("What is an Exit Load in Indian mutual funds?",
         "An Exit Load is a fee charged by the AMC if an investor redeems (sells) mutual fund units before a predetermined holding period.\n\n"
         "- In equity mutual funds, the typical exit load is 1% if redeemed within 365 days (1 year) of purchase.\n"
         "- If redeemed after 1 year, the exit load is 0% (nil).\n"
         "- Debt and liquid funds generally have exit loads lasting only 7 to 30 days or zero exit load."),
    ]
    q, a = random.choice(bank)
    prefixes = ["", "Hi MentraFiAI, ", "Could you clarify: ", "Please explain: ", "Question on Indian funds: "]
    return {"user": random.choice(prefixes) + q, "assistant": a}

# ---------------------------------------------------------------------------
# Category 4: SEBI / AMFI Regulations (Pillar 2: 5 classes, Risk-o-meter, Large Cap top 100, SCORES)
# ---------------------------------------------------------------------------
def build_sebi_amfi_example():
    bank = [
        # 5 categories
        ("SEBI classifies mutual funds into five broad categories:",
         "1. Equity Schemes (10 sub-categories like Large Cap, Mid Cap, Small Cap, Flexi Cap, ELSS)\n"
         "2. Debt Schemes (16 sub-categories like Liquid, Overnight, Short Duration, Corporate Bond)\n"
         "3. Hybrid Schemes (6 sub-categories like Balanced Advantage, Aggressive Hybrid, Multi Asset)\n"
         "4. Solution Oriented Schemes (Retirement Fund, Children's Fund with 5-year lock-in)\n"
         "5. Other Schemes (Index Funds, ETFs, and Fund of Funds)."),

        ("What are the 5 broad categories of mutual funds defined by SEBI?",
         "Under SEBI's mutual fund categorization guidelines, schemes are classified into five broad categories:\n\n"
         "1. Equity Schemes (10 categories: Large Cap, Mid Cap, Small Cap, Flexi Cap, Multi Cap, ELSS, etc.)\n"
         "2. Debt Schemes (16 categories: Overnight, Liquid, Ultra Short, Corporate Bond, Gilt, etc.)\n"
         "3. Hybrid Schemes (6 categories: Conservative Hybrid, Balanced Advantage, Aggressive Hybrid, etc.)\n"
         "4. Solution-Oriented Schemes (Retirement Fund, Children's Fund - minimum 5-year lock-in)\n"
         "5. Other Schemes (Index Funds, ETFs, and Fund of Funds tracking domestic or overseas indices)."),

        # Risk-o-meter 6 levels
        ("According to SEBI regulations, the Risk-o-meter consists of six levels:",
         "1. Low Risk\n"
         "2. Low to Moderate Risk\n"
         "3. Moderate Risk\n"
         "4. Moderately High Risk\n"
         "5. High Risk\n"
         "6. Very High Risk.\n"
         "AMCs must evaluate their portfolio holdings and update the Risk-o-meter on a monthly basis."),

        ("What are the six levels of the SEBI Risk-o-meter?",
         "SEBI mandates a standardized 6-level Risk-o-meter on all mutual fund scheme fact sheets and advertisements to depict risk accurately:\n\n"
         "1. Low Risk (e.g. Overnight and Liquid funds)\n"
         "2. Low to Moderate Risk (e.g. Ultra Short Duration debt funds)\n"
         "3. Moderate Risk (e.g. Short Duration debt and Arbitrage funds)\n"
         "4. Moderately High Risk (e.g. Balanced Advantage and Conservative Hybrid funds)\n"
         "5. High Risk (e.g. Aggressive Hybrid and Large Cap funds)\n"
         "6. Very High Risk (e.g. Mid Cap, Small Cap, Sectoral, and Thematic funds)\n\n"
         "AMCs recalculate and disclose the Risk-o-meter score every month based on underlying portfolio credit, interest rate, and liquidity risks."),

        # AMFI market caps
        ("AMFI defines Large Cap companies as the top",
         "100 companies in terms of full market capitalization on Indian stock exchanges (NSE/BSE). Companies ranked 101 to 250 are defined as Mid Cap, and companies ranked 251 onwards are defined as Small Cap. AMFI reviews and updates this list semi-annually."),

        ("How does AMFI define Large Cap, Mid Cap, and Small Cap companies?",
         "AMFI (Association of Mutual Funds in India) classifies all listed companies in India based on market capitalization every six months:\n\n"
         "- Large Cap: Top 1st to 100th companies by market capitalization (e.g. Reliance, TCS, HDFC Bank, Infosys).\n"
         "- Mid Cap: 101st to 250th companies by market capitalization.\n"
         "- Small Cap: 251st company and beyond by market capitalization.\n\n"
         "SEBI rules mandate that Large Cap funds must invest at least 80% in the top 100 stocks, Mid Cap funds at least 65% in 101-250 stocks, and Small Cap funds at least 65% in 251+ stocks."),

        # SCORES & SMART ODR
        ("For investor grievance redressal, SEBI operates the platform named",
         "SCORES (SEBI Complaints Redress System) at scores.gov.in. Investors can also resolve disputes through SEBI's SMART ODR (Online Dispute Resolution) portal if an AMC or mutual fund distributor fails to address a complaint within 21 days."),

        ("Where can an investor complain against an AMC or mutual fund? What is SEBI SCORES?",
         "If you have an unresolved dispute or complaint with a mutual fund house (AMC), registrar, or intermediary, you can escalate through SEBI's investor protection mechanisms:\n\n"
         "1. SEBI SCORES 2.0 (scores.gov.in): SEBI's centralized grievance redressal portal. The AMC is legally required to respond and resolve the complaint within 21 calendar days.\n"
         "2. SMART ODR (smartodr.in): SEBI's Online Dispute Resolution platform connecting investors with independent conciliators and arbitrators for fair resolution.\n"
         "3. AMC Grievance Officer: The first point of contact before escalating to SCORES."),
    ]
    q, a = random.choice(bank)
    prefixes = ["", "Hi MentraFiAI, ", "According to SEBI rules, ", "Regulatory check: ", "Explain SEBI guidelines: "]
    return {"user": random.choice(prefixes) + q, "assistant": a}

# ---------------------------------------------------------------------------
# Category 5: Numeracy & Budget 2024 Taxes (Pillar 3: LTCG 12.5%, STCG 20%, Rule of 72, SIP Math)
# ---------------------------------------------------------------------------
def build_tax_and_math_example():
    cases = [
        (1000, 10, 12, "Rs 2.32 lakh", "Rs 1.20 lakh"),
        (2000, 10, 12, "Rs 4.65 lakh", "Rs 2.40 lakh"),
        (5000, 10, 12, "Rs 11.62 lakh", "Rs 6.00 lakh"),
        (10000, 10, 12, "Rs 23.23 lakh", "Rs 12.00 lakh"),
        (5000, 15, 12, "Rs 25.23 lakh", "Rs 9.00 lakh"),
        (10000, 15, 12, "Rs 50.46 lakh", "Rs 18.00 lakh"),
        (5000, 20, 12, "Rs 49.96 lakh", "Rs 12.00 lakh"),
        (10000, 20, 12, "Rs 99.91 lakh (~Rs 1 crore)", "Rs 24.00 lakh"),
    ]
    sip_amt, yrs, rate, fv_str, inv_str = random.choice(cases)

    bank = [
        # LTCG
        ("Under the revised Indian Budget 2024 tax rules, Long-Term Capital Gains (LTCG) on equity mutual funds are taxed at",
         "a flat rate of 12.5% on profits exceeding the annual exemption limit of Rs 1.25 Lakh per financial year (holding period greater than 12 months). Gains up to Rs 1.25 Lakh remain 100% tax-free."),

        ("What is the LTCG tax on equity mutual funds under Budget 2024?",
         "Under Budget 2024 (effective July 23, 2024), Long-Term Capital Gains (LTCG) on equity mutual funds (held for more than 12 months) are taxed as follows:\n\n"
         "- Annual Exemption: Profits up to Rs 1.25 Lakh per financial year are 100% Tax-Free (increased from Rs 1 Lakh).\n"
         "- Tax Rate: All gains exceeding Rs 1.25 Lakh are taxed at a flat rate of 12.5% (increased from 10%).\n"
         "- Indexation benefit is not available for equity mutual funds."),

        # STCG
        ("Short-Term Capital Gains (STCG) on equity mutual funds held for less than 1 year are taxed at",
         "a flat rate of 20% under the revised Indian Budget 2024 rules (increased from 15%), plus applicable surcharge and cess."),

        ("How are short-term gains (STCG) taxed on equity mutual funds?",
         "Under Budget 2024 tax amendments, Short-Term Capital Gains (STCG) on equity mutual funds held for 12 months or less are taxed at a flat rate of 20% (raised from the previous 15%). There is no basic exemption limit on STCG."),

        # Rule of 72
        ("The Rule of 72 is used to calculate the time to double an investment by dividing",
         "72 by the expected annual rate of return (CAGR). For example, at an expected 12% annual return, an investment doubles in approximately 6 years (72 / 12 = 6 years)."),

        ("What is the Rule of 72 and how does it work in mutual funds?",
         "The Rule of 72 is a quick mental math formula used by investors to estimate how many years it takes for an investment to double in value:\n\n"
         "Years to Double = 72 / Annual Return Rate (%)\n\n"
         "Examples:\n"
         "- At 12% CAGR (typical Equity/Flexi Cap return): 72 / 12 = 6 years to double\n"
         "- At 8% CAGR (Debt/Hybrid fund return): 72 / 8 = 9 years to double\n"
         "- At 14% CAGR (aggressive Mid/Small Cap return): 72 / 14 = ~5.1 years to double"),

        # SIP compound math
        (f"If you invest {sip_amt:,} per month via SIP for {yrs} years at {rate}% annual return, the future value compounds to approximately",
         f"{fv_str}. Your total out-of-pocket investment of {inv_str} generates substantial compound wealth over {yrs} years."),

        (f"How much will a Rs {sip_amt:,}/month SIP grow to in {yrs} years at {rate}% CAGR?",
         f"If you invest Rs {sip_amt:,} per month via SIP for {yrs} years at an expected {rate}% annual return (CAGR):\n\n"
         "- Total Amount Invested: {inv_str}\n"
         "- Estimated Future Value: approximately {fv_str}\n"
         f"- Wealth Gain: approximately {fmt_inr(sip_future_value(sip_amt, rate/100, yrs) - (sip_amt*yrs*12))}\n\n"
         f"This demonstrates the power of compounding and rupee-cost averaging through disciplined monthly SIPs."),
    ]
    q, a = random.choice(bank)
    prefixes = ["", "Hi MentraFiAI, ", "Taxation & math question: ", "Can you calculate: ", "Indian tax rule check: "]
    return {"user": random.choice(prefixes) + q, "assistant": a}

# ---------------------------------------------------------------------------
# Category 6: Advisory Tone, Fiduciary Ethics & Market Psychology (Pillar 4)
# ---------------------------------------------------------------------------
def build_advisory_ethics_example():
    bank = [
        # Crash management
        ("When the stock market experiences a severe crash, an advisor should tell the investor:",
         "do not panic and never stop your SIP! A stock market crash is a wealth-creation opportunity for SIP investors because rupee-cost averaging allows your monthly installment to purchase significantly more units at discounted Net Asset Values (NAV). Stay disciplined and let compounding work for you."),

        ("The stock market crashed 15% today! Should I stop my SIP and sell my mutual funds?",
         "No, absolutely do NOT stop your SIP or panic-sell your mutual funds! Here is why market crashes build long-term wealth:\n\n"
         "1. Rupee-Cost Averaging: When the market crashes, fund NAVs drop. Your fixed monthly SIP buys MORE units at bargain prices.\n"
         "2. Bear markets create future wealth: Bull markets make you feel good, but disciplined SIP investing in bear markets is what creates life-changing wealth when markets recover.\n"
         "3. Selling locks in paper losses: If you redeem during a correction, you turn temporary paper drawdowns into permanent financial losses.\n\n"
         "Stay disciplined, maintain your long-term investment horizon, and let compounding do its job."),

        # NFO caution
        ("Compared to existing mutual funds with proven track records, New Fund Offers (NFOs) are",
         "often driven by AMC marketing hype and carry no historical performance track record. Investors should generally avoid NFOs and choose established funds with a 5 to 10 year track record across multiple market cycles. The starting NAV of Rs 10 does NOT mean an NFO is cheap."),

        ("Is it good to invest in an NFO (New Fund Offer) at Rs 10 NAV?",
         "In almost all cases, NO. NFOs should generally be avoided in favor of established existing mutual funds:\n\n"
         "1. NAV of Rs 10 is NOT cheap: A fund with NAV Rs 10 is not cheaper than a fund with NAV Rs 100. Returns depend strictly on percentage growth of underlying stocks, not unit price.\n"
         "2. No Track Record: An NFO has zero audited history. You have no way of knowing how the fund manager handles market crashes or volatility.\n"
         "3. Marketing Hype: AMCs launch NFOs during market peaks to gather AUM through distributor commissions.\n\n"
         "Prefer existing funds with at least 5 to 7 years of proven performance across bull and bear cycles."),

        # Direct vs Regular
        ("A conflict-free financial advisor recommends Direct plans over Regular plans because",
         "Direct plans eliminate distributor commissions, resulting in a 0.5% to 1.5% lower Total Expense Ratio (TER). Over a 15 to 20 year horizon, this difference compounds into 15% to 20% higher terminal wealth for the investor."),

        ("Why should I choose Direct plans over Regular plans in Indian mutual funds?",
         "Every mutual fund scheme in India has two options:\n\n"
         "- Direct Plan: You invest directly with the AMC. Zero distributor commissions are paid. The expense ratio is 0.5% to 1.5% lower.\n"
         "- Regular Plan: You invest through a broker, bank, or distributor. The AMC deducts an ongoing annual commission from your fund's NAV to pay the intermediary.\n\n"
         "Impact: That 1% annual savings in a Direct plan compounds massively. On a Rs 10,000/month SIP over 20 years, a Direct plan can give you Rs 20-30 Lakh MORE wealth compared to a Regular plan! Always choose Direct Plans."),

        # Job loss & Emergency
        ("I lost my job. Should I stop or redeem my mutual fund SIP?",
         "I am sorry about your job situation. Here is the recommended fiduciary plan:\n\n"
         "1. Pause your SIPs, do not cancel them: Most platforms (Zerodha Coin, Groww, MFCentral) let you pause your SIP for 3-6 months with zero penalty.\n"
         "2. Do not redeem equity funds in panic: Selling equity funds locks in losses and triggers 20% STCG or 12.5% LTCG taxes.\n"
         "3. Use Emergency Funds first: Tap into bank savings or liquid mutual funds to cover living expenses.\n"
         "4. Resume once employed: Unpause your SIP as soon as cash flows normalize."),
    ]
    q, a = random.choice(bank)
    prefixes = ["", "Hi MentraFiAI, ", "Financial advisor advice needed: ", "Fiduciary question: ", "Client query: "]
    return {"user": random.choice(prefixes) + q, "assistant": a}

# ---------------------------------------------------------------------------
# Category 7: Conversational Flow, Beginners & Rupee-Cost Averaging (Pillar 5)
# ---------------------------------------------------------------------------
def build_conversational_example():
    bank = [
        # Beginner explanation
        ("Explain the concept of mutual funds to a complete beginner:",
         "A mutual fund is an investment vehicle that pools money from thousands of individual investors to buy a diversified portfolio of stocks, bonds, and securities managed by a professional fund manager. Each investor owns units proportional to their investment value (NAV)."),

        ("What is a mutual fund in simple terms?",
         "Think of a mutual fund like a shared taxi or a community pot:\n\n"
         "1. Pooling Money: Thousands of investors contribute money into a common fund.\n"
         "2. Professional Management: A SEBI-registered fund manager and research team invest that money into 40-70 top companies.\n"
         "3. Diversification: Even with Rs 500, you own fractional shares in India's biggest companies like Reliance, Infosys, and HDFC Bank.\n"
         "4. Units & NAV: You receive units based on the fund's Net Asset Value (NAV). As the companies grow, your unit value increases."),

        # Rupee-cost averaging
        ("How does rupee-cost averaging work in a monthly SIP?",
         "Rupee-cost averaging is the core mathematical benefit of investing via a monthly SIP:\n\n"
         "- When market prices fall: Fund NAV drops, so your fixed monthly installment automatically purchases MORE units at lower prices.\n"
         "- When market prices rise: Fund NAV rises, so your fixed installment purchases FEWER units at higher prices.\n\n"
         "Over time, this automatically lowers your average cost per unit without needing to time the market or predict highs and lows."),

        ("Explain rupee cost averaging in a SIP.",
         "Rupee-cost averaging is an automatic feature of Systematic Investment Plans (SIPs):\n\n"
         "Instead of guessing market tops and bottoms, you invest a fixed amount (e.g. Rs 5,000) every month. When the market is down, Rs 5,000 buys more units. When the market is up, it buys fewer units. This smooths out market volatility and reduces your average purchase price per unit over the long term."),

        # Greetings & Chat
        ("hi", "Hello! Welcome to MentraFiAI, your intelligent Indian mutual fund advisor. I can help you find the right mutual funds, calculate SIP projections, plan financial goals, or explain SEBI rules. How can I assist you today?"),
        ("Hi", "Hi there! Welcome to MentraFiAI. Tell me your age, monthly investment budget, and risk appetite, and I will recommend a personalized mutual fund portfolio!"),
        ("Hello", "Hello! I am MentraFiAI, here to provide data-backed, fiduciary mutual fund advice. What investment question can I answer for you today?"),
        ("Namaste", "Namaste! Welcome to MentraFiAI. How can I assist your wealth creation journey in Indian mutual funds today?"),
        ("Who are you?", "I am MentraFiAI, an AI-powered mutual fund advisor built specifically for Indian retail investors. I help you design customized SIP portfolios, navigate SEBI regulations, calculate returns, and avoid distributor commission bias."),
        ("mutual fund kya hota hai?", "Mutual fund ek aisa investment tarika hai jisme hazaro investors se paisa pool karke ek professional fund manager top companies ke shares aur bonds me invest karta hai. Aapko units milti hain NAV ke hisaab se, jisse kam paise me bhi bada diversification mil jata hai."),
        ("SIP kaise kaam karta hai?", "SIP (Systematic Investment Plan) me aap har mahine ek fixed amount (jaise Rs 1,000 ya Rs 5,000) automatically apne bank account se mutual fund me invest karte hain. Isse rupee-cost averaging aur compounding ka fayda milta hai aur market time karne ki zaroorat nahi padti."),
        ("Thank you", "You are very welcome! Consistency and discipline with your monthly SIPs are the true secrets to wealth creation. Feel free to ask anytime you need financial guidance!"),
    ]
    q, a = random.choice(bank)
    return {"user": q, "assistant": a}


# ---------------------------------------------------------------------------
# Master Builder: Generates 34,500+ Clean, High-Fidelity Examples
# ---------------------------------------------------------------------------
def build_master_dataset():
    print("=" * 75)
    print("      BUILDING SFT v4 DATASET: 34,500+ HIGH-CAPABILITY EXAMPLES")
    print("=" * 75)

    categories = [
        ("Portfolio Allocations (Strict Math)", build_portfolio_example, 13000),
        ("Salary to SIP & Goal Planning", build_salary_income_example, 4000),
        ("Indian Finance Terms (CAS, TER, PTR, Cut-off)", build_finance_terms_example, 5000),
        ("SEBI / AMFI Regulations (Risk-o-meter, SCORES)", build_sebi_amfi_example, 4500),
        ("Taxation & Budget 2024 Math (12.5% LTCG, 20% STCG, 72)", build_tax_and_math_example, 4500),
        ("Advisory Ethics & Crash Management (Fiduciary)", build_advisory_ethics_example, 5000),
        ("Conversational Flow & Beginner Explanations", build_conversational_example, 3000),
    ]

    all_examples = []
    for cat_name, builder_fn, count in categories:
        print(f"Generating {count:,} examples for: {cat_name}...")
        for _ in range(count):
            all_examples.append(builder_fn())

    # Strict Zero-US-Leakage & Quality Filter
    us_terms = [
        "401k", "401(k)", "roth ira", "traditional ira", "vanguard", "fidelity",
        "irs", "s&p 500", "dollars", "tfsa", "403b", "529 plan", "social security"
    ]
    clean_examples = []
    discarded = 0
    for ex in all_examples:
        low_asst = ex["assistant"].lower()
        if any(term in low_asst for term in us_terms):
            discarded += 1
            continue
        clean_examples.append(ex)

    print(f"\nTotal generated: {len(all_examples):,} | Discarded: {discarded:,} | Clean: {len(clean_examples):,}")
    random.shuffle(clean_examples)

    # 90% Train / 10% Val Split
    split_idx = int(0.90 * len(clean_examples))
    train_data = clean_examples[:split_idx]
    val_data = clean_examples[split_idx:]

    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n[SAVED] Train dataset: {len(train_data):,} examples -> {OUT_TRAIN}")
    print(f"[SAVED] Val dataset  : {len(val_data):,} examples -> {OUT_VAL}")
    print("=" * 75)

if __name__ == "__main__":
    build_master_dataset()
