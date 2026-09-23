"""
Generate rich, domain-specific Indian mutual fund and capital market training data for pretraining.

Why this is needed:
To make the LLM generate authoritative, nuanced, and articulate answers directly from its
own neural network weights (without relying solely on static lookup files), the model needs
deep parametric exposure to the Indian mutual fund landscape:

Topics covered:
1. SEBI Mutual Fund Categorization Framework (Flexi Cap, Multi Cap, Large Cap, Mid Cap, Small Cap, BAF, ELSS, Arbitrage)
2. Indian AMC (Fund House) Profiles & Philosophies (HDFC, SBI, ICICI Pru, Parag Parikh, Quant, Nippon, Axis, Mirae)
3. Direct Comparisons (Active Large Cap vs Nifty 50 Index, Flexi Cap vs Multi Cap, BAF vs Hybrid, ELSS vs PPF)
4. Indian Investor Personas & Real-life Case Studies (Bengaluru tech worker, mid-career family planner, retiree SWP)
5. Market Infrastructure & Mechanics (AMFI, SEBI, CAMS, KFintech, MFCentral, Growth vs IDCW/Dividend, Exit Loads)
6. Indian Macroeconomic Cycles (Nifty 50, Sensex, historical drawdowns, compounding resilience)

All text is synthesized as high-density, authoritative financial prose and natural advisory discussions.
"""

import argparse
import json
import random
from pathlib import Path

OUT_JSONL = Path("data/raw/corpus_v2/indian_market_knowledge.jsonl")
MIB = 1024 ** 2

NAMES = [
    "Aarav", "Ananya", "Rohan", "Sneha", "Karan", "Pooja", "Vikram", "Meera",
    "Aditya", "Divya", "Suresh", "Kavita", "Arjun", "Neha", "Manish", "Priya",
    "Rahul", "Ritu", "Sanjay", "Farah", "Imran", "Nikhil", "Deepak", "Anjali"
]
CITIES = ["Bengaluru", "Mumbai", "Pune", "Hyderabad", "Delhi NCR", "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Kochi"]


# --------------------------------------------------------------------------- #
# 1. SEBI Mutual Fund Scheme Categorization Deep-Dives
# --------------------------------------------------------------------------- #
def gen_sebi_categories(r):
    category = r.choice([
        "flexi_cap", "multi_cap", "large_cap", "mid_cap", "small_cap",
        "balanced_advantage", "arbitrage", "elss", "index_funds"
    ])
    
    if category == "flexi_cap":
        return """Deep Dive: SEBI Categorization and Strategy of Flexi Cap Mutual Funds

In Indian mutual fund investing, Flexi Cap funds represent the single largest and most popular active equity category. Introduced by SEBI in November 2020, the Flexi Cap classification was established to give fund managers complete mandate freedom across market capitalizations.

1. Regulatory Mandate (SEBI Norms):
• Minimum Equity Allocation: A Flexi Cap fund must invest a minimum of 65% of its total assets in equity and equity-related instruments at all times.
• Market Cap Flexibility: Unlike Multi Cap funds, there are zero restrictions on how the fund manager allocates between large-cap, mid-cap, and small-cap stocks. A manager can hold 80% in large caps during an economic slowdown, or dynamically pivot to 50% in mid/small caps during a high-growth economic expansion.
• Tax Status: Because equity exposure remains strictly above 65%, the scheme is taxed as an equity mutual fund (12.5% LTCG after 1 year above ₹1.25L; 20% STCG under 1 year).

2. Portfolio Role & Investor Fit:
Flexi Cap funds serve as the ideal single-fund equity core for retail investors. Rather than forcing an individual to decide whether large-cap or mid-cap valuations are favorable, the fund manager actively rotates capital where risk-adjusted opportunities are strongest.

3. Benchmark & Risk Profile:
Most Flexi Cap schemes benchmark against the Nifty 500 TRI (Total Returns Index). The category exhibits moderate-to-high volatility, making it best suited for an investment horizon of 5 to 7+ years."""

    elif category == "multi_cap":
        return """Deep Dive: SEBI Mandate and Mechanics of Multi Cap Mutual Funds

Multi Cap funds in India operate under a strict regulatory framework designed to ensure genuine diversification across all three segments of the stock market.

1. The Strict 25-25-25 Rule:
Under SEBI circular mandates, a Multi Cap fund must maintain at least 75% total equity allocation, with an immutable minimum exposure to each market cap segment:
• Minimum 25% in Large Cap companies (Top 100 by market capitalization).
• Minimum 25% in Mid Cap companies (101st to 250th by market capitalization).
• Minimum 25% in Small Cap companies (251st company onwards).
• Remaining 25%: Allocated at the fund manager's discretion across any cap or held in cash/debt.

2. Multi Cap vs. Flexi Cap — The Critical Distinction:
In a Flexi Cap fund, a conservative manager might hold 75% in large caps and only 5% in small caps during volatile times. In a Multi Cap fund, the manager is legally prohibited from dropping small-cap exposure below 25%. Consequently, Multi Cap funds maintain a permanently higher risk-beta profile and higher return potential during broad-based bull markets, but experience steeper drawdowns during mid- and small-cap market corrections.

3. Suitability:
Multi Cap schemes are appropriate for aggressive long-term investors with a horizon of at least 7 to 10 years who want guaranteed exposure to India's high-growth emerging mid-and-small enterprises."""

    elif category == "balanced_advantage":
        return """Deep Dive: Balanced Advantage Funds (Dynamic Asset Allocation)

Balanced Advantage Funds (BAFs), formally classified by SEBI as Dynamic Asset Allocation Funds, are designed to eliminate the emotional trap of market timing for conservative and moderate Indian investors.

1. How Dynamic Allocation Works:
BAFs do not maintain a fixed split between equity and debt. Instead, each AMC uses proprietary, back-tested quantitative valuation models based on metrics such as:
• Trailing Price-to-Earnings (P/E) ratio of Nifty 50
• Price-to-Book (P/B) ratio
• Dividend Yield and G-Sec Yield spread
• Trend-following and momentum indicators

When market valuations become stretched (P/E > 25), the quantitative model automatically reduces net equity exposure to 30%-40% and raises debt/cash. When markets crash and valuations become attractive, the model increases net equity exposure to 70%-80%.

2. The Equity Taxation Advantage via Arbitrage:
To retain equity tax status (65%+ equity), BAFs utilize hedged derivative arbitrage positions. For example, a fund might hold:
• 40% unhedged directional equities
• 30% fully hedged equity arbitrage (buying stock + shorting futures)
• 30% fixed income / debt instruments
Total gross equity equals 70% (qualifying for 12.5% LTCG equity tax), but net directional market risk is only 40%!

3. Ideal Use Case:
BAFs are the premier choice for first-time mutual fund investors, retirees setting up an SWP, or anyone with a 3 to 5-year horizon who wants equity-like compounding with significantly muted drawdowns."""

    elif category == "elss":
        return """Deep Dive: Equity Linked Savings Schemes (ELSS Tax Savers)

Equity Linked Savings Schemes (ELSS) are specialized equity mutual funds offering statutory tax deductions under Section 80C of the Indian Income Tax Act.

1. Key Features & Statutory Lock-In:
• Section 80C Deduction: Investors under the Old Tax Regime can claim deductions of up to ₹1,50,000 per financial year, potentially saving up to ₹46,800 in taxes for individuals in the 30% slab.
• 3-Year Lock-In: ELSS has the shortest lock-in period among all 80C tax-saving instruments (PPF has 15 years, Tax-Saving Bank FDs have 5 years, and NPS locks until age 60).
• Note on SIP Lock-In: Each individual SIP instalment is locked for exactly 36 months from the date of that specific instalment.

2. Investment Strategy:
Most ELSS funds operate essentially as diversified Flexi Cap or Multi Cap funds. The statutory 3-year lock-in actually works as a massive behavioral advantage: because investors cannot redeem during market panics, fund managers do not face redemption pressure and can deploy patient, high-conviction long-term bets.

3. Taxation on Exit:
Gains redeemed after the 3-year lock-in are subject to standard Long-Term Capital Gains tax under Section 112A (12.5% on gains exceeding ₹1,25,000 annually)."""

    elif category == "arbitrage":
        return """Deep Dive: Arbitrage Mutual Funds — The Tax-Efficient Cash Alternative

Arbitrage funds exploit price differentials of the same stock between the cash (spot) market and the derivative (futures) market to generate risk-free returns.

1. Mechanism of Arbitrage:
At any given moment, a stock trading at ₹1,000 in the spot market might trade at ₹1,006 in the one-month futures contract due to the cost of carry. The arbitrage fund manager simultaneously:
1. Buys 1,000 shares in the cash market at ₹1,000.
2. Sells (shorts) 1 futures contract at ₹1,006.
On expiry day, the spot price and futures price must converge by law. The fund pockets the ₹6 spread regardless of whether the market goes up, down, or sideways. The equity market risk is completely neutralized.

2. The Massive Tax Arbitrage:
Because the fund holds physical shares, it legally qualifies as an equity mutual fund (>65% equity exposure).
• Bank FD Interest / Debt Funds: Taxed at marginal income tax slab (up to 30% + surcharge).
• Arbitrage Funds: Short-term gains (held < 1 year) are taxed at 20%; long-term gains (held > 1 year) are taxed at 12.5% (with ₹1.25L exemption).

For investors in the 30% tax bracket parking emergency funds or surplus business capital for 3 to 12 months, Arbitrage funds consistently beat traditional savings accounts and short-term bank deposits on a post-tax basis."""

    else:
        return """Deep Dive: Active Large Cap vs. Nifty 50 Index Funds in India

One of the most consequential debates in Indian investing is whether to choose an actively managed Large Cap mutual fund or a passive Nifty 50 / Sensex Index fund.

1. The SPIVA Reality (Active Underperformance):
According to S&P Indices Versus Active (SPIVA) India scorecards, over 80% of active large-cap mutual funds fail to beat their benchmark (Nifty 50 TRI or BSE 100 TRI) over 3, 5, and 10-year periods.

2. Why Active Large Caps Struggle:
• Regulatory Constraints: SEBI strictly limits large-cap funds to the top 100 companies by market capitalization. In this universe, stocks like Reliance, TCS, Infosys, and HDFC Bank are intensely researched by institutional analysts worldwide, leaving minimal information asymmetry for active managers to generate alpha.
• The Drag of Expense Ratios: A Direct Nifty 50 Index Fund charges an expense ratio of 0.05% to 0.15%. An active large-cap fund charges 0.60% to 1.00% (Direct) or 1.8% to 2.2% (Regular). An active manager must generate at least 1% extra gross return every single year just to break even with the low-cost index!

3. Where Active Management Still Shines:
While Large Caps are best indexed through low-cost Nifty 50 / Nifty Next 50 funds, active fund managers continue to generate substantial alpha in Mid Cap and Small Cap spaces, where broad market inefficiencies and under-researched companies reward skilled bottom-up stock pickers."""


# --------------------------------------------------------------------------- #
# 2. Indian AMC (Fund House) Profiles & Philosophies
# --------------------------------------------------------------------------- #
def gen_amc_profiles(r):
    amc = r.choice([
        "ppfas", "hdfc", "sbi", "icici_pru", "quant", "nippon", "mirae"
    ])
    
    if amc == "ppfas":
        return """Fund House Philosophy: Parag Parikh Financial Advisory Services (PPFAS Mutual Fund)

PPFAS AMC is renowned in the Indian mutual fund ecosystem for its patient, value-investing ethos and exceptional alignment of interests with unitholders.

1. Key Differentiators & Philosophy:
• Value-Oriented Buy-and-Hold: PPFAS seeks fundamentally robust businesses with high return on capital, strong management moats, and reasonable valuations. They avoid momentum chasing and exhibit extremely low portfolio turnover.
• Global Diversification: Parag Parikh Flexi Cap Fund famously utilized international equity allocations (holding stakes in global technology leaders like Alphabet and Microsoft alongside Indian bluechips) to provide geographical diversification.
• Skin in the Game: The AMC's founders, fund managers, and employees invest a significant portion of their personal wealth in their own schemes, ensuring direct skin-in-the-game alignment with investors.
• Cash Calls: When market valuations become frothy, PPFAS has historically demonstrated the discipline to hold 10%–20% of assets in cash and arbitrage rather than blindly buying overvalued stocks.

2. Flagship Schemes:
• Parag Parikh Flexi Cap Fund (Direct - Growth)
• Parag Parikh Conservative Hybrid Fund
• Parag Parikh Arbitrage Fund"""

    elif amc == "hdfc":
        return """Fund House Philosophy: HDFC Mutual Fund

HDFC AMC is one of India's oldest, most trusted, and largest asset management companies, managing trillions of rupees across equity and fixed income.

1. Key Differentiators & Philosophy:
• Bottom-Up Fundamental Research: HDFC relies on extensive proprietary research across hundreds of companies. Their investment framework prioritizes competitive advantages, operating cash flows, and capital allocation discipline.
• Multi-Cycle Consistency: Under veteran fund managers like Chirag Setalvad and Prashant Jain (formerly), HDFC funds are structured to withstand full 5 to 7-year market cycles rather than optimizing for short-term quarterly performance charts.
• Comprehensive Product Suite: Strong performance across Flexi Cap, Mid Cap, and balanced advantage categories.

2. Flagship Schemes:
• HDFC Flexi Cap Fund (Direct - Growth)
• HDFC Mid-Cap Opportunities Fund
• HDFC Balanced Advantage Fund
• HDFC Small Cap Fund"""

    elif amc == "quant":
        return """Fund House Philosophy: Quant Mutual Fund

Quant AMC has disrupted the Indian mutual fund industry through its proprietary, rule-based quantitative investment framework known as the VLRT framework.

1. The VLRT Quantitative Framework:
Unlike traditional fundamental analysis, Quant combines four multidimensional pillars:
• Valuation Analytics: Understanding intrinsic value and relative multiples.
• Liquidity Analytics: Tracking institutional money flows and domestic/FII liquidity.
• Risk Appetite: Gauging sentiment, fear, and greed metrics across asset classes.
• Timing / Momentum: Technical cycles and multi-timeframe price action to identify turning points.

2. Operational Characteristics:
• High Portfolio Turnover: Quant funds frequently rotate across sectors and stocks based on mathematical signals, resulting in higher portfolio turnover than traditional buy-and-hold managers.
• Agility and Aggressive Alpha: By capturing sharp sectoral rotation (e.g., rotating from IT into metals, defense, or PSU banks before market rallies), Quant funds have generated eye-catching alpha in recent market cycles.
• Best Fit: Aggressive, high-risk investors looking for actively managed alpha who can tolerate higher volatility."""

    elif amc == "icici_pru":
        return """Fund House Philosophy: ICICI Prudential Mutual Fund

ICICI Prudential AMC is recognized as one of the pioneers of counter-cyclical and asset allocation investing in India, led for over a decade by CIO S. Naren.

1. Key Differentiators & Philosophy:
• Counter-Cyclical Investing: ICICI Pru is known for buying unloved, undervalued sectors when market sentiment is overly pessimistic, and trimming exposure when retail enthusiasm reaches euphoric levels.
• Asset Allocation Leadership: ICICI Prudential Balanced Advantage Fund pioneered the dynamic use of price-to-book (P/B) models to mechanically manage market drawdowns.
• Risk Management Over Alpha Chasing: Heavy emphasis on capital preservation and downside risk containment across market regimes.

2. Flagship Schemes:
• ICICI Prudential Balanced Advantage Fund (Direct - Growth)
• ICICI Prudential Bluechip Fund
• ICICI Prudential Value Discovery Fund"""

    else:
        return """Fund House Philosophy: Nippon India Mutual Fund

Formerly Reliance Mutual Fund, Nippon India AMC is backed by Nippon Life Insurance, one of the world's largest financial conglomerates.

1. Key Differentiators & Philosophy:
• Dominance in Small Cap and ETF Space: Nippon India manages the single largest and most liquid Small Cap fund in the country (Nippon India Small Cap Fund) and boasts the deepest lineup of Index ETFs on the NSE.
• Broad Pan-Indian Distribution: Extensive reach across Tier 2, Tier 3, and rural Indian cities, democratizing systematic SIP participation.
• High Liquidity and Market Execution: Unmatched scale and execution speed in mid- and small-cap stocks."""


# --------------------------------------------------------------------------- #
# 3. Real-World Indian Investor Personas & Advisory Dialogues
# --------------------------------------------------------------------------- #
def gen_investor_personas(r):
    persona = r.choice(["young_it", "family_planner", "nri_investor", "retiree_swp"])
    name = r.choice(NAMES)
    city = r.choice(CITIES)
    
    if persona == "young_it":
        salary = r.choice([90000, 120000, 150000, 180000, 220000])
        sip = r.choice([25000, 35000, 50000, 60000])
        return f"""User: I am a 25-year-old software engineer in {city} earning ₹{salary:,}/month take-home. I have zero debt and want to start investing ₹{sip:,}/month. How should I allocate this for long-term wealth creation?
Assistant: Hello {name}! At 25 in {city} with a take-home salary of ₹{salary:,} and no liabilities, you possess the single greatest asset an investor can have: **an extensive time horizon of 15 to 20+ years**.

Before deploying your ₹{sip:,}/month SIP, ensure your defensive foundations are set:
1. Health Insurance: A standalone base health policy of ₹10L–₹15L (don't rely solely on corporate employer coverage).
2. Emergency Fund: ₹{salary*4:,} parked in a Liquid Fund or high-yield savings account.

Recommended Growth SIP Portfolio (₹{sip:,}/month, Aggressive Profile):
• 40% in a Flexi Cap Fund (₹{int(sip*0.4):,}/month) — Acts as your all-weather wealth-building engine.
• 35% in a Mid Cap Fund (₹{int(sip*0.35):,}/month) — Captures India's expanding mid-tier business growth.
• 25% in a Small Cap Fund (₹{int(sip*0.25):,}/month) — High-alpha potential over a 10+ year timeframe.

Execution Rule:
Always opt for **Direct Plans (Growth Option)**. Avoid regular plans to save ~1% annually. Implement an annual **10% Step-Up SIP** as your compensation grows; this simple habit will compound your portfolio significantly faster."""

    elif persona == "family_planner":
        sip = r.choice([20000, 30000, 40000, 50000])
        child_age = r.choice([3, 5, 7])
        college_yrs = 18 - child_age
        return f"""User: I am 36, married, living in {city}. I have a {child_age}-year-old child and want to invest ₹{sip:,} monthly specifically for their college education in {college_yrs} years. What strategy should I use?
Assistant: Planning {college_yrs} years in advance for higher education is a proactive and disciplined approach, {name}! College tuition in India and abroad inflates at roughly 8% to 10% annually, meaning standard fixed deposits cannot beat education inflation.

Recommended Dual-Phase Goal Strategy:

Phase 1: Accumulation Phase (Years 1 to {college_yrs - 3}):
With a {college_yrs}-year runway, the portfolio should compound primarily in equities:
• 50% Flexi Cap Fund (₹{int(sip*0.5):,}/month)
• 30% Large & Mid Cap Fund (₹{int(sip*0.3):,}/month)
• 20% Aggressive Hybrid Fund (₹{int(sip*0.2):,}/month)

Phase 2: Systematic De-Risking (Final 3 Years before College):
Never leave educational funds in pure equity right before the admission fees are due!
When your child turns 15 (3 years before matriculation):
• Set up an automated Systematic Transfer Plan (STP) to gradually move accumulated equity units into a high-safety Short Duration Debt Fund or Banking & PSU Debt Fund.
• By the time your child is 17, 100% of the target college corpus will be insulated from any sudden stock market crashes."""

    elif persona == "retiree_swp":
        corpus = r.choice([10000000, 15000000, 20000000, 25000000])
        swp = r.choice([60000, 80000, 100000, 125000])
        return f"""User: I am retiring next month at age 60 in {city} with a provident fund and gratuity corpus of ₹{corpus//10000000} Crore. I need ₹{swp:,}/month to cover household living expenses. How do I structure this safely?
Assistant: Congratulations on reaching your retirement milestone, {name}! 

The goal at age 60 is **capital preservation, regular monthly cashflow, and inflation protection**, not aggressive speculation.

The 3-Bucket Retirement Blueprint:

Bucket 1: Immediate Liquidity (Years 1 to 3 — 15% of Corpus):
• Allocate ₹{int(corpus*0.15):,} in Arbitrage Funds and Liquid Funds.
• Set up your automated monthly SWP of ₹{swp:,} directly from this bucket. Because this bucket carries zero equity market risk, you never have to worry about selling stocks during a market dip.

Bucket 2: Stability and Yield (Years 4 to 8 — 45% of Corpus):
• Allocate ₹{int(corpus*0.45):,} across Balanced Advantage Funds (BAFs) and High-Quality Corporate Bond Funds.
• This bucket generates 7.5%–9% annualized returns, periodically refilling Bucket 1 as liquidity is drawn down.

Bucket 3: Long-Term Growth (Years 9+ — 40% of Corpus):
• Allocate ₹{int(corpus*0.40):,} in conservative Large Cap / Flexi Cap Direct funds.
• This bucket combats inflation over your 20 to 25-year retirement horizon, ensuring your purchasing power does not diminish as medical and lifestyle costs rise."""

    else:
        return f"""User: I am an NRI based in Dubai/Singapore. Can I invest in Indian mutual funds, and what are the KYC requirements?
Assistant: Yes, absolutely {name}! Non-Resident Indians (NRIs) can seamlessly invest in Indian mutual funds on a repatriable or non-repatriable basis.

Key Guidelines for NRIs:
1. Bank Account Requirements:
   • NRE (Non-Resident External) Account: Investments made through an NRE account are fully repatriable (both principal and profits can be remitted abroad in foreign currency without restriction).
   • NRO (Non-Resident Ordinary) Account: Suitable for income originating in India (rental income, dividends). Repatriation is subject to Form 15CA/15CB documentation up to USD 1 Million per financial year.

2. In-Person / Digital KYC:
   NRIs must complete one-time Mutual Fund KYC with foreign address proof, overseas passport copy, valid visa/work permit, and PAN card. Most fund platforms (MFCentral, CAMS, KFintech) support digital KYC.

3. US / Canada Restrictions:
   If you reside in the USA or Canada, FATCA regulations restrict access to certain fund houses. However, major AMCs like ICICI Prudential, Nippon India, and SBI accept investments from US/Canada NRIs under specific compliance conditions."""


# --------------------------------------------------------------------------- #
# 4. Market Mechanics, Growth vs IDCW, and Execution Best Practices
# --------------------------------------------------------------------------- #
def gen_market_mechanics(r):
    topic = r.choice(["growth_vs_idcw", "exit_loads", "mfcentral", "stepup_power"])
    name = r.choice(NAMES)
    
    if topic == "growth_vs_idcw":
        return f"""Analysis: Growth Option vs. IDCW (Dividend) Option in Mutual Funds

When investing in any Indian mutual fund scheme, you must choose between two plan options: **Growth** or **IDCW** (Income Distribution cum Capital Withdrawal, formerly known as the Dividend option).

1. The Tax Trap of IDCW:
Prior to 2020, dividend distribution tax was paid by the fund. Today, under the current Income Tax Act:
• All IDCW payouts are added directly to the investor's taxable income and taxed at their marginal income tax slab (up to 30% + surcharge/cess)!
• If you receive ₹1,00,000 in IDCW and sit in the 30% slab, ₹31,200 is immediately lost to taxes every single year.

2. Why the Growth Option is Strictly Superior:
In the Growth option, no dividends are paid out. All profits, capital gains, and underlying stock dividends are automatically reinvested into the fund's NAV.
• Tax Deferral: You pay zero tax as long as you remain invested!
• Compounding Advantage: Your pre-tax money compounds uninterrupted year after year.
• Favorable Capital Gains: When you eventually redeem units after 1 year, your gains are taxed at the concessional 12.5% LTCG rate (with an annual ₹1.25L tax-free exemption), rather than your high 30% income slab.

Unless you are a senior citizen in the zero-tax bracket requiring mandatory monthly income, **always select the Growth option**."""

    elif topic == "exit_loads":
        return f"""Understanding Exit Loads in Indian Mutual Funds

An Exit Load is a fractional fee charged by an asset management company if an investor redeems their units before a predetermined holding threshold.

1. Purpose of Exit Loads:
Exit loads are not intended as revenue for the AMC. Instead, they protect long-term unitholders by discouraging short-term churning and penalizing hot money that disrupts fund liquidity. The collected exit load is credited back into the scheme's assets.

2. Typical Exit Load Structures:
• Equity Funds (Flexi Cap, Mid Cap, Large Cap): Commonly charge a 1.0% exit load if redeemed within 365 days (1 year) of investment. Redemptions after 365 days have zero exit load.
• Arbitrage Funds: Typically charge 0.25% to 0.50% if redeemed within 15 to 30 days; zero exit load thereafter.
• Liquid Funds: Use a graded 7-day exit load (0.0070% on Day 1, decreasing daily to 0.0045% on Day 6, and exactly 0% on Day 7).
• Overnight Funds: Zero exit load from Day 1.

3. FIFO Application:
Mutual fund redemptions follow the First-In, First-Out (FIFO) accounting rule. The units bought earliest are deemed to be sold first, ensuring you clear exit load thresholds systematically."""

    elif topic == "mfcentral":
        return f"""Platform Mechanics: How to Manage All Indian Mutual Funds in One Place

Many Indian retail investors accidentally fragment their portfolios by purchasing funds across multiple apps (like Groww, Zerodha Coin, Indmoney, and direct AMC websites), making tracking and service requests cumbersome.

1. The Official Unified Solution — MFCentral:
MFCentral (mfcentral.com) is the official joint initiative created by India's two registered Registrar and Transfer Agents (RTAs) — **CAMS and KFintech** — under the directive of SEBI.

2. Unique Advantages:
• Single Consolidated View: Automatically discovers every mutual fund folio linked to your PAN card across all 44 AMCs in India, regardless of where or when you purchased them.
• Non-Commercial: Free of ads, marketing calls, and cross-selling.
• Unified Non-Financial Service Requests: Update your registered mobile number, email address, bank mandate change, nominee addition, or address correction across all mutual funds simultaneously with a single OTP-verified request.
• Direct Plan Execution: Allows seamless redemption and investment in direct plans with zero distributor intermediation."""

    else:
        return f"""The Wealth Multiplier: Why a 10% Annual Step-Up SIP Outperforms

A regular SIP keeps the monthly investment amount constant. A Step-Up SIP (or Top-Up SIP) automatically increases your contribution by a fixed percentage (typically 10%) once every 12 months to match your annual career salary increments.

The Mathematical Contrast:
Investor A starts a flat ₹10,000/month SIP for 20 years at 12% CAGR:
• Total Contributed: ₹24.0 Lakh
• Final Accumulated Wealth: ~₹99.9 Lakh (~₹1.0 Crore)

Investor B starts the same ₹10,000/month SIP, but adds a 10% annual Step-Up (Year 1: ₹10k, Year 2: ₹11k, Year 3: ₹12.1k...):
• Total Contributed: ₹68.7 Lakh
• Final Accumulated Wealth: ~₹2.05 Crore!

By stepping up contributions as your income rises, Investor B accumulated more than double the final wealth of Investor A. Most modern investment platforms and AMC portals allow you to check the 'Auto Step-Up' box during SIP registration."""


GENERATORS = [
    (gen_sebi_categories, 0.30),
    (gen_amc_profiles, 0.25),
    (gen_investor_personas, 0.25),
    (gen_market_mechanics, 0.20),
]


def main():
    parser = argparse.ArgumentParser(description="Generate Indian market knowledge and advisory data")
    parser.add_argument("--target_mib", type=float, default=50.0, help="Target file size in MiB (~12-15M tokens)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default=str(OUT_JSONL))
    args = parser.parse_args()

    r = random.Random(args.seed)
    fns = [f for f, _ in GENERATORS]
    weights = [w for _, w in GENERATORS]
    target_bytes = args.target_mib * MIB

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    n = 0
    counts = {}

    print(f"Generating ~{args.target_mib:.1f} MiB of rich Indian mutual fund market knowledge...")
    with open(out_path, "w", encoding="utf-8") as f:
        while written < target_bytes:
            fn = r.choices(fns, weights=weights, k=1)[0]
            text = fn(r).strip()
            line = json.dumps({"text": text, "source": "indian_market_knowledge", "domain": "finance_india"}, ensure_ascii=False) + "\n"
            f.write(line)
            written += len(line.encode("utf-8"))
            n += 1
            counts[fn.__name__] = counts.get(fn.__name__, 0) + 1
            if n % 5000 == 0:
                print(f"  {n:,} articles written | {written / MIB:.1f}/{args.target_mib:.1f} MiB", flush=True)

    print(f"\n[DONE] Wrote {n:,} Indian market knowledge passages ({written / MIB:.1f} MiB) to {out_path}")
    for k, v in counts.items():
        print(f"  - {k:25}: {v:,} ({v/n*100:.1f}%)")


if __name__ == "__main__":
    main()
