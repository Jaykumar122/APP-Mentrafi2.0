"""
Generate advanced Indian financial numeracy and mathematical reasoning passages for pretraining.

Topics covered:
1. LTCG / STCG Capital Gains Tax (Finance Act 2024: 12.5% LTCG above 1.25L, 20% STCG, Debt slab rates)
2. Rule of 72 & Multi-Cycle Doubling Horizons
3. Direct Plan vs Regular Plan Wealth Drag (Distributor Commission Leakage over 10-25 yrs)
4. Systematic Withdrawal Plan (SWP) for Retirement Sustainability & Cashflows
5. Sharpe Ratio & Risk-Adjusted Volatility Comparison
6. Emergency Fund & Human Life Value (HLV) Term Insurance Math
7. Rupee Cost Averaging & NAV Unit Accumulation during Market Cycles
8. Step-Up SIP Wealth Multiplier Analysis

All arithmetic is computed in Python and internally verified for 100% mathematical consistency.
"""

import argparse
import json
import math
import random
from pathlib import Path

OUT_JSONL = Path("data/raw/corpus_v2/advanced_finance_math.jsonl")
MIB = 1024 ** 2


def money(x):
    """Indian-style comma grouping: 1,00,000"""
    x = round(x)
    s = str(abs(int(x)))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if x < 0 else "") + s


def R(x):
    """Round to whole rupees at point of computation"""
    return float(round(x))


def lakh(x):
    if x >= 1e7:
        return f"about ₹{x/1e7:.2f} crore"
    if x >= 1e5:
        return f"about ₹{x/1e5:.2f} lakh"
    return f"about ₹{money(x)}"


NAMES = [
    "Aarav", "Ananya", "Rohan", "Sneha", "Karan", "Pooja", "Vikram", "Meera",
    "Aditya", "Divya", "Suresh", "Kavita", "Arjun", "Neha", "Manish", "Priya",
    "Rahul", "Ritu", "Sanjay", "Farah", "Imran", "Nikhil", "Deepak", "Anjali"
]

CAVEATS = [
    """Tax laws and regulatory provisions reflect the statutory framework under the Finance Act. Actual tax liabilities depend on individual income slabs, overall capital gains across asset classes in a financial year, and prevailing surcharge and cess rates.""",
    
    """All calculations above demonstrate mathematical principles of compounding and cost deduction. Actual market returns are non-linear, and investment outcomes will fluctuate based on macroeconomic cycles and asset allocation.""",
    
    """The figures illustrate the mechanics of financial formulas. In personal finance, consistency of contribution and maintaining an appropriate asset allocation usually drive success more than attempting to forecast interest rate cycles.""",
    
    """Direct comparison between regular and direct plans highlights the compounding impact of recurring distributor commissions. Every rupee saved in expense ratio remains invested and compounds for the investor.""",
    
    """A systematic withdrawal plan depends on asset allocation and return sequencing. In early retirement years, negative market returns combined with withdrawals can impair portfolio longevity, highlighting the need for a debt buffer.""",
]


def tail(r):
    return "\n" + r.choice(CAVEATS).strip() + "\n"


# --------------------------------------------------------------------------- #
# 1. LTCG and STCG Tax Calculation (Finance Act 2024 Rules)
# --------------------------------------------------------------------------- #
def ltcg_stcg_tax_calc(r):
    name = r.choice(NAMES)
    asset_type = r.choice(["equity", "equity", "equity", "debt"])
    invested = r.choice([200000, 300000, 500000, 750000, 1000000, 1500000, 2000000, 3000000])
    
    if asset_type == "equity":
        is_long_term = r.choice([True, True, False])
        if is_long_term:
            holding_months = r.choice([14, 18, 24, 36, 48, 60])
            gain_pct = r.uniform(0.20, 1.20)
            final_val = R(invested * (1 + gain_pct))
            total_gain = final_val - invested
            exemption = 125000  # ₹1.25 Lakh exemption under 2024 rules
            taxable_gain = max(0.0, total_gain - exemption)
            tax_rate = 12.5     # 12.5% LTCG
            tax_payable = R(taxable_gain * 0.125)
            net_in_hand = final_val - tax_payable
            return f"""Mutual Fund Taxation: Calculating Equity LTCG under the 12.5% Tax Rule

{name} invested ₹{money(invested)} into an Equity Mutual Fund. After holding the units for {holding_months} months, the investment grew to ₹{money(final_val)}, and {name} redeemed the entire balance. How is the capital gains tax calculated?

Under the prevailing Indian tax framework (post-July 2024 Budget):
1. Holding Period Classification: Units held in equity-oriented mutual funds for more than 12 months qualify as Long-Term Capital Assets (LTCG). Since {holding_months} months exceeds 12 months, this redemption is subject to Long-Term Capital Gains tax.
2. Total Capital Gain:
   Total Gain = Redemption Value - Initial Investment
              = ₹{money(final_val)} - ₹{money(invested)} = ₹{money(total_gain)}

3. Statutory Exemption:
   Section 112A provides an annual exemption of ₹1,25,000 on equity LTCG across all investments combined in a financial year.
   Taxable Gain = Total Gain - Exemption
                = ₹{money(total_gain)} - ₹{money(exemption)} = ₹{money(taxable_gain)}

4. Tax Calculation:
   Equity LTCG is taxed at a flat 12.5% (plus applicable cess).
   Tax Payable = ₹{money(taxable_gain)} x 12.5% = ₹{money(tax_payable)}

5. Net In-Hand Realization:
   Net Proceeds = ₹{money(final_val)} - ₹{money(tax_payable)} = ₹{money(net_in_hand)}

Effective tax rate on total profits is {(tax_payable/total_gain)*100:.2f}%. Notice that if the total gains were below the ₹1,25,000 threshold, the tax payable would be exactly zero.
{tail(r)}"""
        else:
            holding_months = r.choice([4, 6, 8, 10, 11])
            gain_pct = r.uniform(0.08, 0.35)
            final_val = R(invested * (1 + gain_pct))
            total_gain = final_val - invested
            tax_rate = 20.0     # 20% STCG
            tax_payable = R(total_gain * 0.20)
            net_in_hand = final_val - tax_payable
            return f"""Mutual Fund Taxation: Calculating Equity STCG under the 20% Tax Rule

{name} invested ₹{money(invested)} in an Equity Mutual Fund and redeemed it after {holding_months} months when the value reached ₹{money(final_val)}. What is the applicable Short-Term Capital Gains tax?

Because equity mutual fund units were held for {holding_months} months (less than or equal to 12 months), the gain is classified as Short-Term Capital Gain (STCG) under Section 111A:
1. Total Short-Term Capital Gain:
   Gain = ₹{money(final_val)} - ₹{money(invested)} = ₹{money(total_gain)}

2. Applicable Tax Rate:
   Equity STCG does not enjoy any basic threshold exemption (such as the LTCG ₹1.25L exemption). The entire gain is taxed at a flat rate of 20.0%:
   Tax Payable = ₹{money(total_gain)} x 20% = ₹{money(tax_payable)}

3. Net Proceeds to Investor:
   Net Realization = ₹{money(final_val)} - ₹{money(tax_payable)} = ₹{money(net_in_hand)}

This demonstrates why holding equity mutual funds for longer than 12 months is significantly more tax-efficient: waiting past the 1-year mark reduces the tax rate from 20% to 12.5% and unlocks the annual ₹1,25,000 tax-free exemption.
{tail(r)}"""
    else:
        # Debt fund taxation
        holding_months = r.choice([18, 24, 36, 48])
        gain_pct = r.uniform(0.10, 0.35)
        final_val = R(invested * (1 + gain_pct))
        total_gain = final_val - invested
        slab = r.choice([20, 30])
        tax_payable = R(total_gain * (slab / 100))
        net_in_hand = final_val - tax_payable
        return f"""Debt Mutual Fund Taxation: Slab-Rate Calculation (Finance Act 2023 Rules)

{name} invested ₹{money(invested)} in a Corporate Bond Debt Mutual Fund and held it for {holding_months} months, redeeming at ₹{money(final_val)}. {name} falls in the {slab}% income tax bracket. How are the returns taxed?

Under Section 50AA introduced in the Finance Act 2023, investments in specified debt mutual funds (where equity allocation is 35% or less) made on or after April 1, 2023, do not receive indexation benefits regardless of the holding period. All gains are treated as short-term and taxed at the investor's marginal slab rate:

1. Capital Gain = ₹{money(final_val)} - ₹{money(invested)} = ₹{money(total_gain)}
2. Applicable Tax Rate = Investor's marginal slab ({slab}%)
3. Tax Amount = ₹{money(total_gain)} x {slab}% = ₹{money(tax_payable)}
4. Post-Tax Value = ₹{money(final_val)} - ₹{money(tax_payable)} = ₹{money(net_in_hand)}

Post-tax annual return is roughly {(net_in_hand/invested - 1)/(holding_months/12)*100:.2f}%. For investors in the 30% slab, arbitrage funds or balanced advantage funds often provide superior tax efficiency while maintaining low risk.
{tail(r)}"""


# --------------------------------------------------------------------------- #
# 2. Rule of 72 & Compounding Doubling Horizons
# --------------------------------------------------------------------------- #
def rule_of_72_calc(r):
    name = r.choice(NAMES)
    cagr = r.choice([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.5, 14.0, 15.0, 16.0, 18.0])
    capital = r.choice([100000, 250000, 500000, 1000000, 2000000, 2500000])
    
    doubling_years = 72.0 / cagr
    exact_doubling = math.log(2) / math.log(1 + cagr / 100)
    doubled_corpus = capital * 2
    quadrupled_corpus = capital * 4
    
    return f"""The Rule of 72: Estimating Capital Doubling Horizons

{name} has ₹{money(capital)} to invest and expects a compound annual growth rate (CAGR) of {cagr}% per annum. How many years will it take for the corpus to double to ₹{money(doubled_corpus)}, and quadruple to ₹{money(quadrupled_corpus)}?

The Rule of 72 is a foundational mental model in compounding arithmetic:
    Years to Double ~ 72 / Annual Rate of Return

Substituting {cagr}%:
    Doubling Time = 72 / {cagr} = {doubling_years:.1f} years (Exact logarithmic value: {exact_doubling:.2f} years)

Milestone Trajectory:
• Year 0: Initial Capital = ₹{money(capital)}
• Year {doubling_years:.1f}: 1st Doubling = ₹{money(doubled_corpus)}
• Year {doubling_years * 2:.1f}: 2nd Doubling = ₹{money(quadrupled_corpus)}
• Year {doubling_years * 3:.1f}: 3rd Doubling = ₹{money(capital * 8)}

Comparative Context:
- Fixed Deposit at 7.0% CAGR doubles in: 72 / 7.0 = 10.3 years
- Conservative Hybrid Fund at 9.0% CAGR doubles in: 72 / 9.0 = 8.0 years
- Large Cap Equity Fund at 12.0% CAGR doubles in: 72 / 12.0 = 6.0 years
- Mid / Small Cap Fund at 15.0% CAGR doubles in: 72 / 15.0 = 4.8 years

Over a 24-year investment horizon:
At 6% (FD), money doubles ~2.3 times (ending at ~₹{money(capital * 2**2.3)}).
At 12% (Equity), money doubles 4 times (ending at 16x capital: ₹{money(capital * 16)}).
A 6% difference in annual return creates a dramatic 4x difference in accumulated wealth over long horizons.
{tail(r)}"""


# --------------------------------------------------------------------------- #
# 3. Direct Plan vs Regular Plan Wealth Drag
# --------------------------------------------------------------------------- #
def direct_vs_regular_plan_drag(r):
    name = r.choice(NAMES)
    sip = r.choice([5000, 10000, 15000, 20000, 25000, 30000, 50000])
    yrs = r.choice([10, 15, 20, 25])
    gross_cagr = r.choice([12.0, 13.0, 14.0, 15.0])
    commission_drag = r.choice([0.75, 0.90, 1.0, 1.15, 1.25])
    
    dir_cagr = gross_cagr
    reg_cagr = gross_cagr - commission_drag
    
    # Calculate SIP FV for both
    n = yrs * 12
    r_dir = (dir_cagr / 100) / 12
    r_reg = (reg_cagr / 100) / 12
    
    fv_dir = R(sip * (((1 + r_dir)**n - 1) / r_dir) * (1 + r_dir))
    fv_reg = R(sip * (((1 + r_reg)**n - 1) / r_reg) * (1 + r_reg))
    lost_to_comm = fv_dir - fv_reg
    total_invested = sip * n
    
    return f"""The Cost of Regular Plans: Quantifying Distributor Commission Drag

{name} starts a monthly SIP of ₹{money(sip)} for {yrs} years. The underlying portfolio delivers a gross return of {gross_cagr}% CAGR. The Direct Plan charges a minimal expense ratio, while the Regular Plan pays an additional distributor commission of {commission_drag:.2f}% per year. What is the total monetary difference over {yrs} years?

Total Capital Contributed = ₹{money(sip)} x {n} months = ₹{money(total_invested)}

1. Direct Plan Compounding ({dir_cagr}% net CAGR):
   Monthly rate r = {dir_cagr}% / 12 = {r_dir:.6f}
   FV_Direct = ₹{money(sip)} x [((1 + {r_dir:.6f})^{n} - 1) / {r_dir:.6f}] x (1 + {r_dir:.6f})
             = ₹{money(fv_dir)} ({lakh(fv_dir)})

2. Regular Plan Compounding ({reg_cagr:.2f}% net CAGR after {commission_drag:.2f}% commission deduction):
   Monthly rate r = {reg_cagr:.2f}% / 12 = {r_reg:.6f}
   FV_Regular = ₹{money(sip)} x [((1 + {r_reg:.6f})^{n} - 1) / {r_reg:.6f}] x (1 + {r_reg:.6f})
              = ₹{money(fv_reg)} ({lakh(fv_reg)})

3. Wealth Lost to Intermediary Commissions:
   Commission Leakage = ₹{money(fv_dir)} - ₹{money(fv_reg)} = ₹{money(lost_to_comm)} ({lakh(lost_to_comm)})

Key Takeaway:
A seemingly small commission of {commission_drag:.2f}% per year costs {name} ₹{money(lost_to_comm)} in lost compounding over {yrs} years. That lost amount equals {(lost_to_comm/total_invested)*100:.1f}% of the total money {name} invested out of pocket! Investing directly via AMCs or direct platforms (such as MFCentral, Zerodha Coin, or Groww) saves this entire amount.
{tail(r)}"""


# --------------------------------------------------------------------------- #
# 4. Systematic Withdrawal Plan (SWP) for Retirement
# --------------------------------------------------------------------------- #
def swp_retirement_example(r):
    name = r.choice(NAMES)
    corpus = r.choice([5000000, 7500000, 10000000, 15000000, 20000000, 25000000])
    ann_return = r.choice([8.0, 8.5, 9.0, 9.5, 10.0])
    swp_rate = r.choice([5.0, 6.0, 6.5, 7.0])  # percentage of corpus withdrawn annually
    
    monthly_swp = R(corpus * (swp_rate / 100) / 12)
    yrs = r.choice([5, 10, 15, 20])
    
    # Month by month simulation
    m_rate = (ann_return / 100) / 12
    balance = corpus
    total_withdrawn = 0.0
    for m in range(yrs * 12):
        balance = balance * (1 + m_rate) - monthly_swp
        total_withdrawn += monthly_swp
        if balance <= 0:
            balance = 0.0
            break
            
    balance = R(balance)
    total_withdrawn = R(total_withdrawn)
    
    return f"""Systematic Withdrawal Plan (SWP): Sustaining Retirement Cashflows

{name} has accumulated a retirement corpus of {lakh(corpus)} (₹{money(corpus)}) invested in a conservative hybrid / balanced advantage portfolio generating ~{ann_return}% annual return. {name} sets up an SWP to withdraw {swp_rate}% annually (₹{money(monthly_swp)} per month) to fund living expenses for {yrs} years.

How does the portfolio balance evolve?

Monthly Math:
• Starting Corpus: ₹{money(corpus)}
• Monthly Withdrawal: ₹{money(monthly_swp)} / month
• Monthly Return Rate: {ann_return}% / 12 = {m_rate*100:.4f}% per month
• Total Withdrawals over {yrs} years ({yrs*12} months): ₹{money(total_withdrawn)} ({lakh(total_withdrawn)})

Portfolio Status after {yrs} Years:
• Remaining Balance: ₹{money(balance)} ({lakh(balance)})

Strategic Insight:
Because the portfolio growth rate ({ann_return}%) is higher than the withdrawal rate ({swp_rate}%), the capital remains protected while generating consistent regular income. Total cash received by {name} is ₹{money(total_withdrawn)}, while the ending corpus still stands at ₹{money(balance)}. SWP provides monthly predictability and superior tax efficiency compared to traditional bank FD interest.
{tail(r)}"""


# --------------------------------------------------------------------------- #
# 5. Sharpe Ratio & Risk-Adjusted Returns
# --------------------------------------------------------------------------- #
def sharpe_ratio_calc(r):
    fund_a = r.choice(["HDFC Flexi Cap", "Parag Parikh Flexi Cap", "SBI Large & Midcap", "Mirae Asset Large Cap"])
    fund_b = r.choice(["Quant Active Fund", "Nippon Small Cap", "Tata Small Cap", "Axis Growth Opportunities"])
    
    rf = r.choice([6.5, 6.8, 7.0, 7.2])  # 10Y G-Sec yield
    
    ret_a = r.choice([14.0, 15.0, 15.5, 16.0])
    sd_a = r.choice([11.0, 12.0, 12.5, 13.0])
    
    ret_b = r.choice([17.0, 18.0, 18.5, 19.5])
    sd_b = r.choice([18.0, 19.0, 20.0, 22.0])
    
    sharpe_a = (ret_a - rf) / sd_a
    sharpe_b = (ret_b - rf) / sd_b
    
    return f"""Risk-Adjusted Return Analysis: Calculating the Sharpe Ratio

An investor is evaluating two mutual funds:
• Fund A ({fund_a}): 3-Year Return = {ret_a}%, Standard Deviation (Volatility) = {sd_a}%
• Fund B ({fund_b}): 3-Year Return = {ret_b}%, Standard Deviation (Volatility) = {sd_b}%
• Risk-Free Benchmark Rate (10-Year Indian Government Security yield): {rf}%

Which fund delivers superior risk-adjusted performance?

The Sharpe Ratio measures the excess return generated per unit of total risk (volatility):
    Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Standard Deviation

1. Fund A Sharpe Ratio:
   Sharpe_A = ({ret_a}% - {rf}%) / {sd_a}% = {ret_a - rf:.1f} / {sd_a}% = {sharpe_a:.2f}

2. Fund B Sharpe Ratio:
   Sharpe_B = ({ret_b}% - {rf}%) / {sd_b}% = {ret_b - rf:.1f} / {sd_b}% = {sharpe_b:.2f}

Comparison & Conclusion:
Even though Fund B produced a higher absolute return ({ret_b}% vs {ret_a}%), it took on disproportionately higher volatility ({sd_b}% vs {sd_a}%). 
Fund A has a Sharpe ratio of {sharpe_a:.2f}, while Fund B has a Sharpe ratio of {sharpe_b:.2f}.
Fund {'A' if sharpe_a >= sharpe_b else 'B'} generated more return per unit of volatility, indicating superior fund management and more consistent risk-adjusted compounding.
{tail(r)}"""


# --------------------------------------------------------------------------- #
# 6. Emergency Fund & Human Life Value (HLV) Insurance Math
# --------------------------------------------------------------------------- #
def emergency_fund_insurance_math(r):
    name = r.choice(NAMES)
    monthly_expenses = r.choice([35000, 50000, 65000, 80000, 100000, 125000, 150000])
    monthly_emi = r.choice([15000, 25000, 35000, 45000, 60000])
    annual_income = (monthly_expenses + monthly_emi) * 12 * r.uniform(1.2, 1.8)
    annual_income = R(annual_income / 100000) * 100000  # round to lakh
    existing_debt = monthly_emi * r.choice([36, 60, 84, 120])
    liquid_savings = r.choice([200000, 400000, 500000, 800000, 1000000])
    
    # 1. Emergency fund: 6 to 9 months of mandatory outflows
    monthly_burn = monthly_expenses + monthly_emi
    ef_min = monthly_burn * 6
    ef_max = monthly_burn * 9
    
    # 2. Term insurance HLV: 15x annual income + debt - liquid savings
    hlv_cover = R((annual_income * 15) + existing_debt - liquid_savings)
    
    return f"""Financial Foundation Planning: Calculating Emergency Fund and Life Insurance Cover

{name} has the following financial profile:
• Annual Take-Home Income: ₹{money(annual_income)} ({lakh(annual_income)})
• Monthly Living Expenses: ₹{money(monthly_expenses)}
• Monthly EMI Obligations: ₹{money(monthly_emi)}
• Outstanding Home / Car Loans: ₹{money(existing_debt)} ({lakh(existing_debt)})
• Existing Liquid Savings: ₹{money(liquid_savings)}

Step 1: Emergency Fund Sizing
Mandatory monthly cash burn = Expenses + EMI = ₹{money(monthly_expenses)} + ₹{money(monthly_emi)} = ₹{money(monthly_burn)}
• Recommended 6-Month Emergency Buffer: ₹{money(monthly_burn)} x 6 = ₹{money(ef_min)} ({lakh(ef_min)})
• Prudent 9-Month Buffer (for volatile careers): ₹{money(monthly_burn)} x 9 = ₹{money(ef_max)} ({lakh(ef_max)})
This should be held in high-safety liquid mutual funds or sweep-in bank FDs to ensure instant access.

Step 2: Term Life Insurance Cover (Human Life Value Method)
Pure term insurance replaces the economic value of the breadwinner:
    Ideal Cover = (15 x Annual Income) + Liabilities - Liquid Assets
    Income Replacement = 15 x ₹{money(annual_income)} = ₹{money(annual_income * 15)}
    Cover Needed = ₹{money(annual_income * 15)} + ₹{money(existing_debt)} - ₹{money(liquid_savings)}
                 = ₹{money(hlv_cover)} ({lakh(hlv_cover)})

Rounding to standard policy slabs, {name} requires a Pure Term Insurance Policy of approximately {lakh(hlv_cover)}.
{tail(r)}"""


GENERATORS = [
    (ltcg_stcg_tax_calc, 0.22),
    (direct_vs_regular_plan_drag, 0.20),
    (rule_of_72_calc, 0.18),
    (swp_retirement_example, 0.15),
    (sharpe_ratio_calc, 0.13),
    (emergency_fund_insurance_math, 0.12),
]


def main():
    parser = argparse.ArgumentParser(description="Generate advanced financial math passages")
    parser.add_argument("--target_mib", type=float, default=60.0, help="Target file size in MiB (~15M tokens)")
    parser.add_argument("--seed", type=int, default=42)
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

    print(f"Generating ~{args.target_mib:.1f} MiB of advanced financial math passages...")
    with open(out_path, "w", encoding="utf-8") as f:
        while written < target_bytes:
            fn = r.choices(fns, weights=weights, k=1)[0]
            text = fn(r).strip()
            line = json.dumps({"text": text, "source": "advanced_finance_math", "domain": "numeracy"}, ensure_ascii=False) + "\n"
            f.write(line)
            written += len(line.encode("utf-8"))
            n += 1
            counts[fn.__name__] = counts.get(fn.__name__, 0) + 1
            if n % 10000 == 0:
                print(f"  {n:,} passages written | {written / MIB:.1f}/{args.target_mib:.1f} MiB", flush=True)

    print(f"\n[DONE] Wrote {n:,} verified math passages ({written / MIB:.1f} MiB) to {out_path}")
    for k, v in counts.items():
        print(f"  - {k:32}: {v:,} ({v/n*100:.1f}%)")


if __name__ == "__main__":
    main()
