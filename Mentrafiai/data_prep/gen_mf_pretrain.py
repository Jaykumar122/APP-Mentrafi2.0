"""Generate Indian mutual-fund pretraining documents from verified sources.

1. db_funds     : one document per active scheme from the PostgreSQL funds DB,
                  using ONLY fields that come from AMFI/NAV data (name, fund
                  house, plan, option, NAV and date, launch date, NAV-based
                  returns). Expense ratio, AUM, riskometer, manager and
                  benchmark columns were filled by enrich_database.py with
                  formulas, so they are deliberately NOT used.
2. db_families  : Direct vs Regular and Growth vs IDCW comparisons of the same
                  scheme, and fund-house overviews (real NAV-based numbers).
3. db_rankings  : category tables of NAV-based returns, dated.
4. concepts     : SEBI scheme categories, tax rules, and the curriculum topics,
                  written as prose (hand-checked definitions).
5. calc         : worked SIP / goal / step-up / lump-sum / SWP / tax / 80C /
                  inflation examples; every number from inference.financial_calculator.

Output: data/raw/mf_corpus/gen_<name>.jsonl with {"text", "source"}.
Run: python -m data_prep.gen_mf_pretrain
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from inference import financial_calculator as fc

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "mf_corpus"

# ───────────────────────────── SEBI categories ─────────────────────────────
# (pattern in scheme name, display name, kind, definition) — SEBI scheme
# categorisation (Oct 2017, multi cap revised 2020, flexi cap added Nov 2020).
CATEGORIES = [
    (r"\bovernight\b", "Overnight Fund", "debt",
     "An overnight fund invests in securities that mature in one day, so its interest-rate and credit risk are very low; it is used for parking money for a few days."),
    (r"\bliquid\b", "Liquid Fund", "debt",
     "A liquid fund invests in debt and money market instruments with a maturity of up to 91 days. It is commonly used for emergency money and short-term parking, but it is not a bank deposit and its value can fall in rare credit events."),
    (r"\bultra short\b", "Ultra Short Duration Fund", "debt",
     "An ultra short duration fund keeps the Macaulay duration of its portfolio between three and six months."),
    (r"\blow duration\b", "Low Duration Fund", "debt",
     "A low duration fund keeps the Macaulay duration of its portfolio between six and twelve months."),
    (r"\bmoney market\b", "Money Market Fund", "debt",
     "A money market fund invests in money market instruments with a maturity of up to one year."),
    (r"\bshort (?:duration|term)\b", "Short Duration Fund", "debt",
     "A short duration fund keeps the Macaulay duration of its portfolio between one and three years."),
    (r"\bmedium to long\b", "Medium to Long Duration Fund", "debt",
     "A medium to long duration fund keeps the Macaulay duration between four and seven years, so it is more sensitive to interest-rate changes."),
    (r"\bmedium (?:duration|term)\b", "Medium Duration Fund", "debt",
     "A medium duration fund keeps the Macaulay duration of its portfolio between three and four years."),
    (r"\blong duration\b", "Long Duration Fund", "debt",
     "A long duration fund keeps the Macaulay duration above seven years; its NAV can move sharply when interest rates change."),
    (r"\bdynamic bond\b", "Dynamic Bond Fund", "debt",
     "A dynamic bond fund can change the duration of its portfolio across all maturities depending on the fund manager's view of interest rates."),
    (r"\bcorporate bond\b", "Corporate Bond Fund", "debt",
     "A corporate bond fund invests at least 80% of its assets in corporate bonds rated AA+ and above."),
    (r"\bcredit risk\b", "Credit Risk Fund", "debt",
     "A credit risk fund invests at least 65% of its assets in corporate bonds rated below the highest grades (AA and below), so it carries higher default risk in exchange for higher interest."),
    (r"\bbanking (?:&|and) psu\b", "Banking and PSU Fund", "debt",
     "A banking and PSU fund invests at least 80% of its assets in debt of banks, public sector undertakings and public financial institutions."),
    (r"\bgilt\b", "Gilt Fund", "debt",
     "A gilt fund invests at least 80% of its assets in government securities. It has very low credit risk but can carry high interest-rate risk."),
    (r"\bfloat(?:er|ing)\b", "Floater Fund", "debt",
     "A floater fund invests at least 65% of its assets in floating-rate instruments, whose interest resets with market rates."),
    (r"\bfixed (?:maturity|term)\b|\bfmp\b", "Fixed Maturity Plan", "debt",
     "A fixed maturity plan is a closed-ended debt scheme that holds bonds maturing around the scheme's own maturity date; units can usually be sold only on the stock exchange before maturity."),
    (r"\binterval\b", "Interval Fund", "debt",
     "An interval fund allows purchase and redemption only during specified transaction windows."),
    (r"\barbitrage\b", "Arbitrage Fund", "hybrid_equity",
     "An arbitrage fund buys shares in the cash market and sells them in the futures market to earn the price difference; it keeps at least 65% in equity and related instruments, so it is taxed like an equity fund while behaving more like a low-risk debt fund."),
    (r"\bequity savings\b", "Equity Savings Fund", "hybrid_equity",
     "An equity savings fund invests at least 65% in equity including arbitrage positions and at least 10% in debt, aiming for lower volatility than a pure equity fund."),
    (r"\bbalanced advantage\b|\bdynamic asset allocation\b", "Balanced Advantage Fund", "hybrid",
     "A balanced advantage (dynamic asset allocation) fund changes its equity and debt mix based on market valuations or a model."),
    (r"\bmulti asset\b", "Multi Asset Allocation Fund", "hybrid",
     "A multi asset allocation fund invests in at least three asset classes, such as equity, debt and gold, with at least 10% in each."),
    (r"\bconservative hybrid\b|\bdebt hybrid\b|\bmonthly income\b|\bregular savings\b", "Conservative Hybrid Fund", "debt",
     "A conservative hybrid fund invests 75% to 90% in debt and 10% to 25% in equity."),
    (r"\bbalanced hybrid\b", "Balanced Hybrid Fund", "hybrid",
     "A balanced hybrid fund keeps between 40% and 60% each in equity and debt."),
    (r"\baggressive hybrid\b|\bequity hybrid\b|\bhybrid equity\b", "Aggressive Hybrid Fund", "equity",
     "An aggressive hybrid fund invests 65% to 80% in equity and 20% to 35% in debt."),
    (r"\belss\b|\btax saver\b|\btax saving\b|\btaxsaver\b|\blong term equity\b", "ELSS Tax Saver Fund", "equity",
     "An ELSS (equity linked savings scheme) invests at least 80% in equity and has a three-year lock-in for every investment, including each SIP instalment. Under the old tax regime, investments of up to Rs 1.5 lakh a year qualify for deduction under Section 80C."),
    (r"\bfund of funds?\b|\bfof\b", "Fund of Funds", "other",
     "A fund of funds invests in units of other mutual fund schemes, so investors pay the expenses of both the outer fund and the underlying funds."),
    (r"\bgold\b", "Gold Fund", "other",
     "A gold ETF holds physical gold of high purity, and a gold fund of funds invests in gold ETFs; their value follows the domestic price of gold."),
    (r"\b(?:etf|exchange traded)\b", "Exchange Traded Fund", "equity",
     "An exchange traded fund tracks an index or commodity and is bought and sold on the stock exchange during market hours through a demat account."),
    (r"\bindex\b|\bnifty\b|\bsensex\b", "Index Fund", "equity",
     "An index fund buys the same stocks as its index in the same proportions, so its costs are usually low and its returns stay close to the index, minus expenses and tracking error."),
    (r"\blarge (?:&|and) mid ?cap\b", "Large and Mid Cap Fund", "equity",
     "A large and mid cap fund invests at least 35% each in large cap and mid cap stocks."),
    (r"\bmulti cap\b|\bmulticap\b", "Multi Cap Fund", "equity",
     "A multi cap fund must invest at least 25% each in large cap, mid cap and small cap stocks, with at least 75% in equity overall."),
    (r"\bflexi cap\b|\bflexicap\b", "Flexi Cap Fund", "equity",
     "A flexi cap fund keeps at least 65% in equity and can move freely between large, mid and small cap stocks."),
    (r"\blarge cap\b|\bbluechip\b|\bblue chip\b|\btop 100\b", "Large Cap Fund", "equity",
     "A large cap fund invests at least 80% in large cap stocks, which SEBI defines as the 100 largest listed companies by full market capitalisation."),
    (r"\bmid ?cap\b|\bemerging\b", "Mid Cap Fund", "equity",
     "A mid cap fund invests at least 65% in mid cap stocks, the 101st to 250th largest companies by market capitalisation, which can be more volatile than large caps."),
    (r"\bsmall ?cap\b", "Small Cap Fund", "equity",
     "A small cap fund invests at least 65% in small cap stocks, companies ranked 251st and below by market capitalisation; these can rise and fall sharply."),
    (r"\bfocused\b", "Focused Fund", "equity",
     "A focused fund holds at most 30 stocks, so each holding has a larger effect on returns."),
    (r"\bvalue\b|\bcontra\b", "Value or Contra Fund", "equity",
     "A value fund buys stocks it considers undervalued, and a contra fund takes positions against prevailing market trends; SEBI allows an AMC to offer only one of the two."),
    (r"\bdividend yield\b", "Dividend Yield Fund", "equity",
     "A dividend yield fund invests at least 65% in stocks with high dividend yields."),
    (r"\b(?:banking|financial services|pharma|healthcare|infrastructure|technology|digital|consumption|manufacturing|psu|energy|auto|fmcg|it)\b", "Sectoral or Thematic Fund", "equity",
     "A sectoral or thematic fund invests at least 80% in one sector or theme, so it is concentrated and can underperform the broad market for long periods."),
    (r"\binternational\b|\bglobal\b|\bus\b|\bnasdaq\b|\bworld\b", "International Fund", "other",
     "An international fund invests in foreign stocks directly or through overseas funds, adding currency risk."),
    (r"\bretirement\b|\bchildren'?s?\b|\bchild\b", "Solution Oriented Fund", "other",
     "A solution oriented fund (retirement or children's fund) has a lock-in of at least five years or until the goal age, whichever is earlier."),
]

TAX_TEXT = {
    "equity": ("For tax, this is an equity-oriented fund: gains on units held for more than 12 months are long-term and taxed at 12.5% on gains above Rs 1.25 lakh in a financial year, and units held for 12 months or less are taxed at 20%, plus 4% cess (rules for transfers on or after 23 July 2024)."),
    "hybrid_equity": ("For tax, it is treated as an equity-oriented fund because it keeps at least 65% in equity: long-term gains after 12 months are taxed at 12.5% above Rs 1.25 lakh a year and short-term gains at 20%, plus cess."),
    "debt": ("For tax, gains on debt-oriented funds bought on or after 1 April 2023 are added to income and taxed at the investor's slab rate, whatever the holding period."),
    "hybrid": ("For tax, the treatment depends on the actual equity share: at 65% or more in domestic equity it is taxed like an equity fund; otherwise long-term gains after 24 months are taxed at 12.5% and shorter-term gains at the slab rate."),
    "other": ("For tax, funds of this type are usually not equity-oriented: gains on units held for more than 24 months are taxed at 12.5%, and gains within 24 months at the slab rate."),
}


def classify(name: str):
    low = name.lower()
    for pattern, display, kind, definition in CATEGORIES:
        if re.search(pattern, low):
            return display, kind, definition
    return None


def base_scheme(name: str) -> str:
    cut = re.split(r"\s*[-–(]\s*(?:direct|regular|growth|idcw|dividend|bonus|payout|reinvest|cumulative|plan)\b", name, flags=re.I)[0]
    return " ".join(cut.split()).strip(" -")


def pct(value) -> str:
    return f"{float(value):.2f}%"


def nav_text(value) -> str:
    return f"Rs {float(value):,.4f}".rstrip("0").rstrip(".")


def date_text(value) -> str:
    return value.strftime("%d %B %Y").lstrip("0") if value else "an unknown date"


def load_funds():
    import sys
    sys.path.insert(0, str(ROOT))
    from inference.db_fund_retriever import FundDatabase

    db = FundDatabase()
    if not db.is_connected:
        raise SystemExit("PostgreSQL funds database is not reachable")
    conn = db._get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT f.scheme_code, f.name, f.fund_house, f.plan_type, f.option_type, f.nav, f.nav_date,
               f.launch_date, f.isin, p.return_1m, p.return_3m, p.return_6m, p.return_1y, p.return_3y,
               p.return_5y, p.return_10y
        FROM funds f LEFT JOIN fund_performance p ON p.scheme_code = f.scheme_code
        WHERE f.is_active AND f.nav IS NOT NULL AND f.nav > 0 AND f.nav_date IS NOT NULL
    """)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    # Matured FMPs / wound-up schemes stay is_active with years-old NAVs: keep only
    # schemes whose NAV is within 45 days of the latest NAV date in the data.
    latest = max(r["nav_date"] for r in rows)
    return [r for r in rows if (latest - r["nav_date"]).days <= 45]


RETURN_LABELS = [("return_1y", "1 year"), ("return_3y", "3 years"), ("return_5y", "5 years"), ("return_10y", "10 years")]


def returns_sentence(rng, f) -> str | None:
    parts = [(label, f[key]) for key, label in RETURN_LABELS if f.get(key) is not None]
    if not parts:
        return None
    joined = ", ".join(f"{pct(v)} over {label}" for label, v in parts)
    return rng.choice([
        f"Based on its NAV history, the scheme returned {joined}; returns for periods above one year are annualised.",
        f"Its NAV-based returns were {joined} (annualised for periods longer than a year).",
        f"Looking at past NAVs, it delivered {joined}, with multi-year figures shown as annual rates.",
    ]) + " Past returns do not guarantee future performance."


def fund_doc(rng, f) -> str:
    cat = classify(f["name"])
    plan = (f.get("plan_type") or "").strip()
    option = (f.get("option_type") or "").strip()
    house = f.get("fund_house") or "its asset management company"
    s = []
    s.append(rng.choice([
        f"{f['name']} is a mutual fund scheme managed by {house}.",
        f"{f['name']} is offered by {house}.",
        f"Scheme profile: {f['name']}, from {house}.",
    ]))
    if cat:
        display, kind, definition = cat
        s.append(f"By its name it belongs to the {display} category. {definition}")
    if plan:
        s.append({"direct": "This is the Direct plan, bought straight from the fund house without a distributor, so it has a lower expense ratio than the Regular plan of the same scheme.",
                  "regular": "This is the Regular plan, sold through distributors who are paid a commission from the expense ratio, so its returns are usually a little lower than the Direct plan of the same scheme."}
                 .get(plan.lower(), f"Plan: {plan}."))
    if option:
        low = option.lower()
        if "growth" in low or "cumulative" in low:
            s.append("Under the Growth option, gains stay invested and are reflected in a rising NAV; nothing is paid out until you redeem.")
        elif "idcw" in low or "dividend" in low:
            s.append("Under the IDCW option (income distribution cum capital withdrawal), the fund may pay out part of the NAV; such payouts are not extra profit, the NAV falls by the amount paid, and payouts are taxed at your slab rate.")
    s.append(rng.choice([
        f"Its NAV was {nav_text(f['nav'])} on {date_text(f['nav_date'])}.",
        f"On {date_text(f['nav_date'])}, one unit was valued at {nav_text(f['nav'])}.",
        f"The latest NAV in the data is {nav_text(f['nav'])}, dated {date_text(f['nav_date'])}.",
    ]))
    if f.get("launch_date"):
        s.append(f"The scheme was launched on {date_text(f['launch_date'])}.")
    sentence = returns_sentence(rng, f)
    if sentence:
        s.append(sentence)
    if cat:
        s.append(TAX_TEXT[cat[1]])
    if f.get("isin"):
        s.append(f"Its ISIN is {f['isin']}.")
    s.append(rng.choice([
        "A NAV level by itself says nothing about whether a fund is cheap or expensive; compare returns, costs and risk instead.",
        "Before investing, check the latest factsheet and scheme information document for costs, holdings and risk.",
        "Whether it suits an investor depends on the goal, the time horizon and the ability to accept losses.",
    ]))
    return " ".join(s)


def family_docs(rng, funds) -> list[str]:
    families = defaultdict(list)
    for f in funds:
        families[(f.get("fund_house"), base_scheme(f["name"]).lower())].append(f)
    docs = []
    for (_, base), members in families.items():
        direct = [m for m in members if (m.get("plan_type") or "").lower() == "direct" and "growth" in (m.get("option_type") or "").lower()]
        regular = [m for m in members if (m.get("plan_type") or "").lower() == "regular" and "growth" in (m.get("option_type") or "").lower()]
        if direct and regular:
            d, r = direct[0], regular[0]
            lines = [f"Direct versus Regular plan: {base_scheme(d['name'])}.",
                     f"The Direct Growth plan had a NAV of {nav_text(d['nav'])} on {date_text(d['nav_date'])}, while the Regular Growth plan had {nav_text(r['nav'])} on {date_text(r['nav_date'])}.",
                     "Both plans hold the same portfolio; the difference comes from the distributor commission included in the Regular plan's expense ratio."]
            for key, label in RETURN_LABELS:
                if d.get(key) is not None and r.get(key) is not None and abs(float(d[key]) - float(r[key])) >= 0.01:
                    gap = float(d[key]) - float(r[key])
                    lines.append(f"Over {label}, the Direct plan returned {pct(d[key])} a year against {pct(r[key])} for the Regular plan, a gap of {gap:.2f} percentage points.")
            lines.append("Over long periods a small yearly gap compounds into a noticeable difference in the final corpus.")
            docs.append(" ".join(lines))
        if len(members) >= 3 and rng.random() < 0.5:
            names = sorted({m["name"] for m in members})[:6]
            docs.append(f"{base_scheme(members[0]['name'])} is available in {len(members)} plan and option combinations in the data, "
                        f"including {', '.join(names)}. The plans share one portfolio; they differ in cost (Direct or Regular) and in how gains are handled (Growth or IDCW).")
    houses = defaultdict(list)
    for f in funds:
        if f.get("fund_house"):
            houses[f["fund_house"]].append(f)
    for house, members in houses.items():
        counts = defaultdict(int)
        for m in members:
            cat = classify(m["name"])
            counts[cat[0] if cat else "other"] += 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
        docs.append(f"{house} has {len(members)} active scheme plans in the data. By scheme name, the largest groups are "
                    + ", ".join(f"{n} {c} plans" for c, n in top) + ". "
                    "Each scheme usually has Direct and Regular plans, with Growth and IDCW options.")
    return docs


def ranking_docs(funds) -> list[str]:
    groups = defaultdict(list)
    for f in funds:
        cat = classify(f["name"])
        if (cat and (f.get("plan_type") or "").lower() == "direct" and "growth" in (f.get("option_type") or "").lower()
                and "bonus" not in f["name"].lower()):
            groups[cat[0]].append(f)
    docs = []
    for category, members in groups.items():
        for key, label in RETURN_LABELS[:3]:
            ranked = sorted((m for m in members if m.get(key) is not None), key=lambda m: -float(m[key]))
            if len(ranked) < 5:
                continue
            latest = max(m["nav_date"] for m in ranked)
            values = sorted(float(m[key]) for m in ranked)
            median = values[len(values) // 2]
            top = ranked[:8]
            docs.append(
                f"{category} category, Direct Growth plans, ranked by NAV-based return over {label} (data up to {date_text(latest)}): "
                + "; ".join(f"{i}. {m['name']} {pct(m[key])}" for i, m in enumerate(top, 1))
                + f". Across {len(ranked)} such plans the median was {median:.2f}%, ranging from {values[0]:.2f}% to {values[-1]:.2f}%. "
                "Rankings change often, and the top performer in one period is frequently not the top performer in the next, so a ranking alone is not a reason to invest.")
    return docs


# ───────────────────────────── concepts ─────────────────────────────
def concept_docs() -> list[str]:
    from data_prep.mutual_fund_curriculum import TOPICS

    docs = []
    for topic, entry in TOPICS.items():
        docs.append(f"{entry['question']} {entry['answer']}")
    for _, display, kind, definition in CATEGORIES:
        docs.append(f"{display}. {definition} {TAX_TEXT[kind]}")
    docs.append(fc.TAX_NOTE.replace("**", "").replace("₹", "Rs "))
    return docs


# ───────────────────────────── calculator examples ─────────────────────────────
NAMES = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Arjun", "Kavya", "Rahul", "Meera", "Imran",
         "Fatima", "Joseph", "Neha", "Siddharth", "Lakshmi", "Harpreet", "Deepak", "Pooja", "Karthik", "Zoya",
         "Manish", "Divya", "Suresh", "Anjali", "Farhan", "Gauri", "Nikhil", "Ritu", "Tenzin", "Sanjay", "Asha"]
CITIES = ["Mumbai", "Pune", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Kolkata", "Jaipur", "Lucknow", "Kochi",
          "Indore", "Ahmedabad", "Nagpur", "Bhopal", "Guwahati", "Chandigarh", "Coimbatore", "Surat"]
GOALS = ["retirement", "a child's college education", "a house down payment", "a car", "a wedding", "an emergency fund",
         "a foreign holiday", "starting a business", "early retirement", "long-term wealth"]


def money(value: float) -> str:
    return fc.format_inr_exact(value).replace("₹", "Rs ")


def short(value: float) -> str:
    return fc.format_inr(value).replace("₹", "Rs ")


def calc_doc(rng) -> str:
    kind = rng.choices(["sip", "goal", "stepup", "lumpsum", "swp", "tax", "80c", "inflation", "compare", "delay"],
                       [22, 14, 10, 12, 7, 10, 5, 6, 8, 6])[0]
    name, city = rng.choice(NAMES), rng.choice(CITIES)
    rate = rng.choice([6, 7, 8, 9, 10, 11, 12, 13, 14])
    years = rng.choice([1, 2, 3, 5, 7, 8, 10, 12, 15, 18, 20, 25, 30])
    sip = rng.choice([500, 1000, 2000, 2500, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 30000, 50000])
    yrs = fc._fmt_years(years)
    if kind == "sip":
        r = fc.calculate_sip(sip, years, rate)
        return rng.choice([
            f"{name} from {city} invests {money(sip)} at the start of every month for {yrs}. Assuming {rate}% a year, "
            f"the monthly rate is {rate}/12 = {r['monthly_rate']*100:.4f}%. Over {r['months']} instalments the total invested is {short(r['invested'])}. "
            f"Using FV = P x [((1 + r)^n - 1) / r] x (1 + r), the projected value is {short(r['corpus'])}, so the estimated gain is {short(r['wealth_gain'])}. "
            f"At 6% inflation this equals about {short(r['real_corpus'])} in today's money. Actual fund returns vary and can be negative.",
            f"How much does a {money(sip)} monthly SIP grow to in {yrs}? With an assumed return of {rate}% a year and instalments at the start of each month, "
            f"{r['months']} payments add up to {short(r['invested'])} and grow to roughly {short(r['corpus'])}. The gain of {short(r['wealth_gain'])} comes from compounding. "
            f"Remember that this is an illustration with a constant return; real markets move up and down.",
        ])
    if kind == "goal":
        years = max(years, 3)
        yrs = fc._fmt_years(years)
        while True:  # keep goals realistic: required SIP at most Rs 2 lakh a month
            target = rng.choice([500000, 1000000, 2000000, 2500000, 5000000, 10000000, 20000000, 50000000])
            g = fc.calculate_goal_sip(target, years, rate)
            if g["required_sip"] <= 200000:
                break
        st = fc.calculate_goal_step_up_sip(target, years, rate, 10)
        return (f"{name} wants {short(target)} in {yrs} for {rng.choice(GOALS)}. At an assumed {rate}% a year, the required SIP is "
                f"{money(g['required_sip'])} a month for {g['months']} months, a total of {short(g['total_invested'])} invested. "
                f"If {name} can raise the SIP by 10% every year, starting at {money(st['starting_sip'])} a month is enough. "
                f"Note that {short(target)} after {yrs} buys only about {short(fc.real_value(target, years))} of today's goods at 6% inflation, "
                "so the target itself may need to be inflation-adjusted.")
    if kind == "stepup":
        step = rng.choice([5, 10, 15])
        a, b = fc.calculate_sip(sip, years, rate), fc.calculate_step_up_sip(sip, years, rate, step)
        return (f"Step-up SIP example: starting at {money(sip)} a month and increasing it by {step}% every year for {yrs} at {rate}% a year. "
                f"The fixed SIP invests {short(a['invested'])} and reaches about {short(a['corpus'])}. "
                f"With the step-up, total investment becomes {short(b['invested'])}, the last instalment is about {money(b['final_sip'])}, "
                f"and the projected corpus rises to {short(b['corpus'])}. A step-up suits investors whose income rises over time.")
    if kind == "lumpsum":
        principal = rng.choice([25000, 50000, 100000, 200000, 300000, 500000, 1000000, 2500000])
        r = fc.calculate_lumpsum(principal, years, rate)
        return (f"A one-time investment of {short(principal)} for {yrs} at {rate}% a year compounds as FV = P x (1 + {rate/100:.2f})^{years}. "
                f"The result is about {short(r['corpus'])}, a gain of {short(r['wealth_gain'])}, worth roughly {short(r['real_corpus'])} in today's money at 6% inflation. "
                "A lump sum is fully exposed to the market level on the day it is invested, while a SIP spreads purchases over time.")
    if kind == "swp":
        corpus = rng.choice([1000000, 2000000, 2500000, 5000000, 10000000])
        draw = rng.choice([5000, 10000, 15000, 20000, 25000, 40000, 50000])
        ret = rng.choice([6, 7, 8, 9])
        w = fc.calculate_swp(corpus, draw, ret, max_years=40)
        outcome = (f"the money lasts the full 40 years and {short(w['remaining_corpus'])} would still remain" if w["lasts_full_period"]
                   else f"the full withdrawal can be paid for {w['months_fully_paid']} months, about {w['years_fully_paid']:.1f} years")
        return (f"Systematic withdrawal plan: {name} has {short(corpus)} and withdraws {money(draw)} every month while the balance earns {ret}% a year. "
                f"Under these assumptions {outcome}. A withdrawal smaller than the monthly return keeps the capital intact; a larger one slowly uses it up. "
                "Each withdrawal redeems units, so capital gains tax applies to the gain portion.")
    if kind == "tax":
        gain = rng.choice([50000, 80000, 125000, 150000, 200000, 300000, 450000, 600000, 1000000])
        fund_type = rng.choice(["equity", "equity", "debt", "other"])
        months = rng.choice([6, 11, 12, 13, 18, 24, 25, 36, 60])
        slab = rng.choice([5, 20, 30])
        t = fc.calculate_capital_gains_tax(gain, months, fund_type, slab_rate_pct=slab)
        kind_text = {"equity": "an equity fund", "debt": "a debt fund bought after April 2023", "other": "a fund of funds"}[fund_type]
        text = (f"Tax example: {name} sells units of {kind_text} after {months} months with a gain of {money(gain)}. "
                f"The gain is {t['term']} under {t['section']}. ")
        if t["exemption_used"]:
            text += f"The yearly exemption covers {money(t['exemption_used'])}, leaving {money(t['taxable_gain'])} taxable. "
        text += (f"At {t['rate_pct']:g}% the tax is {money(t['base_tax'])}, plus 4% cess of {money(t['cess'])}, "
                 f"for a total of {money(t['total_tax'])}. The gain after tax is {money(t['post_tax_gain'])}.")
        if fund_type != "equity":
            text += f" This assumes a {slab}% slab rate where the slab applies."
        return text + " Surcharge and other income are ignored in this example."
    if kind == "80c":
        invest = rng.choice([25000, 50000, 100000, 150000, 200000])
        other = rng.choice([0, 0, 50000, 100000])
        slab = rng.choice([5, 20, 30])
        s = fc.calculate_80c_saving(invest, slab, other_80c=other)
        return (f"Section 80C example: under the old tax regime, {name} invests {money(invest)} in an ELSS fund and already has {money(other)} of other 80C investments. "
                f"The 80C limit is Rs 1,50,000, so the deduction from this ELSS is {money(s['eligible_deduction'])}. At a {slab}% slab plus 4% cess, the tax saved is about {money(s['tax_saved'])}. "
                "Under the new tax regime there is no 80C deduction, and each ELSS investment is locked in for three years.")
    if kind == "inflation":
        cost = rng.choice([500000, 1000000, 1500000, 2500000, 4000000])
        infl = rng.choice([6, 8, 10])
        future = cost * (1 + infl / 100) ** years
        real = fc.real_return_pct(rate, infl)
        return (f"Inflation example: something that costs {short(cost)} today will cost about {short(future)} in {yrs} if prices rise {infl}% a year. "
                f"An investment earning {rate}% a year then has a real return of only about {real:.2f}% a year, because (1 + {rate/100:.2f}) / (1 + {infl/100:.2f}) - 1 = {real/100:.4f}. "
                "This is why long-term goals are usually planned in future rupees, not today's rupees.")
    if kind == "compare":
        low, high = rng.sample([6, 8, 10, 12, 14], 2)
        a, b = fc.calculate_sip(sip, years, min(low, high)), fc.calculate_sip(sip, years, max(low, high))
        return (f"Why a few percentage points matter: a {money(sip)} monthly SIP for {yrs} grows to about {short(a['corpus'])} at {min(low, high)}% a year "
                f"but to about {short(b['corpus'])} at {max(low, high)}% a year, although the amount invested is the same {short(a['invested'])} in both cases. "
                "Higher expected returns come with higher risk, so the rate should match the kind of fund actually chosen.")
    later = rng.choice([3, 5, 10])
    total = years + later
    a, b = fc.calculate_sip(sip, total, rate), fc.calculate_sip(sip, years, rate)
    return (f"The cost of waiting: investing {money(sip)} a month for {fc._fmt_years(total)} at {rate}% a year projects to {short(a['corpus'])}. "
            f"Starting {later} years later and investing for only {yrs} projects to {short(b['corpus'])}. "
            f"The delay costs about {short(a['corpus'] - b['corpus'])}, while the extra money invested in the longer plan is only {short(a['invested'] - b['invested'])}.")


def write(name: str, docs: list[str]) -> None:
    with (OUT / f"gen_{name}.jsonl").open("w", encoding="utf-8") as out:
        for text in docs:
            out.write(json.dumps({"text": text, "source": name}, ensure_ascii=False) + "\n")
    print(f"{name}: {len(docs)} documents", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)
    funds = load_funds()
    print(f"funds with NAV: {len(funds)}", flush=True)
    write("db_funds", [fund_doc(rng, f) for f in funds])
    write("db_families", family_docs(rng, funds))
    write("db_rankings", ranking_docs(funds))
    write("concepts", concept_docs())
    seen, docs = set(), []
    while len(docs) < 90000:
        text = calc_doc(rng)
        if text not in seen:
            seen.add(text)
            docs.append(text)
    write("calc", docs)


if __name__ == "__main__":
    main()
