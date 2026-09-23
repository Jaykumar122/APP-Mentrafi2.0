"""
Generate worked financial-arithmetic passages for the pretraining corpus.

Why synthesise this: "basic SIP or numerical reasoning" is one of the evaluation
axes for this model, and none of the open corpora contain Indian-market SIP /
CAGR / expense-drag arithmetic worked out step by step. GSM8K supplies generic
numeracy; this supplies the domain-specific kind.

Every number is computed in Python and formatted from the computed value, so the
arithmetic is correct by construction — the same faithful-by-construction rule
used for the SFT augmentation. Nothing here asserts a fact about a real fund.

Parameter values, names, phrasings and framings are all randomised so the model
learns the *method* rather than memorising instances (the failure mode the
original SFT set had, where 17 distinct picks covered 81% of examples).

Usage:
    python data_prep/gen_finance_numeracy.py --target_mib 80 --seed 7
"""

import argparse
import json
import math
import random
from pathlib import Path

OUT = Path("data/raw/corpus_v2/finance_numeracy.jsonl")
MIB = 1024 ** 2


def money(x):
    """Indian-style grouping, since the domain is Indian mutual funds."""
    x = round(x)
    s = str(abs(int(x)))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:]); head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if x < 0 else "") + s


def R(x):
    """Round to whole rupees AT THE POINT OF COMPUTATION, not at display time.

    Without this, a passage computes P = 32017.47, displays "Rs 32,017", then
    displays P x 60 as 19,21,048 — because the product used the unrounded value.
    The stated multiplication is then wrong by 28 rupees, and a model learning
    arithmetic from these passages learns the wrong product. Rounding first makes
    every displayed equation internally consistent.
    """
    return float(round(x))


def lakh(x):
    if x >= 1e7:
        return f"about Rs {x/1e7:.2f} crore"
    if x >= 1e5:
        return f"about Rs {x/1e5:.2f} lakh"
    return f"about Rs {money(x)}"


NAMES = ["Anita", "Rahul", "Priya", "Vikram", "Meera", "Arjun", "Sneha", "Karan",
         "Divya", "Rohit", "Neha", "Suresh", "Kavya", "Manish", "Pooja", "Aditya",
         "Farah", "Imran", "Lakshmi", "Nikhil", "Ritu", "Sanjay", "Tara", "Yusuf"]
GOALS = ["a home down payment", "their child's college fees", "retirement",
         "a car purchase", "an emergency buffer", "a wedding", "a sabbatical",
         "starting a business", "a family holiday", "postgraduate study"]

# Shared closing paragraphs. Eight templates repeated tens of thousands of times
# would teach the model the template rather than the method, and would waste
# budget re-reading identical prose. Drawing the closer from a shared pool makes
# the surface form combinatorial while every number stays computed, and it also
# means the passages are no longer near-duplicates of each other by construction.
CAVEATS = [
    """Every figure above follows from the stated assumptions. Change the assumed
return and the conclusion changes with it, which is why an assumed rate should
always be stated explicitly rather than buried in a projection.""",

    """A calculation like this is a planning tool, not a forecast. The arithmetic
is exact; the inputs are estimates, and the return assumption is the least
reliable of them.""",

    """Worth restating: this is arithmetic on an assumed return, not a prediction.
Market returns arrive unevenly, and a plan that only works at the assumed rate is
a plan without margin for error.""",

    """The sensitivity is worth checking before relying on any single number.
Recomputing at two or three percentage points lower shows how much of the outcome
depends on the return assumption rather than on the contributions.""",

    """None of this substitutes for matching the investment to the horizon. The
arithmetic is the same whatever the instrument, but the probability of actually
realising the assumed return is not.""",

    """Two figures are worth separating whenever a result like this is quoted: what
was contributed, and what compounding added. The first is certain, the second is
an assumption.""",

    """Numbers of this kind are best treated as a range rather than a point. The
same contributions at two percentage points either side of the assumption produce
materially different outcomes, and both are plausible.""",

    """Rounding is applied at each step above, so recomputing with unrounded
intermediate values will shift the final figure slightly. The method, not the last
rupee, is the point.""",

    """A plan built on this arithmetic still needs reviewing as circumstances
change. Income, horizon and goal amount all move over a period this long, and the
required contribution moves with them.""",

    """The assumed rate is a long-run average, not an annual entitlement. Realised
returns in individual years will sit well above and well below it, which matters
if the money might be needed at a bad moment.""",
]


def tail(r):
    return "\n" + r.choice(CAVEATS).strip() + "\n"


# --------------------------------------------------------------------------- #
# Each generator returns a prose passage with the arithmetic worked out.
# --------------------------------------------------------------------------- #
def sip_future_value(r):
    p = r.choice([1000, 1500, 2000, 2500, 3000, 5000, 7500, 10000, 12000, 15000, 20000, 25000])
    yrs = r.choice([3, 5, 7, 8, 10, 12, 15, 20, 25])
    ann = r.choice([8, 9, 10, 10.5, 11, 12, 12.5, 13, 14])
    n = yrs * 12
    i = ann / 100 / 12
    fv = R(p * (((1 + i) ** n - 1) / i) * (1 + i))
    invested = p * n
    gain = fv - invested
    name = r.choice(NAMES)
    goal = r.choice(GOALS)
    return f"""How a monthly SIP compounds: a worked example

{name} invests Rs {money(p)} per month through a systematic investment plan for
{yrs} years, targeting {goal}. The plan is assumed to earn {ann}% per year. What
does the investment grow to, and how much of that is contribution versus growth?

A SIP is a series of equal monthly investments, each of which compounds for a
different length of time. The first instalment compounds for the whole {n}
months; the last one compounds for a single month. Rather than adding up {n}
separate compound-interest calculations, the standard future-value-of-an-annuity
formula does it in one step:

    FV = P x [ ((1 + i)^n - 1) / i ] x (1 + i)

where P is the monthly instalment, i is the monthly rate, and n is the number of
instalments. The annual rate has to be converted to a monthly rate first:

    i = {ann}% / 12 = {i:.6f} per month
    n = {yrs} years x 12 = {n} instalments

Substituting:

    (1 + i)^n = (1 + {i:.6f})^{n} = {(1+i)**n:.4f}
    ((1 + i)^n - 1) / i = ({(1+i)**n:.4f} - 1) / {i:.6f} = {((1+i)**n - 1)/i:.4f}
    FV = {money(p)} x {((1+i)**n - 1)/i:.4f} x {1+i:.6f} = Rs {money(fv)}

So the corpus of the investment is {lakh(fv)}.

Separating contribution from growth matters for understanding the result:

    total invested = Rs {money(p)} x {n} months = Rs {money(invested)}
    growth        = Rs {money(fv)} - Rs {money(invested)} = Rs {money(gain)}

That is {gain/invested*100:.1f}% growth on the amount actually contributed, and
growth accounts for {gain/fv*100:.1f}% of the final value. Note that the growth
share rises sharply with tenure: the early instalments are the ones with enough
time to compound, which is why starting earlier usually matters more than
increasing the instalment later.
{tail(r)}"""


def required_sip(r):
    target = r.choice([500000, 1000000, 1500000, 2000000, 2500000, 5000000, 10000000])
    yrs = r.choice([3, 5, 7, 10, 12, 15, 20])
    ann = r.choice([8, 9, 10, 11, 12, 13])
    n = yrs * 12
    i = ann / 100 / 12
    p = R(target * i / (((1 + i) ** n - 1) * (1 + i)))
    name = r.choice(NAMES)
    goal = r.choice(GOALS)
    return f"""Working backwards from a goal to a monthly instalment

{name} wants {lakh(target)} in {yrs} years for {goal}. Assuming {ann}% per year,
how much needs to be invested every month?

This is the SIP future-value relationship solved for the instalment instead of
the final amount. Starting from

    FV = P x [ ((1 + i)^n - 1) / i ] x (1 + i)

and rearranging for P:

    P = FV x i / ( ((1 + i)^n - 1) x (1 + i) )

With i = {ann}%/12 = {i:.6f} and n = {n} months:

    (1 + i)^n = {(1+i)**n:.4f}
    ((1 + i)^n - 1) = {(1+i)**n - 1:.4f}
    denominator = {(1+i)**n - 1:.4f} x {1+i:.6f} = {((1+i)**n - 1)*(1+i):.4f}
    P = {money(target)} x {i:.6f} / {((1+i)**n - 1)*(1+i):.4f} = Rs {money(p)}

So roughly Rs {money(p)} per month is required. Over the full term that adds up
to Rs {money(p*n)} of contributions, with the remaining
Rs {money(target - p*n)} expected to come from compounding.

Two sanity checks are worth doing on any answer of this kind. First, the
contribution total must be less than the target, otherwise the assumed return is
doing no work. Second, a longer horizon should reduce the required instalment
steeply, because compounding has more time to contribute. If the required
instalment is uncomfortably large, the realistic levers are extending the
horizon, reducing the target, or accepting that a lower-risk allocation with a
lower expected return will need a bigger contribution.
{tail(r)}"""


def cagr_calc(r):
    start = r.choice([50000, 100000, 150000, 200000, 250000, 500000, 1000000])
    yrs = r.choice([2, 3, 4, 5, 6, 7, 8, 10, 12])
    growth = r.uniform(1.15, 3.6)
    end = R(start * growth)
    cagr = (end / start) ** (1 / yrs) - 1
    absolute = (end / start - 1) * 100
    return f"""Absolute return versus CAGR: why the two numbers differ

An investment of Rs {money(start)} is worth Rs {money(end)} after {yrs} years.
The absolute return is straightforward:

    absolute return = (final / initial - 1) x 100
                    = ({money(end)} / {money(start)} - 1) x 100
                    = {absolute:.2f}%

That {absolute:.2f}% figure is often quoted on its own, and on its own it is
close to meaningless, because it says nothing about how long the money was
invested. A {absolute:.2f}% gain over {yrs} years is a very different thing from
the same gain in one year. The compound annual growth rate answers "what steady
annual rate would have produced this result?":

    CAGR = (final / initial)^(1/years) - 1
         = ({end/start:.4f})^(1/{yrs}) - 1
         = {(end/start)**(1/yrs):.6f} - 1
         = {cagr*100:.2f}% per year

Verifying by compounding forward: Rs {money(start)} x (1 + {cagr:.6f})^{yrs}
= Rs {money(start * (1+cagr)**yrs)}, which recovers the final value, so the rate
is consistent.

CAGR is a smoothed figure. It describes the start and end points only and hides
the path taken, so two funds with the same CAGR can have had very different
year-to-year volatility. That is why CAGR is normally read alongside a measure of
variability rather than by itself, and why comparing a {yrs}-year CAGR against a
one-year return is not a like-for-like comparison.
{tail(r)}"""


def expense_drag(r):
    p = r.choice([50000, 75000, 100000, 150000, 200000, 250000, 400000, 500000,
                  750000, 1000000, 1500000, 2000000])
    yrs = r.choice([3, 5, 7, 10, 12, 15, 20, 25])
    gross = r.choice([9, 10, 10.5, 11, 11.5, 12, 13])
    er_a = r.choice([0.2, 0.3, 0.35, 0.4, 0.5, 0.6])
    er_b = r.choice([1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2])
    na = R(p * (1 + (gross - er_a) / 100) ** yrs)
    nb = R(p * (1 + (gross - er_b) / 100) ** yrs)
    return f"""What an expense ratio actually costs over time

The expense ratio is the annual cost of running a fund, expressed as a
percentage of assets and deducted from returns before the investor sees them. A
quoted return is already net of it. The reason a difference of one or two
percentage points deserves attention is that the cost recurs every year and is
therefore compounded against the investor.

Take Rs {money(p)} invested for {yrs} years in two funds that both earn {gross}%
per year before costs. One charges {er_a}%, the other {er_b}%.

    net return, lower-cost fund = {gross}% - {er_a}% = {gross - er_a}%
    net return, higher-cost fund = {gross}% - {er_b}% = {gross - er_b}%

Compounding each forward:

    lower-cost  = {money(p)} x (1 + {(gross-er_a)/100:.4f})^{yrs} = Rs {money(na)}
    higher-cost = {money(p)} x (1 + {(gross-er_b)/100:.4f})^{yrs} = Rs {money(nb)}

    difference = Rs {money(na)} - Rs {money(nb)} = Rs {money(na - nb)}

The annual cost gap is only {er_b - er_a:.2f} percentage points, yet over
{yrs} years it accounts for Rs {money(na - nb)}, which is {(na-nb)/nb*100:.1f}%
of the higher-cost outcome and {(na-nb)/p*100:.1f}% of the original investment.
The gap widens with the holding period, because the cost is charged on a base
that is itself growing.

The correct conclusion is not "always choose the cheapest fund". Cost is one
input among several: an actively managed fund can justify a higher ratio if it
delivers enough extra return net of that cost, though it has to clear the hurdle
every year. What the arithmetic does establish is that cost differences are not
a rounding error, and that they should be compared explicitly rather than
ignored in favour of headline past performance.
{tail(r)}"""


def real_return(r):
    nom = r.choice([6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 11, 12, 13, 14])
    inf = r.choice([3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5])
    real = (1 + nom / 100) / (1 + inf / 100) - 1
    naive = nom - inf
    amt = r.choice([50000, 100000, 200000, 300000, 500000, 750000, 1000000,
                    1500000, 2500000, 5000000])
    yrs = r.choice([3, 5, 7, 10, 12, 15, 20, 25])
    nominal_fv = R(amt * (1 + nom / 100) ** yrs)
    real_fv = R(amt * (1 + real) ** yrs)
    return f"""Nominal return, inflation, and what the money can actually buy

A return of {nom}% per year with inflation running at {inf}% does not leave the
investor {nom}% better off in purchasing-power terms. The common shortcut is to
subtract:

    approximate real return = {nom}% - {inf}% = {naive}%

The exact relationship divides rather than subtracts, because the return and the
price rise both apply to the same base over the same period:

    1 + real = (1 + nominal) / (1 + inflation)
    1 + real = (1 + {nom/100:.4f}) / (1 + {inf/100:.4f}) = {(1+nom/100)/(1+inf/100):.6f}
    real return = {real*100:.2f}% per year

The shortcut overstates the real return by
{(naive - real*100):.2f} percentage points here. The error is small at low rates
and grows as inflation rises, which is exactly when the distinction matters.

Applied to Rs {money(amt)} held for {yrs} years:

    nominal value = {money(amt)} x (1 + {nom/100:.4f})^{yrs} = Rs {money(nominal_fv)}
    value in today's purchasing power = {money(amt)} x (1 + {real:.6f})^{yrs} = Rs {money(real_fv)}

The account statement would show {lakh(nominal_fv)}, but it would buy what
{lakh(real_fv)} buys today. The Rs {money(nominal_fv - real_fv)} gap is not a
loss in any accounting sense; it is the part of the nominal gain consumed by
rising prices.

This is the main argument against holding a long-horizon goal entirely in very
low-return instruments. An instrument returning less than {inf}% has a negative
real return: the balance rises every year while the purchasing power falls.
{tail(r)}"""


def stepup_sip(r):
    p = r.choice([1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000, 12000,
                  15000, 20000])
    yrs = r.choice([8, 10, 12, 15, 18, 20, 22, 25])
    ann = r.choice([9, 10, 10.5, 11, 11.5, 12, 13])
    step = r.choice([5, 6, 7, 8, 10, 12])
    i = ann / 100 / 12
    # Year by year: each year's 12 instalments compound for the remaining months.
    fv = 0.0
    inst = float(p)
    for y in range(yrs):
        months_left = (yrs - y) * 12
        fv_year = inst * (((1 + i) ** 12 - 1) / i) * (1 + i) ** (months_left - 12)
        fv += fv_year
        inst *= (1 + step / 100)
    fv = R(fv)
    flat = R(p * (((1 + i) ** (yrs * 12) - 1) / i) * (1 + i))
    total_step = R(sum(R(p * (1 + step / 100) ** y) * 12 for y in range(yrs)))
    return f"""Step-up SIP: raising the instalment with income

A plain SIP keeps the instalment fixed for the whole term. A step-up SIP raises
it by a set percentage each year, which tracks salary growth and offsets the
fact that a fixed instalment loses real value to inflation over a long horizon.

Compare Rs {money(p)} per month for {yrs} years at {ann}%, with and without a
{step}% annual step-up.

The flat case uses the standard annuity formula over all {yrs*12} months:

    FV_flat = {money(p)} x [((1+{i:.6f})^{yrs*12} - 1)/{i:.6f}] x {1+i:.6f}
            = Rs {money(flat)}
    contributed = Rs {money(p)} x {yrs*12} = Rs {money(p*yrs*12)}

The step-up case cannot use one formula, because the instalment changes. It is
handled year by year: each year's twelve instalments are compounded as their own
small annuity, then carried forward for the months remaining after that year.

    year 1 instalment  = Rs {money(p)}
    year 2 instalment  = Rs {money(p*(1+step/100))}
    year {yrs} instalment = Rs {money(p*(1+step/100)**(yrs-1))}

    FV_step-up = Rs {money(fv)}
    contributed = Rs {money(total_step)}

    extra final value = Rs {money(fv - flat)}  ({(fv/flat - 1)*100:.1f}% more)
    extra contributed = Rs {money(total_step - p*yrs*12)}

The step-up produces {(fv/flat-1)*100:.1f}% more at the end, but it also
required more money in, so the honest comparison is between the two ratios of
final value to contribution: {fv/total_step:.3f} for the step-up against
{flat/(p*yrs*12):.3f} for the flat plan. The flat plan shows the higher ratio,
because its money went in earlier on average and so compounded longer. The
step-up wins on absolute outcome, not on efficiency per rupee, and it is chosen
because the larger later instalments are affordable, not because the mechanism
is inherently superior.
{tail(r)}"""


def allocation_split(r):
    amt = r.choice([50000, 100000, 200000, 300000, 500000, 750000, 1000000,
                    1500000, 2000000, 3000000, 5000000, 7500000])
    profile, eq, dt, gold = r.choice([
        ("conservative", 20, 70, 10), ("conservative", 25, 65, 10),
        ("moderately conservative", 35, 57, 8),
        ("moderately conservative", 40, 52, 8),
        ("moderate", 50, 42, 8), ("moderate", 55, 37, 8),
        ("moderately aggressive", 65, 30, 5),
        ("moderately aggressive", 70, 25, 5),
        ("aggressive", 75, 20, 5), ("aggressive", 80, 15, 5),
    ])
    yrs = {"conservative": 3, "moderately conservative": 4, "moderate": 6,
           "moderately aggressive": 9, "aggressive": 12}[profile]
    drift = r.choice([4, 5, 6, 7, 8, 9, 10, 12, 14])
    new_eq = eq + drift
    name = r.choice(NAMES)
    return f"""Turning a risk profile into rupee amounts, and rebalancing later

{name} is a {profile} investor with Rs {money(amt)} to allocate and a horizon of
roughly {yrs} years and above. A target allocation of {eq}% equity, {dt}% debt and
{gold}% gold translates into:

    equity = {eq}% of {money(amt)} = Rs {money(amt*eq/100)}
    debt   = {dt}% of {money(amt)} = Rs {money(amt*dt/100)}
    gold   = {gold}% of {money(amt)} = Rs {money(amt*gold/100)}
    check: {money(amt*eq/100)} + {money(amt*dt/100)} + {money(amt*gold/100)} = Rs {money(amt)}

The horizon is what makes this allocation defensible rather than the risk label
alone. Equity needs time to recover from drawdowns, so a {eq}% equity weight is
appropriate when the money is not required for {yrs}+ years. The same {profile}
label with an eighteen-month horizon would call for far less equity, because the
investor could be forced to sell during a decline.

Allocations drift as markets move. Suppose equity outperforms and the weight
rises from {eq}% to {new_eq}%. The portfolio now carries more risk than was
chosen:

    equity value = {new_eq}% of {money(amt)} = Rs {money(amt*new_eq/100)}
    target       = {eq}% of {money(amt)} = Rs {money(amt*eq/100)}
    excess       = Rs {money(amt*(new_eq-eq)/100)}

Rebalancing means moving that Rs {money(amt*(new_eq-eq)/100)} back into debt and
gold in their target proportions. Mechanically this sells what has risen and buys
what has not, which feels wrong and is the point: the purpose is to hold risk at
the chosen level, not to forecast which asset does better next. Rebalancing has a
cost in transaction charges and possibly tax, so it is usually done on a schedule
or when a band such as five percentage points is breached, rather than
continuously.
{tail(r)}"""


def compounding_frequency(r):
    p = r.choice([25000, 40000, 50000, 60000, 75000, 100000, 125000, 150000,
                  200000, 250000, 300000, 400000, 500000, 750000, 1000000])
    rate = r.choice([5, 5.5, 6, 6.5, 7, 7.25, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11])
    yrs = r.choice([2, 3, 4, 5, 6, 7, 8, 10, 12, 15])
    rows = []
    for label, m in (("annually", 1), ("half-yearly", 2), ("quarterly", 4),
                     ("monthly", 12), ("daily", 365)):
        v = R(p * (1 + rate / 100 / m) ** (m * yrs))
        rows.append((label, m, v))
    cont = R(p * math.exp(rate / 100 * yrs))
    name = r.choice(NAMES)
    return f"""Why compounding frequency changes the answer

{name} is comparing deposits that quote the same headline rate.
Rs {money(p)} at {rate}% for {yrs} years does not have a single answer until the
compounding frequency is specified. The general form is

    FV = P x (1 + r/m)^(m x t)

where m is the number of compounding periods per year. More frequent
compounding means interest starts earning interest sooner:

""" + "\n".join(
        f"    {label:12} (m={m:3}) : {money(p)} x (1 + {rate/100:.4f}/{m})^({m}x{yrs}) = Rs {money(v)}"
        for label, m, v in rows) + f"""

    continuous limit : {money(p)} x e^({rate/100:.4f} x {yrs}) = Rs {money(cont)}

The spread between annual and daily compounding is
Rs {money(rows[-1][2] - rows[0][2])}, which is
{(rows[-1][2]/rows[0][2] - 1)*100:.2f}% of the annually compounded result. The
increments shrink quickly: going from annual to quarterly gains far more than
going from monthly to daily, and the continuous case sets a ceiling that no
finite frequency exceeds.

This is why a stated rate is not comparable across products without knowing the
frequency. The effective annual rate puts them on the same footing:

    effective annual rate = (1 + r/m)^m - 1

At {rate}% compounded monthly that is
{((1 + rate/100/12)**12 - 1)*100:.3f}%, against the {rate}% nominal figure. Mutual
funds sidestep this particular confusion because they report realised NAV-based
returns rather than a promised compounding rate, but the same arithmetic governs
the fixed-income instruments they are usually compared against.
{tail(r)}"""


GENERATORS = [
    (sip_future_value, 0.20), (required_sip, 0.16), (cagr_calc, 0.14),
    (expense_drag, 0.14), (real_return, 0.12), (stepup_sip, 0.09),
    (allocation_split, 0.09), (compounding_frequency, 0.06),
]


def main():
    ap = argparse.ArgumentParser()
    # 48 MiB is ~12M tokens, roughly 1% of the 1.25B-token budget the
    # architecture benchmark solved for. Sizing note: an earlier 120 MiB run
    # repeated each of the 8 templates ~10,000 times. Numbers vary every time,
    # but the surrounding prose does not, so past a few thousand instances the
    # marginal passage teaches nothing new about the METHOD and only reinforces
    # the template -- the exact failure mode the old SFT set had. 48 MiB is
    # ~4,000 per template, which is enough to make the arithmetic patterns
    # dominant without spending 2.4% of the budget re-reading the same prose.
    ap.add_argument("--target_mib", type=float, default=48.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    r = random.Random(args.seed)
    fns = [f for f, _ in GENERATORS]
    weights = [w for _, w in GENERATORS]
    budget = args.target_mib * MIB

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = n = 0
    per_type = {}
    with open(out, "w", encoding="utf-8") as fh:
        while written < budget:
            fn = r.choices(fns, weights=weights, k=1)[0]
            text = fn(r).strip()
            fh.write(json.dumps({"text": text, "source": "finance_numeracy",
                                 "domain": "numeracy"}, ensure_ascii=False) + "\n")
            written += len(text.encode("utf-8"))
            n += 1
            per_type[fn.__name__] = per_type.get(fn.__name__, 0) + 1
            if n % 20000 == 0:
                print(f"  {n:,} passages  {written/MIB:.1f}/{args.target_mib:.0f} MiB",
                      flush=True)

    print(f"\nwrote {n:,} passages, {written/MIB:.1f} MiB "
          f"(~{written/4.2/1e6:.1f}M tokens) -> {out}")
    print(f'{"generator":24} {"count":>9} {"share":>7}')
    print("-" * 42)
    for k, v in sorted(per_type.items(), key=lambda kv: -kv[1]):
        print(f"{k:24} {v:9,} {v/n*100:6.1f}%")


if __name__ == "__main__":
    main()
