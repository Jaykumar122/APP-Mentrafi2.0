"""Synthetic GROUNDED SFT conversations for Network 3 (llm120).

Each example is what the live pipeline gives the model: Network 1 intent and
risk, Network 2 slots, exact numbers from inference/financial_calculator.py,
and fund candidates from retrieval, rendered by llm120.grounding. The target
answer uses only those facts, so the model learns to COPY figures and scheme
names from context instead of inventing them.

Fund names are fictional on purpose (no real AMC brands): the skill being
taught is copying whatever the retriever supplies, not memorizing schemes.
Every answer is checked with llm120.grounding.ungrounded_numbers.

Used by data_prep/build_llm120_sft.py; run alone to preview:
    python -m data_prep.build_grounded_sft --count 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

from inference import financial_calculator as fc
from llm120.grounding import (TAX_AS_OF, TAX_DEBT, TAX_EQUITY, Allocation, FundFact, GroundedContext,
                              build_messages, to_model_text, ungrounded_numbers)

AMCS = ["Aravalli", "Kaveri", "Nilgiri", "Sahyadri", "Narmada", "Vindhya", "Godavari", "Lotus Bay",
        "Meridian India", "Saptarishi", "Konark", "Shivalik", "Tapti", "Indus Crest", "Brahmaputra",
        "Malabar", "Deccan Peak", "Chambal", "Ajanta", "Mahanadi", "Coromandel", "Satpura", "Ellora",
        "Kanchan", "Tungabhadra", "Pench", "Sunderban", "Doon Valley", "Kalinga", "Thar Horizon"]

# category -> (display name, assumed return %, 1y range, 3y/5y range, expense range, riskometer, equity?)
CATEGORIES = {
    "liquid": ("Liquid Fund", 6.5, (6.0, 7.5), (5.5, 7.0), (0.10, 0.30), "Low to Moderate", False),
    "arbitrage": ("Arbitrage Fund", 6.5, (6.0, 8.0), (5.5, 7.0), (0.20, 0.45), "Low", True),
    "short_debt": ("Short Duration Fund", 7.0, (6.5, 8.5), (5.8, 7.5), (0.25, 0.60), "Moderate", False),
    "corp_bond": ("Corporate Bond Fund", 7.0, (6.5, 8.5), (6.0, 7.8), (0.25, 0.55), "Moderate", False),
    "cons_hybrid": ("Conservative Hybrid Fund", 8.0, (7.0, 12.0), (7.0, 10.0), (0.40, 1.10), "Moderately High", False),
    "baf": ("Balanced Advantage Fund", 10.0, (6.0, 18.0), (9.0, 14.0), (0.50, 1.00), "Very High", True),
    "large_cap": ("Large Cap Fund", 11.0, (2.0, 22.0), (11.0, 17.0), (0.50, 1.10), "Very High", True),
    "index": ("Nifty 50 Index Fund", 11.0, (3.0, 20.0), (11.0, 15.0), (0.10, 0.30), "Very High", True),
    "flexi_cap": ("Flexi Cap Fund", 12.0, (0.0, 28.0), (12.0, 20.0), (0.50, 1.10), "Very High", True),
    "mid_cap": ("Mid Cap Fund", 13.0, (-5.0, 35.0), (15.0, 25.0), (0.50, 1.10), "Very High", True),
    "small_cap": ("Small Cap Fund", 14.0, (-10.0, 40.0), (16.0, 28.0), (0.50, 1.20), "Very High", True),
    "elss": ("ELSS Tax Saver Fund", 12.0, (0.0, 28.0), (12.0, 20.0), (0.50, 1.10), "Very High", True),
}

ALLOCATIONS = {
    ("conservative", "short"): [("liquid", 60), ("short_debt", 40)],
    ("conservative", "medium"): [("short_debt", 50), ("cons_hybrid", 35), ("index", 15)],
    ("conservative", "long"): [("cons_hybrid", 45), ("corp_bond", 35), ("large_cap", 20)],
    ("moderate", "short"): [("liquid", 50), ("arbitrage", 50)],
    ("moderate", "medium"): [("baf", 50), ("short_debt", 30), ("large_cap", 20)],
    ("moderate", "long"): [("flexi_cap", 45), ("large_cap", 35), ("mid_cap", 20)],
    ("aggressive", "short"): [("arbitrage", 60), ("liquid", 40)],
    ("aggressive", "medium"): [("flexi_cap", 40), ("large_cap", 35), ("baf", 25)],
    ("aggressive", "long"): [("flexi_cap", 40), ("mid_cap", 35), ("small_cap", 25)],
}

NAMES = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Arjun", "Kavya", "Rahul", "Meera", "Imran",
         "Fatima", "Joseph", "Neha", "Siddharth", "Lakshmi", "Harpreet", "Deepak", "Pooja", "Karthik", "Zoya",
         "Manish", "Divya", "Suresh", "Anjali", "Farhan", "Gauri", "Nikhil", "Ritu", "Tenzin"]
OCCUPATIONS = ["software engineer", "teacher", "doctor", "business owner", "student", "government employee",
               "freelancer", "bank employee", "nurse", "shop owner", "salaried professional", "homemaker"]
GOALS = ["retirement", "child's higher education", "buying a house", "wealth creation", "an emergency fund",
         "a car purchase", "a foreign trip", "my daughter's wedding", "early retirement", "tax saving"]

CLOSINGS = [
    "Please treat this as an illustration based on the figures above, not a guarantee, and review it once a year.",
    "These projections are assumptions, not promises; actual returns can be lower or even negative.",
    "Do check each scheme's latest factsheet before investing, since the figures above are dated.",
    "Remember that past returns do not predict future returns, so review the plan as your situation changes.",
]


def horizon_bucket(years: float) -> str:
    return "short" if years < 3 else "medium" if years <= 5 else "long"


def pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}%"


SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "mf_corpus" / "fund_snapshot.json"
_SNAPSHOT: dict | None = None


def snapshot() -> dict:
    """Real Direct-Growth schemes (name, NAV, NAV-based returns) from the funds DB, if snapshotted."""
    global _SNAPSHOT
    if _SNAPSHOT is None:
        _SNAPSHOT = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8")) if SNAPSHOT_PATH.exists() else {}
    return _SNAPSHOT


def make_fund(rng: random.Random, key: str, used: set[str]) -> FundFact:
    display, _, r1, r35, er, risk, _ = CATEGORIES[key]
    real = [f for f in snapshot().get(key, []) if f["name"] not in used]
    if real:
        f = rng.choice(real)
        used.add(f["name"])
        pct1 = lambda v: None if v is None else pct(v)
        return FundFact(name=f["name"], category=display, plan="Direct Growth", nav=f"₹{f['nav']:,.2f}",
                        return_1y=pct1(f["return_1y"]), return_3y=pct1(f["return_3y"]), return_5y=pct1(f["return_5y"]))
    while True:
        name = f"{rng.choice(AMCS)} {display} - Direct Growth"
        if name not in used:
            used.add(name)
            break
    return FundFact(
        name=name, category=display, plan="Direct Growth",
        nav=f"₹{rng.uniform(10, 950):,.2f}", return_1y=pct(rng.uniform(*r1)),
        return_3y=pct(rng.uniform(*r35)), return_5y=None if rng.random() < 0.25 else pct(rng.uniform(*r35)))


def round_budget(rng: random.Random) -> int:
    return rng.choice([500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7500, 8000, 10000, 12000, 15000,
                       20000, 25000, 30000, 40000, 50000, 75000, 100000])


def years_text(years: float) -> str:
    return fc._fmt_years(years)


def split_amounts(budget: int, weights: list[int]) -> list[int]:
    amounts, used = [], 0
    for index, weight in enumerate(weights):
        amount = budget - used if index == len(weights) - 1 else int(round(budget * weight / 100))
        used += amount
        amounts.append(amount)
    return amounts


def blended_rate(keys_weights) -> float:
    return round(sum(CATEGORIES[key][1] * weight for key, weight in keys_weights) / 100.0, 1)


def money(value: float) -> str:
    return fc.format_inr_exact(value)


# ─────────────────────────────── question templates ───────────────────────────────
def portfolio_question(rng, p) -> str:
    pieces = []
    if p.get("name") and rng.random() < 0.6:
        pieces.append(rng.choice([f"Hi, I am {p['name']}.", f"My name is {p['name']}.", f"Hello, {p['name']} here."]))
    if p.get("age"):
        pieces.append(rng.choice([f"I am {p['age']} years old", f"I'm {p['age']}", f"Age {p['age']}"]) +
                      (f" and work as a {p['occupation']}." if p.get("occupation") else "."))
    budget = money(p["budget"])
    pieces.append(rng.choice([
        f"I can invest {budget} every month for {years_text(p['years'])}.",
        f"I want to start a SIP of {budget} per month for {years_text(p['years'])}.",
        f"Monthly budget {budget}, horizon {years_text(p['years'])}.",
        f"Planning to put {budget} monthly for the next {years_text(p['years'])}.",
    ]))
    if p.get("goal"):
        pieces.append(rng.choice([f"The goal is {p['goal']}.", f"This is for {p['goal']}.", f"Goal: {p['goal']}."]))
    if p.get("risk_stated"):
        pieces.append(rng.choice([f"My risk appetite is {p['risk']}.", f"I am a {p['risk']} investor.",
                                  f"Risk: {p['risk']}."]))
    pieces.append(rng.choice(["Which funds should I choose?", "Please suggest a portfolio.", "What should I invest in?",
                              "Can you build a plan for me?", "Suggest the best mutual funds for me.",
                              "How should I split this money?"]))
    return " ".join(pieces)


# ─────────────────────────────── scenario builders ───────────────────────────────
def profile(rng: random.Random, *, risk=None, years=None, budget=None) -> dict:
    p = {"risk": risk or rng.choice(["conservative", "moderate", "aggressive"]),
         "years": years or rng.choice([1, 2, 3, 4, 5, 7, 8, 10, 12, 15, 20, 25]),
         "budget": budget or round_budget(rng)}
    if rng.random() < 0.6:
        p["name"] = rng.choice(NAMES)
    if rng.random() < 0.7:
        p["age"] = rng.randint(19, 58)
    if rng.random() < 0.5:
        p["occupation"] = rng.choice(OCCUPATIONS)
    if rng.random() < 0.6:
        p["goal"] = rng.choice(GOALS)
    p["risk_stated"] = rng.random() < 0.7
    return p


def profile_fields(p: dict) -> dict[str, str]:
    fields = {"monthly_budget": money(p["budget"]), "horizon": years_text(p["years"])}
    for key in ("name", "occupation", "goal"):
        if p.get(key):
            fields[key] = p[key]
    if p.get("age"):
        fields["age"] = str(p["age"])
    return fields


def portfolio_example(rng: random.Random) -> tuple[str, GroundedContext, str]:
    p = profile(rng)
    bucket = horizon_bucket(p["years"])
    plan = ALLOCATIONS[(p["risk"], bucket)]
    amounts = split_amounts(p["budget"], [w for _, w in plan])
    used: set[str] = set()
    funds, allocation, missing_funds = [], [], []
    for (key, weight), amount in zip(plan, amounts):
        if rng.random() < 0.1:  # retrieval found nothing for this category
            allocation.append(Allocation(CATEGORIES[key][0], money(amount), f"{weight}%"))
            missing_funds.append(CATEGORIES[key][0])
            continue
        funds.append(make_fund(rng, key, used))
        allocation.append(Allocation(CATEGORIES[key][0], money(amount), f"{weight}%", len(funds)))
    rate = blended_rate(plan)
    sip = fc.calculate_sip(p["budget"], p["years"], rate)
    step = fc.calculate_step_up_sip(p["budget"], p["years"], rate, 10.0)
    calcs = {"months": str(sip["months"]), "total invested": sip["invested_fmt"],
             "projected corpus": sip["corpus_fmt"], "estimated gain": sip["gain_fmt"],
             "value in today's money": sip["real_corpus_fmt"], "with 10% yearly step-up": step["corpus_fmt"]}
    any_equity = any(CATEGORIES[key][6] for key, _ in plan)
    ctx = GroundedContext(
        intent="PORTFOLIO_RECOMMENDATION", risk=p["risk"], profile=profile_fields(p), calculations=calcs,
        assumptions=[f"blended return {rate}% a year", "start-of-month installments", "6% inflation"],
        allocation=allocation, funds=funds, data_as_of=rand_date(rng),
        tax_notes=[TAX_EQUITY if any_equity else TAX_DEBT, TAX_AS_OF])
    if bucket == "short":
        ctx.notes.append("Horizon under 3 years, so no equity.")
    question = portfolio_question(rng, p)

    opener = (f"{rng.choice(['Thank you', 'Thanks', 'Happy to help'])}, {p['name']}. " if p.get("name") else "")
    s = [opener + rng.choice([
        f"For a {p['risk']} investor putting {money(p['budget'])} a month for {years_text(p['years'])}, here is a plan based on the verified data.",
        f"Based on your {money(p['budget'])} monthly budget, {years_text(p['years'])} horizon and {p['risk']} risk profile, this is how the engine splits your SIP.",
        f"Here is a {p['risk']} plan for {money(p['budget'])} a month over {years_text(p['years'])}, using only the verified figures.",
    ])]
    if not p["risk_stated"]:
        s.append(f"You did not state your risk appetite, so the plan uses the {p['risk']} profile from your earlier details; tell me if that is wrong.")
    for alloc in allocation:
        if alloc.fund_index:
            fund = funds[alloc.fund_index - 1]
            facts = [f"3-year return {fund.return_3y}"]
            if rng.random() < 0.6:
                facts.insert(0, f"NAV {fund.nav}")
            if fund.return_1y and rng.random() < 0.4:
                facts.append(f"1-year return {fund.return_1y}")
            if fund.return_5y and rng.random() < 0.5:
                facts.append(f"5-year return {fund.return_5y}")
            s.append(rng.choice([
                f"Put {alloc.amount} ({alloc.percent}) in {fund.name}, with {', '.join(facts)}.",
                f"{alloc.percent}, or {alloc.amount} a month, goes to {fund.name} ({', '.join(facts)}).",
                f"For {alloc.category}, {alloc.amount} a month ({alloc.percent}) can go to {fund.name}, which shows {', '.join(facts)}.",
            ]))
        else:
            s.append(f"The {alloc.category} share of {alloc.amount} ({alloc.percent}) has no verified scheme in the data, "
                     f"so I cannot name one; I can search again if you like.")
    s.append(rng.choice([
        f"At an assumed {rate}% a year, investing {calcs['total invested']} over {calcs['months']} months projects to about {calcs['projected corpus']}, an estimated gain of {calcs['estimated gain']}.",
        f"Over {calcs['months']} months you would invest {calcs['total invested']}; at the assumed {rate}% a year this could grow to {calcs['projected corpus']}.",
    ]))
    todays = calcs["value in today's money"]
    if rng.random() < 0.7:
        s.append(f"In today's money, after 6% inflation, that is roughly {todays}.")
    if rng.random() < 0.6:
        s.append(f"If you raise the SIP by 10% every year, the projection becomes {calcs['with 10% yearly step-up']}.")
    if bucket == "short":
        s.append("Because the horizon is under 3 years, the plan avoids equity, since losses near withdrawal are hard to recover.")
    if rng.random() < 0.5:
        s.append(rng.choice([
            "On tax, equity fund gains are taxed at 12.5% above ₹1.25 Lakh a year after 12 months, or 20% within 12 months, plus 4% cess."
            if any_equity else "On tax, gains from these debt funds are taxed at your slab rate.",
        ]))
    s.append(rng.choice(CLOSINGS))
    return question, ctx, " ".join(s)


def sip_calculation_example(rng: random.Random) -> tuple[str, GroundedContext, str]:
    budget = round_budget(rng)
    years = rng.choice([1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30])
    rate = rng.choice([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    rate_given = rng.random() < 0.6
    if not rate_given:
        rate = 12
    sip = fc.calculate_sip(budget, years, rate)
    calcs = {"monthly SIP": money(budget), "months": str(sip["months"]), "total invested": sip["invested_fmt"],
             "projected corpus": sip["corpus_fmt"], "estimated gain": sip["gain_fmt"],
             "value in today's money": sip["real_corpus_fmt"]}
    step_pct = rng.choice([5, 10, 15])
    with_step = rng.random() < 0.5
    if with_step:
        step = fc.calculate_step_up_sip(budget, years, rate, step_pct)
        calcs[f"with {step_pct}% yearly step-up"] = step["corpus_fmt"]
        calcs["step-up total invested"] = step["invested_fmt"]
    ctx = GroundedContext(intent="SIP_CALCULATION", risk=None, profile={"monthly_budget": money(budget),
                          "horizon": years_text(years)}, calculations=calcs,
                          assumptions=[f"return {rate}% a year" + ("" if rate_given else " (default, user gave none)"),
                                       "start-of-month installments", "6% inflation"])
    rate_q = f" at {rate}% return" if rate_given else ""
    question = rng.choice([
        f"If I invest {money(budget)} per month for {years_text(years)}{rate_q}, how much will I get?",
        f"Calculate SIP of {money(budget)} for {years_text(years)}{rate_q}.",
        f"What will a {money(budget)} monthly SIP become in {years_text(years)}{rate_q}?",
        f"How much corpus for {money(budget)}/month SIP over {years_text(years)}{rate_q}?",
    ]) + (f" Also show a {step_pct}% annual step-up." if with_step else "")
    s = [rng.choice([
        f"A SIP of {money(budget)} a month for {years_text(years)} means {calcs['months']} installments, so you invest {calcs['total invested']} in total.",
        f"Over {years_text(years)} you make {calcs['months']} monthly payments of {money(budget)}, which adds up to {calcs['total invested']}.",
    ])]
    s.append(f"At an assumed {rate}% a year" + ("" if rate_given else ", the default since you did not give a rate,") +
             f" the projected corpus is {calcs['projected corpus']}, an estimated gain of {calcs['estimated gain']}.")
    todays = calcs["value in today's money"]
    s.append(f"Adjusted for 6% inflation, that corpus is worth about {todays} in today's money.")
    if with_step:
        step_corpus = calcs[f"with {step_pct}% yearly step-up"]
        s.append(f"With a {step_pct}% step-up every year you would invest {calcs['step-up total invested']} and the projection rises to {step_corpus}.")
    s.append(rng.choice(["The calculation assumes installments at the start of each month and a constant return, which real funds do not deliver.",
                         "Real returns vary year to year and can be negative, so treat this as an estimate."]))
    return question, ctx, " ".join(s)


def goal_example(rng: random.Random) -> tuple[str, GroundedContext, str]:
    target = rng.choice([500000, 1000000, 2000000, 2500000, 5000000, 7500000, 10000000, 20000000, 30000000, 50000000])
    years = rng.choice([3, 5, 7, 8, 10, 12, 15, 18, 20, 25])
    rate = rng.choice([8, 10, 11, 12, 13])
    goal = fc.calculate_goal_sip(target, years, rate)
    step = fc.calculate_goal_step_up_sip(target, years, rate, 10)
    calcs = {"target": goal["target_fmt"], "months": str(goal["months"]), "required monthly SIP": goal["required_sip_fmt"],
             "total invested": goal["invested_fmt"], "step-up starting SIP (10% yearly)": step["starting_sip_fmt"],
             "target in today's money": fc.format_inr(fc.real_value(target, years))}
    budget = round_budget(rng) if rng.random() < 0.5 else None
    if budget:
        projection = fc.calculate_sip(budget, years, rate)
        calcs["user budget projects to"] = projection["corpus_fmt"]
        gap = target - projection["corpus"]
        calcs["shortfall" if gap > 0 else "surplus"] = fc.format_inr(abs(gap))
    goal_name = rng.choice(GOALS)
    ctx = GroundedContext(intent="GOAL_PLANNING", risk=None,
                          profile={"goal": goal_name, "horizon": years_text(years),
                                   **({"monthly_budget": money(budget)} if budget else {})},
                          calculations=calcs, assumptions=[f"return {rate}% a year", "start-of-month installments",
                                                           "6% inflation"])
    question = rng.choice([
        f"I need {goal['target_fmt']} in {years_text(years)} for {goal_name}. How much SIP should I do?",
        f"How much should I invest monthly to reach {goal['target_fmt']} in {years_text(years)}?",
        f"Target {goal['target_fmt']} for {goal_name} in {years_text(years)}, what monthly SIP is needed?",
    ]) + (f" I can manage {money(budget)} a month." if budget else "")
    s = [f"To reach {calcs['target']} in {years_text(years)} at an assumed {rate}% a year, you need a SIP of about {calcs['required monthly SIP']} a month for {calcs['months']} months, investing {calcs['total invested']} in total."]
    s.append(f"If your income grows, you could instead start at {calcs['step-up starting SIP (10% yearly)']} and raise it by 10% every year.")
    if budget:
        if "shortfall" in calcs:
            s.append(f"Your budget of {money(budget)} a month projects to {calcs['user budget projects to']}, which is short of the target by {calcs['shortfall']}; you could extend the horizon, step up the SIP, or revise the target.")
        else:
            s.append(f"Your budget of {money(budget)} a month projects to {calcs['user budget projects to']}, which covers the target with {calcs['surplus']} to spare under these assumptions.")
    target_today = calcs["target in today's money"]
    if rng.random() < 0.6:
        s.append(f"Keep in mind that {calcs['target']} after {years_text(years)} is worth about {target_today} in today's money at 6% inflation.")
    if years <= 5:
        s.append("With a horizon this short, a large equity share could lose value near the deadline, so a lower-risk mix and a lower return assumption may be more realistic.")
    s.append(rng.choice(CLOSINGS))
    return question, ctx, " ".join(s)


def missing_info_example(rng: random.Random) -> tuple[str, GroundedContext, str]:
    p = profile(rng)
    missing = rng.choice([["monthly SIP budget"], ["horizon"], ["monthly SIP budget", "horizon"], ["risk appetite"]])
    fields = profile_fields(p)
    if "monthly SIP budget" in missing:
        fields.pop("monthly_budget")
    if "horizon" in missing:
        fields.pop("horizon")
    risk = None if "risk appetite" in missing else p["risk"]
    ctx = GroundedContext(intent="PORTFOLIO_RECOMMENDATION", risk=risk, profile=fields, missing=missing)
    bits = []
    if fields.get("age"):
        bits.append(f"I am {fields['age']}.")
    if fields.get("monthly_budget"):
        bits.append(f"I can invest {fields['monthly_budget']} a month.")
    if fields.get("horizon"):
        bits.append(f"I want to stay invested for {fields['horizon']}.")
    if risk and rng.random() < 0.6:
        bits.append(f"I am a {risk} investor.")
    bits.append(rng.choice(["Which mutual funds should I buy?", "Suggest a portfolio for me.", "Where should I invest?"]))
    question = " ".join(bits)
    asks = {"monthly SIP budget": "how much you can invest each month",
            "horizon": "how many years you can stay invested before you need the money",
            "risk appetite": "how much temporary loss you can tolerate (conservative, moderate or aggressive)"}
    s = [rng.choice(["I can help with that, but I need a little more information first.",
                     "Happy to help. Before I suggest an allocation, a few details are missing.",
                     "Sure. To build a sensible plan I need a couple more details."])]
    s.append("Please tell me " + " and ".join(asks[m] for m in missing) + ".")
    if "horizon" in missing:
        s.append("The horizon matters most, because money needed within three years should not be in equity funds.")
    if "risk appetite" in missing:
        s.append("Your risk appetite decides how much of the SIP can go into equity.")
    if rng.random() < 0.5:
        s.append("It also helps to know your goal and whether you already have an emergency fund.")
    s.append("Once I have these, I will show an allocation with verified schemes and exact projections.")
    return question, ctx, " ".join(s)


def comparison_example(rng: random.Random) -> tuple[str, GroundedContext, str]:
    key = rng.choice(["large_cap", "flexi_cap", "mid_cap", "small_cap", "index", "baf", "elss", "short_debt"])
    used: set[str] = set()
    funds = [make_fund(rng, key, used) for _ in range(rng.choice([2, 2, 3]))]
    ctx = GroundedContext(intent="FUND_COMPARISON", risk=rng.choice([None, "moderate", "aggressive", "conservative"]),
                          funds=funds, data_as_of=rand_date(rng))
    names = [f.name for f in funds]
    question = rng.choice([f"Compare {' and '.join(names)}.", f"Which is better: {' or '.join(names)}?",
                           f"{' vs '.join(names)} - which one should I pick?"])
    s = [f"Here is a side-by-side view using the data as of {ctx.data_as_of}."]
    for f in funds:
        bits = [f"NAV {f.nav}", f"3-year return {f.return_3y}"]
        if f.return_1y:
            bits.insert(1, f"1-year return {f.return_1y}")
        if f.return_5y:
            bits.append(f"5-year return {f.return_5y}")
        s.append(f"{f.name} shows {', '.join(bits)}.")
    best3 = max(funds, key=lambda f: float(f.return_3y.rstrip("%")))
    s.append(f"{best3.name} has the highest 3-year return at {best3.return_3y}, but past returns do not guarantee future performance.")
    s.append(f"All of them are in the {funds[0].category} category, so the category risk is similar.")
    if ctx.risk == "conservative" and CATEGORIES[key][6]:
        s.append("Since your profile is conservative, a high-risk equity category like this may not suit you at all.")
    s.append("Also compare costs, portfolio overlap, fund manager tenure and consistency across market cycles in the latest factsheets before deciding.")
    return question, ctx, " ".join(s)


def not_in_context_example(rng: random.Random) -> tuple[str, GroundedContext, str]:
    key = rng.choice(list(CATEGORIES))
    used: set[str] = set()
    funds = [make_fund(rng, key, used) for _ in range(rng.choice([0, 1, 2]))]
    wanted_key = rng.choice([k for k in CATEGORIES if k != key])
    asked = f"{rng.choice([a for a in AMCS])} {CATEGORIES[wanted_key][0]}"
    asked = asked if asked not in used else asked + " Plus"
    ctx = GroundedContext(intent=rng.choice(["FUND_COMPARISON", "GENERAL_CHAT"]), funds=funds,
                          data_as_of=rand_date(rng) if funds else None)
    question = rng.choice([f"What is the current NAV and return of {asked}?", f"Is {asked} a good fund?",
                           f"Tell me about {asked}.", f"Should I invest in {asked}?"])
    s = [f"I do not have verified data for {asked}, so I cannot quote its NAV, returns or costs."]
    if funds:
        s.append("The schemes I do have verified figures for are " + " and ".join(f.name for f in funds) +
                 f", as of {ctx.data_as_of}.")
        s.append("I can explain those, or you can check the latest factsheet of the scheme you asked about on its AMC or AMFI website.")
    else:
        s.append("Please check the scheme's latest factsheet on the AMC or AMFI website, or ask me to search the database again with the exact scheme name.")
    s.append("I will not guess figures, because an invented number could mislead your decision.")
    return question, ctx, " ".join(s)


def risk_mismatch_example(rng: random.Random) -> tuple[str, GroundedContext, str]:
    years = rng.choice([1, 2, 3, 4, 5, 10, 15])
    risk = "conservative" if years > 3 or rng.random() < 0.5 else rng.choice(["moderate", "conservative"])
    budget = round_budget(rng)
    wanted = rng.choice(["small_cap", "mid_cap", "flexi_cap"]) if years > 3 else rng.choice(["small_cap", "mid_cap", "large_cap"])
    used: set[str] = set()
    funds = [make_fund(rng, wanted, used)]
    reason = (f"Horizon {years_text(years)} is under 3 years; equity can fall sharply before the money is needed."
              if years < 3 else f"Requested {CATEGORIES[wanted][0]} is a high-volatility equity category, above a {risk} profile.")
    ctx = GroundedContext(intent="PORTFOLIO_RECOMMENDATION", risk=risk,
                          profile={"monthly_budget": money(budget), "horizon": years_text(years)},
                          funds=funds, data_as_of=rand_date(rng), notes=["Suitability warning: " + reason])
    question = rng.choice([
        f"I want to put my {money(budget)} monthly SIP in {funds[0].name} for {years_text(years)}. Good idea?",
        f"Should I invest {money(budget)} a month in {funds[0].name}? I need the money in {years_text(years)}.",
    ]) + (f" I am a {risk} investor." if rng.random() < 0.6 else "")
    s = [f"{funds[0].name} shows a 3-year return of {funds[0].return_3y}, but I would be cautious here."]
    if years < 3:
        s.append(f"You need the money in {years_text(years)}, and a {funds[0].category} can fall sharply in that time with little chance to recover before withdrawal.")
        s.append("For money needed this soon, lower-volatility options such as liquid or short duration debt funds are usually considered, though they also carry some risk.")
    else:
        s.append(f"It is a {funds[0].category}, an equity category that can swing sharply, which is above what a {risk} risk profile usually accepts.")
        s.append("If you still want some equity, a smaller share in a large cap or hybrid fund, with the rest in debt, would match your profile better.")
    s.append("Past returns, however strong, do not reduce future volatility. If your risk appetite has changed, tell me and I will rebuild the plan with the verified data.")
    return question, ctx, " ".join(s)


def tax_example(rng: random.Random) -> tuple[str, GroundedContext, str]:
    gain = rng.choice([40000, 80000, 100000, 150000, 200000, 250000, 300000, 500000, 750000])
    fund_type = rng.choice(["equity", "equity", "debt", "other"])
    holding = rng.choice([6, 10, 12, 13, 18, 24, 30, 48])
    slab = rng.choice([None, 5, 20, 30]) if fund_type != "equity" else None
    res = fc.calculate_capital_gains_tax(gain, holding, fund_type, slab_rate_pct=slab)
    calcs = {"gain": money(gain), "holding": f"{holding} months", "classification": f"{res['term']} ({res['section']})"}
    if res["needs_slab_rate"]:
        calcs["tax"] = "needs slab rate"
    else:
        if res["exemption_used"]:
            calcs["exemption used"] = money(res["exemption_used"])
        calcs.update({"taxable gain": money(res["taxable_gain"]), "rate": f"{res['rate_pct']:g}%",
                      "tax before cess": money(res["base_tax"]), "cess": money(res["cess"]),
                      "total tax": money(res["total_tax"]), "gain after tax": money(res["post_tax_gain"])})
    ctx = GroundedContext(intent="TAX_CALCULATION", calculations=calcs,
                          assumptions=[res["fund_class"], "no other equity gains this year", "surcharge not included"],
                          tax_notes=[TAX_AS_OF])
    kind = {"equity": "equity fund", "debt": "debt fund", "other": "hybrid fund"}[fund_type]
    question = rng.choice([
        f"I sold my {kind} units after {holding} months with a profit of {money(gain)}. How much tax will I pay?",
        f"Tax on {money(gain)} gain from a {kind} held for {holding} months?",
    ]) + (f" My slab is {slab}%." if slab else "")
    s = [f"Your gain of {money(gain)} after {holding} months is treated as {calcs['classification']}."]
    if res["needs_slab_rate"]:
        s.append("These gains are added to your income and taxed at your slab rate, so please tell me your income-tax slab rate and I can give the exact amount.")
    else:
        if res["exemption_used"]:
            s.append(f"The yearly exemption covers {calcs['exemption used']} of it, leaving {calcs['taxable gain']} taxable at {calcs['rate']}.")
        else:
            s.append(f"The taxable gain is {calcs['taxable gain']} at {calcs['rate']}.")
        s.append(f"That is {calcs['tax before cess']} plus {calcs['cess']} cess, so the total tax is about {calcs['total tax']} and you keep {calcs['gain after tax']}.")
    s.append("This assumes no other equity gains this year and excludes surcharge, so please confirm with a tax professional.")
    return question, ctx, " ".join(s)


def rand_date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28):02d}-{rng.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])}-{rng.choice([2025, 2026])}"


GENERATORS: list[tuple[str, Callable, float]] = [
    ("grounded_portfolio", portfolio_example, 0.34),
    ("grounded_sip_calculation", sip_calculation_example, 0.16),
    ("grounded_goal", goal_example, 0.14),
    ("grounded_missing_info", missing_info_example, 0.09),
    ("grounded_comparison", comparison_example, 0.09),
    ("grounded_not_in_context", not_in_context_example, 0.06),
    ("grounded_risk_mismatch", risk_mismatch_example, 0.06),
    ("grounded_tax", tax_example, 0.06),
]


SHORT_HISTORY_TOPICS = ["grounded_missing_info", "grounded_sip_calculation", "grounded_goal", "grounded_tax"]


def grounded_records(count: int = 3000, seed: int = 20260520, *, fits: Callable[[list], bool] | None = None) -> list[dict]:
    """Deterministic grounded conversations; ~15% carry one earlier turn as history."""
    rng = random.Random(seed)
    topics = [name for name, _, _ in GENERATORS]
    weights = [weight for _, _, weight in GENERATORS]
    builders = {name: builder for name, builder, _ in GENERATORS}
    records, seen = [], set()
    attempts = 0
    while len(records) < count:
        attempts += 1
        if attempts > count * 20:
            raise RuntimeError("Could not generate enough distinct grounded records")
        topic = rng.choices(topics, weights)[0]
        question, ctx, answer = builders[topic](rng)
        answer = to_model_text(answer)
        history = None
        # One earlier short turn, so the model also sees multi-turn layouts; long
        # portfolio/comparison answers would not fit in the 1,024-token context.
        if topic not in ("grounded_portfolio", "grounded_comparison") and rng.random() < 0.2:
            prev_q, prev_ctx, prev_a = builders[rng.choice(SHORT_HISTORY_TOPICS)](rng)
            history = [{"role": "user", "content": prev_q}, {"role": "assistant", "content": to_model_text(prev_a)}]
        messages = build_messages(question, ctx, history)
        messages.append({"role": "assistant", "content": answer})
        sources = [message["content"] for message in messages[:-1]]
        leaked = ungrounded_numbers(answer, *sources)
        if leaked:
            raise AssertionError(f"{topic}: answer has numbers not in context: {sorted(leaked)}\n{answer}")
        if question in seen or (fits is not None and not fits(messages)):
            # Too long for the model context (or a repeated question): draw another.
            continue
        seen.add(question)
        records.append({"topic": topic, "messages": messages})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260520)
    args = parser.parse_args()
    for record in grounded_records(args.count, args.seed):
        print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
