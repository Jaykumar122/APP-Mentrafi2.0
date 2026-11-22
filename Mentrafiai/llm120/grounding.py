"""Shared grounded-context format for Network 3 (the 120M generative model).

The SAME function builds the system message for SFT training examples
(data_prep/build_grounded_sft.py) and for live inference (serve.py), so the
model sees at serving time exactly the layout it was trained on.

Pipeline: Network 1 (intent + risk) and Network 2 (slots) -> deterministic
math engine (inference/financial_calculator.py) -> fund retrieval (PostgreSQL)
-> this context -> Network 3 writes the explanation.

The tokenizer collapses whitespace, so the context is one line with explicit
separators. All numbers are pre-formatted; the model is trained to copy them,
never to compute new ones. `ungrounded_numbers` checks a reply afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

GROUNDED_SYSTEM_PROMPT = (
    "You are MentraFi, a polite mutual fund explainer for Indian investors. Use only the VERIFIED CONTEXT. "
    "Copy numbers exactly; never calculate or invent figures. Name only listed fund candidates. "
    "If something is missing, ask for it. Projections are not guarantees and past returns do not predict "
    "future returns. You are not a SEBI-registered adviser."
)
CONTEXT_START = "VERIFIED CONTEXT:"
# Compact tax lines (numbers match inference/financial_calculator.py constants).
TAX_EQUITY = "Equity funds: 12.5% on gains above Rs 1.25 Lakh a year if held over 12 months, else 20%; plus 4% cess."
TAX_DEBT = "Debt funds bought from April 2023: gains taxed at slab rate."
TAX_AS_OF = "FY 2025-26 rules."
CONTEXT_END = "END OF CONTEXT."

PROFILE_LABELS = (
    ("name", "name"), ("age", "age"), ("occupation", "occupation"), ("goal", "goal"),
    ("monthly_budget", "monthly SIP budget"), ("lumpsum", "lump sum"), ("horizon", "horizon"),
)
# Only NAV-derived fields. The expense ratio, AUM and riskometer columns in the app
# database were filled by formulas, not read from factsheets, so the model never sees them.
FUND_FIELDS = (
    ("nav", "NAV"), ("return_1y", "1y"), ("return_3y", "3y"), ("return_5y", "5y"),
)


def clean_fund_name(name: str) -> str:
    """Drop parenthetical notes ("(Formerly Known as ...)") and extra spaces from a scheme name."""
    name = re.sub(r"\s*\([^)]*\)", "", name)
    return re.sub(r"\s+", " ", name).strip(" -")


def to_model_text(text: str) -> str:
    """The tokenizer has no ₹ piece (4 byte tokens each); the model reads and writes 'Rs'."""
    return re.sub(r"₹\s*", "Rs ", text)


def to_display_text(text: str) -> str:
    """Convert the model's 'Rs 1,234' back to '₹1,234' for the app."""
    return re.sub(r"\bRs\.?\s?(?=\d)", "₹", text)

@dataclass
class FundFact:
    """One retrieved scheme. Every value is a pre-formatted string or None."""
    name: str
    category: str | None = None
    plan: str | None = "Direct Growth"
    nav: str | None = None
    return_1y: str | None = None
    return_3y: str | None = None
    return_5y: str | None = None
    expense_ratio: str | None = None
    riskometer: str | None = None
    aum: str | None = None

    def render(self, index: int) -> str:
        facts = [f"{label} {getattr(self, key)}" for key, label in FUND_FIELDS if getattr(self, key)]
        if self.category and self.category.lower() not in self.name.lower():
            facts.insert(0, self.category)
        return f"[{index}] {self.name}: " + ", ".join(facts)

@dataclass
class Allocation:
    category: str
    amount: str
    percent: str
    fund_index: int | None = None  # 1-based index into GroundedContext.funds

    def render(self) -> str:
        target = f"fund [{self.fund_index}]" if self.fund_index else "no verified scheme"
        return f"{self.category} {self.percent} {self.amount}/month to {target}"


@dataclass
class GroundedContext:
    intent: str
    risk: str | None = None
    profile: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    calculations: dict[str, str] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    allocation: list[Allocation] = field(default_factory=list)
    funds: list[FundFact] = field(default_factory=list)
    data_as_of: str | None = None
    tax_notes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        sections = [CONTEXT_START, f"Intent: {self.intent}."]
        sections.append(f"Risk profile: {self.risk}." if self.risk else "Risk profile: not stated.")
        profile = [f"{label} {self.profile[key]}" for key, label in PROFILE_LABELS if self.profile.get(key)]
        sections.append("Investor: " + ("; ".join(profile) if profile else "no details given") + ".")
        sections.append("Missing: " + (", ".join(self.missing) if self.missing else "none") + ".")
        if self.calculations:
            sections.append("Calculations: " +
                            "; ".join(f"{key} {value}" for key, value in self.calculations.items()) + ".")
        if self.assumptions:
            sections.append("Assumptions: " + "; ".join(self.assumptions) + ".")
        if self.allocation:
            sections.append("Allocation: " + "; ".join(a.render() for a in self.allocation) + ".")
        if self.funds:
            dated = f" (data as of {self.data_as_of})" if self.data_as_of else ""
            sections.append(f"Fund candidates{dated}: " + " ".join(f.render(i) + "." for i, f in
                                                                    enumerate(self.funds, 1)))
        else:
            sections.append("Fund candidates: none retrieved.")
        if self.tax_notes:
            sections.append("Tax: " + " ".join(self.tax_notes))
        if self.notes:
            sections.append("Notes: " + " ".join(self.notes))
        sections.append(CONTEXT_END)
        return to_model_text(" ".join(sections))


def build_messages(question: str, context: GroundedContext,
                   history: Sequence[Mapping[str, str]] | None = None) -> list[dict[str, str]]:
    """System (instructions + context), optional prior turns, then the user question."""
    messages = [{"role": "system", "content": f"{GROUNDED_SYSTEM_PROMPT} {context.render()}"}]
    for turn in history or ():
        messages.append({"role": turn["role"], "content": to_model_text(turn["content"])})
    messages.append({"role": "user", "content": to_model_text(question)})
    return messages


def unlisted_scheme_mentions(reply: str, context: str) -> bool:
    """True if the reply names a scheme ("... Direct Plan/Growth") that is not exactly a listed candidate."""
    remainder = reply
    for name in sorted(set(re.findall(r"\[\d+\] (.+?): ", context)), key=len, reverse=True):
        remainder = remainder.replace(name, " ")
    return re.search(r"\bdirect (?:plan|growth)\b", remainder, re.I) is not None


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return {match.group(0).rstrip(",").replace(",", "") for match in _NUMBER.finditer(text)}


def ungrounded_numbers(reply: str, *sources: str) -> set[str]:
    """Numbers in reply that appear in none of the sources (context, question, history).

    Single-digit numbers are ignored (list ordinals, "3 schemes"). A non-empty
    result means the model produced a figure the engine did not supply.
    """
    allowed: set[str] = set()
    for source in sources:
        allowed |= _numbers(source)
    return {value for value in _numbers(reply) - allowed if len(value.replace(".", "")) > 1}


def fund_fact_from_row(row: Mapping[str, Any]) -> FundFact:
    """Convert a db_fund_retriever row into display strings (missing -> omitted)."""
    def pct(key, digits=1):
        value = row.get(key)
        return None if value is None else f"{float(value):.{digits}f}%"

    plan = " ".join(str(row[key]) for key in ("plan_type", "option_type") if row.get(key)) or None
    category = row.get("subcategory") or row.get("category")
    nav = row.get("nav")
    aum = row.get("aum_crores")
    return FundFact(
        name=clean_fund_name(str(row["name"])), category=str(category) if category else None, plan=plan,
        nav=None if nav is None else f"₹{float(nav):,.2f}",
        return_1y=pct("return_1y"), return_3y=pct("return_3y"), return_5y=pct("return_5y"),
    )
