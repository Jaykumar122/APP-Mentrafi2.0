"""Route concept questions ("what is SIP?", "SIP vs lump sum") to the reviewed curriculum.

The 120M model was fine-tuned on 91 reviewed definition topics
(data_prep/mutual_fund_curriculum.py). It reproduces a topic reliably only when it is
asked in the trained wording, so `match_topic` maps free-form questions to a topic,
serve.py asks the model the trained question, and `faithful` checks the reply against
the reviewed answer. If the reply drifts, the reviewed answer itself is shown.
"""
from __future__ import annotations

import difflib
import re
from typing import Optional

from data_prep.mutual_fund_curriculum import SYSTEM_PROMPT, TOPICS

# (pattern, topic). Ordered: comparisons and specific terms before general ones.
ALIASES: list[tuple[str, str]] = [
    # comparisons ("difference between X and Y", "X vs Y")
    (r"\bsip\b.*\b(lump ?sum|one[- ]time)\b|\b(lump ?sum|one[- ]time)\b.*\bsip\b", "lump_sum"),
    (r"\bdirect\b.*\bregular\b|\bregular\b.*\bdirect\b", "direct_and_regular"),
    (r"\bmulti ?cap\b.*\bflexi ?cap\b|\bflexi ?cap\b.*\bmulti ?cap\b", "multi_cap_fund"),
    (r"\betf\b.*\bindex\b|\bindex\b.*\betf\b", "etf_vs_index_fund"),
    (r"\b(open|close)[- ]?ended\b.*\b(open|close)[- ]?ended\b|\bclosed[- ]?ended\b", "closed_ended"),
    (r"\brolling\b.*\btrailing\b|\btrailing\b.*\brolling\b|\brolling returns?\b", "rolling_return"),
    (r"\btracking difference\b", "tracking_difference"),
    (r"\bduration\b.*\bmaturity\b|\bmaturity\b.*\bduration\b", "duration_and_maturity"),
    (r"\bexit load\b.*\block[- ]?in\b|\block[- ]?in\b.*\bexit load\b", "exit_load"),
    (r"\b(nominal|real) returns?\b|\binflation[- ]adjusted return", "inflation_real_return"),
    (r"\bcagr\b.*\bxirr\b|\bxirr\b.*\bcagr\b|\bxirr\b", "xirr"),
    (r"\bgrowth\b.*\bidcw\b|\bidcw\b.*\bgrowth\b|\bidcw\b|\bdividend option\b", "idcw"),
    # SIP family
    (r"\bstep[- ]?up\b|\btop[- ]?up sip\b", "step_up_sip"),
    (r"\bswp\b|systematic withdrawal", "swp"),
    (r"\bstp\b|systematic transfer", "stp"),
    (r"\b(pause|stop|skip|miss)\w*\b.*\bsip\b|\bsip\b.*\b(pause|stop|skip|miss)\w*", "sip_pause"),
    (r"rupee[- ]cost averaging", "rupee_cost_averaging"),
    (r"\bsip\b.*\bguarantee|\bguarantee\w*\b.*\bsip\b", "sip_is_a_method"),
    (r"\bsip\b|systematic investment plan", "sip_definition"),
    (r"\blump ?sum\b", "lump_sum"),
    # NAV, units, costs
    (r"\b(low|lower|cheap|high)\w* nav\b|\bnav\b.*\b(cheap|better|myth)\b", "nav_and_price_myth"),
    (r"\bnav\b|net asset value", "nav_formula"),
    (r"\bunits?\b", "units"),
    (r"\baum\b|assets under management", "aum"),
    (r"\bamc\b|asset management compan", "amc"),
    (r"\bexpense ratio\b|\bter\b", "total_expense_ratio"),
    (r"\bexit load\b", "exit_load"),
    (r"\bgrowth option\b|\bgrowth plan\b", "growth_option"),
    (r"\bnfo\b|new fund offer", "nfo"),
    # returns and risk metrics
    (r"\babsolute return", "absolute_return"),
    (r"\bcagr\b|compound(ed)? annual growth", "cagr"),
    (r"\btrailing return", "trailing_return"),
    (r"\bstandard deviation\b", "standard_deviation"),
    (r"\bvolatil", "volatility"),
    (r"\balpha\b", "alpha"),
    (r"\bbeta\b", "beta"),
    (r"\bsharpe\b", "sharpe_ratio"),
    (r"\bdrawdown\b", "drawdown"),
    (r"\bcompound", "compounding"),
    (r"\basset allocation\b", "asset_allocation"),
    (r"\bdiversif", "diversification"),
    (r"\brebalanc", "rebalancing"),
    (r"\boverlap\b", "portfolio_overlap"),
    (r"\briskometer\b|\brisk ?o ?meter\b", "riskometer"),
    (r"\btracking error\b", "tracking_error"),
    (r"\bbenchmark\b", "benchmark"),
    (r"\bpast (performance|returns?)\b|\bhighest return", "past_performance"),
    # debt
    (r"\bmodified duration\b", "modified_duration"),
    (r"\bytm\b|yield to maturity", "yield_to_maturity"),
    (r"\bcredit risk fund\b", "credit_risk_fund"),
    (r"\bcredit risk\b", "credit_risk"),
    (r"\bliquidity risk\b", "liquidity_risk"),
    (r"\binterest[- ]rate risk\b|\brates? (rise|go up)\b", "interest_rate_risk"),
    (r"\bovernight fund", "overnight_fund"),
    (r"\bliquid fund", "liquid_fund"),
    (r"\bgilt\b|government securit", "gilt_fund"),
    (r"\bdebt (mutual )?funds?\b", "debt_fund"),
    # equity and other categories
    (r"\blarge[- ]?cap\b", "large_cap_fund"),
    (r"\bmid[- ]?cap\b", "mid_cap_fund"),
    (r"\bsmall[- ]?cap\b", "small_cap_fund"),
    (r"\bflexi[- ]?cap\b", "flexi_cap_fund"),
    (r"\bmulti[- ]?cap\b", "multi_cap_fund"),
    (r"\bindex funds?\b|\bpassive fund", "index_fund"),
    (r"\betfs?\b|exchange[- ]traded", "etf_vs_index_fund"),
    (r"\bfund of funds?\b|\bfof\b", "fund_of_funds"),
    (r"\bhybrid\b|\bbalanced fund", "hybrid_fund"),
    (r"\barbitrage\b", "arbitrage_fund"),
    (r"\binternational\b|\boverseas\b|\bglobal fund", "international_fund"),
    (r"\b(sector|sectoral|thematic)\b", "sector_thematic"),
    (r"\bequity (mutual )?funds?\b", "equity_fund"),
    (r"\bopen[- ]?ended\b", "open_ended"),
    # tax and ELSS
    (r"\belss\b.*\b(new|old) (tax )?regime\b|\b80c\b", "elss_tax_regime"),
    (r"\belss\b|tax saver|tax saving fund|equity linked saving", "elss_lock_in"),
    (r"\bsip\b.*\btax\b|\btax\b.*\b(each|every) (sip|installment)", "sip_installment_tax"),
    # operations
    (r"\bkyc\b|know your customer", "kyc"),
    (r"\bfolio\b", "folio"),
    (r"\bnomin", "nomination"),
    (r"\bcut[- ]?off\b", "cutoff_and_nav"),
    (r"\bsettle|\bwhen will .*money|\bredemption (time|money)", "settlement"),
    (r"\bredeem|\bredemption\b", "redemption"),
    (r"\block[- ]?in\b", "lock_in_vs_horizon"),
    (r"\bswitch(ing)?\b", "switch_transactions"),
    (r"\bcomplain|\bgrievance|\bunresolved\b", "fund_complaints"),
    (r"\bcustodian\b|\btrustee\b", "custodian_trustee"),
    (r"\bscheme (information )?documents?\b|\bsid\b|\bkim\b|\bfactsheet\b", "scheme_documents"),
    (r"\bemergency\b", "emergency_fund"),
    (r"\bmutual funds?\b|\bwhat is (an? )?mf\b", "mutual_fund"),
]
_COMPILED = [(re.compile(p, re.I), t) for p, t in ALIASES if t in TOPICS]

CONCEPT_CUE = re.compile(
    r"^\s*(what|whats|what's|explain|define|meaning|tell me about|how (does|do|is)|why|is|are|does|can|difference|"
    r"compare|which is better)\b|\b(vs\.?|versus|difference|meaning|means|explain|definition)\b|\?\s*$", re.I)


def match_topic(question: str) -> Optional[str]:
    """Curriculum topic for a concept question, or None."""
    text = " ".join(question.lower().split())
    for pattern, topic in _COMPILED:
        if pattern.search(text):
            return topic
    return None


ADVICE_CUE = re.compile(
    r"\b(suggest|recommend|which (mutual )?funds?|best|top \d*|good funds?|where (should|can|do) i|"
    r"should i (invest|buy|start|choose)|portfolio for me|for me\b|i (want|plan|need) to invest|i can invest|"
    r"calculate|how much (will|should|do)|nav of|price of|returns? of)\b", re.I)


def is_concept_question(question: str) -> bool:
    """A question about what something is or how it works, not a request for advice."""
    return bool(CONCEPT_CUE.search(question)) and not ADVICE_CUE.search(question)


def reference(topic: str) -> tuple[str, str]:
    """(trained question, reviewed answer) for a topic."""
    entry = TOPICS[topic]
    return entry["question"], " ".join(entry["answer"].split())


def messages_for(topic: str) -> list[dict[str, str]]:
    question, _ = reference(topic)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]


def faithful(reply: str, topic: str, threshold: float = 0.85) -> bool:
    """True if the model reply is (almost) the reviewed answer."""
    _, answer = reference(topic)
    return difflib.SequenceMatcher(None, " ".join(reply.split()), answer).ratio() >= threshold
