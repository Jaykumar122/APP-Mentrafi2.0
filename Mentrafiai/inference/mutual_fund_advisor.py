"""
Grounded mutual-fund recommendation pipeline.

The language model is responsible for conversation and explanations, not for
inventing fund facts or selecting a winner from 38,000 rows. PostgreSQL retrieves
eligible funds, a deterministic suitability layer ranks them, and MentraFiAI
explains the highest-ranked result using only the supplied evidence.

Usage:
    python inference/mutual_fund_advisor.py \
        --checkpoint checkpoints/finetune_v2/best_model.pt \
        --tokenizer tokenizer/tokenizer.model \
        --model_config configs/model_config.yaml
"""

import argparse
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import NotRequired, TypedDict

# The advisor prints ✓, ⭐ and ₹; Windows consoles default to cp1252 and raise
# UnicodeEncodeError on those. Force UTF-8 so the pipeline runs on any console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import psycopg2
import torch
from psycopg2 import sql
from psycopg2.extensions import connection

from inference.database_schema import FUND_COLUMNS, USER_COLUMNS, FUNDS_TABLE, USERS_TABLE
from inference.generate import (
    GREETING_REPLY, generate_chat_reply, informational_reply, is_greeting, is_informational,
)
from model.architecture import MentraFiAI
from model.config import ModelConfig
from tokenizer.tokenizer_utils import Tokenizer
from training.utils import load_config, setup_device


RISK_SCORE = {"low": 1, "moderate": 2, "medium": 2, "high": 3, "very high": 4}
NumericValue = int | float | str | Decimal | None

NEED_PROFILE_REPLY = (
    "I can recommend a fund once I know a bit about you. Please share:\n"
    "  • your age\n"
    "  • how much you want to invest each month (SIP amount)\n"
    "  • your risk tolerance (low / moderate / high)\n"
    "  • your investment horizon (in years)\n"
    "For example: \"I am 30, SIP 7000/month, moderate risk, 8-year horizon.\""
)


def _extract_age(query: str) -> int | None:
    """Detect a stated age ('I am 30', '30 years old', 'age 30') — used only to
    decide whether the query carries enough signal to attempt a recommendation."""
    m = re.search(r"\bage\D{0,4}(\d{1,3})\b|\b(\d{1,3})\s*(?:years?|yrs?)\s*old\b"
                  r"|\bi\s*am\s*(\d{1,3})\b", query, re.IGNORECASE)
    if not m:
        return None
    val = int(next(g for g in m.groups() if g))
    return val if 5 <= val <= 120 else None


def has_sufficient_profile(query: str, profile: "UserProfile") -> bool:
    """True if the query carries at least one concrete investment signal — risk
    tolerance, age, SIP amount, horizon, or an explicit fund category. If nothing
    was stated (nonsense/unrelated input), we ask for details instead of inventing
    a profile and producing a bogus recommendation panel."""
    return bool(
        profile.risk_level
        or profile.horizon_years
        or profile.preferred_category
        or extract_sip_amount(query)
        or _extract_age(query)
    )


_SINGLE_ASSET_TERMS = ("gold", "silver", "commodity", "commodities", "bullion",
                       "precious metal", "platinum")

# Narrow sector / thematic exposure. A fund tracking one industry behaves like a
# single-asset bet: its trailing return is often a one-off industry cycle rather
# than a repeatable, diversified return, so it gets the same down-rank as gold.
_SECTOR_THEME_TERMS = (
    "ev", "electric vehicle", "auto", "automobile", "automotive",
    "bank", "banking", "financial service", "financial services", "finserv",
    "pharma", "pharmaceutical", "healthcare", "health care",
    "infra", "infrastructure", "psu", "digital", "technology",
    "information technology", "consumption", "consumer", "fmcg",
    "energy", "oil", "gas", "metal", "mining", "realty", "real estate",
    "media", "entertainment", "telecom", "defence", "defense",
    "manufacturing", "transportation", "logistics", "tourism", "housing",
    "internet", "innovation", "commodities ex", "services",
)
# Whole-word matching only: "bank" must not fire on "Bandhan", "ev" not on "value".
_SECTOR_TERM_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in
                      sorted(_SECTOR_THEME_TERMS, key=len, reverse=True)) + r")\b"
)
# Debt schemes legitimately carry sector words in their name — "Nifty PSU Bond
# Plus SDL Index Fund", "Banking & PSU Debt Fund", "CRISIL IBX AAA Bond Financial
# Services Index Fund" are broad BOND baskets, not industry bets. Gate the sector
# rule on the equity side so those are not wrongly penalized.
_DEBT_CATEGORY_HINTS = ("debt", "liquid", "gilt", "bond", "money market",
                        "overnight", "duration")

# Two fund HOUSES carry a sector word in their own name, so "BANK OF INDIA Small
# Cap Fund" and "BAJAJ FINSERV SMALL CAP FUND" would otherwise look thematic.
# Strip the house prefix before testing the mandate. (Verified against the 82
# distinct fund_house values in the funds table — only these two match.)
_AMC_SECTOR_PREFIXES = ("bank of india", "boi axa", "bajaj finserv")

# A declared broad mandate outranks an incidental sector word: a small-cap or
# Nifty-50 fund is diversified even if its house or index label contains
# "bank"/"finserv". Deliberately excludes bare "index fund", since "Nifty Bank
# Index Fund" IS a single-industry bet.
_BROAD_MANDATE_RE = re.compile(
    r"\b(small\s?cap|mid\s?cap|large\s?cap|flexi\s?cap|multi\s?cap|micro\s?cap|"
    r"large\s*&\s*mid|large and mid|elss|tax saver|tax advantage|balanced|hybrid|"
    r"arbitrage|equity savings|multi asset|asset allocation|value|contra|focused|"
    r"dividend yield|blue\s?chip|nifty 50|nifty next 50|nifty 100|nifty 200|"
    r"nifty 500|nifty midcap|nifty smallcap|sensex|total market|broad)\b"
)


def _is_single_asset(fund: "Fund") -> bool:
    """True for single-commodity or narrowly-concentrated funds (gold/silver ETFs,
    sectoral/thematic funds). Their trailing returns are often one-off spikes, so
    they shouldn't win the default #1 slot over diversified funds."""
    cat = str(fund.get("category", "")).lower()
    name = str(fund.get("fund_name", "")).lower()
    hay = f"{cat} {name}"
    if cat == "gold" or any(t in hay for t in _SINGLE_ASSET_TERMS):
        return True
    # Explicitly labelled sectoral / thematic exposure.
    if "sectoral" in hay or "thematic" in hay or "sector fund" in name:
        return True
    # Named single-industry exposure, equity side only (see _DEBT_CATEGORY_HINTS),
    # and only when the fund does not declare a broad mandate.
    if any(h in cat for h in _DEBT_CATEGORY_HINTS) or _BROAD_MANDATE_RE.search(name):
        return False
    mandate = name
    for prefix in _AMC_SECTOR_PREFIXES:
        if mandate.startswith(prefix):
            mandate = mandate[len(prefix):]
            break
    return bool(_SECTOR_TERM_RE.search(mandate))


def _user_wants_single_asset(profile: "UserProfile") -> bool:
    """The penalty is suppressed when the user explicitly asked for this exposure
    (e.g. 'suggest a gold fund', 'a pharma fund'), so these funds stay rankable
    on demand — down-ranked by default, never excluded."""
    pref = str(profile.preferred_category or "").lower()
    if not pref:
        return False
    return (pref in _SINGLE_ASSET_TERMS
            or pref in _SECTOR_THEME_TERMS
            or "sector" in pref
            or "thematic" in pref)


# A sector word only counts as a user REQUEST when a fund/ETF word sits within a
# short window of it, in either order ("a pharma fund", "fund for banking").
_SECTOR_REQUEST_RE = re.compile(
    r"\b(?P<a>" + "|".join(re.escape(t) for t in
                           sorted(_SECTOR_THEME_TERMS, key=len, reverse=True)) +
    r")\b.{0,25}?\b(?:funds?|etfs?|scheme|sector|theme|thematic)\b"
    r"|\b(?:funds?|etfs?|scheme|invest\w*)\b.{0,25}?\b(?P<b>" +
    "|".join(re.escape(t) for t in
             sorted(_SECTOR_THEME_TERMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE | re.DOTALL,
)


def _requested_sector(query: str) -> str | None:
    """Return the industry the user explicitly asked for, else None."""
    m = _SECTOR_REQUEST_RE.search(query or "")
    if not m:
        return None
    return (m.group("a") or m.group("b") or "").lower() or None


# This database encodes risk implicitly through category/subcategory (there is no
# risk-level column — `rating` is a 1-5 quality score, not a risk grade). Map the
# fund's asset class to a conventional risk level so suitability ranking can honor
# a user's stated risk tolerance (e.g. "Low" → Debt, not Small Cap).
def _risk_from_asset_class(category: str | None, subcategory: str | None) -> str:
    cat = str(category or "").strip().lower()
    sub = str(subcategory or "").strip().lower()
    if "small cap" in sub:
        return "Very High"
    if "mid cap" in sub:
        return "High"
    if "large cap" in sub:
        return "Moderate"
    if cat == "debt" or "debt" in sub or "liquid" in sub or "money market" in sub:
        return "Low"
    if cat == "gold" or "gold" in sub:
        return "Moderate"
    if cat == "hybrid" or "balanced" in sub:
        return "Moderate"
    if cat == "elss" or "tax" in sub:
        return "High"
    if cat == "equity":
        return "High"
    return "Moderate"


# Some rows are mislabeled by the upstream (MFapi.in) sync — e.g. a fund named
# "... MID & SMALL CAP EQUITY & DEBT FUND" is stored as category='Debt'. Reading
# the fund NAME lets us catch that conflict: if the name implies a riskier asset
# class than the stored category, we don't trust the "Low" label.
def _risk_from_name(name: str | None) -> str | None:
    n = str(name or "").lower()
    if "small cap" in n:
        return "Very High"
    if "mid cap" in n:
        return "High"
    if any(k in n for k in ("small cap", "flexi cap", "aggressive", "sectoral", "thematic")):
        return "High"
    if any(k in n for k in ("large cap", "bluechip", "blue chip", "index", "nifty", "sensex")):
        return "Moderate"
    if any(k in n for k in ("liquid", "overnight", "gilt", "money market", "ultra short")):
        return "Low"
    # A generic "equity" mention only bumps risk if the category disagrees.
    if "equity" in n:
        return "High"
    return None


_RISK_ORDER = {"low": 0, "moderate": 1, "medium": 1, "high": 2, "very high": 3}


def _derive_risk_level(category: str | None, subcategory: str | None,
                       name: str | None = None) -> tuple[str, bool]:
    """Return (risk_level, conflict) for a fund.

    Risk is derived from the stored asset class, but the fund NAME is used as a
    cross-check. When the name implies a materially riskier asset class than the
    category (a data-quality conflict), we adopt the HIGHER risk so a mislabeled
    aggressive fund can't masquerade as "Low" for a conservative investor.
    `conflict` is True in that case so callers can down-rank / flag it.
    """
    by_class = _risk_from_asset_class(category, subcategory)
    by_name = _risk_from_name(name)
    if by_name is None:
        return by_class, False
    if _RISK_ORDER[by_name.lower()] > _RISK_ORDER[by_class.lower()]:
        return by_name, True  # name says riskier than category → trust the name
    return by_class, False


class Fund(TypedDict):
    fund_name: str
    category: str
    nav: NumericValue
    expense_ratio: NumericValue
    performance: NumericValue
    risk_level: str
    data_conflict: NotRequired[bool]
    suitability_score: NotRequired[float]
    score_reasons: NotRequired[list[str]]


@dataclass(frozen=True)
class UserProfile:
    risk_level: str | None = None
    horizon_years: int | None = None
    preferred_category: str | None = None
    age: int | None = None
    monthly_income: int | None = None      # ₹ per month
    monthly_budget: int | None = None      # SIP budget ₹ per month (explicit)


def merge_profiles(stored: UserProfile, requested: UserProfile) -> UserProfile:
    """Prefer explicit request details while retaining stored suitability data."""
    return UserProfile(
        risk_level=requested.risk_level or stored.risk_level,
        horizon_years=requested.horizon_years or stored.horizon_years,
        preferred_category=requested.preferred_category or stored.preferred_category,
        age=requested.age or stored.age,
        monthly_income=requested.monthly_income or stored.monthly_income,
        monthly_budget=requested.monthly_budget or stored.monthly_budget,
    )


def _number(value: NumericValue, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _risk_value(value: str | None) -> int:
    text = str(value or "").strip().lower()
    return RISK_SCORE.get(text, 2)


def score_fund(fund: Fund, profile: UserProfile) -> tuple[float, list[str]]:
    """Return an auditable suitability score and concise reasons.

    Performance remains useful but cannot dominate suitability. Risk alignment,
    category alignment, horizon compatibility, and expense ratio collectively
    carry more weight than trailing performance.
    """
    score = 0.0
    reasons: list[str] = []

    performance = _number(fund.get("performance"))
    raw_er = fund.get("expense_ratio")
    expense_ratio = max(0.0, _number(raw_er))
    score += max(-20.0, min(30.0, performance)) * 0.8
    score += max(-5.0, 5.0 - expense_ratio * 2.0)
    reasons.append(f"reported performance {performance:.2f}%")
    # Only surface an expense-ratio reason when we actually have the value. The
    # DB currently has no expense_ratio data, so appending "expense ratio 0.00%"
    # here fed a fabricated 0.00% into the model prompt (score_reasons flows into
    # build_prompt), which the model then parroted in its reasoning. Scoring still
    # uses the neutral 0.0 default above; we just don't claim a number we lack.
    if raw_er is not None:
        reasons.append(f"expense ratio {expense_ratio:.2f}%")

    if profile.risk_level:
        difference = abs(_risk_value(fund.get("risk_level")) - _risk_value(profile.risk_level))
        risk_points = 25.0 - difference * 15.0
        score += risk_points
        if difference == 0:
            reasons.append(f"matches the requested {profile.risk_level.lower()} risk level")
        else:
            reasons.append(f"risk differs from the requested {profile.risk_level.lower()} level")

    if profile.preferred_category:
        matches = profile.preferred_category.lower() in str(fund.get("category", "")).lower()
        score += 20.0 if matches else -10.0
        reasons.append(
            "matches the requested category" if matches else "does not match the requested category"
        )

    if profile.horizon_years is not None:
        fund_risk = _risk_value(fund.get("risk_level"))
        compatible = not (
            (profile.horizon_years < 3 and fund_risk >= 3)
            or (profile.horizon_years < 5 and fund_risk >= 4)
        )
        score += 15.0 if compatible else -30.0
        reasons.append(
            f"is compatible with a {profile.horizon_years}-year horizon"
            if compatible
            else f"may be too volatile for a {profile.horizon_years}-year horizon"
        )

    # Penalize rows whose stored category conflicts with the fund name (a data
    # quality issue): their risk classification is untrustworthy, so keep them
    # from topping the ranking rather than surfacing a mislabeled fund.
    if fund.get("data_conflict"):
        score -= 40.0
        reasons.append("category/name mismatch — risk classification uncertain")

    # Down-rank single-commodity / narrowly-concentrated funds so a one-off price
    # spike (e.g. a silver ETF up 100%) can't win the default #1 slot over a
    # diversified fund. Suppressed when the user explicitly asked for this exposure.
    if _is_single_asset(fund) and not _user_wants_single_asset(profile):
        score -= 40.0
        reasons.append("single-asset/commodity — down-ranked vs diversified funds")

    return score, reasons


def rank_funds(funds: list[Fund], profile: UserProfile) -> list[Fund]:
    ranked: list[Fund] = []
    for fund in funds:
        score, reasons = score_fund(fund, profile)
        ranked.append({**fund, "suitability_score": round(score, 3), "score_reasons": reasons})
    return sorted(
        ranked,
        key=lambda fund: (-fund.get("suitability_score", float("-inf")), fund["fund_name"]),
    )


def _load_dotenv(path: str | None = None) -> None:
    """Best-effort load of a .env file so DATABASE_URL / DB_* vars are available
    even when the shell did not export them. Existing env vars win (setdefault),
    so an explicitly exported variable always overrides the file."""
    path = path or os.getenv("MENTRAFI_ENV_FILE", ".env")
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
    except OSError as exc:
        print(f"⚠️  Could not read env file {path}: {exc}")


def _connect() -> connection:
    """Open a PostgreSQL connection.

    Priority:
      1. DATABASE_URL (a full DSN, e.g. postgresql://user:pass@host:5432/db) —
         this is the canonical way to supply the password and avoids the
         'fe_sendauth: no password supplied' failure that occurs when the
         password is never passed.
      2. Individual DB_* variables as a fallback.
    """
    timeout = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn, connect_timeout=timeout)

    password = os.getenv("DB_PASSWORD")
    if not password:
        raise psycopg2.OperationalError(
            "No database password supplied. Set DATABASE_URL "
            "(e.g. postgresql://postgres:PASSWORD@localhost:5432/mentrafi) "
            "or the DB_PASSWORD environment variable."
        )
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "mentrafi"),
        user=os.getenv("DB_USER", "postgres"),
        password=password,
        connect_timeout=timeout,
    )


class PostgresFundStore:
    def __init__(self):
        self.conn: connection | None = None
        _load_dotenv()
        try:
            self.conn = _connect()
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            print("✓ Connected to PostgreSQL")
        except Exception as exc:
            self.conn = None
            print(f"⚠️  PostgreSQL connection failed: {exc}")

    def parse_profile(self, query: str) -> UserProfile:
        q = query.lower()
        category = next(
            (name.title() for name in ("large cap", "mid cap", "small cap", "index", "debt", "hybrid", "elss", "gold", "silver", "commodity", "sector", "thematic", "equity") if name in q),
            None,
        )
        # An explicit industry request ("suggest a pharma fund") counts as a
        # requested category so _user_wants_single_asset suppresses the thematic
        # down-rank. Requires a nearby fund/ETF word so an incidental mention
        # ("money sitting in the bank") does not register as a sector request.
        if category is None:
            category = _requested_sector(q)
        # Horizon: prefer explicit horizon context and never mistake "N years
        # old" (the user's age) for the investment horizon.
        horizon = None
        hm = (
            re.search(r"horizon\D{0,8}(\d+)\s*(?:year|yr)", q)
            or re.search(r"(\d+)\s*(?:year|yr)s?\s*horizon", q)
            or re.search(r"for\s*(\d+)\s*(?:year|yr)", q)
            or re.search(r"(\d+)\s*(?:year|yr)s?\b(?!\s*old)", q)
        )
        if hm:
            horizon = int(hm.group(1))
        risk = None
        # Explicit multi-word phrases first (either word order).
        if any(p in q for p in ("aggressive", "high growth", "high risk", "risk tolerance high", "high risk tolerance")):
            risk = "High"
        elif any(p in q for p in ("moderate risk", "balanced", "medium risk", "risk tolerance moderate", "risk tolerance medium")):
            risk = "Moderate"
        elif any(p in q for p in ("low risk", "risk tolerance low", "low risk tolerance", "safe", "conservative")):
            risk = "Low"
        else:
            # Generic "<level> risk/tolerance/appetite ..." or "risk tolerance <level>".
            rm = (
                re.search(r"risk\s*(?:tolerance|appetite|profile|level)?\s*(?:is|of|:|=)?\s*(very high|high|moderate|medium|low|conservative|aggressive)", q)
                or re.search(r"(very high|high|moderate|medium|low|conservative|aggressive)\s*(?:risk|tolerance|appetite)", q)
            )
            if rm:
                risk = {
                    "low": "Low", "conservative": "Low",
                    "moderate": "Moderate", "medium": "Moderate",
                    "high": "High", "very high": "High", "aggressive": "High",
                }[rm.group(1)]
            elif "moderate" in q or "medium" in q:
                risk = "Moderate"
        # Age: "22 years old", "age 45", "aged 30", "22yo". Guarded to 10-100 so
        # a horizon or SIP figure is never mistaken for age. Grounds the prompt so
        # the model states the real age instead of inventing one.
        age = None
        am = (
            re.search(r"\bage\s*(?:is|:|=)?\s*(\d{1,3})\b", q)
            or re.search(r"\baged\s*(\d{1,3})\b", q)
            or re.search(r"\b(\d{1,3})\s*(?:years?\s*old|yo|y/o|yrs?\s*old)\b", q)
            or re.search(r"\bi\s*am\s*(\d{1,3})\b(?!\s*(?:year|yr)?s?\s*(?:horizon|SIP))", q)
        )
        if am:
            a = int(am.group(1))
            if 10 <= a <= 100:
                age = a
        # Monthly income: "earn 50000", "salary 80k", "income 1.2 lakh/month"
        monthly_income = None
        im = (
            re.search(r"(?:earn|salary|income|take.?home|ctc)\D{0,15}?([\d.]+)\s*(lakh|lac|k|thousand)?", q)
        )
        if im:
            val_str = im.group(1).replace(",", "")
            try:
                val = float(val_str)
                unit = (im.group(2) or "").lower()
                if unit in ("lakh", "lac"):
                    val = val * 100000
                elif unit == "k":
                    val = val * 1000
                elif unit == "thousand":
                    val = val * 1000
                monthly_income = int(val) if 3000 <= val <= 10000000 else None
            except ValueError:
                pass

        # Monthly SIP budget — explicit budget/invest amount.
        # Strip the income clause first so "earn 60000 per month" doesn't
        # confuse extract_sip_amount into returning the income as the SIP.
        sip_query = re.sub(
            r"(?:earn|salary|income|take.?home|ctc)\D{0,15}?[\d.]+\s*(?:lakh|lac|k|thousand)?\s*(?:per\s*month|/\s*month|a\s*month|monthly|p\.?m\.?)?",
            "",
            query,
            flags=re.IGNORECASE,
        )
        monthly_budget = extract_sip_amount(sip_query)

        return UserProfile(risk, horizon, category, age, monthly_income, monthly_budget)

    def get_user_profile(self, user_id: str | None) -> UserProfile:
        """Load only suitability attributes, not unrelated personal information.

        This schema (users table) lacks dedicated suitability fields. Subclasses
        or extended tables (e.g., user_preferences, user_suitability) would
        store risk_level, investment_horizon_years, and preferred_category.
        For now, return an empty profile; parse_profile() will extract from the query.
        """
        if not self.conn or not user_id:
            return UserProfile()
        if USER_COLUMNS.risk_level == "None":
            # Suitability profile table does not exist; rely on text parsing
            return UserProfile()
        try:
            query = sql.SQL("SELECT {}, {}, {} FROM {} WHERE {} = %s LIMIT 1").format(
                sql.Identifier(USER_COLUMNS.risk_level),
                sql.Identifier(USER_COLUMNS.horizon_years),
                sql.Identifier(USER_COLUMNS.preferred_category),
                sql.Identifier(USERS_TABLE),
                sql.Identifier(USER_COLUMNS.user_id),
            )
            with self.conn.cursor() as cursor:
                cursor.execute(query, [user_id])
                row = cursor.fetchone()
            if not row:
                return UserProfile()
            horizon = int(row[1]) if row[1] is not None else None
            return UserProfile(
                risk_level=str(row[0]) if row[0] else None,
                horizon_years=horizon,
                preferred_category=str(row[2]) if row[2] else None,
            )
        except Exception as exc:
            self.conn.rollback()
            print(f"⚠️  User profile query failed: {exc}")
            return UserProfile()

    # Standard broad categories we hard-filter on when the user explicitly asks.
    # Sector/thematic and gold/commodity are deliberately NOT here: those stay on
    # the soft down-rank-not-exclude path so they remain rankable on demand.
    _HARD_FILTER_CATEGORIES = {
        "large cap": "large cap", "mid cap": "mid cap", "small cap": "small cap",
        "elss": "elss", "debt": "debt", "index": "index", "hybrid": "hybrid",
    }

    def _rows_to_funds(self, rows) -> list[Fund]:
        funds: list[Fund] = []
        for row in rows:
            risk, conflict = _derive_risk_level(row[1], row[2], row[0])
            funds.append({
                "fund_name": str(row[0] or "Unknown"),
                "category": str(row[1] or "Other"),
                "nav": _number(row[3]),
                # No expense-ratio column exists in this schema — leave it
                # unknown (None) rather than inventing a misleading 0.0%.
                "expense_ratio": None,
                "performance": _number(row[4]),
                # Derived from asset class, cross-checked against the name.
                "risk_level": risk,
                "data_conflict": conflict,
            })
        return funds

    def get_candidate_funds(self, profile: UserProfile, limit: int = 0) -> list[Fund]:
        """Retrieve eligible funds; ranking happens in application code.

        When the user explicitly requests a standard broad category (e.g. "small
        cap fund"), hard-filter the SQL to that category/subcategory so an
        off-category high performer can never win. If the filtered set is too thin
        (< MIN_FILTERED rows) we fall back to the full pool so the user still gets
        a recommendation rather than an empty panel. Sector/thematic/gold requests
        are left to the soft down-rank path and are not hard-filtered here.

        Maps real database columns to advisor format using environment-configurable
        column names. Missing fields (e.g., expense_ratio) default to None.
        """
        if not self.conn:
            return []
        MIN_FILTERED = 3
        cols = [
            FUND_COLUMNS.fund_name,
            FUND_COLUMNS.category,
            FUND_COLUMNS.subcategory,
            FUND_COLUMNS.nav,
            FUND_COLUMNS.performance,
        ]
        cols_str = ", ".join(cols)
        base = f"SELECT {cols_str} FROM {FUNDS_TABLE}"
        order = f" ORDER BY {FUND_COLUMNS.performance} DESC NULLS LAST"
        tail = ""
        base_params: list[object] = []
        if limit > 0:
            tail = " LIMIT %s"
            base_params.append(limit)

        pref = (profile.preferred_category or "").lower().strip()
        cat_term = self._HARD_FILTER_CATEGORIES.get(pref)

        try:
            with self.conn.cursor() as cursor:
                if cat_term:
                    # Match either the category or subcategory column (schema stores
                    # cap tiers in either depending on the AMC feed).
                    where = (f" WHERE LOWER({FUND_COLUMNS.category}) LIKE %s "
                             f"OR LOWER({FUND_COLUMNS.subcategory}) LIKE %s")
                    like = f"%{cat_term}%"
                    cursor.execute(base + where + order + tail,
                                   [like, like, *base_params])
                    rows = cursor.fetchall()
                    if len(rows) >= MIN_FILTERED:
                        return self._rows_to_funds(rows)
                    # Too few — fall back to the full pool below.
                    print(f"ℹ️  Only {len(rows)} '{pref}' funds; using full pool.")
                cursor.execute(base + order + tail, base_params)
                rows = cursor.fetchall()
            return self._rows_to_funds(rows)
        except Exception as exc:
            self.conn.rollback()
            print(f"⚠️  Fund query failed: {exc}")
            return []

    def close(self) -> None:
        if self.conn:
            self.conn.close()


def build_prompt(user_query: str, profile: UserProfile, ranked_funds: list[Fund]) -> str:
    if not ranked_funds:
        return (
            f"User request: {user_query}\n\nNo matching funds were returned by the database. "
            "Apologise warmly, explain you could not find a matching fund right now, and ask the "
            "user to try specifying risk tolerance, investment horizon, or a broader category. "
            "Be helpful and suggest they try 'low risk', 'moderate risk', or a category like equity or debt."
        )

    winner = ranked_funds[0]

    # Derive suggested SIP if income is known but budget not stated
    sip_hint = ""
    if profile.monthly_income and not profile.monthly_budget:
        suggested = int(profile.monthly_income * 0.20)  # 20% of income rule
        sip_hint = f"\n  Note: User earns \u20b9{profile.monthly_income:,}/month — suggest \u20b9{suggested:,}/month SIP (20% rule)."

    # SIP projection math for the prompt
    sip_projection = ""
    sip_amt = profile.monthly_budget or 5000
    if profile.horizon_years and sip_amt:
        r = 0.12 / 12  # 12% annual equity rate, monthly
        n = profile.horizon_years * 12
        fv = sip_amt * (((1 + r) ** n - 1) / r) * (1 + r)
        invested = sip_amt * n
        sip_projection = (
            f"\n  SIP math: \u20b9{sip_amt:,}/month for {profile.horizon_years} years "
            f"= \u20b9{invested/100000:.1f}L invested \u2192 \u20b9{fv/100000:.1f}L at 12% CAGR "
            f"(wealth ratio {fv/max(invested,1):.1f}x). Include this in your response."
        )

    lines = [
        "INSTRUCTIONS: You are MentraFiAI, an expert Indian mutual fund advisor.",
        "Your reply must be warm, specific, and expert — better than a generic chatbot.",
        "Structure your response as:",
        "  1. Acknowledge the user's situation personally (1-2 sentences).",
        f"  2. Recommend {winner['fund_name']} as the top pick and explain WHY it fits their exact profile.",
        "  3. Mention 1-2 alternative fund categories (not specific funds) for diversification.",
        "  4. Include the SIP math projection below if available.",
        "  5. End with one actionable tip (e.g. step-up SIP, ELSS for 80C, direct plan).",
        "  6. Add one-line disclaimer: past performance doesn't guarantee future returns.",
        "",
        "RULES:",
        "- Cite ONLY the fund data provided below. Do not invent NAV, returns, or expense ratios.",
        "- Use Indian context: \u20b9, lakh, crore, SIP, ELSS, NAV, SEBI, AMFI.",
        "- Be specific about the user's age, risk level, and horizon — don't be generic.",
        "- DO NOT say 'consult a financial advisor' — YOU are the advisor.",
        "- Aim for 150-200 words.",
        "",
        f"USER REQUEST: {user_query}",
        f"USER PROFILE: Age={profile.age or 'not stated'} | "
        f"Risk={profile.risk_level or 'not stated'} | "
        f"Horizon={profile.horizon_years or '?'} years | "
        f"Category={profile.preferred_category or 'any'} | "
        f"SIP Budget=\u20b9{profile.monthly_budget or 'not stated'}/month | "
        f"Income=\u20b9{profile.monthly_income or 'not stated'}/month",
        sip_hint,
        sip_projection,
        "",
        "FUND DATA (use only these facts):",
    ]
    for rank, fund in enumerate(ranked_funds[:3], 1):
        marker = "TOP PICK" if rank == 1 else f"Alternative #{rank-1}"
        lines.append(
            f"\n[{marker}] {fund['fund_name']}"
            f"\n  Category: {fund['category']} | Risk: {fund['risk_level']} | Performance: {fund['performance']}%"
            f"\n  NAV: {fund['nav']} | Suitability score: {fund.get('suitability_score', 0.0):.0f}/100"
            f"\n  Why it fits: {'; '.join(fund.get('score_reasons', []))}"
        )
    return "\n".join(lines)


def extract_sip_amount(query: str) -> int | None:
    """Extract the monthly SIP amount the user actually stated (Fix 3).

    The naive "first number in the string" approach is wrong because the first
    number is often the *horizon* ('5 years') or a risk figure, not the SIP.
    Priority order:
      1. A number carrying an explicit currency marker (₹ / Rs / INR).
      2. A number in SIP / investment / monthly context.
      3. Fallback: the largest remaining number, after discarding anything that
         is clearly a horizon ('N years/yr') or a percentage.
    Returns an int, or None if nothing plausible is found. This value — not any
    number the model restates in its generated text — is what gets displayed.
    """
    q = query.lower()

    def to_int(s: str) -> int | None:
        s = s.replace(",", "").strip()
        return int(s) if s.isdigit() else None

    # 1. Currency-tagged amount, e.g. ₹20,000 / Rs. 20000 / INR 5000
    m = re.search(r"(?:₹|rs\.?|inr)\s*([\d,]+)", q)
    if m and (val := to_int(m.group(1))):
        return val

    # 2. SIP / investment / monthly context
    ctx = re.search(
        r"(?:sip|invest(?:ing|ment)?|contribut\w*|put(?:ting)?)\D{0,20}?([\d,]{3,})"
        r"|([\d,]{3,})\s*(?:per\s*month|/\s*month|a\s*month|monthly|p\.?m\.?\b)",
        q,
    )
    if ctx and (val := to_int(ctx.group(1) or ctx.group(2))):
        return val

    # 3. Fallback: largest number that is not a horizon ('N years') or percent.
    candidates: list[int] = []
    for m in re.finditer(r"([\d,]+)\s*(years?|yrs?|%|percent)?", q):
        if m.group(2):  # trailed by year/yr/% → horizon or rate, not a SIP
            continue
        val = to_int(m.group(1))
        if val is not None and val >= 100:  # realistic SIPs are >= 100
            candidates.append(val)
    return max(candidates) if candidates else None


def _normalize_name(text: str) -> str:
    """Lowercase and collapse punctuation to spaces for robust name matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def fuzzy_match_fund(generated_text: str, candidate_funds: list[Fund]) -> Fund | None:
    """Identify which candidate fund the model recommends, by fund NAME only (Fix 2).

    The text is used solely to pick *which* candidate. The caller then reads the
    category / expense ratio / performance from that candidate's ground-truth
    entry, never from numbers/categories the model restated in prose.

    Scoring:
      1. Exact normalized full-name substring wins (earliest mention breaks ties).
      2. Otherwise the candidate with the highest fraction of its significant
         name words present in the text (>= 0.6), earliest mention as tie-break.
    """
    if not candidate_funds:
        return None
    norm_text = _normalize_name(generated_text)

    # 1. Exact (normalized) full-name substring — strongest, unambiguous signal.
    best_exact: Fund | None = None
    best_pos: int | None = None
    for fund in candidate_funds:
        name = _normalize_name(fund["fund_name"])
        if name and name in norm_text:
            pos = norm_text.find(name)
            if best_pos is None or pos < best_pos:
                best_pos, best_exact = pos, fund
    if best_exact is not None:
        return best_exact

    # 2. Significant-word overlap fallback.
    text_words = set(norm_text.split())
    best_fund: Fund | None = None
    best_score = 0.0
    best_first = 10**9
    for fund in candidate_funds:
        words = [w for w in _normalize_name(fund["fund_name"]).split() if len(w) > 3]
        if not words:
            continue
        hits = [w for w in words if w in text_words]
        score = len(hits) / len(words)
        positions = [norm_text.find(w) for w in hits if norm_text.find(w) >= 0]
        first = min(positions) if positions else 10**9
        if score > best_score or (score == best_score and first < best_first):
            best_score, best_fund, best_first = score, fund, first
    return best_fund if best_score >= 0.6 else None


def _scrub_expense_ratio(text: str) -> str:
    """Remove any fabricated expense-ratio mention from model prose.

    The model sometimes invents an expense ratio in its unverified reasoning even
    when we have no such data (learned prior from synthetic training examples).
    We call this ONLY when the matched fund's real expense_ratio is None, so a
    genuine value is never stripped. Cleans up the surrounding connectors/
    punctuation so removal never leaves a dangling fragment like "at a  ." """
    if not text:
        return text
    # Phrasings, longest/most-specific first:
    #   "at a 0.20% expense ratio", "with an expense ratio of 0.2%",
    #   "0.20% expense ratio", "expense ratio of 0.2%", "0.2% expense".
    # The model also uses "cost"/"fee" as synonyms for the fabricated fee number
    #   ("comes at a low 0.95% cost", "a 0.5% annual fee"), so those are stripped
    #   too — we have no fee data of any kind, so no real value is at risk.
    patterns = [
        r"\b(?:at|with|and|,)?\s*(?:an?|the)?\s*expense ratio of\s*[\d.]+\s*%",
        r"\b(?:at|with|and|,)?\s*(?:an?|the)?\s*[\d.]+\s*%\s*expense ratio",
        r"\bexpense ratio of\s*[\d.]+\s*%",
        r"\b[\d.]+\s*%\s*expense ratio",
        r"\b[\d.]+\s*%\s*expense\b",
        r"\b(?:at|with|for|and|,)?\s*(?:an?|the)?\s*(?:low|high|modest|small)?\s*"
        r"[\d.]+\s*%\s*(?:annual\s+)?(?:cost|fee|charges?)\b",
        r"\b(?:cost|fee|charges?)\s*of\s*[\d.]+\s*%",
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    # Repair connectors/punctuation left dangling by the removal.
    cleaned = re.sub(r"\s+(?:at|with|and)\s*(?=[.,;])", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\breturns?\s+and\s*(?=[.,;])", "returns", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r"\s+([.,;])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\.\s*\.", ".", cleaned)
    return cleaned.strip()


def _scrub_wrong_age(text: str, real_age: int | None) -> str:
    """Remove age claims in the prose whose number doesn't match the user's real
    age (or any age claim at all when we don't know it). The model tends to
    invent "age (32)" / "based on your age (32)" phrases; a wrong one is worse
    than none, so drop the mismatched clause and repair punctuation."""
    if not text:
        return text
    def _bad(m):
        num = int(m.group(1))
        return "" if (real_age is None or num != real_age) else m.group(0)
    patterns = [
        r"\bbased on your age\s*\(?\s*(\d{1,3})\s*\)?",
        r"\byour age\s*(?:is\s*|of\s*)?\(?\s*(\d{1,3})\s*\)?",
        r"\bage\s*\(?\s*(\d{1,3})\s*\)?",
        r"\baged\s*(\d{1,3})\b",
        r"\b(\d{1,3})[\s-]*years?[\s-]*old\b",
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, _bad, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:at|with|and|,)\s*(?=[.,;])", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r"\s+([.,;])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _article(word: str) -> str:
    """'a'/'an' for the following word (vowel-sound heuristic, good enough here)."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def _looks_like_garbage(text: str) -> bool:
    """Detect the SLM's degeneration failure modes so the caller can suppress them.

    Two distinct styles observed on this 100M model:
      (a) char-level: dense 1-2 char fragments + commas ("DIRE30, N,N, N; N, D, P").
      (b) morphological drift: one stem mutated over and over, often with colons
          ("Factories: Factures Factress Factresses ... Factored Factually").
    (a) is caught by short-token / comma / colon density; (b) by a dominant
    short word-stem prefix (stem length 2-4 chars varies, so several are tested;
    no_repeat_ngram_size=3 misses it because each token differs slightly).
    Thresholds calibrated so coherent prose (incl. repeated fund names) passes."""
    toks = text.split()
    if not toks:
        return True
    stripped = [t.strip(".,;:%()-/") for t in toks]
    short_frac = sum(1 for t in stripped if 0 < len(t) <= 2) / len(toks)
    comma_ratio = text.count(",") / len(toks)
    colon_ratio = text.count(":") / len(toks)
    if short_frac > 0.30 or comma_ratio > 0.28 or colon_ratio > 0.15:
        return True
    # Morphological-drift check: one word-stem mutated over and over
    # ("Ax" -> Axi/Axe/Axus/Axiality/Axem/Axer, or "Fact" -> Factures/Factress).
    # The shared stem length varies (2-4 chars), so test several prefix lengths
    # and flag if ANY length has a bucket that dominates. Calibrated on real
    # drift samples (top-bucket count 9-12, frac 0.32-0.47) vs clean prose incl.
    # repeated fund names (count <=4, frac <=0.17) — count>=6 is a wide margin.
    for plen in (2, 3, 4):
        words = [re.sub(r"[^a-z]", "", w.lower()) for w in toks]
        words = [w for w in words if len(w) >= plen]
        if len(words) < 8:
            continue
        top = Counter(w[:plen] for w in words).most_common(1)[0][1]
        if top >= 6 and top / len(words) > 0.25:
            return True
    return False


def generate_reasoning(model, tokenizer, prompt: str, device) -> str:
    """Generate the SLM reasoning prose with a coherence backstop.

    CRITICAL FIX: Improved sampling params to reduce degeneration:
    - Increased repetition_penalty from 1.0 to 1.2
    - Increased no_repeat_ngram_size from 3 to 4
    - Added more aggressive garbage detection
    - Retry with progressively higher penalties if needed
    """
    attempts = [
        (0.5, 1.2, 4),  # (temp, repetition_penalty, no_repeat_ngram_size)
        (0.6, 1.3, 4),
        (0.7, 1.4, 4),
    ]

    for temp, rep_penalty, ngram_size in attempts:
        text = generate_chat_reply(
            model, tokenizer, prompt, device,
            max_new_tokens=160, temperature=temp, top_k=40, top_p=0.9,
            repetition_penalty=rep_penalty, no_repeat_ngram_size=ngram_size,
        )
        if not _looks_like_garbage(text):
            return text
    return ""


def build_verified_response(generated_text: str, matched_fund: Fund | None,
                           sip_amount: int | None, profile: UserProfile) -> str:
    """Build the displayed answer: a fixed panel of GROUND-TRUTH facts, plus the
    model's own generated reasoning shown separately (and clearly labeled).

    PostgreSQL + the deterministic ranker own every FACT (name, category, risk,
    performance, expense, NAV) and the user's parsed profile owns age/horizon/
    risk/SIP — those go in the verified panel and cannot be altered by the model.
    The model's generated prose is displayed as REASONING so this stays a genuine
    from-scratch SLM project. Two light scrubs keep the prose from contradicting
    the panel (fabricated expense numbers, wrong age); repetition/looping is
    handled upstream in generate() (repetition_penalty + no_repeat_ngram_size)
    and by _truncate_on_repeat in inference/generate.py.
    """
    reasoning = generated_text.strip()
    if not matched_fund:
        # No candidate to ground the panel on → return the model prose alone.
        return reasoning or (
            "I couldn't find a fund that matches your criteria. Try adjusting your "
            "risk tolerance, horizon, or the category you're interested in."
        )

    er = matched_fund.get("expense_ratio")
    # No real expense data → strip any expense-ratio number the model invented,
    # so the reasoning never contradicts the "n/a" panel.
    if er is None:
        reasoning = _scrub_expense_ratio(reasoning)
    # Drop any age claim that disagrees with the user's stated age (or any age
    # claim when we don't know it), so the prose can't contradict the real profile.
    reasoning = _scrub_wrong_age(reasoning, profile.age)

    headline = (
        f"I recommend {matched_fund['fund_name']} — {_article(matched_fund['category'])} "
        f"{matched_fund['category']} fund with {matched_fund['risk_level']} risk."
    )
    er_str = f"{er}%" if er is not None else "n/a (not in dataset)"
    facts = [
        f"  • Category:      {matched_fund['category']}",
        f"  • Risk level:    {matched_fund['risk_level']}",
        f"  • Performance:   {matched_fund['performance']}%",
        f"  • Expense ratio: {er_str}",
        f"  • NAV:           {matched_fund['nav']}",
    ]

    # Profile line from the user's ACTUAL inputs (ground truth), when available.
    prof_bits: list[str] = []
    if sip_amount:
        prof_bits.append(f"\u20b9{sip_amount:,}/month SIP")
    elif profile.monthly_budget:
        prof_bits.append(f"\u20b9{profile.monthly_budget:,}/month SIP")
    if profile.monthly_income:
        prof_bits.append(f"\u20b9{profile.monthly_income:,}/month income")
    if profile.age:
        prof_bits.append(f"age {profile.age}")
    if profile.horizon_years:
        prof_bits.append(f"{profile.horizon_years}-year horizon")
    if profile.risk_level:
        prof_bits.append(f"{profile.risk_level.lower()} risk tolerance")

    # SIP projection in verified panel
    sip_proj_line = ""
    sip_amt = sip_amount or profile.monthly_budget
    if sip_amt and profile.horizon_years:
        import math as _math
        r = 0.12 / 12
        n = profile.horizon_years * 12
        fv = sip_amt * (((1 + r) ** n - 1) / r) * (1 + r)
        invested = sip_amt * n
        sip_proj_line = (
            f"  \u2022 SIP projection: \u20b9{sip_amt:,}/month \u00d7 {profile.horizon_years}yr "
            f"\u2192 \u20b9{invested/100000:.1f}L invested, ~\u20b9{fv/100000:.1f}L at 12% CAGR"
        )

    parts = [headline, *facts]
    if sip_proj_line:
        parts.append(sip_proj_line)
    if prof_bits:
        parts.append("  Matched to your profile: " + ", ".join(prof_bits))

    # Model-generated reasoning, shown separately from the verified facts above.
    # Facts in the panel are authoritative; this prose is the SLM's own text.
    if reasoning:
        parts.append("")
        parts.append(f"Reasoning (model-generated, unverified): {reasoning}")
    return "\n".join(parts)


def demonstrate_scoring():
    """Quick smoke test: load a few funds and score them."""
    import sys
    sys.path.insert(0, ".")
    from inference.database_schema import FUND_COLUMNS

    print("\n=== Fund Column Mapping ===")
    print(f"  fund_name        → {FUND_COLUMNS.fund_name}")
    print(f"  category         → {FUND_COLUMNS.category}")
    print(f"  nav              → {FUND_COLUMNS.nav}")
    print(f"  expense_ratio    → {FUND_COLUMNS.expense_ratio}")
    print(f"  performance      → {FUND_COLUMNS.performance}")
    print(f"  risk_level       → {FUND_COLUMNS.risk_level}")
    print()

    store = PostgresFundStore()
    profile = UserProfile(risk_level="Medium", horizon_years=5, preferred_category="Equity")
    print(f"Profile: {profile}")

    candidates = store.get_candidate_funds(profile, limit=10)
    if candidates:
        print(f"\nLoaded {len(candidates)} sample funds")
        ranked = rank_funds(candidates, profile)
        print(f"\nTop 3 by suitability:")
        for i, fund in enumerate(ranked[:3], 1):
            print(
                f"  {i}. {fund['fund_name'][:60]}\n"
                f"     Score: {fund.get('suitability_score', 0):.1f}, "
                f"Performance: {fund.get('performance')}%, "
                f"Risk: {fund.get('risk_level')}\n"
            )
    else:
        print("No funds loaded.")

    store.close()


def test_advisory_loop():
    """Quick interactive test of the full pipeline."""
    print("\n" + "="*80)
    print("MENTRAIFAI INTERACTIVE TEST")
    print("="*80)
    print("Type fund queries (or 'hello', 'quit' to exit)\n")

    device = setup_device()
    cfg_dict = load_config("configs/model_config.yaml")
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
    model = MentraFiAI(cfg).to(device)

    # Try to load v2, fall back to old
    for ckpt_path in ["checkpoints/finetune_v2/best_model.pt", "checkpoints/finetune_old/best_model.pt"]:
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            print(f"✓ Loaded checkpoint: {ckpt_path}\n")
            break
    else:
        print("✗ No checkpoint found")
        return

    tokenizer = Tokenizer("tokenizer/tokenizer.model")
    store = PostgresFundStore()

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "/quit"):
            break
        if not query:
            continue

        if is_greeting(query):
            print(f"MentraFiAI: {GREETING_REPLY}\n")
            continue
        if is_informational(query):
            print(f"MentraFiAI: {informational_reply(query)}\n")
            continue

        profile = merge_profiles(store.get_user_profile(None), store.parse_profile(query))
        if not has_sufficient_profile(query, profile):
            print(f"MentraFiAI: {NEED_PROFILE_REPLY}\n")
            continue
        candidates = store.get_candidate_funds(profile)
        if not candidates:
            print("MentraFiAI: No matching funds found. Try different criteria.\n")
            continue

        ranked = rank_funds(candidates, profile)
        prompt = build_prompt(query, profile, ranked)

        # SLM generates the reasoning via generate_reasoning(), which uses the
        # swept-optimal sampling (temp=0.5/top_k=40/top_p=0.9/rp=1.0/ng=3) and a
        # garbage-detection backstop (retry once, else show panel only).
        raw_response = generate_reasoning(model, tokenizer, prompt, device)

        sip_amount = extract_sip_amount(query)
        matched_fund = fuzzy_match_fund(raw_response, ranked[:5]) or (ranked[0] if ranked else None)
        verified_response = build_verified_response(raw_response, matched_fund, sip_amount, profile)

        print(f"MentraFiAI: {verified_response}\n")

    store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/finetune_v2/best_model.pt")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.model")
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--user_id", help="Application user ID used to load suitability fields")
    parser.add_argument(
        "--candidate_limit",
        type=int,
        default=0,
        help="Maximum eligible funds to score; 0 evaluates all matching database rows",
    )
    args = parser.parse_args()

    device = setup_device()
    cfg_dict = load_config(args.model_config)
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})

    model = MentraFiAI(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = Tokenizer(args.tokenizer)
    store = PostgresFundStore()

    print("\nMentraFiAI — Mutual Fund Research Assistant. Type /quit to exit.\n")
    try:
        while True:
            query = input("You: ").strip()
            if query.lower() in ("/quit", "quit", "exit"):
                break
            if not query:
                continue
            if is_greeting(query):
                print(f"MentraFiAI: {GREETING_REPLY}\n")
                continue
            # Definitional questions ("what is SIP") get an explanation, never a
            # recommendation panel with a hallucinated profile.
            if is_informational(query):
                print(f"MentraFiAI: {informational_reply(query)}\n")
                continue

            stored_profile = store.get_user_profile(args.user_id)
            requested_profile = store.parse_profile(query)
            profile = merge_profiles(stored_profile, requested_profile)
            # No usable signal (nonsense/unrelated input) → ask for details rather
            # than fabricating a profile and recommending on it.
            if not has_sufficient_profile(query, profile):
                print(f"MentraFiAI: {NEED_PROFILE_REPLY}\n")
                continue
            candidates = store.get_candidate_funds(profile, args.candidate_limit)
            ranked = rank_funds(candidates, profile)
            prompt = build_prompt(query, profile, ranked)

            # SLM reasoning via generate_reasoning(): swept-optimal sampling
            # (temp=0.5/top_k=40/top_p=0.9/rp=1.0/ng=3) + garbage backstop.
            raw_reply = generate_reasoning(model, tokenizer, prompt, device)

            # Facts are authoritative: we already instructed the model to
            # recommend ranked[0], so if its prose names no other matchable
            # candidate, default to ranked[0] rather than dropping the panel.
            sip_amount = extract_sip_amount(query)
            matched_fund = fuzzy_match_fund(raw_reply, ranked[:5]) or (ranked[0] if ranked else None)
            verified_reply = build_verified_response(raw_reply, matched_fund, sip_amount, profile)

            print(f"MentraFiAI: {verified_reply}\n")
    finally:
        store.close()


if __name__ == "__main__":
    main()
