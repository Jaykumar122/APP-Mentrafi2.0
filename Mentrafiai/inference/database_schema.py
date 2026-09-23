"""
Database schema adapter for PostgreSQL mutual fund and user profile data.

Maps real application database columns to MentraFiAI advisor expectations.
Configure through environment variables if your schema differs.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FundColumnMapping:
    """Maps real database columns to advisor expectations."""

    fund_name: str = os.getenv("DB_FUND_NAME_COLUMN", "name")
    category: str = os.getenv("DB_FUND_CATEGORY_COLUMN", "category")
    subcategory: str = os.getenv("DB_FUND_SUBCATEGORY_COLUMN", "subcategory")
    nav: str = os.getenv("DB_FUND_NAV_COLUMN", "nav")
    expense_ratio: str = os.getenv("DB_FUND_EXPENSE_RATIO_COLUMN", "None")
    performance: str = os.getenv("DB_FUND_PERFORMANCE_COLUMN", "one_year_return")
    # NOTE: this schema has no risk-level column. `rating` is a 1-5 quality score,
    # NOT a risk level, so risk suitability is derived from category/subcategory
    # (see _derive_risk_level in mutual_fund_advisor.py) rather than read here.
    risk_level: str = os.getenv("DB_FUND_RISK_LEVEL_COLUMN", "None")


@dataclass(frozen=True)
class UserProfileColumnMapping:
    """Maps real database columns to advisor suitability expectations."""

    user_id: str = os.getenv("DB_USER_ID_COLUMN", "id")
    risk_level: str = os.getenv("DB_USER_RISK_COLUMN", "None")
    horizon_years: str = os.getenv("DB_USER_HORIZON_COLUMN", "None")
    preferred_category: str = os.getenv("DB_USER_CATEGORY_COLUMN", "None")


FUND_COLUMNS = FundColumnMapping()
USER_COLUMNS = UserProfileColumnMapping()

FUNDS_TABLE = os.getenv("DB_FUNDS_TABLE", "funds")
USERS_TABLE = os.getenv("DB_USERS_TABLE", "users")
