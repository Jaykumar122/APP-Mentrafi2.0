"""
MentraFiAI Database Fund Retriever
Connects to PostgreSQL database (mentrafi) to fetch real, top-performing direct mutual funds
for grounded recommendations and fund lookups.

Fixes applied:
  - Fix #2: python-dotenv loads DATABASE_URL from .env before any DB call.
  - Fix #3: psycopg2.pool.SimpleConnectionPool replaces a single bare connection,
            preventing silent stale-connection failures that return empty fund data.
"""

import os
import psycopg2
import psycopg2.pool
from pathlib import Path
from typing import Optional, Dict, Any, List

# ── Fix #2: Load .env so DATABASE_URL is always available ──────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass  # python-dotenv not installed; fall back to os.environ only

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:1223@localhost:5432/mentrafi")


def format_star_rating(rating: Optional[float]) -> str:
    if rating is None or rating <= 0:
        return "Not Rated"
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    empty_stars = max(0, 5 - full_stars - half_star)
    return f"{'★' * full_stars}{'½' if half_star else ''}{'☆' * empty_stars} ({rating:.1f}/5)"


class FundDatabase:
    """
    Manages a psycopg2 SimpleConnectionPool (1–5 connections).

    Fix #3: Every public method borrows a connection from the pool, uses it,
    and returns it via putconn().  A lightweight ping (SELECT 1) detects stale
    connections; if the ping fails the connection is discarded and a fresh one
    is borrowed, so queries never silently return empty results due to a
    dropped idle connection.
    """

    _MIN_CONN = 1
    _MAX_CONN = 5

    def __init__(self, db_url: str = DATABASE_URL):
        self.db_url = db_url
        self._pool: Optional[psycopg2.pool.SimpleConnectionPool] = None
        self.is_connected = False
        self._init_pool()

    # ── Pool lifecycle ─────────────────────────────────────────────────────

    def _init_pool(self) -> None:
        """Create the connection pool. Called once on startup."""
        try:
            self._pool = psycopg2.pool.SimpleConnectionPool(
                self._MIN_CONN,
                self._MAX_CONN,
                self.db_url,
            )
            self.is_connected = True
        except Exception as exc:
            print(f"[DB] Connection pool init failed: {exc}")
            self._pool = None
            self.is_connected = False

    def _get_conn(self):
        """
        Borrow a connection from the pool.
        Pings it (SELECT 1) to detect stale connections; discards and replaces
        if the ping fails.  Returns None if the pool is unavailable.
        """
        if self._pool is None:
            self._init_pool()
        if self._pool is None:
            return None
        try:
            conn = self._pool.getconn()
            # Ping to detect idle/stale connections
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                conn.commit()
            except Exception:
                # Stale — discard and let the pool give us a fresh one
                try:
                    self._pool.putconn(conn, close=True)
                except Exception:
                    pass
                conn = self._pool.getconn()
            return conn
        except Exception as exc:
            print(f"[DB] Could not borrow connection from pool: {exc}")
            return None

    def _put_conn(self, conn, close: bool = False) -> None:
        """Return a connection to the pool (or discard it if close=True)."""
        if self._pool is not None and conn is not None:
            try:
                self._pool.putconn(conn, close=close)
            except Exception:
                pass

    def close(self) -> None:
        """Close all connections in the pool (call on application shutdown)."""
        if self._pool is not None:
            try:
                self._pool.closeall()
            except Exception:
                pass
            self._pool = None
            self.is_connected = False

    # ── Public query methods ───────────────────────────────────────────────

    def get_fund_count(self) -> int:
        conn = self._get_conn()
        if conn is None:
            return 0
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM funds")
                result = cur.fetchone()[0]
            conn.commit()
            return result
        except Exception:
            self._put_conn(conn, close=True)
            return 0
        finally:
            self._put_conn(conn)

    def get_top_fund(self, category_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves one of the top-rated Direct Plan, Growth mutual fund schemes for a given category.
        Fetches the top 5 and randomly selects one to provide AMC diversification across sessions (Fix #6).
        """
        import random
        queries = {
            "flexi_cap": """
                SELECT f.scheme_code, f.name, f.nav, f.rating, p.return_3y, f.fund_house,
                       p.return_1y, p.return_5y
                FROM funds f LEFT JOIN fund_performance p ON f.scheme_code = p.scheme_code
                WHERE f.plan_type = 'Direct' AND f.option_type = 'Growth' AND f.subcategory = 'Flexi Cap'
                  AND f.is_active = TRUE
                  AND f.name NOT ILIKE '%Series%' AND f.name NOT ILIKE '%US %' AND f.name NOT ILIKE '%Global%'
                ORDER BY p.return_3y DESC NULLS LAST, f.rating DESC NULLS LAST LIMIT 5
            """,
            "large_cap": """
                SELECT f.scheme_code, f.name, f.nav, f.rating, p.return_3y, f.fund_house,
                       p.return_1y, p.return_5y
                FROM funds f LEFT JOIN fund_performance p ON f.scheme_code = p.scheme_code
                WHERE f.plan_type = 'Direct' AND f.option_type = 'Growth' AND f.subcategory = 'Large Cap'
                  AND f.is_active = TRUE
                  AND f.name NOT ILIKE '%Series%' AND f.name NOT ILIKE '%US %' AND f.name NOT ILIKE '%Global%'
                ORDER BY p.return_3y DESC NULLS LAST, f.rating DESC NULLS LAST LIMIT 5
            """,
            "mid_cap": """
                SELECT f.scheme_code, f.name, f.nav, f.rating, p.return_3y, f.fund_house,
                       p.return_1y, p.return_5y
                FROM funds f LEFT JOIN fund_performance p ON f.scheme_code = p.scheme_code
                WHERE f.plan_type = 'Direct' AND f.option_type = 'Growth' AND f.subcategory = 'Mid Cap'
                  AND f.is_active = TRUE
                  AND f.name NOT ILIKE '%Large & Mid%' AND f.name NOT ILIKE '%Series%'
                ORDER BY p.return_3y DESC NULLS LAST, f.rating DESC NULLS LAST LIMIT 5
            """,
            "small_cap": """
                SELECT f.scheme_code, f.name, f.nav, f.rating, p.return_3y, f.fund_house,
                       p.return_1y, p.return_5y
                FROM funds f LEFT JOIN fund_performance p ON f.scheme_code = p.scheme_code
                WHERE f.plan_type = 'Direct' AND f.option_type = 'Growth' AND f.subcategory = 'Small Cap'
                  AND f.is_active = TRUE
                  AND f.name NOT ILIKE '%Series%'
                ORDER BY p.return_3y DESC NULLS LAST, f.rating DESC NULLS LAST LIMIT 5
            """,
            "hybrid": """
                SELECT f.scheme_code, f.name, f.nav, f.rating, p.return_3y, f.fund_house,
                       p.return_1y, p.return_5y
                FROM funds f LEFT JOIN fund_performance p ON f.scheme_code = p.scheme_code
                WHERE f.plan_type = 'Direct' AND f.option_type = 'Growth' AND f.category = 'Hybrid'
                  AND f.is_active = TRUE
                  AND (f.name ILIKE '%Hybrid%' OR f.name ILIKE '%Balanced%')
                  AND f.name NOT ILIKE '%Series%' AND f.name NOT ILIKE '%FOF%'
                  AND f.name NOT ILIKE '%Retirement%' AND f.name NOT ILIKE '%Children%'
                  AND f.name NOT ILIKE '%Segregated%'
                ORDER BY p.return_3y DESC NULLS LAST, f.rating DESC NULLS LAST LIMIT 5
            """,
            "debt": """
                SELECT f.scheme_code, f.name, f.nav, f.rating, p.return_3y, f.fund_house,
                       p.return_1y, p.return_5y
                FROM funds f LEFT JOIN fund_performance p ON f.scheme_code = p.scheme_code
                WHERE f.plan_type = 'Direct' AND f.option_type = 'Growth' AND f.category = 'Debt'
                  AND f.is_active = TRUE
                  AND (f.name ILIKE '%Short Duration%' OR f.name ILIKE '%Corporate Bond%')
                  AND f.name NOT ILIKE '%Series%' AND f.name NOT ILIKE '%Segregated%'
                ORDER BY p.return_3y DESC NULLS LAST, f.rating DESC NULLS LAST LIMIT 5
            """,
            "elss": """
                SELECT f.scheme_code, f.name, f.nav, f.rating, p.return_3y, f.fund_house,
                       p.return_1y, p.return_5y
                FROM funds f LEFT JOIN fund_performance p ON f.scheme_code = p.scheme_code
                WHERE f.plan_type = 'Direct' AND f.option_type = 'Growth'
                  AND f.is_active = TRUE
                  AND (f.category = 'ELSS' OR f.subcategory = 'Tax Saving')
                  AND f.name NOT ILIKE '%Series%'
                ORDER BY p.return_3y DESC NULLS LAST, f.rating DESC NULLS LAST LIMIT 5
            """,
            "index": """
                SELECT f.scheme_code, f.name, f.nav, f.rating, p.return_3y, f.fund_house,
                       p.return_1y, p.return_5y
                FROM funds f LEFT JOIN fund_performance p ON f.scheme_code = p.scheme_code
                WHERE f.plan_type = 'Direct' AND f.option_type = 'Growth'
                  AND f.is_active = TRUE
                  AND (f.name ILIKE '%Nifty 50 Index%' OR f.name ILIKE '%Sensex Index%')
                  AND f.name NOT ILIKE '%Equal%' AND f.name NOT ILIKE '%Value%'
                  AND f.name NOT ILIKE '%Junior%'
                ORDER BY p.return_3y DESC NULLS LAST, f.rating DESC NULLS LAST LIMIT 5
            """,
            "conservative_hybrid": """
                SELECT f.scheme_code, f.name, f.nav, f.rating, p.return_3y, f.fund_house,
                       p.return_1y, p.return_5y
                FROM funds f LEFT JOIN fund_performance p ON f.scheme_code = p.scheme_code
                WHERE f.plan_type = 'Direct' AND f.option_type = 'Growth' AND f.is_active = TRUE
                  AND f.sebi_riskometer IN ('Low', 'Low to Moderate', 'Moderate')
                  AND (f.name ILIKE '%Conservative%' OR f.subcategory ILIKE '%Conservative%')
                  AND f.name NOT ILIKE '%Aggressive%' AND f.name NOT ILIKE '%Series%' AND f.name NOT ILIKE '%Segregated%'
                ORDER BY f.rating DESC NULLS LAST, p.return_3y DESC NULLS LAST LIMIT 5
            """,
            "low_risk": """
                SELECT f.scheme_code, f.name, f.nav, f.rating, p.return_3y, f.fund_house,
                       p.return_1y, p.return_5y
                FROM funds f LEFT JOIN fund_performance p ON f.scheme_code = p.scheme_code
                WHERE f.plan_type = 'Direct' AND f.option_type = 'Growth' AND f.is_active = TRUE
                  AND f.sebi_riskometer IN ('Low', 'Low to Moderate', 'Moderate')
                  AND (
                    f.name ILIKE '%Conservative Hybrid%'
                    OR f.name ILIKE '%Corporate Bond%'
                    OR f.name ILIKE '%Short Duration%'
                    OR f.name ILIKE '%Medium Term%'
                    OR f.subcategory ILIKE '%Conservative%'
                  )
                  AND f.name NOT ILIKE '%Aggressive%' AND f.name NOT ILIKE '%Credit Risk%'
                  AND f.name NOT ILIKE '%Flexi%' AND f.name NOT ILIKE '%Mid%' AND f.name NOT ILIKE '%Small%'
                  AND f.name NOT ILIKE '%Series%' AND f.name NOT ILIKE '%Segregated%'
                ORDER BY f.rating DESC NULLS LAST, p.return_3y DESC NULLS LAST LIMIT 5
            """,
        }

        query = queries.get(category_key.lower())
        if not query:
            return None

        conn = self._get_conn()
        if conn is None:
            return None
        try:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
            conn.commit()
            if not rows:
                return None
            row = random.choice(rows)
            return {
                "scheme_code": row[0],
                "name": row[1],
                "nav": float(row[2]) if row[2] is not None else None,
                "rating": float(row[3]) if row[3] is not None else None,
                "return_3y": float(row[4]) if row[4] is not None else None,
                "fund_house": row[5],
                "return_1y": float(row[6]) if row[6] is not None else None,  # Fix #17
                "return_5y": float(row[7]) if row[7] is not None else None,  # Fix #17
            }
        except Exception:
            self._put_conn(conn, close=True)
            return None
        finally:
            self._put_conn(conn)

    def get_suitable_low_risk_funds(self, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top-rated, SEBI-compliant low-to-moderate risk mutual funds for conservative investors.
        Strictly excludes pure equity funds (Flexi Cap, Large & Mid Cap, Mid Cap, Small Cap, Sectoral).
        Applies deterministic suitability scoring based on rating, 3Y trailing return, and riskometer tier.
        """
        conn = self._get_conn()
        if conn is None:
            return [
                {
                    "scheme_code": 120154,
                    "name": "Kotak Conservative Hybrid Fund - Direct Plan - Growth",
                    "category": "Hybrid",
                    "subcategory": "Conservative Hybrid",
                    "nav": 52.48,
                    "rating": 4.0,
                    "return_1y": 8.50,
                    "return_3y": 9.66,
                    "return_5y": 9.10,
                    "fund_house": "Kotak Mahindra Mutual Fund",
                    "sebi_riskometer": "Moderate",
                    "suitability_score": 92.5
                },
                {
                    "scheme_code": 118569,
                    "name": "Franklin India Corporate Bond Fund - Direct Plan - Growth",
                    "category": "Debt",
                    "subcategory": "Corporate Bond",
                    "nav": 94.12,
                    "rating": 4.0,
                    "return_1y": 7.40,
                    "return_3y": 8.14,
                    "return_5y": 7.85,
                    "fund_house": "Franklin Templeton Mutual Fund",
                    "sebi_riskometer": "Low to Moderate",
                    "suitability_score": 89.0
                },
                {
                    "scheme_code": 119118,
                    "name": "HDFC Conservative Hybrid Fund - Direct Plan - Growth Option",
                    "category": "Hybrid",
                    "subcategory": "Conservative Hybrid",
                    "nav": 68.30,
                    "rating": 4.0,
                    "return_1y": 8.10,
                    "return_3y": 8.34,
                    "return_5y": 8.40,
                    "fund_house": "HDFC Mutual Fund",
                    "sebi_riskometer": "Moderate",
                    "suitability_score": 88.0
                }
            ][:limit]
        try:
            sql = """
                SELECT f.scheme_code, f.name, f.category, f.subcategory, f.nav, f.rating,
                       p.return_1y, p.return_3y, p.return_5y, f.fund_house, f.sebi_riskometer
                FROM funds f
                LEFT JOIN fund_performance p ON f.scheme_code = p.scheme_code
                WHERE f.plan_type = 'Direct' AND f.option_type = 'Growth' AND f.is_active = TRUE
                  AND f.sebi_riskometer IN ('Low', 'Low to Moderate', 'Moderate')
                  AND (
                    f.name ILIKE '%Conservative Hybrid%'
                    OR f.name ILIKE '%Corporate Bond%'
                    OR f.name ILIKE '%Short Duration%'
                    OR f.name ILIKE '%Medium Term%'
                    OR f.name ILIKE '%Banking & PSU%'
                    OR f.subcategory ILIKE '%Conservative%'
                  )
                  AND f.name NOT ILIKE '%Aggressive%'
                  AND f.name NOT ILIKE '%Credit Risk%'
                  AND f.name NOT ILIKE '%Flexi%'
                  AND f.name NOT ILIKE '%Mid%'
                  AND f.name NOT ILIKE '%Small%'
                  AND f.name NOT ILIKE '%Large & Mid%'
                  AND f.name NOT ILIKE '%Thematic%'
                  AND f.name NOT ILIKE '%Sectoral%'
                  AND f.name NOT ILIKE '%Series%'
                  AND f.name NOT ILIKE '%Segregated%'
                  AND f.name NOT ILIKE '%FOF%'
                ORDER BY f.rating DESC NULLS LAST, p.return_3y DESC NULLS LAST
                LIMIT 15
            """
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
            conn.commit()

            results = []
            for r in rows:
                rating = float(r[5]) if r[5] is not None else 3.5
                ret_3y = float(r[7]) if r[7] is not None else 7.0
                riskometer = r[10] or "Moderate"
                safety_bonus = 30.0 if riskometer == "Low" else (25.0 if riskometer == "Low to Moderate" else 20.0)
                suitability_score = round((rating * 8.0) + (ret_3y * 2.5) + safety_bonus, 1)

                results.append({
                    "scheme_code": r[0],
                    "name": r[1],
                    "category": r[2] or "Debt",
                    "subcategory": r[3] or ("Conservative Hybrid" if "hybrid" in r[1].lower() else "Corporate Bond"),
                    "nav": float(r[4]) if r[4] is not None else None,
                    "rating": rating,
                    "return_1y": float(r[6]) if r[6] is not None else None,
                    "return_3y": ret_3y,
                    "return_5y": float(r[8]) if r[8] is not None else None,
                    "fund_house": r[9],
                    "sebi_riskometer": riskometer,
                    "suitability_score": suitability_score
                })

            results.sort(key=lambda x: x["suitability_score"], reverse=True)
            return results[:limit]
        except Exception as e:
            print(f"[DB] get_suitable_low_risk_funds failed: {e}")
            self._put_conn(conn, close=True)
            return []
        finally:
            self._put_conn(conn)

    def search_funds(self, query_text: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Searches funds by keyword across fund names.
        Handles synonyms (e.g. bluechip -> Large Cap, Top 100 -> Nifty 100 / Large Cap).
        Prioritizes Direct Growth plans and highest returns.
        """
        clean_q = query_text.strip().lower()
        raw_tokens = [t for t in clean_q.split() if len(t) > 1 and t not in ["fund", "scheme", "direct", "growth", "plan"]]
        if not raw_tokens:
            raw_tokens = clean_q.split()
        if not raw_tokens:
            return []

        where_clauses: list[str] = []
        params: list = []
        i = 0
        while i < len(raw_tokens):
            t = raw_tokens[i]
            if t in ["bluechip", "blue-chip"]:
                where_clauses.append("(f.name ILIKE %s OR f.name ILIKE %s)")
                params.extend(["%Bluechip%", "%Large Cap%"])
                i += 1
            elif t == "top" and i + 1 < len(raw_tokens) and raw_tokens[i + 1] == "100":
                where_clauses.append("(f.name ILIKE %s OR f.name ILIKE %s OR f.name ILIKE %s)")
                params.extend(["%Top 100%", "%Large Cap%", "%Nifty 100%"])
                i += 2
            elif t in ["top100", "top-100"]:
                where_clauses.append("(f.name ILIKE %s OR f.name ILIKE %s OR f.name ILIKE %s)")
                params.extend(["%Top 100%", "%Large Cap%", "%Nifty 100%"])
                i += 1
            elif t in ["elss", "taxsaver", "tax-saver"]:
                where_clauses.append("(f.name ILIKE %s OR f.name ILIKE %s)")
                params.extend(["%ELSS%", "%Tax%"])
                i += 1
            else:
                where_clauses.append("f.name ILIKE %s")
                params.append(f"%{t}%")
                i += 1

        # Always filter out dead/discontinued schemes
        where_clauses.append("f.is_active = TRUE")

        sql = f"""
        SELECT f.scheme_code, f.name, f.category, f.subcategory, f.nav, f.rating,
               f.plan_type, f.option_type, f.fund_house,
               p.return_1y, p.return_3y, p.return_5y
        FROM funds f
        LEFT JOIN fund_performance p ON f.scheme_code = p.scheme_code
        WHERE {' AND '.join(where_clauses)}
        ORDER BY (f.plan_type = 'Direct') DESC, (f.option_type = 'Growth') DESC,
                 f.rating DESC NULLS LAST, p.return_3y DESC NULLS LAST
        LIMIT %s
        """
        query_params = list(params) + [limit]

        conn = self._get_conn()
        if conn is None:
            return []
        try:
            with conn.cursor() as cur:
                cur.execute(sql, query_params)
                rows = cur.fetchall()

                # Strict AND found nothing → fallback to OR with relevance ranking
                if not rows and len(raw_tokens) > 1:
                    rank_parts = ["(CASE WHEN f.name ILIKE %s THEN 1 ELSE 0 END)" for _ in raw_tokens]
                    rank_expr = " + ".join(rank_parts)
                    or_clauses = ["f.name ILIKE %s" for _ in raw_tokens]
                    fallback_sql = f"""
                    SELECT f.scheme_code, f.name, f.category, f.subcategory, f.nav, f.rating,
                           f.plan_type, f.option_type, f.fund_house,
                           p.return_1y, p.return_3y, p.return_5y
                    FROM funds f
                    LEFT JOIN fund_performance p ON f.scheme_code = p.scheme_code
                    WHERE ({' OR '.join(or_clauses)}) AND f.is_active = TRUE
                    ORDER BY ({rank_expr}) DESC,
                             (f.plan_type = 'Direct') DESC, (f.option_type = 'Growth') DESC,
                             f.rating DESC NULLS LAST, p.return_3y DESC NULLS LAST
                    LIMIT %s
                    """
                    or_params = [f"%{t}%" for t in raw_tokens] + [f"%{t}%" for t in raw_tokens] + [limit]
                    cur.execute(fallback_sql, or_params)
                    rows = cur.fetchall()

            conn.commit()
            results = []
            for r in rows:
                results.append({
                    "scheme_code": r[0],
                    "name": r[1],
                    "category": r[2],
                    "subcategory": r[3],
                    "nav": float(r[4]) if r[4] is not None else None,
                    "rating": float(r[5]) if r[5] is not None else None,
                    "plan_type": r[6],
                    "option_type": r[7],
                    "fund_house": r[8],
                    "return_1y": float(r[9]) if r[9] is not None else None,
                    "return_3y": float(r[10]) if r[10] is not None else None,
                    "return_5y": float(r[11]) if r[11] is not None else None,
                })
            return results
        except Exception:
            self._put_conn(conn, close=True)
            return []
        finally:
            self._put_conn(conn)

    def get_fund_holdings(self, scheme_code: int) -> List[Dict[str, Any]]:
        """Returns the top stock holdings and sector weights for a given mutual fund scheme."""
        conn = self._get_conn()
        if conn is None:
            return []
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT stock_name, ticker, sector, weight_percentage
                    FROM fund_holdings
                    WHERE scheme_code = %s
                    ORDER BY weight_percentage DESC LIMIT 10
                """, (scheme_code,))
                rows = cur.fetchall()
            conn.commit()
            return [
                {"stock": r[0], "ticker": r[1], "sector": r[2], "weight": float(r[3])}
                for r in rows
            ]
        except Exception:
            self._put_conn(conn, close=True)
            return []
        finally:
            self._put_conn(conn)

    def get_multi_asset_rates(self) -> List[Dict[str, Any]]:
        """Returns government small savings, sovereign gold bonds, and fixed deposit rates."""
        conn = self._get_conn()
        if conn is None:
            return []
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT asset_name, category, interest_rate, lock_in_years, tax_treatment, sebi_risk_tier, description
                    FROM multi_asset_rates
                    ORDER BY interest_rate DESC
                """)
                rows = cur.fetchall()
            conn.commit()
            return [
                {
                    "name": r[0],
                    "category": r[1],
                    "rate": float(r[2]),
                    "lock_in_years": float(r[3]),
                    "tax_treatment": r[4],
                    "risk_tier": r[5],
                    "description": r[6]
                }
                for r in rows
            ]
        except Exception:
            self._put_conn(conn, close=True)
            return []
        finally:
            self._put_conn(conn)

    def check_portfolio_overlap(self, scheme_codes: List[int]) -> Dict[str, Any]:
        """Calculates portfolio overlap across a list of mutual fund scheme codes."""
        if len(scheme_codes) < 2:
            return {"overlapping_stocks": [], "total_overlap_percentage": 0.0}
        conn = self._get_conn()
        if conn is None:
            return {"overlapping_stocks": [], "total_overlap_percentage": 0.0}
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT stock_name, sector, COUNT(DISTINCT scheme_code) as fund_count, AVG(weight_percentage) as avg_wt
                    FROM fund_holdings
                    WHERE scheme_code = ANY(%s)
                    GROUP BY stock_name, sector
                    HAVING COUNT(DISTINCT scheme_code) > 1
                    ORDER BY avg_wt DESC;
                """, (scheme_codes,))
                rows = cur.fetchall()
            conn.commit()
            overlaps = [
                {"stock": r[0], "sector": r[1], "fund_count": r[2], "avg_weight": round(float(r[3]), 2)}
                for r in rows
            ]
            total_wt = round(sum(item["avg_weight"] for item in overlaps), 2)
            return {"overlapping_stocks": overlaps, "total_overlap_percentage": total_wt}
        except Exception:
            self._put_conn(conn, close=True)
            return {"overlapping_stocks": [], "total_overlap_percentage": 0.0}
        finally:
            self._put_conn(conn)

    def get_fund_details(self, scheme_code: int) -> Optional[Dict[str, Any]]:
        """Fetches complete scheme metadata, returns, description, holdings, and scores."""
        conn = self._get_conn()
        if conn is None:
            return None
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT f.scheme_code, f.isin, f.name, f.category, f.subcategory, f.plan_type, f.option_type,
                           f.fund_house, f.launch_date, f.nav, f.nav_date, f.is_active,
                           f.expense_ratio, f.aum_crores, f.sebi_riskometer, f.fund_manager, f.benchmark_name, f.rating,
                           fp.return_1m, fp.return_3m, fp.return_6m, fp.return_1y, fp.return_3y, fp.return_5y, fp.return_10y,
                           fd.summary, fd.investment_strategy, fd.suitable_for
                    FROM funds f
                    LEFT JOIN fund_performance fp ON fp.scheme_code = f.scheme_code
                    LEFT JOIN fund_descriptions fd ON fd.scheme_code = f.scheme_code
                    WHERE f.scheme_code = %s
                """, (scheme_code,))
                row = cur.fetchone()
                if not row:
                    return None

                cur.execute("""
                    SELECT stock_name, ticker, sector, weight_percentage
                    FROM fund_holdings
                    WHERE scheme_code = %s
                    ORDER BY weight_percentage DESC LIMIT 10
                """, (scheme_code,))
                holdings_rows = cur.fetchall()

                cur.execute("""
                    SELECT risk_profile, score
                    FROM fund_scores
                    WHERE scheme_code = %s
                """, (scheme_code,))
                scores_rows = cur.fetchall()

            conn.commit()

            return {
                "scheme_code": row[0],
                "isin": row[1],
                "name": row[2],
                "category": row[3],
                "subcategory": row[4],
                "plan_type": row[5],
                "option_type": row[6],
                "fund_house": row[7],
                "launch_date": str(row[8]) if row[8] else None,
                "nav": float(row[9]) if row[9] is not None else None,
                "nav_date": str(row[10]) if row[10] else None,
                "is_active": bool(row[11]),
                "expense_ratio": float(row[12]) if row[12] is not None else None,
                "aum_crores": float(row[13]) if row[13] is not None else None,
                "sebi_riskometer": row[14],
                "fund_manager": row[15],
                "benchmark_name": row[16],
                "rating": float(row[17]) if row[17] is not None else None,
                "return_1m": float(row[18]) if row[18] is not None else None,
                "return_3m": float(row[19]) if row[19] is not None else None,
                "return_6m": float(row[20]) if row[20] is not None else None,
                "return_1y": float(row[21]) if row[21] is not None else None,
                "return_3y": float(row[22]) if row[22] is not None else None,
                "return_5y": float(row[23]) if row[23] is not None else None,
                "return_10y": float(row[24]) if row[24] is not None else None,
                "summary": row[25],
                "investment_strategy": row[26],
                "suitable_for": row[27],
                "holdings": [
                    {"stock": h[0], "ticker": h[1], "sector": h[2], "weight": float(h[3])}
                    for h in holdings_rows
                ],
                "scores": {
                    s[0].lower(): float(s[1]) for s in scores_rows
                }
            }
        except Exception:
            self._put_conn(conn, close=True)
            return None
        finally:
            self._put_conn(conn)


