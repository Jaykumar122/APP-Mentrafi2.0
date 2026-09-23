import { Router } from "express";
import pool from "../config/db";
import { syncScheme, seedStarterFunds, syncAllFunds, syncAllFundsFast } from "../services/fundSync";
import { syncFromAmfiOfficial, isAmfiSyncRunning } from "../services/amfiSync";

const router = Router();

// GET /api/funds?category=Equity&q=hdfc&schemeCode=120154&tenure=new&page=1&limit=20
router.get("/", async (req, res) => {
  try {
    const { category, q, schemeCode, tenure } = req.query;
    const page = Math.max(1, Number(req.query.page) || 1);
    const limit = Math.min(50, Number(req.query.limit) || 20);
    const offset = (page - 1) * limit;

    // Enforce fiduciary invariant: only display currently active, live schemes
    const conditions: string[] = ["f.is_active = TRUE"];
    const values: any[] = [];

    if (schemeCode) {
      values.push(Number(schemeCode));
      conditions.push(`f.scheme_code = $${values.length}`);
    }
    if (category && category !== "All") {
      values.push(category);
      conditions.push(`f.category = $${values.length}`);
    }
    if (q) {
      values.push(`%${(q as string).toLowerCase()}%`);
      conditions.push(`LOWER(f.name) LIKE $${values.length}`);
    }
    if (tenure === "new") {
      // Funds launched less than 1 year ago or without full 1Y history
      conditions.push("(f.one_year_return IS NULL OR f.launch_date >= NOW() - INTERVAL '1 year')");
    } else if (tenure === "growth") {
      // 1 to 3 years track record
      conditions.push("(f.one_year_return IS NOT NULL AND fp.return_3y IS NOT NULL AND fp.return_5y IS NULL)");
    } else if (tenure === "established") {
      // 5+ years track record
      conditions.push("(fp.return_5y IS NOT NULL)");
    }

    const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";

    const countResult = await pool.query(
      `SELECT COUNT(*) FROM funds f
       LEFT JOIN fund_performance fp ON fp.scheme_code = f.scheme_code
       ${where}`,
      values
    );
    const total = Number(countResult.rows[0].count);

    values.push(limit, offset);
    const result = await pool.query(
      `SELECT f.id, f.scheme_code, f.name, f.category, f.subcategory, f.nav, f.nav_date AS "navDate",
              f.fund_house AS "fundHouse", f.launch_date AS "launchDate", f.rating,
              f.expense_ratio AS "expenseRatio", f.aum_crores AS "aumCrores", f.sebi_riskometer AS "sebiRiskometer",
              fp.return_1m AS "return1m",
              fp.return_3m AS "return3m",
              fp.return_6m AS "return6m",
              COALESCE(fp.return_1y, f.one_year_return) AS "oneYearReturn",
              fp.return_3y AS "threeYearReturn",
              fp.return_5y AS "fiveYearReturn"
       FROM funds f
       LEFT JOIN fund_performance fp ON fp.scheme_code = f.scheme_code
       ${where}
       ORDER BY COALESCE(f.one_year_return, fp.return_6m, fp.return_3m, fp.return_1m) DESC NULLS LAST
       LIMIT $${values.length - 1} OFFSET $${values.length}`,
      values
    );

    res.json({
      funds: result.rows,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
        hasMore: offset + result.rows.length < total,
      },
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch funds" });
  }
});

// GET /api/funds/:schemeCode/performance — 1M/3M/6M/1Y/3Y/5Y/10Y returns
router.get("/:schemeCode/performance", async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT scheme_code, return_1m, return_3m, return_6m, return_1y, return_3y, return_5y, return_10y, updated_at
       FROM fund_performance WHERE scheme_code = $1`,
      [req.params.schemeCode]
    );
    if (result.rows.length === 0) {
      return res.status(404).json({ error: "No performance data yet for this fund" });
    }
    res.json(result.rows[0]);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch performance" });
  }
});

// Cache for chart data: schemeCode -> { timestamp, raw }
const chartCache = new Map<string, { timestamp: number; raw: any[] }>();
const CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour

function parseMfapiDate(dateStr: string): Date {
  const parts = dateStr.split("-");
  if (parts.length === 3) {
    return new Date(Number(parts[2]), Number(parts[1]) - 1, Number(parts[0]));
  }
  return new Date(dateStr);
}

// GET /api/funds/:schemeCode/nav-chart?range=1M|3M|6M|1Y|3Y|5Y|ALL
router.get("/:schemeCode/nav-chart", async (req, res) => {
  try {
    const schemeCode = req.params.schemeCode;
    const range = ((req.query.range as string) || "1Y").toUpperCase();

    let rawData: { date: string; nav: string }[] = [];
    const cached = chartCache.get(schemeCode);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
      rawData = cached.raw;
    } else {
      const response = await fetch(`https://api.mfapi.in/mf/${schemeCode}`);
      if (response.ok) {
        const json = (await response.json()) as any;
        rawData = json.data || [];
        if (rawData.length > 0) {
          chartCache.set(schemeCode, { timestamp: Date.now(), raw: rawData });
        }
      }
    }

    if (!rawData || rawData.length === 0) {
      const fundRes = await pool.query(
        "SELECT nav, nav_date FROM funds WHERE scheme_code = $1",
        [schemeCode]
      );
      if (fundRes.rows.length === 0) {
        return res.status(404).json({ error: "Fund not found" });
      }
      const nav = Number(fundRes.rows[0].nav) || 10;
      return res.json({
        schemeCode: Number(schemeCode),
        range,
        periodReturn: 0,
        minNav: nav,
        maxNav: nav,
        startNav: nav,
        latestNav: nav,
        startDate: "Today",
        latestDate: "Today",
        totalPoints: 1,
        points: [{ date: "Today", nav }],
      });
    }

    const latestDate = parseMfapiDate(rawData[0].date);
    let cutoffDate: Date | null = new Date(latestDate);

    if (range === "1M") {
      cutoffDate.setMonth(cutoffDate.getMonth() - 1);
    } else if (range === "3M") {
      cutoffDate.setMonth(cutoffDate.getMonth() - 3);
    } else if (range === "6M") {
      cutoffDate.setMonth(cutoffDate.getMonth() - 6);
    } else if (range === "1Y") {
      cutoffDate.setFullYear(cutoffDate.getFullYear() - 1);
    } else if (range === "3Y") {
      cutoffDate.setFullYear(cutoffDate.getFullYear() - 3);
    } else if (range === "5Y") {
      cutoffDate.setFullYear(cutoffDate.getFullYear() - 5);
    } else {
      cutoffDate = null;
    }

    let filtered: { date: string; nav: number; time: number }[] = [];
    for (let i = rawData.length - 1; i >= 0; i--) {
      const d = parseMfapiDate(rawData[i].date);
      const navVal = parseFloat(rawData[i].nav);
      if (isNaN(navVal) || isNaN(d.getTime())) continue;

      if (!cutoffDate || d >= cutoffDate) {
        filtered.push({
          date: rawData[i].date,
          nav: navVal,
          time: d.getTime(),
        });
      }
    }

    if (filtered.length < 2 && rawData.length >= 2) {
      const slice = rawData.slice(0, Math.min(10, rawData.length)).reverse();
      filtered = slice.map((p) => ({
        date: p.date,
        nav: parseFloat(p.nav),
        time: parseMfapiDate(p.date).getTime(),
      }));
    }

    const TARGET_POINTS = 35;
    let sampled: { date: string; nav: number }[] = [];
    if (filtered.length <= TARGET_POINTS) {
      sampled = filtered.map((p) => ({ date: p.date, nav: p.nav }));
    } else {
      const step = (filtered.length - 1) / (TARGET_POINTS - 1);
      for (let i = 0; i < TARGET_POINTS; i++) {
        const idx = Math.round(i * step);
        sampled.push({
          date: filtered[idx].date,
          nav: filtered[idx].nav,
        });
      }
      sampled[sampled.length - 1] = {
        date: filtered[filtered.length - 1].date,
        nav: filtered[filtered.length - 1].nav,
      };
    }

    const navs = sampled.map((p) => p.nav);
    const minNav = Math.min(...navs);
    const maxNav = Math.max(...navs);
    const startNav = sampled[0].nav;
    const latestNav = sampled[sampled.length - 1].nav;
    const periodReturn = startNav > 0 ? Number((((latestNav - startNav) / startNav) * 100).toFixed(2)) : 0;

    res.json({
      schemeCode: Number(schemeCode),
      range,
      periodReturn,
      minNav,
      maxNav,
      startNav,
      latestNav,
      startDate: sampled[0].date,
      latestDate: sampled[sampled.length - 1].date,
      totalPoints: filtered.length,
      points: sampled,
    });
  } catch (err) {
    console.error("nav-chart error:", err);
    res.status(500).json({ error: "Failed to fetch nav chart" });
  }
});

// GET /api/funds/:schemeCode/details — complete fund metadata from both AMFI and MFAPI
router.get("/:schemeCode/details", async (req, res) => {
  try {
    const schemeCode = req.params.schemeCode;
    const [fundResult, holdingsResult, scoresResult] = await Promise.all([
      pool.query(
        `SELECT f.scheme_code, f.isin, f.name, f.category, f.subcategory, f.plan_type, f.option_type,
                f.fund_house, f.launch_date, f.nav, f.nav_date, f.is_active,
                f.expense_ratio, f.aum_crores, f.sebi_riskometer, f.fund_manager, f.benchmark_name, f.rating,
                fp.return_1m, fp.return_3m, fp.return_6m, fp.return_1y, fp.return_3y, fp.return_5y, fp.return_10y,
                fd.summary, fd.investment_strategy, fd.suitable_for
         FROM funds f
         LEFT JOIN fund_performance fp ON fp.scheme_code = f.scheme_code
         LEFT JOIN fund_descriptions fd ON fd.scheme_code = f.scheme_code
         WHERE f.scheme_code = $1`,
        [schemeCode]
      ),
      pool.query(
        `SELECT stock_name, ticker, sector, weight_percentage
         FROM fund_holdings
         WHERE scheme_code = $1
         ORDER BY weight_percentage DESC
         LIMIT 10`,
        [schemeCode]
      ),
      pool.query(
        `SELECT risk_profile, score
         FROM fund_scores
         WHERE scheme_code = $1`,
        [schemeCode]
      ),
    ]);

    if (fundResult.rows.length === 0) {
      return res.status(404).json({ error: "Fund not found" });
    }

    const fundData = fundResult.rows[0];
    const holdings = holdingsResult.rows.map((h) => ({
      stock: h.stock_name,
      ticker: h.ticker,
      sector: h.sector,
      weight: parseFloat(h.weight_percentage),
    }));

    const scores: Record<string, number> = {};
    for (const row of scoresResult.rows) {
      scores[row.risk_profile.toLowerCase()] = parseFloat(row.score);
    }

    res.json({
      ...fundData,
      holdings,
      scores,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch fund details" });
  }
});

// POST /api/funds/sync/:schemeCode — admin-only in production; manually add a fund by code
router.post("/sync/:schemeCode", async (req, res) => {
  try {
    await syncScheme(Number(req.params.schemeCode));
    res.json({ success: true });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Sync failed" });
  }
});

// POST /api/funds/seed — run once to populate starter funds
router.post("/seed", async (_req, res) => {
  await seedStarterFunds();
  res.json({ success: true });
});

router.post("/sync-fast", async (_req, res) => {
  res.json({ started: true, message: "Fast NAV sync running in background" });
  syncAllFundsFast().catch((err) => console.error("Fast sync failed:", err));
});

router.post("/sync-all", async (_req, res) => {
  res.json({ started: true, message: "Full sync running in background — watch server logs" });
  syncAllFunds().catch((err) => console.error("Full sync failed:", err));
});

router.post("/sync-amfi", async (_req, res) => {
  if (isAmfiSyncRunning()) {
    return res.status(409).json({ error: "AMFI sync is already running in background" });
  }
  res.json({ started: true, message: "Official AMFI portal sync running in background — watch server logs" });
  syncFromAmfiOfficial().catch((err) => console.error("AMFI sync failed:", err));
});

export default router;