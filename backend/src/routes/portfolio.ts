// src/routes/portfolio.ts
import { Router } from "express";
import pool from "../config/db";
import { authenticateToken, AuthRequest } from "../middleware/auth";

const router = Router();

type AssetClass = "Equity" | "Debt" | "Hybrid";

// Funds are classified into many categories (Equity, Debt, Hybrid, ELSS, Gold)
// by fundSync.ts, but the Portfolio/Home UI only understands three buckets.
// ELSS is equity-oriented and Gold behaves like an alternative/defensive
// asset, so both are folded into the closest bucket for allocation purposes.
function mapAssetClass(category: string | null): AssetClass {
  if (category === "Debt") return "Debt";
  if (category === "Hybrid") return "Hybrid";
  return "Equity"; // Equity, ELSS, Gold, and anything unrecognized
}

// For each scheme code, look up the latest two distinct NAV dates from
// nav_history so we can compute a real day-over-day change. Falls back to
// {latestNav: null, prevNav: null} when history hasn't been synced yet.
async function getDayChangeMap(schemeCodes: number[]) {
  const map = new Map<number, { latestNav: number | null; prevNav: number | null }>();
  if (schemeCodes.length === 0) return map;

  const result = await pool.query(
    `WITH ranked AS (
       SELECT scheme_code, nav, nav_date,
              ROW_NUMBER() OVER (PARTITION BY scheme_code ORDER BY nav_date DESC) AS rn
       FROM nav_history
       WHERE scheme_code = ANY($1::int[])
     )
     SELECT scheme_code,
            MAX(CASE WHEN rn = 1 THEN nav END) AS latest_nav,
            MAX(CASE WHEN rn = 2 THEN nav END) AS prev_nav
     FROM ranked
     GROUP BY scheme_code`,
    [schemeCodes]
  );

  for (const row of result.rows) {
    map.set(row.scheme_code, {
      latestNav: row.latest_nav !== null ? Number(row.latest_nav) : null,
      prevNav: row.prev_nav !== null ? Number(row.prev_nav) : null,
    });
  }
  return map;
}

// ---------------------------------------------------------------------------
// GET /api/portfolio — holdings, totals & allocation breakdown for the
// logged-in user. Backs both the Home screen's portfolio card / holdings
// preview and the full Portfolio screen.
// ---------------------------------------------------------------------------
router.get("/", authenticateToken, async (req: AuthRequest, res) => {
  try {
    const holdingsResult = await pool.query(
      `SELECT up.id, up.scheme_code, up.units, up.avg_purchase_nav, up.invested_amount, up.created_at,
              f.name, f.category, f.nav AS current_nav, f.one_year_return
       FROM user_portfolio up
       JOIN funds f ON f.scheme_code = up.scheme_code
       WHERE up.user_id = $1
       ORDER BY (up.units * f.nav) DESC NULLS LAST`,
      [req.userId]
    );

    const rows = holdingsResult.rows;
    const schemeCodes = rows.map((r) => r.scheme_code);
    const dayChangeMap = await getDayChangeMap(schemeCodes);

    let totalValue = 0;
    let investedValue = 0;
    let todayChangeAmount = 0;
    let earliestDate: Date | null = null;

    const allocationTotals: Record<AssetClass, number> = { Equity: 0, Debt: 0, Hybrid: 0 };

    const holdings = rows.map((r) => {
      const units = Number(r.units);
      const currentNav = Number(r.current_nav ?? r.avg_purchase_nav ?? 0);
      const value = units * currentNav;
      const invested = Number(r.invested_amount ?? 0);
      const gainAmount = value - invested;
      const gainPercent = invested > 0 ? (gainAmount / invested) * 100 : 0;
      const assetClass = mapAssetClass(r.category);

      const dayInfo = dayChangeMap.get(r.scheme_code);
      let dayChangeAmt = 0;
      let dayChangePct = 0;
      if (dayInfo?.latestNav != null && dayInfo?.prevNav != null && dayInfo.prevNav > 0) {
        dayChangeAmt = units * (dayInfo.latestNav - dayInfo.prevNav);
        dayChangePct = ((dayInfo.latestNav - dayInfo.prevNav) / dayInfo.prevNav) * 100;
      } else if (r.one_year_return != null) {
        const annual = Number(r.one_year_return);
        dayChangePct = Math.max(-2.5, Math.min(2.5, annual / 250));
        dayChangeAmt = (value * dayChangePct) / 100;
      }

      todayChangeAmount += dayChangeAmt;
      totalValue += value;
      investedValue += invested;
      allocationTotals[assetClass] += value;

      const createdAt = new Date(r.created_at);
      if (!earliestDate || createdAt < earliestDate) earliestDate = createdAt;

      return {
        id: String(r.id),
        schemeCode: r.scheme_code,
        name: r.name,
        category: r.category || "Equity",
        assetClass,
        units: Number(units.toFixed(4)),
        avgPurchaseNav: Number(Number(r.avg_purchase_nav ?? 0).toFixed(4)),
        currentNav: Number(currentNav.toFixed(4)),
        value: Number(value.toFixed(2)),
        investedAmount: Number(invested.toFixed(2)),
        gainAmount: Number(gainAmount.toFixed(2)),
        gainPercent: Number(gainPercent.toFixed(2)),
        dayChangeAmount: Number(dayChangeAmt.toFixed(2)),
        dayChangePercent: Number(dayChangePct.toFixed(2)),
        // 1-year return %, used for the badge/color in the holdings list
        change: Number(r.one_year_return ?? 0),
      };
    });

    const gainValue = totalValue - investedValue;
    const gainPercent = investedValue > 0 ? (gainValue / investedValue) * 100 : 0;
    const previousTotalValue = totalValue - todayChangeAmount;
    const todayChangePercent = previousTotalValue > 0 ? (todayChangeAmount / previousTotalValue) * 100 : 0;

    // Simplified annualized return (not true multi-cash-flow XIRR — we don't
    // track individual transaction dates yet, only the first purchase).
    let xirr = gainPercent;
    if (earliestDate && investedValue > 0 && totalValue > 0) {
      const days = (Date.now() - (earliestDate as Date).getTime()) / (1000 * 60 * 60 * 24);
      const years = days / 365;
      if (years > 0.05) {
        xirr = (Math.pow(totalValue / investedValue, 1 / years) - 1) * 100;
      }
    }

    const allocations = (Object.keys(allocationTotals) as AssetClass[])
      .map((label) => ({
        label,
        value: Number(allocationTotals[label].toFixed(2)),
        percent: totalValue > 0 ? Number(((allocationTotals[label] / totalValue) * 100).toFixed(1)) : 0,
      }))
      .filter((a) => a.value > 0);

    const holdingsWithWeights = holdings.map((h) => ({
      ...h,
      portfolioWeightPercent: totalValue > 0 ? Number(((h.value / totalValue) * 100).toFixed(1)) : 0,
    }));

    res.json({
      totalValue: Number(totalValue.toFixed(2)),
      investedValue: Number(investedValue.toFixed(2)),
      gainValue: Number(gainValue.toFixed(2)),
      gainPercent: Number(gainPercent.toFixed(2)),
      todayChange: Number(todayChangeAmount.toFixed(2)),
      todayChangePercent: Number(todayChangePercent.toFixed(2)),
      xirr: Number(xirr.toFixed(2)),
      totalHoldingsCount: holdings.length,
      holdings: holdingsWithWeights,
      allocations,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch portfolio" });
  }
});

// ---------------------------------------------------------------------------
// POST /api/portfolio/invest — buy units of a fund at its current NAV.
// body: { schemeCode: number, amount: number, paymentMethod?: string }
// ---------------------------------------------------------------------------
router.post("/invest", authenticateToken, async (req: AuthRequest, res) => {
  const client = await pool.connect();
  try {
    const schemeCode = Number(req.body.schemeCode);
    const amount = Number(req.body.amount);
    const paymentMethod = String(req.body.paymentMethod || "UPI");

    if (!schemeCode || !amount || amount <= 0) {
      return res.status(400).json({ error: "schemeCode and a positive amount are required" });
    }

    const fundResult = await pool.query(
      `SELECT scheme_code, nav, name FROM funds WHERE scheme_code = $1`,
      [schemeCode]
    );
    if (fundResult.rows.length === 0) {
      return res.status(404).json({ error: "Fund not found" });
    }

    const fund = fundResult.rows[0];
    const nav = Number(fund.nav);
    if (!nav || nav <= 0) {
      return res.status(400).json({ error: "Fund NAV unavailable, try again later" });
    }

    const units = amount / nav;

    await client.query("BEGIN");

    const existing = await client.query(
      `SELECT id, units, invested_amount FROM user_portfolio
       WHERE user_id = $1 AND scheme_code = $2 FOR UPDATE`,
      [req.userId, schemeCode]
    );

    let holding;
    if (existing.rows.length > 0) {
      const prevUnits = Number(existing.rows[0].units);
      const prevInvested = Number(existing.rows[0].invested_amount);
      const newUnits = prevUnits + units;
      const newInvested = prevInvested + amount;
      const newAvgNav = newInvested / newUnits;

      const updated = await client.query(
        `UPDATE user_portfolio
         SET units = $1, invested_amount = $2, avg_purchase_nav = $3, updated_at = NOW()
         WHERE id = $4
         RETURNING id, units, invested_amount, avg_purchase_nav`,
        [newUnits, newInvested, newAvgNav, existing.rows[0].id]
      );
      holding = updated.rows[0];
    } else {
      const inserted = await client.query(
        `INSERT INTO user_portfolio (user_id, scheme_code, units, avg_purchase_nav, invested_amount)
         VALUES ($1, $2, $3, $4, $5)
         RETURNING id, units, invested_amount, avg_purchase_nav`,
        [req.userId, schemeCode, units, nav, amount]
      );
      holding = inserted.rows[0];
    }

    // Record into transactions history
    const txResult = await client.query(
      `INSERT INTO transactions
       (user_id, scheme_code, type, amount, units, nav, status, payment_method)
       VALUES ($1, $2, 'BUY', $3, $4, $5, 'COMPLETED', $6)
       RETURNING id, type, amount, units, nav, status, payment_method, created_at`,
      [req.userId, schemeCode, amount, units, nav, paymentMethod]
    );

    await client.query("COMMIT");

    res.status(201).json({
      message: `Successfully invested ₹${amount.toLocaleString("en-IN")} in ${fund.name}`,
      holding,
      transaction: txResult.rows[0],
    });
  } catch (err) {
    await client.query("ROLLBACK");
    console.error(err);
    res.status(500).json({ error: "Investment failed" });
  } finally {
    client.release();
  }
});

// ---------------------------------------------------------------------------
// POST /api/portfolio/redeem — sell units of a currently held fund.
// body: { schemeCode: number, units: number, payoutMethod?: string }
// ---------------------------------------------------------------------------
router.post("/redeem", authenticateToken, async (req: AuthRequest, res) => {
  const client = await pool.connect();
  try {
    const schemeCode = Number(req.body.schemeCode);
    const unitsToSell = Number(req.body.units);
    const payoutMethod = String(req.body.payoutMethod || "BANK_TRANSFER");

    if (!schemeCode || !unitsToSell || unitsToSell <= 0) {
      return res.status(400).json({ error: "schemeCode and a positive units are required" });
    }

    await client.query("BEGIN");

    const existing = await client.query(
      `SELECT id, units, invested_amount, avg_purchase_nav FROM user_portfolio
       WHERE user_id = $1 AND scheme_code = $2 FOR UPDATE`,
      [req.userId, schemeCode]
    );

    if (existing.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "You don't hold this fund" });
    }

    const row = existing.rows[0];
    const currentUnits = Number(row.units);

    if (unitsToSell > currentUnits + 0.0001) {
      await client.query("ROLLBACK");
      return res.status(400).json({ error: "Cannot redeem more units than you hold" });
    }

    const fundResult = await client.query(
      `SELECT scheme_code, nav, name FROM funds WHERE scheme_code = $1`,
      [schemeCode]
    );
    const fund = fundResult.rows[0];
    const currentNav = Number(fund?.nav || row.avg_purchase_nav || 0);
    const redeemAmount = Number((unitsToSell * currentNav).toFixed(2));

    const remainingUnits = Math.max(0, currentUnits - unitsToSell);
    const investedPerUnit = currentUnits > 0 ? Number(row.invested_amount) / currentUnits : 0;
    const remainingInvested = remainingUnits * investedPerUnit;

    if (remainingUnits <= 0.0001) {
      await client.query(`DELETE FROM user_portfolio WHERE id = $1`, [row.id]);
    } else {
      await client.query(
        `UPDATE user_portfolio SET units = $1, invested_amount = $2, updated_at = NOW() WHERE id = $3`,
        [remainingUnits, remainingInvested, row.id]
      );
    }

    // Record into transactions history
    const txResult = await client.query(
      `INSERT INTO transactions
       (user_id, scheme_code, type, amount, units, nav, status, payment_method)
       VALUES ($1, $2, 'REDEEM', $3, $4, $5, 'COMPLETED', $6)
       RETURNING id, type, amount, units, nav, status, payment_method, created_at`,
      [req.userId, schemeCode, redeemAmount, unitsToSell, currentNav, payoutMethod]
    );

    await client.query("COMMIT");

    res.json({
      message: `Redeemed ${unitsToSell.toFixed(4)} units (₹${redeemAmount.toLocaleString("en-IN")}) from ${fund?.name || "fund"}`,
      redeemAmount,
      remainingUnits: Number(remainingUnits.toFixed(4)),
      transaction: txResult.rows[0],
    });
  } catch (err) {
    await client.query("ROLLBACK");
    console.error(err);
    res.status(500).json({ error: "Redemption failed" });
  } finally {
    client.release();
  }
});

// ---------------------------------------------------------------------------
// POST /api/portfolio/switch — transfer units from one held fund to another fund.
// body: { sourceSchemeCode: number, targetSchemeCode: number, units: number }
// ---------------------------------------------------------------------------
router.post("/switch", authenticateToken, async (req: AuthRequest, res) => {
  const client = await pool.connect();
  try {
    const sourceSchemeCode = Number(req.body.sourceSchemeCode);
    const targetSchemeCode = Number(req.body.targetSchemeCode);
    const unitsToSwitch = Number(req.body.units);

    if (!sourceSchemeCode || !targetSchemeCode || !unitsToSwitch || unitsToSwitch <= 0) {
      return res.status(400).json({
        error: "sourceSchemeCode, targetSchemeCode, and positive units are required",
      });
    }

    if (sourceSchemeCode === targetSchemeCode) {
      return res.status(400).json({ error: "Source and target funds must be different" });
    }

    await client.query("BEGIN");

    // 1. Verify user holds source scheme
    const sourceHoldingResult = await client.query(
      `SELECT id, units, invested_amount, avg_purchase_nav FROM user_portfolio
       WHERE user_id = $1 AND scheme_code = $2 FOR UPDATE`,
      [req.userId, sourceSchemeCode]
    );

    if (sourceHoldingResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "You don't hold the source fund" });
    }

    const sourceHolding = sourceHoldingResult.rows[0];
    const sourceCurrentUnits = Number(sourceHolding.units);

    if (unitsToSwitch > sourceCurrentUnits + 0.0001) {
      await client.query("ROLLBACK");
      return res.status(400).json({ error: "Cannot switch more units than currently held" });
    }

    // 2. Fetch source & target fund info and live NAVs
    const fundsResult = await client.query(
      `SELECT scheme_code, nav, name FROM funds WHERE scheme_code IN ($1, $2)`,
      [sourceSchemeCode, targetSchemeCode]
    );

    const sourceFund = fundsResult.rows.find((f) => f.scheme_code === sourceSchemeCode);
    const targetFund = fundsResult.rows.find((f) => f.scheme_code === targetSchemeCode);

    if (!sourceFund) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "Source fund not found in database" });
    }
    if (!targetFund) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "Target fund not found in database" });
    }

    const sourceNav = Number(sourceFund.nav || sourceHolding.avg_purchase_nav || 0);
    const targetNav = Number(targetFund.nav || 0);

    if (sourceNav <= 0 || targetNav <= 0) {
      await client.query("ROLLBACK");
      return res.status(400).json({ error: "NAV unavailable for one of the funds" });
    }

    // 3. Compute switch amount & target units
    const switchAmount = Number((unitsToSwitch * sourceNav).toFixed(2));
    const targetUnits = switchAmount / targetNav;

    // 4. Deduct units from source fund
    const remainingSourceUnits = Math.max(0, sourceCurrentUnits - unitsToSwitch);
    const investedPerUnit = sourceCurrentUnits > 0 ? Number(sourceHolding.invested_amount) / sourceCurrentUnits : 0;
    const remainingInvested = remainingSourceUnits * investedPerUnit;

    if (remainingSourceUnits <= 0.0001) {
      await client.query(`DELETE FROM user_portfolio WHERE id = $1`, [sourceHolding.id]);
    } else {
      await client.query(
        `UPDATE user_portfolio SET units = $1, invested_amount = $2, updated_at = NOW() WHERE id = $3`,
        [remainingSourceUnits, remainingInvested, sourceHolding.id]
      );
    }

    // 5. Add units to target fund in user_portfolio
    const targetHoldingResult = await client.query(
      `SELECT id, units, invested_amount FROM user_portfolio
       WHERE user_id = $1 AND scheme_code = $2 FOR UPDATE`,
      [req.userId, targetSchemeCode]
    );

    if (targetHoldingResult.rows.length > 0) {
      const prevUnits = Number(targetHoldingResult.rows[0].units);
      const prevInvested = Number(targetHoldingResult.rows[0].invested_amount);
      const newUnits = prevUnits + targetUnits;
      const newInvested = prevInvested + switchAmount;
      const newAvgNav = newInvested / newUnits;

      await client.query(
        `UPDATE user_portfolio
         SET units = $1, invested_amount = $2, avg_purchase_nav = $3, updated_at = NOW()
         WHERE id = $4`,
        [newUnits, newInvested, newAvgNav, targetHoldingResult.rows[0].id]
      );
    } else {
      await client.query(
        `INSERT INTO user_portfolio (user_id, scheme_code, units, avg_purchase_nav, invested_amount)
         VALUES ($1, $2, $3, $4, $5)`,
        [req.userId, targetSchemeCode, targetUnits, targetNav, switchAmount]
      );
    }

    // 6. Record transaction history
    const txResult = await client.query(
      `INSERT INTO transactions
       (user_id, scheme_code, type, amount, units, nav, status, payment_method, target_scheme_code, target_units, target_nav)
       VALUES ($1, $2, 'SWITCH', $3, $4, $5, 'COMPLETED', 'PORTFOLIO_SWITCH', $6, $7, $8)
       RETURNING id, type, amount, units, nav, status, payment_method, target_scheme_code, target_units, target_nav, created_at`,
      [req.userId, sourceSchemeCode, switchAmount, unitsToSwitch, sourceNav, targetSchemeCode, targetUnits, targetNav]
    );

    await client.query("COMMIT");

    res.json({
      message: `Successfully switched ₹${switchAmount.toLocaleString("en-IN")} from ${sourceFund.name} to ${targetFund.name}`,
      switchAmount,
      sourceUnits: Number(unitsToSwitch.toFixed(4)),
      targetUnits: Number(targetUnits.toFixed(4)),
      transaction: txResult.rows[0],
    });
  } catch (err) {
    await client.query("ROLLBACK");
    console.error(err);
    res.status(500).json({ error: "Fund switch failed" });
  } finally {
    client.release();
  }
});

// ---------------------------------------------------------------------------
// GET /api/portfolio/transactions — full transaction history for logged-in user.
// optional query: ?type=BUY|REDEEM|SWITCH|ALL&limit=50&page=1
// ---------------------------------------------------------------------------
router.get("/transactions", authenticateToken, async (req: AuthRequest, res) => {
  try {
    const { type } = req.query;
    const page = Math.max(1, Number(req.query.page) || 1);
    const limit = Math.min(100, Number(req.query.limit) || 50);
    const offset = (page - 1) * limit;

    const conditions: string[] = ["t.user_id = $1"];
    const values: any[] = [req.userId];

    if (type && type !== "ALL") {
      values.push(String(type).toUpperCase());
      conditions.push(`t.type = $${values.length}`);
    }

    const where = `WHERE ${conditions.join(" AND ")}`;

    const countResult = await pool.query(
      `SELECT COUNT(*) FROM transactions t ${where}`,
      values
    );
    const total = Number(countResult.rows[0].count);

    values.push(limit, offset);
    const txResult = await pool.query(
      `SELECT t.id, t.scheme_code AS "schemeCode", t.type, t.amount, t.units, t.nav,
              t.status, t.payment_method AS "paymentMethod",
              t.target_scheme_code AS "targetSchemeCode",
              t.target_units AS "targetUnits",
              t.target_nav AS "targetNav",
              t.created_at AS "createdAt",
              f.name AS "fundName", f.category AS "fundCategory",
              tf.name AS "targetFundName", tf.category AS "targetFundCategory"
       FROM transactions t
       JOIN funds f ON f.scheme_code = t.scheme_code
       LEFT JOIN funds tf ON tf.scheme_code = t.target_scheme_code
       ${where}
       ORDER BY t.created_at DESC
       LIMIT $${values.length - 1} OFFSET $${values.length}`,
      values
    );

    const formatted = txResult.rows.map((r) => ({
      id: String(r.id),
      schemeCode: r.schemeCode,
      fundName: r.fundName,
      fundCategory: r.fundCategory,
      type: r.type,
      amount: Number(Number(r.amount).toFixed(2)),
      units: Number(Number(r.units).toFixed(4)),
      nav: Number(Number(r.nav).toFixed(4)),
      status: r.status,
      paymentMethod: r.paymentMethod,
      targetSchemeCode: r.targetSchemeCode,
      targetFundName: r.targetFundName,
      targetFundCategory: r.targetFundCategory,
      targetUnits: r.targetUnits ? Number(Number(r.targetUnits).toFixed(4)) : null,
      targetNav: r.targetNav ? Number(Number(r.targetNav).toFixed(4)) : null,
      createdAt: r.createdAt,
    }));

    res.json({
      transactions: formatted,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch transactions" });
  }
});

export default router;

