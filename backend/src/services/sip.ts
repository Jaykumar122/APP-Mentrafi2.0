import pool from "../config/db";

export type SIPStatus = "active" | "paused" | "completed" | "cancelled";

export interface SIPCard {
  id: string;
  schemeCode: number;
  name: string;
  type: "Equity" | "Debt";
  status: SIPStatus;
  completed: number;
  total: number;
  monthlyAmount: number;
  installmentDay: number;
  nextDate: string; // ISO date, or "Paused" when the SIP is paused
  totalInvested: number;
  currentValue: number;
  returns: number;
}

function clampInstallmentDay(day: number): number {
  if (!Number.isFinite(day)) return 1;
  return Math.min(28, Math.max(1, Math.round(day)));
}

// Next occurrence of `installmentDay` on/after `fromDate`.
function computeNextInstallmentDate(fromDate: Date, installmentDay: number): Date {
  const next = new Date(fromDate);
  next.setHours(0, 0, 0, 0);
  next.setDate(installmentDay);
  if (next < fromDate) {
    next.setMonth(next.getMonth() + 1);
    next.setDate(installmentDay);
  }
  return next;
}

// One month after `date`, landing on `installmentDay` — used to advance a
// SIP to its next due date once an installment has been executed.
function addOneMonth(date: Date, installmentDay: number): Date {
  const next = new Date(date);
  next.setMonth(next.getMonth() + 1);
  next.setDate(installmentDay);
  return next;
}

function toDateString(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function mapRow(row: any): SIPCard {
  const totalInvested = Number(row.total_invested);
  const totalUnits = Number(row.total_units);
  const currentNav = Number(row.current_nav ?? 0);
  const currentValue = totalUnits * currentNav;
  const returns = totalInvested > 0 ? ((currentValue - totalInvested) / totalInvested) * 100 : 0;

  return {
    id: String(row.id),
    schemeCode: row.scheme_code,
    name: row.name,
    type: row.category === "Debt" ? "Debt" : "Equity",
    status: row.status,
    completed: row.completed_installments,
    total: row.total_installments,
    monthlyAmount: Number(row.monthly_amount),
    installmentDay: row.installment_day,
    nextDate: row.status === "paused" ? "Paused" : row.next_installment_date,
    totalInvested: Number(totalInvested.toFixed(2)),
    currentValue: Number(currentValue.toFixed(2)),
    returns: Number(returns.toFixed(2)),
  };
}

// ---------------------------------------------------------------------------
// getSIPsForUser — joins fund info + aggregates real installment history
// (from sip_installments) so totalInvested/currentValue/returns reflect
// actual purchases rather than an estimate.
// ---------------------------------------------------------------------------
export async function getSIPsForUser(userId: number): Promise<SIPCard[]> {
  const result = await pool.query(
    `SELECT s.id, s.scheme_code, s.monthly_amount, s.installment_day, s.total_installments,
            s.completed_installments, s.status, s.next_installment_date, s.start_date,
            f.name, f.category, f.nav AS current_nav,
            COALESCE(SUM(si.amount), 0) AS total_invested,
            COALESCE(SUM(si.units), 0) AS total_units
     FROM sips s
     JOIN funds f ON f.scheme_code = s.scheme_code
     LEFT JOIN sip_installments si ON si.sip_id = s.id
     WHERE s.user_id = $1 AND s.status != 'cancelled'
     GROUP BY s.id, f.name, f.category, f.nav
     ORDER BY s.created_at DESC`,
    [userId]
  );

  return result.rows.map(mapRow);
}

// ---------------------------------------------------------------------------
// createSIP — sets up a new recurring mandate; next_installment_date is the
// next occurrence of installmentDay on/after today.
// ---------------------------------------------------------------------------
export async function createSIP(
  userId: number,
  schemeCode: number,
  monthlyAmount: number,
  installmentDay: number,
  totalInstallments: number
): Promise<SIPCard> {
  const day = clampInstallmentDay(installmentDay);
  const duration = Math.max(1, Math.round(totalInstallments) || 60);
  const nextDate = computeNextInstallmentDate(new Date(), day);

  const result = await pool.query(
    `INSERT INTO sips (user_id, scheme_code, monthly_amount, installment_day, total_installments, next_installment_date)
     VALUES ($1, $2, $3, $4, $5, $6)
     RETURNING id`,
    [userId, schemeCode, monthlyAmount, day, duration, toDateString(nextDate)]
  );

  const [sip] = await getSIPById(userId, result.rows[0].id);
  return sip;
}

async function getSIPById(userId: number, sipId: number): Promise<SIPCard[]> {
  const result = await pool.query(
    `SELECT s.id, s.scheme_code, s.monthly_amount, s.installment_day, s.total_installments,
            s.completed_installments, s.status, s.next_installment_date, s.start_date,
            f.name, f.category, f.nav AS current_nav,
            COALESCE(SUM(si.amount), 0) AS total_invested,
            COALESCE(SUM(si.units), 0) AS total_units
     FROM sips s
     JOIN funds f ON f.scheme_code = s.scheme_code
     LEFT JOIN sip_installments si ON si.sip_id = s.id
     WHERE s.id = $1 AND s.user_id = $2
     GROUP BY s.id, f.name, f.category, f.nav`,
    [sipId, userId]
  );
  return result.rows.map(mapRow);
}

export async function pauseSIP(userId: number, sipId: number): Promise<SIPCard | null> {
  const result = await pool.query(
    `UPDATE sips SET status = 'paused', updated_at = NOW()
     WHERE id = $1 AND user_id = $2 AND status = 'active'
     RETURNING id`,
    [sipId, userId]
  );
  if (result.rows.length === 0) return null;
  const [sip] = await getSIPById(userId, sipId);
  return sip ?? null;
}

export async function resumeSIP(userId: number, sipId: number): Promise<SIPCard | null> {
  const existing = await pool.query(
    `SELECT installment_day FROM sips WHERE id = $1 AND user_id = $2 AND status = 'paused'`,
    [sipId, userId]
  );
  if (existing.rows.length === 0) return null;

  const day = existing.rows[0].installment_day;
  const nextDate = computeNextInstallmentDate(new Date(), day);

  await pool.query(
    `UPDATE sips SET status = 'active', next_installment_date = $1, updated_at = NOW()
     WHERE id = $2 AND user_id = $3`,
    [toDateString(nextDate), sipId, userId]
  );

  const [sip] = await getSIPById(userId, sipId);
  return sip ?? null;
}

export async function updateSIP(
  userId: number,
  sipId: number,
  updates: { monthlyAmount?: number; installmentDay?: number; totalInstallments?: number }
): Promise<SIPCard | null> {
  const result = await pool.query(
    `UPDATE sips SET
       monthly_amount = COALESCE($1, monthly_amount),
       installment_day = COALESCE($2, installment_day),
       total_installments = COALESCE($3, total_installments),
       updated_at = NOW()
     WHERE id = $4 AND user_id = $5
     RETURNING id`,
    [
      updates.monthlyAmount ?? null,
      updates.installmentDay != null ? clampInstallmentDay(updates.installmentDay) : null,
      updates.totalInstallments != null ? Math.max(1, Math.round(updates.totalInstallments)) : null,
      sipId,
      userId,
    ]
  );
  if (result.rows.length === 0) return null;
  const [sip] = await getSIPById(userId, sipId);
  return sip ?? null;
}

export async function cancelSIP(userId: number, sipId: number): Promise<boolean> {
  const result = await pool.query(
    `UPDATE sips SET status = 'cancelled', updated_at = NOW()
     WHERE id = $1 AND user_id = $2
     RETURNING id`,
    [sipId, userId]
  );
  return result.rows.length > 0;
}

// ---------------------------------------------------------------------------
// processDueSIPs — executes every active SIP whose next_installment_date has
// arrived: buys units at the fund's current NAV, logs the installment, feeds
// the purchase into user_portfolio (so it shows up on the Portfolio screen
// too), and advances the mandate to its next due date or marks it completed.
// Meant to be run by a daily cron job, same pattern as fundSync's NAV syncs.
// ---------------------------------------------------------------------------
export async function processDueSIPs(): Promise<{ processed: number; failed: number }> {
  const dueResult = await pool.query(
    `SELECT s.id, s.user_id, s.scheme_code, s.monthly_amount, s.installment_day,
            s.total_installments, s.completed_installments, f.nav
     FROM sips s
     JOIN funds f ON f.scheme_code = s.scheme_code
     WHERE s.status = 'active' AND s.next_installment_date <= CURRENT_DATE`
  );

  let processed = 0;
  let failed = 0;

  for (const sip of dueResult.rows) {
    const client = await pool.connect();
    try {
      const nav = Number(sip.nav);
      if (!nav || nav <= 0) {
        failed++;
        continue;
      }

      const amount = Number(sip.monthly_amount);
      const units = amount / nav;
      const installmentDate = toDateString(new Date());

      await client.query("BEGIN");

      await client.query(
        `INSERT INTO sip_installments (sip_id, amount, nav, units, installment_date)
         VALUES ($1, $2, $3, $4, $5)`,
        [sip.id, amount, nav, units, installmentDate]
      );

      const existingHolding = await client.query(
        `SELECT id, units, invested_amount FROM user_portfolio
         WHERE user_id = $1 AND scheme_code = $2 FOR UPDATE`,
        [sip.user_id, sip.scheme_code]
      );

      if (existingHolding.rows.length > 0) {
        const prevUnits = Number(existingHolding.rows[0].units);
        const prevInvested = Number(existingHolding.rows[0].invested_amount);
        const newUnits = prevUnits + units;
        const newInvested = prevInvested + amount;
        await client.query(
          `UPDATE user_portfolio
           SET units = $1, invested_amount = $2, avg_purchase_nav = $3, updated_at = NOW()
           WHERE id = $4`,
          [newUnits, newInvested, newInvested / newUnits, existingHolding.rows[0].id]
        );
      } else {
        await client.query(
          `INSERT INTO user_portfolio (user_id, scheme_code, units, avg_purchase_nav, invested_amount)
           VALUES ($1, $2, $3, $4, $5)`,
          [sip.user_id, sip.scheme_code, units, nav, amount]
        );
      }

      const newCompleted = sip.completed_installments + 1;
      const isDone = newCompleted >= sip.total_installments;
      const nextDate = addOneMonth(new Date(), sip.installment_day);

      await client.query(
        `UPDATE sips SET
           completed_installments = $1,
           status = $2,
           next_installment_date = $3,
           updated_at = NOW()
         WHERE id = $4`,
        [newCompleted, isDone ? "completed" : "active", toDateString(nextDate), sip.id]
      );

      await client.query("COMMIT");
      processed++;
    } catch (err) {
      await client.query("ROLLBACK");
      console.error(`SIP installment failed for sip #${sip.id}:`, err);
      failed++;
    } finally {
      client.release();
    }
  }

  return { processed, failed };
}
