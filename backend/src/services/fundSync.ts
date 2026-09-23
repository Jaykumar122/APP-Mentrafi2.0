// src/services/fundSync.ts
import pool from "../config/db";

const MFAPI_BASE = "https://api.mfapi.in/mf";

let isSyncing = false;
export function isSyncRunning() {
  return isSyncing;
}

function classify(name: string): { category: string; subcategory: string } {
  const n = name.toLowerCase();
  if (n.includes("elss") || n.includes("tax saver")) return { category: "ELSS", subcategory: "Tax Saving" };
  if (n.includes("gold")) return { category: "Gold", subcategory: "Gold" };
  if (n.includes("debt") || n.includes("bond") || n.includes("liquid") || n.includes("duration") || n.includes("gilt"))
    return { category: "Debt", subcategory: "Debt Fund" };
  if (n.includes("hybrid") || n.includes("balanced") || n.includes("advantage"))
    return { category: "Hybrid", subcategory: "Balanced" };
  if (n.includes("small cap")) return { category: "Equity", subcategory: "Small Cap" };
  if (n.includes("mid cap")) return { category: "Equity", subcategory: "Mid Cap" };
  if (n.includes("large cap") || n.includes("bluechip")) return { category: "Equity", subcategory: "Large Cap" };
  if (n.includes("flexi cap") || n.includes("multi cap")) return { category: "Equity", subcategory: "Flexi Cap" };
  return { category: "Equity", subcategory: "Diversified" };
}

function detectPlanType(name: string): { planType: string; optionType: string } {
  const n = name.toLowerCase();
  return {
    planType: n.includes("direct") ? "Direct" : "Regular",
    optionType: n.includes("idcw") || n.includes("dividend") ? "IDCW" : "Growth",
  };
}

function toPostgresDate(mfapiDate: string): string {
  const [d, m, y] = mfapiDate.split("-");
  return `${y}-${m}-${d}`;
}

type ReturnPeriods = {
  return1m: number | null;
  return3m: number | null;
  return6m: number | null;
  return1y: number | null;
  return3y: number | null;
  return5y: number | null;
  return10y: number | null;
};

// ---------------------------------------------------------------------------
// findNavAtDate — finds the NAV entry closest to `monthsAgo` months back.
// Returns null (rather than a bogus fallback) when the fund doesn't have
// enough real history for that period, or when the closest match found is
// too far from the target date to be trustworthy. This prevents funds
// younger than 3/5/10 years from silently reporting since-inception
// performance mislabeled as a 3Y/5Y/10Y return.
// ---------------------------------------------------------------------------
function findNavAtDate(
  navHistory: { date: string; nav: string }[],
  monthsAgo: number
): number | null {
  if (!navHistory.length) return null;
  const target = new Date();
  target.setMonth(target.getMonth() - monthsAgo);

  // MFapi returns history newest-first, so the oldest entry is the last one.
  const oldestEntry = navHistory[navHistory.length - 1];
  const [od, om, oy] = oldestEntry.date.split("-").map(Number);
  const oldestDate = new Date(oy, om - 1, od);

  // Allow a small grace window (funds don't trade every calendar day),
  // but if the target predates the fund's earliest NAV by more than that,
  // there simply isn't a valid data point for this period.
  const graceMs = 20 * 24 * 60 * 60 * 1000;
  if (target.getTime() < oldestDate.getTime() - graceMs) {
    return null;
  }

  let closest = navHistory[navHistory.length - 1];
  let minDiff = Infinity;
  for (const entry of navHistory) {
    const [d, m, y] = entry.date.split("-").map(Number);
    const entryDate = new Date(y, m - 1, d);
    const diff = Math.abs(entryDate.getTime() - target.getTime());
    if (diff < minDiff) {
      minDiff = diff;
      closest = entry;
    }
  }

  // Reject matches that are still too far from the target (>45 days) —
  // a sparse/stale match isn't a reliable basis for a return calculation.
  const [cd, cm, cy] = closest.date.split("-").map(Number);
  const closestDate = new Date(cy, cm - 1, cd);
  const closestDiffMs = Math.abs(closestDate.getTime() - target.getTime());
  if (closestDiffMs > 45 * 24 * 60 * 60 * 1000) {
    return null;
  }

  return parseFloat(closest.nav);
}

function computeReturn(latest: number, past: number | null): number | null {
  if (!past) return null;
  return Number((((latest - past) / past) * 100).toFixed(2));
}

// Annualized (CAGR) return — used for multi-year periods (3Y/5Y/10Y) so
// numbers are comparable across funds and don't look inflated the way
// raw cumulative returns do over longer windows.
function computeCAGR(latest: number, past: number | null, years: number): number | null {
  if (!past || past <= 0) return null;
  const cagr = (Math.pow(latest / past, 1 / years) - 1) * 100;
  return Number(cagr.toFixed(2));
}

function computeAllReturns(navHistory: { date: string; nav: string }[]): ReturnPeriods {
  if (!navHistory.length) {
    return {
      return1m: null, return3m: null, return6m: null,
      return1y: null, return3y: null, return5y: null, return10y: null,
    };
  }
  const latest = parseFloat(navHistory[0].nav);

  return {
    return1m: computeReturn(latest, findNavAtDate(navHistory, 1)),
    return3m: computeReturn(latest, findNavAtDate(navHistory, 3)),
    return6m: computeReturn(latest, findNavAtDate(navHistory, 6)),
    return1y: computeReturn(latest, findNavAtDate(navHistory, 12)),
    return3y: computeCAGR(latest, findNavAtDate(navHistory, 36), 3),
    return5y: computeCAGR(latest, findNavAtDate(navHistory, 60), 5),
    return10y: computeCAGR(latest, findNavAtDate(navHistory, 120), 10),
  };
}

// Earliest entry in NAV history is the fund's launch/inception date proxy.
// MFapi returns history newest-first, so the earliest is the last element.
function extractLaunchDate(navHistory: { date: string; nav: string }[]): string | null {
  if (!navHistory.length) return null;
  const earliest = navHistory[navHistory.length - 1];
  return toPostgresDate(earliest.date);
}

export async function syncScheme(schemeCode: number) {
  const res = await fetch(`${MFAPI_BASE}/${schemeCode}`);
  if (!res.ok) throw new Error(`MFapi fetch failed for ${schemeCode}`);
  const json = await res.json();

  const name: string = json.meta?.scheme_name ?? "Unknown Fund";
  const fundHouse: string | null = json.meta?.fund_house ?? null;
  const navHistory = json.data ?? [];
  const latestNav = navHistory[0] ? parseFloat(navHistory[0].nav) : null;
  const latestDate = navHistory[0]?.date ? toPostgresDate(navHistory[0].date) : null;
  const launchDate = extractLaunchDate(navHistory);
  const { category, subcategory } = classify(name);
  const { planType, optionType } = detectPlanType(name);
  const returns = computeAllReturns(navHistory);

  await pool.query(
    `INSERT INTO funds (scheme_code, name, category, subcategory, nav, nav_date, one_year_return, plan_type, option_type, fund_house, launch_date, updated_at)
     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
     ON CONFLICT (scheme_code) DO UPDATE SET
       name = EXCLUDED.name, category = EXCLUDED.category, subcategory = EXCLUDED.subcategory,
       nav = EXCLUDED.nav, nav_date = EXCLUDED.nav_date, one_year_return = EXCLUDED.one_year_return,
       plan_type = EXCLUDED.plan_type, option_type = EXCLUDED.option_type,
       fund_house = EXCLUDED.fund_house, launch_date = EXCLUDED.launch_date,
       updated_at = NOW()`,
    [schemeCode, name, category, subcategory, latestNav, latestDate, returns.return1y, planType, optionType, fundHouse, launchDate]
  );

  await pool.query(
    `INSERT INTO fund_performance (scheme_code, return_1m, return_3m, return_6m, return_1y, return_3y, return_5y, return_10y, updated_at)
     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
     ON CONFLICT (scheme_code) DO UPDATE SET
       return_1m = EXCLUDED.return_1m, return_3m = EXCLUDED.return_3m, return_6m = EXCLUDED.return_6m,
       return_1y = EXCLUDED.return_1y, return_3y = EXCLUDED.return_3y, return_5y = EXCLUDED.return_5y,
       return_10y = EXCLUDED.return_10y,
       updated_at = NOW()`,
    [schemeCode, returns.return1m, returns.return3m, returns.return6m, returns.return1y, returns.return3y, returns.return5y, returns.return10y]
  );
}

async function fetchLatestNav(schemeCode: number): Promise<{
  schemeCode: number; name: string; nav: number; navDate: string; planType: string; optionType: string;
} | null> {
  const res = await fetch(`${MFAPI_BASE}/${schemeCode}/latest`);
  if (!res.ok) return null;
  const json = await res.json();
  const entry = json.data?.[0];
  if (!entry) return null;

  const name = json.meta?.scheme_name ?? "Unknown Fund";
  const { planType, optionType } = detectPlanType(name);

  return {
    schemeCode,
    name,
    nav: parseFloat(entry.nav),
    navDate: toPostgresDate(entry.date),
    planType,
    optionType,
  };
}

export const STARTER_SCHEME_CODES = [
  118955, 125497, 120586, 120510, 120847, 118663,
];

export async function seedStarterFunds() {
  for (const code of STARTER_SCHEME_CODES) {
    try {
      await syncScheme(code);
      console.log(`✅ Synced scheme ${code}`);
    } catch (err) {
      console.error(`Failed to sync scheme ${code}:`, err);
    }
    await new Promise((resolve) => setTimeout(resolve, 800));
  }
}

// Fetches the master scheme list — no filtering, includes all plan/option
// variants (Direct/Regular, Growth/IDCW). Deduped by schemeCode only.
export async function fetchAllDirectGrowthSchemes(): Promise<{ schemeCode: number; schemeName: string }[]> {
  const res = await fetch(MFAPI_BASE);
  if (!res.ok) throw new Error("Failed to fetch scheme master list");
  const all: { schemeCode: number; schemeName: string }[] = await res.json();

  const seen = new Set<number>();
  return all.filter((s) => {
    if (seen.has(s.schemeCode)) return false;
    seen.add(s.schemeCode);
    return true;
  });
}

export async function syncAllFundsFast() {
  if (isSyncing) {
    console.log("⚠️ Sync already in progress, skipping duplicate trigger.");
    return;
  }
  isSyncing = true;
  const startTime = Date.now();

  try {
    const schemes = await fetchAllDirectGrowthSchemes();
    console.log(`Found ${schemes.length} schemes to sync (fast mode)`);

    const BATCH_SIZE = 20;
    let done = 0;
    const failedSchemes: typeof schemes = [];

    for (let i = 0; i < schemes.length; i += BATCH_SIZE) {
      const batch = schemes.slice(i, i + BATCH_SIZE);
      const results = await Promise.allSettled(batch.map((s) => fetchLatestNav(s.schemeCode)));

      const rawRows: NonNullable<Awaited<ReturnType<typeof fetchLatestNav>>>[] = [];
      results.forEach((r, idx) => {
        if (r.status === "fulfilled" && r.value !== null) {
          rawRows.push(r.value);
        } else {
          failedSchemes.push(batch[idx]);
        }
      });

      const seen = new Set<number>();
      const validRows = rawRows.filter((row) => {
        if (seen.has(row.schemeCode)) return false;
        seen.add(row.schemeCode);
        return true;
      });

      if (validRows.length > 0) {
        const valuePlaceholders = validRows
          .map((_, idx) => {
            const base = idx * 7;
            return `($${base + 1}, $${base + 2}, $${base + 3}, $${base + 4}, $${base + 5}, $${base + 6}, $${base + 7}, NOW())`;
          })
          .join(", ");

        const queryValues = validRows.flatMap((row) => [
          row.schemeCode,
          row.name,
          row.nav,
          row.navDate,
          classify(row.name).category,
          row.planType,
          row.optionType,
        ]);

        await pool.query(
          `INSERT INTO funds (scheme_code, name, nav, nav_date, category, plan_type, option_type, updated_at)
           VALUES ${valuePlaceholders}
           ON CONFLICT (scheme_code) DO UPDATE SET
             name = EXCLUDED.name,
             nav = EXCLUDED.nav,
             nav_date = EXCLUDED.nav_date,
             plan_type = EXCLUDED.plan_type,
             option_type = EXCLUDED.option_type,
             updated_at = NOW()`,
          queryValues
        );
      }

      done += batch.length;
      if (done % 200 === 0 || done === schemes.length) {
        console.log(`Progress: ${done}/${schemes.length} (${failedSchemes.length} failed so far)`);
      }
    }

    let finalFailed = failedSchemes.length;
    if (failedSchemes.length > 0) {
      console.log(`Retrying ${failedSchemes.length} failed schemes...`);
      const RETRY_BATCH = 5;
      const stillFailed: typeof schemes = [];

      for (let i = 0; i < failedSchemes.length; i += RETRY_BATCH) {
        const batch = failedSchemes.slice(i, i + RETRY_BATCH);
        const results = await Promise.allSettled(batch.map((s) => fetchLatestNav(s.schemeCode)));
        const rows = results
          .map((r) => (r.status === "fulfilled" ? r.value : null))
          .filter((r): r is NonNullable<typeof r> => r !== null);

        for (const row of rows) {
          await pool.query(
            `INSERT INTO funds (scheme_code, name, nav, nav_date, category, plan_type, option_type, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
             ON CONFLICT (scheme_code) DO UPDATE SET
               name = EXCLUDED.name, nav = EXCLUDED.nav, nav_date = EXCLUDED.nav_date,
               plan_type = EXCLUDED.plan_type, option_type = EXCLUDED.option_type, updated_at = NOW()`,
            [row.schemeCode, row.name, row.nav, row.navDate, classify(row.name).category, row.planType, row.optionType]
          );
        }

        results.forEach((r, idx) => {
          if (r.status !== "fulfilled" || r.value === null) stillFailed.push(batch[idx]);
        });

        await new Promise((r) => setTimeout(r, 500));
      }

      finalFailed = stillFailed.length;
      console.log(`Retry recovered ${failedSchemes.length - finalFailed} schemes; ${finalFailed} still failed.`);
    }

    const seconds = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`✅ Fast sync complete in ${seconds}s. ${schemes.length - finalFailed} succeeded, ${finalFailed} failed.`);
  } finally {
    isSyncing = false;
  }
}

// ---------------------------------------------------------------------------
// SLOW full sync: fetches complete NAV history per fund, needed to (re)compute
// multi-period returns (now incl. 10Y) and derive fund_house/launch_date.
// BATCH_SIZE 8 (lower than fast sync since payloads are heavier) + a gentler
// retry pass for anything that failed the first time.
// ---------------------------------------------------------------------------
export async function syncAllFunds() {
  if (isSyncing) {
    console.log("⚠️ Sync already in progress, skipping duplicate trigger.");
    return;
  }
  isSyncing = true;
  const startTime = Date.now();

  try {
    const schemes = await fetchAllDirectGrowthSchemes();
    console.log(`Found ${schemes.length} schemes to sync (full mode)`);

    const BATCH_SIZE = 8;
    let done = 0;
    const failedSchemes: typeof schemes = [];

    for (let i = 0; i < schemes.length; i += BATCH_SIZE) {
      const batch = schemes.slice(i, i + BATCH_SIZE);
      const results = await Promise.allSettled(batch.map((s) => syncScheme(s.schemeCode)));
      results.forEach((r, idx) => {
        if (r.status === "rejected") failedSchemes.push(batch[idx]);
      });
      done += batch.length;
      if (done % 200 === 0 || done === schemes.length) {
        console.log(`Progress: ${done}/${schemes.length} (${failedSchemes.length} failed so far)`);
      }
    }

    let finalFailed = failedSchemes.length;
    if (failedSchemes.length > 0) {
      console.log(`Retrying ${failedSchemes.length} failed schemes...`);
      const RETRY_BATCH = 3;
      const stillFailed: typeof schemes = [];

      for (let i = 0; i < failedSchemes.length; i += RETRY_BATCH) {
        const batch = failedSchemes.slice(i, i + RETRY_BATCH);
        const results = await Promise.allSettled(batch.map((s) => syncScheme(s.schemeCode)));
        results.forEach((r, idx) => {
          if (r.status === "rejected") stillFailed.push(batch[idx]);
        });
        await new Promise((r) => setTimeout(r, 500));
      }

      finalFailed = stillFailed.length;
      console.log(`Retry recovered ${failedSchemes.length - finalFailed} schemes; ${finalFailed} still failed.`);
    }

    const seconds = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`✅ Full sync complete in ${seconds}s. ${schemes.length - finalFailed} succeeded, ${finalFailed} failed.`);
  } finally {
    isSyncing = false;
  }
}