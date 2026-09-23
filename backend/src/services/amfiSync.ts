// src/services/amfiSync.ts
import pool from "../config/db";

const AMFI_NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt";

let isAmfiSyncing = false;
export function isAmfiSyncRunning(): boolean {
  return isAmfiSyncing;
}

const MONTH_MAP: Record<string, string> = {
  jan: "01", feb: "02", mar: "03", apr: "04", may: "05", jun: "06",
  jul: "07", aug: "08", sep: "09", oct: "10", nov: "11", dec: "12",
};

function parseAmfiDate(dateStr: string): string {
  const parts = dateStr.trim().split("-");
  if (parts.length === 3) {
    const day = parts[0].padStart(2, "0");
    const mon = MONTH_MAP[parts[1].toLowerCase()] || "01";
    const year = parts[2];
    return `${year}-${mon}-${day}`;
  }
  return new Date().toISOString().split("T")[0];
}

function parseAmfiCategory(headerStr: string | null): { category: string; subcategory: string } {
  if (!headerStr) return { category: "Equity", subcategory: "Diversified" };
  const h = headerStr.toLowerCase();

  let category = "Equity";
  let subcategory = "Diversified";

  if (
    h.includes("debt") ||
    h.includes("income") ||
    h.includes("gilt") ||
    h.includes("liquid") ||
    h.includes("money market")
  ) {
    category = "Debt";
    subcategory = "Debt Fund";
  } else if (h.includes("hybrid") || h.includes("balanced") || h.includes("advantage")) {
    category = "Hybrid";
    subcategory = "Balanced";
  } else if (h.includes("elss") || h.includes("tax")) {
    category = "ELSS";
    subcategory = "Tax Saving";
  } else if (h.includes("gold") || h.includes("silver") || h.includes("precious")) {
    category = "Commodity";
    subcategory = "Gold";
  }

  if (h.includes("large cap") || h.includes("bluechip")) {
    subcategory = "Large Cap";
  } else if (h.includes("mid cap")) {
    subcategory = "Mid Cap";
  } else if (h.includes("small cap")) {
    subcategory = "Small Cap";
  } else if (h.includes("flexi cap") || h.includes("multi cap")) {
    subcategory = "Flexi Cap";
  }

  return { category, subcategory };
}

export interface AmfiScheme {
  schemeCode: number;
  isin: string | null;
  name: string;
  planType: string;
  optionType: string;
  nav: number;
  navDate: string;
  fundHouse: string;
  category: string;
  subcategory: string;
}

export async function syncFromAmfiOfficial(): Promise<{
  totalParsed: number;
  totalUpserted: number;
  durationSeconds: string;
}> {
  if (isAmfiSyncing) {
    console.log("⚠️ AMFI sync already in progress, skipping duplicate call.");
    return { totalParsed: 0, totalUpserted: 0, durationSeconds: "0" };
  }

  isAmfiSyncing = true;
  const startTime = Date.now();

  try {
    console.log("📥 Fetching master NAVAll.txt feed from AMFI portal (amfiindia.com)...");
    const response = await fetch(AMFI_NAV_URL, {
      headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" },
    });

    if (!response.ok) {
      throw new Error(`AMFI portal returned HTTP ${response.status}: ${response.statusText}`);
    }

    const text = await response.text();
    const lines = text.split(/\r?\n/);
    console.log(`✓ Downloaded ${lines.length.toLocaleString()} raw lines from official AMFI feed.`);

    const schemes: AmfiScheme[] = [];
    let currentCategory: string | null = null;
    let currentAmc: string | null = null;

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      if (
        trimmed.startsWith("Open Ended Schemes") ||
        trimmed.startsWith("Close Ended Schemes") ||
        trimmed.startsWith("Interval Fund")
      ) {
        currentCategory = trimmed;
        continue;
      }

      if (!trimmed.includes(";") && trimmed.length > 3 && !/^\d/.test(trimmed)) {
        currentAmc = trimmed;
        continue;
      }

      const parts = trimmed.split(";");
      if (parts.length >= 8 && /^\d+$/.test(parts[0])) {
        const code = parseInt(parts[0], 10);
        const isin1 = parts[1]?.trim();
        const isin2 = parts[2]?.trim();
        const isin = isin1 && isin1 !== "-" ? isin1 : (isin2 && isin2 !== "-" ? isin2 : null);
        const name = parts[3].trim();
        const planRaw = parts[4]?.trim() || "";
        const optionRaw = parts[5]?.trim() || "";
        const navStr = parts[6]?.trim() || "";
        const dateStr = parts[7]?.trim() || "";

        const nav = parseFloat(navStr);
        if (isNaN(nav)) continue;

        const navDate = parseAmfiDate(dateStr);
        const planType =
          planRaw.toLowerCase().includes("direct") || name.toLowerCase().includes("direct")
            ? "Direct"
            : "Regular";
        const optionType =
          optionRaw.toLowerCase().includes("idcw") || name.toLowerCase().includes("dividend")
            ? "IDCW"
            : "Growth";

        const { category, subcategory } = parseAmfiCategory(currentCategory);

        schemes.push({
          schemeCode: code,
          isin,
          name,
          planType,
          optionType,
          nav,
          navDate,
          fundHouse: currentAmc || "Mutual Fund AMC",
          category,
          subcategory,
        });
      }
    }

    console.log(`✓ Parsed ${schemes.length.toLocaleString()} active mutual fund schemes from AMFI.`);

    // Ensure ISIN column exists in funds table
    await pool.query("ALTER TABLE funds ADD COLUMN IF NOT EXISTS isin TEXT;");

    // Chunked batch upsert using UNNEST for maximum performance
    const BATCH_SIZE = 1000;
    let upserted = 0;

    for (let i = 0; i < schemes.length; i += BATCH_SIZE) {
      const chunk = schemes.slice(i, i + BATCH_SIZE);

      const query = `
        INSERT INTO funds (
          scheme_code, isin, name, plan_type, option_type, nav, nav_date,
          fund_house, category, subcategory, is_active, updated_at
        )
        SELECT * FROM UNNEST(
          $1::int[],
          $2::text[],
          $3::text[],
          $4::text[],
          $5::text[],
          $6::numeric[],
          $7::date[],
          $8::text[],
          $9::text[],
          $10::text[],
          $11::boolean[],
          $12::timestamp[]
        )
        ON CONFLICT (scheme_code) DO UPDATE
        SET isin = COALESCE(EXCLUDED.isin, funds.isin),
            nav = EXCLUDED.nav,
            nav_date = EXCLUDED.nav_date,
            is_active = TRUE,
            updated_at = CURRENT_TIMESTAMP;
      `;

      const now = new Date();
      const codes = chunk.map((s) => s.schemeCode);
      const isins = chunk.map((s) => s.isin);
      const names = chunk.map((s) => s.name);
      const planTypes = chunk.map((s) => s.planType);
      const optionTypes = chunk.map((s) => s.optionType);
      const navs = chunk.map((s) => s.nav);
      const navDates = chunk.map((s) => s.navDate);
      const fundHouses = chunk.map((s) => s.fundHouse);
      const categories = chunk.map((s) => s.category);
      const subcategories = chunk.map((s) => s.subcategory);
      const isActives = chunk.map(() => true);
      const timestamps = chunk.map(() => now);

      await pool.query(query, [
        codes,
        isins,
        names,
        planTypes,
        optionTypes,
        navs,
        navDates,
        fundHouses,
        categories,
        subcategories,
        isActives,
        timestamps,
      ]);

      upserted += chunk.length;
      if (upserted % 3000 === 0 || upserted === schemes.length) {
        console.log(`  Progress: ${upserted.toLocaleString()} / ${schemes.length.toLocaleString()} schemes upserted into PostgreSQL...`);
      }
    }

    const durationSeconds = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`✅ AMFI official sync completed in ${durationSeconds}s. Successfully synced ${upserted.toLocaleString()} schemes.`);

    return {
      totalParsed: schemes.length,
      totalUpserted: upserted,
      durationSeconds,
    };
  } finally {
    isAmfiSyncing = false;
  }
}
