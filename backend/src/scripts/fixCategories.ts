// Re-label funds.category / funds.subcategory with the SEBI categories from fundCategory.ts.
//
//   npx tsx src/scripts/fixCategories.ts --dry-run   # show what would change
//   npx tsx src/scripts/fixCategories.ts             # apply (one transaction)
//   npx tsx src/scripts/fixCategories.ts --restore   # put the backed-up labels back
//
// Schemes in today's AMFI NAVAll.txt use their official section header; others (closed,
// matured, merged) are classified by name. Before the first change the old labels are
// copied to funds_category_backup (created once, never overwritten), so --restore can
// always return to the original state.
import dotenv from "dotenv";
dotenv.config();
import pool from "../config/db";
import { classifyFund } from "../services/fundCategory";

const AMFI_NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt";

async function amfiHeaders(): Promise<Map<number, string>> {
  const res = await fetch(AMFI_NAV_URL, { headers: { "User-Agent": "Mozilla/5.0" } });
  if (!res.ok) throw new Error(`AMFI returned HTTP ${res.status}`);
  const headers = new Map<number, string>();
  let header: string | null = null;
  for (const raw of (await res.text()).split(/\r?\n/)) {
    const line = raw.trim();
    if (/^(Open Ended|Close Ended|Interval Fund)/.test(line)) { header = line; continue; }
    const code = line.split(";")[0];
    if (header && /^\d+$/.test(code)) headers.set(Number(code), header);
  }
  return headers;
}

async function main() {
  const dryRun = process.argv.includes("--dry-run");
  const client = await pool.connect();
  try {
    if (process.argv.includes("--restore")) {
      const r = await client.query(
        `UPDATE funds f SET category = b.category, subcategory = b.subcategory
         FROM funds_category_backup b WHERE b.scheme_code = f.scheme_code`);
      console.log(`Restored original labels for ${r.rowCount} funds.`);
      return;
    }
    const headers = await amfiHeaders();
    console.log(`AMFI feed: ${headers.size.toLocaleString()} schemes with official categories.`);
    const { rows } = await client.query("SELECT scheme_code, name, category, subcategory FROM funds");
    const changes: { code: number; category: string; subcategory: string }[] = [];
    const moves = new Map<string, number>();
    const samples = new Map<string, string[]>();
    let official = 0;
    for (const row of rows) {
      const header = headers.get(row.scheme_code) ?? null;
      if (header) official++;
      const next = classifyFund(row.name, header);
      if (next.category !== row.category || next.subcategory !== row.subcategory) {
        changes.push({ code: row.scheme_code, ...next });
        const key = `${row.category}/${row.subcategory} -> ${next.category}/${next.subcategory}`;
        moves.set(key, (moves.get(key) ?? 0) + 1);
        const list = samples.get(key) ?? [];
        if (list.length < 3 && Math.random() < 0.3) list.push(row.name);
        samples.set(key, list);
      }
    }
    console.log(`${rows.length.toLocaleString()} funds (${official.toLocaleString()} matched to AMFI); ${changes.length.toLocaleString()} labels change.`);
    [...moves.entries()].sort((a, b) => b[1] - a[1]).slice(0, 25).forEach(([k, v]) => {
      console.log(`  ${String(v).padStart(6)}  ${k}`);
      if (dryRun) (samples.get(k) ?? []).forEach((s) => console.log(`            e.g. ${s}`));
    });
    if (dryRun || changes.length === 0) {
      console.log(dryRun ? "Dry run: nothing written." : "Nothing to change.");
      return;
    }
    await client.query("BEGIN");
    await client.query(`CREATE TABLE IF NOT EXISTS funds_category_backup AS
                        SELECT scheme_code, category, subcategory, NOW() AS saved_at FROM funds`);
    const BATCH = 2000;
    for (let i = 0; i < changes.length; i += BATCH) {
      const part = changes.slice(i, i + BATCH);
      await client.query(
        `UPDATE funds f SET category = u.category, subcategory = u.subcategory
         FROM UNNEST($1::int[], $2::text[], $3::text[]) AS u(scheme_code, category, subcategory)
         WHERE f.scheme_code = u.scheme_code`,
        [part.map((c) => c.code), part.map((c) => c.category), part.map((c) => c.subcategory)]);
    }
    await client.query("COMMIT");
    console.log(`Updated ${changes.length.toLocaleString()} funds. Original labels are in funds_category_backup.`);
  } catch (err) {
    await client.query("ROLLBACK").catch(() => undefined);
    throw err;
  } finally {
    client.release();
    // Not awaited: the pool's startup check in config/db.ts can hold a client open.
    void pool.end().catch(() => undefined);
  }
}

// Explicit exit: fetch keep-alive sockets would otherwise keep the CLI running.
main().then(() => process.exit(0), (err) => { console.error("fixCategories failed:", err); process.exit(1); });
