// src/scripts/syncAmfi.ts
import dotenv from "dotenv";
dotenv.config();

import { syncFromAmfiOfficial } from "../services/amfiSync";
import pool from "../config/db";

async function main() {
  console.log("🚀 Starting official AMFI sync directly from amfiindia.com (Node.js / TypeScript)...");
  try {
    const result = await syncFromAmfiOfficial();
    console.log(`🎉 Official AMFI sync completed successfully in ${result.durationSeconds}s!`);
    console.log(`   Total parsed: ${result.totalParsed.toLocaleString()}, Total upserted: ${result.totalUpserted.toLocaleString()}`);
  } catch (err) {
    console.error("❌ AMFI sync failed:", err);
  } finally {
    await pool.end();
    process.exit(0);
  }
}

main();
