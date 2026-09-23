import dotenv from "dotenv";
dotenv.config();

import { syncAllFundsFast } from "../services/fundSync";
import pool from "../config/db";

async function main() {
  console.log("🚀 Starting manual Fast NAV sync from mfapi.in...");
  try {
    await syncAllFundsFast();
    console.log("🎉 Manual Fast NAV sync completed successfully!");
  } catch (err) {
    console.error("❌ Sync failed:", err);
  } finally {
    await pool.end();
  }
}

main();
