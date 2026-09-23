import dotenv from "dotenv";
dotenv.config();

import { seedStarterFunds } from "../services/fundSync";
import pool from "../config/db";

async function main() {
  console.log("🌱 Starting manual seed of starter mutual funds from mfapi.in...");
  try {
    await seedStarterFunds();
    console.log("🎉 Starter mutual funds seeded successfully!");
  } catch (err) {
    console.error("❌ Seeding failed:", err);
  } finally {
    await pool.end();
  }
}

main();
