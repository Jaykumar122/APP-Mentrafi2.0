import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import cron from "node-cron";
import authRoutes from "./routes/auth";
import fundsRouter from "./routes/funds";
import { syncAllFunds, syncAllFundsFast, isSyncRunning } from "./services/fundSync";
import { syncFromAmfiOfficial } from "./services/amfiSync";
import profileRoutes from "./routes/profile";
import path from "path";
import advisorRoutes from "./routes/advisor";
import portfolioRoutes from "./routes/portfolio";
import sipRoutes from "./routes/sip";
import { processDueSIPs } from "./services/sip";
dotenv.config();

const app = express();

app.use(cors());
app.use(express.json());

// Routes
app.use("/api/auth", authRoutes);
app.use("/api/funds", fundsRouter);
app.use("/api/profile", profileRoutes);
app.use("/api/advisor", advisorRoutes);
app.use("/api/portfolio", portfolioRoutes);
app.use("/api/sip", sipRoutes);
app.use("/uploads", express.static(path.join(__dirname, "../uploads")));

// Health check
app.get("/", (_req, res) => {
  res.json({ status: "ok", message: "Mentrafi API is running" });
});

// ---------------------------------------------------------------------------
// If a sync is already running when a scheduled job fires (e.g. a manual
// trigger overlapped with a cron job), wait for it to finish instead of
// silently skipping — avoids the "missed execution" node-cron warning
// leaving a scheduled refresh un-run.
// ---------------------------------------------------------------------------
async function runSyncWhenFree(syncFn: () => Promise<void>, label: string) {
  if (isSyncRunning()) {
    console.log(`⏳ ${label} delayed — another sync is already running. Waiting...`);
    while (isSyncRunning()) {
      await new Promise((r) => setTimeout(r, 30_000));
    }
    console.log(`▶️ ${label} starting now that the other sync finished.`);
  }
  await syncFn();
}

// ---------------------------------------------------------------------------
// Scheduled NAV syncs.
//
// Mutual fund NAV is calculated once per day, after market close, and
// published by AMFI typically between 9-11 PM IST.
//
// Daily runs use syncAllFundsFast() — lightweight /latest endpoint, only
// updates NAV + date, finishes in a couple minutes instead of 30+.
//
// A weekly run uses the full syncAllFunds() — fetches complete NAV history
// per fund, needed to recompute accurate 1-year returns. Returns barely
// move day-to-day, so this doesn't need to run daily.
// ---------------------------------------------------------------------------

// 6 times daily Fast NAV syncs (Optimized for AMFI & mfapi.in publication windows):
// 1. 8:30 AM IST  — Morning pre-market (ready before 9:05 AM SIP execution & 9:15 AM market open)
// 2. 10:30 AM IST — Overseas Fund of Funds (SEBI next-day 10:00 AM deadline)
// 3. 6:30 PM IST  — Early evening debt/liquid fund NAV releases
// 4. 9:30 PM IST  — Prime AMFI wave 1 (first major batch of equity funds on mfapi.in)
// 5. 10:45 PM IST — Peak AMFI wave 2 (HDFC, SBI, ICICI, Nippon, Quant bulk uploads)
// 6. 11:45 PM IST — Late-night final sweep (captures all delayed AMCs past 11:00 PM SEBI deadline)
const FAST_SYNC_SCHEDULES = [
  { cron: "30 8 * * *", label: "8:30 AM" },
  { cron: "30 10 * * *", label: "10:30 AM" },
  { cron: "30 18 * * *", label: "6:30 PM" },
  { cron: "30 21 * * *", label: "9:30 PM" },
  { cron: "45 22 * * *", label: "10:45 PM" },
  { cron: "45 23 * * *", label: "11:45 PM" },
];

FAST_SYNC_SCHEDULES.forEach(({ cron: expr, label }) => {
  cron.schedule(
    expr,
    () => {
      console.log(`🔄 [${label}] Fast NAV refresh starting...`);
      runSyncWhenFree(syncAllFundsFast, `${label} fast sync`)
        .then(() => console.log(`✅ [${label}] Fast NAV refresh finished.`))
        .catch((err) => console.error(`❌ [${label}] Fast NAV refresh failed:`, err));
    },
    { timezone: "Asia/Kolkata" }
  );
});

// Nightly official AMFI master feed sync (11:15 PM IST Monday-Friday)
// Pulls master NAVAll.txt from AMFI, updating official NAVs and ISIN codes.
cron.schedule(
  "15 23 * * 1-5",
  () => {
    console.log("🔄 [11:15 PM IST] Daily Official AMFI Master Feed Sync starting...");
    syncFromAmfiOfficial()
      .then((res) => console.log(`✅ [11:15 PM] Official AMFI sync finished: ${res.totalUpserted.toLocaleString()} schemes updated in ${res.durationSeconds}s.`))
      .catch((err) => console.error("❌ [11:15 PM] Official AMFI sync failed:", err));
  },
  { timezone: "Asia/Kolkata" }
);

// Weekly full sync (Sunday 6:00 AM IST) — recomputes 1-year returns
// using complete NAV history. Heavier, so it runs only once a week.
cron.schedule(
  "0 6 * * 0",
  () => {
    console.log("🔄 [Sun 6:00 AM] Weekly full sync starting...");
    runSyncWhenFree(syncAllFunds, "Sunday weekly full sync")
      .then(() => console.log("✅ [Sun 6:00 AM] Weekly full sync finished."))
      .catch((err) => console.error("❌ [Sun 6:00 AM] Weekly full sync failed:", err));
  },
  { timezone: "Asia/Kolkata" }
);

// ---------------------------------------------------------------------------
// Daily SIP installment processing (9:05 AM IST) — runs after the
// morning NAV safety-check sync so due SIPs buy units at a fresh NAV.
// ---------------------------------------------------------------------------
cron.schedule(
  "5 9 * * *",
  () => {
    console.log("🔄 [9:05 AM] Processing due SIP installments...");
    processDueSIPs()
      .then(({ processed, failed }) =>
        console.log(`✅ SIP installments processed: ${processed}, failed: ${failed}`)
      )
      .catch((err) => console.error("❌ SIP installment processing failed:", err));
  },
  { timezone: "Asia/Kolkata" }
);

const PORT = process.env.PORT || 3001;

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`⏰ Fast NAV refresh scheduled: 6 times a day (8:30 AM, 10:30 AM, 6:30 PM, 9:30 PM, 10:45 PM, 11:45 PM IST)`);
  console.log(`⏰ Full sync (returns recalc) scheduled: Sunday 6:00 AM IST`);
});