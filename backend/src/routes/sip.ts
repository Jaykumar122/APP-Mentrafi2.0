// src/routes/sip.ts
import { Router } from "express";
import pool from "../config/db";
import { authenticateToken, AuthRequest } from "../middleware/auth";
import { cancelSIP, createSIP, getSIPsForUser, pauseSIP, resumeSIP, updateSIP } from "../services/sip";

const router = Router();

// ---------------------------------------------------------------------------
// GET /api/sip — active + paused SIPs for the logged-in user, plus summary
// totals for the dashboard header (monthly SIP total, total invested).
// ---------------------------------------------------------------------------
router.get("/", authenticateToken, async (req: AuthRequest, res) => {
  try {
    const sips = await getSIPsForUser(req.userId!);
    const activeSIPs = sips.filter((s) => s.status === "active");
    const pausedSIPs = sips.filter((s) => s.status === "paused");
    const completedSIPs = sips.filter((s) => s.status === "completed");

    const monthlySIPTotal = activeSIPs.reduce((sum, s) => sum + s.monthlyAmount, 0);
    const totalInvested = sips.reduce((sum, s) => sum + s.totalInvested, 0);

    res.json({
      activeSIPs,
      pausedSIPs,
      completedSIPs,
      monthlySIPTotal: Number(monthlySIPTotal.toFixed(2)),
      totalInvested: Number(totalInvested.toFixed(2)),
    });
  } catch (err: any) {
    console.error("GET /api/sip failed:", err);
    res.status(500).json({ error: `Failed to fetch SIPs: ${err.message ?? "unknown error"}` });
  }
});

// ---------------------------------------------------------------------------
// POST /api/sip — start a new SIP mandate.
// body: { schemeCode, monthlyAmount, installmentDay?, totalInstallments? }
// ---------------------------------------------------------------------------
router.post("/", authenticateToken, async (req: AuthRequest, res) => {
  try {
    const schemeCode = Number(req.body.schemeCode);
    const monthlyAmount = Number(req.body.monthlyAmount);
    const installmentDay = req.body.installmentDay != null ? Number(req.body.installmentDay) : 1;
    const totalInstallments = req.body.totalInstallments != null ? Number(req.body.totalInstallments) : 60;

    if (!schemeCode || !monthlyAmount || monthlyAmount <= 0) {
      return res.status(400).json({ error: "schemeCode and a positive monthlyAmount are required" });
    }

    const fundResult = await pool.query(`SELECT scheme_code FROM funds WHERE scheme_code = $1`, [schemeCode]);
    if (fundResult.rows.length === 0) {
      return res.status(404).json({ error: "Fund not found" });
    }

    const sip = await createSIP(req.userId!, schemeCode, monthlyAmount, installmentDay, totalInstallments);
    res.status(201).json({ message: "SIP started", sip });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to create SIP" });
  }
});

// ---------------------------------------------------------------------------
// PATCH /api/sip/:id — edit monthly amount / installment day / duration.
// ---------------------------------------------------------------------------
router.patch("/:id", authenticateToken, async (req: AuthRequest, res) => {
  try {
    const sipId = Number(req.params.id);
    const updated = await updateSIP(req.userId!, sipId, {
      monthlyAmount: req.body.monthlyAmount != null ? Number(req.body.monthlyAmount) : undefined,
      installmentDay: req.body.installmentDay != null ? Number(req.body.installmentDay) : undefined,
      totalInstallments: req.body.totalInstallments != null ? Number(req.body.totalInstallments) : undefined,
    });
    if (!updated) return res.status(404).json({ error: "SIP not found" });
    res.json({ message: "SIP updated", sip: updated });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to update SIP" });
  }
});

// ---------------------------------------------------------------------------
// PATCH /api/sip/:id/pause
// ---------------------------------------------------------------------------
router.patch("/:id/pause", authenticateToken, async (req: AuthRequest, res) => {
  try {
    const updated = await pauseSIP(req.userId!, Number(req.params.id));
    if (!updated) return res.status(404).json({ error: "Active SIP not found" });
    res.json({ message: "SIP paused", sip: updated });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to pause SIP" });
  }
});

// ---------------------------------------------------------------------------
// PATCH /api/sip/:id/resume
// ---------------------------------------------------------------------------
router.patch("/:id/resume", authenticateToken, async (req: AuthRequest, res) => {
  try {
    const updated = await resumeSIP(req.userId!, Number(req.params.id));
    if (!updated) return res.status(404).json({ error: "Paused SIP not found" });
    res.json({ message: "SIP resumed", sip: updated });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to resume SIP" });
  }
});

// ---------------------------------------------------------------------------
// DELETE /api/sip/:id — cancel a SIP mandate.
// ---------------------------------------------------------------------------
router.delete("/:id", authenticateToken, async (req: AuthRequest, res) => {
  try {
    const cancelled = await cancelSIP(req.userId!, Number(req.params.id));
    if (!cancelled) return res.status(404).json({ error: "SIP not found" });
    res.json({ message: "SIP cancelled" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to cancel SIP" });
  }
});

export default router;
