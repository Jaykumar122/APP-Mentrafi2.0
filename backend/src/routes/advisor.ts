import { Router } from "express";
import pool from "../config/db";
import { authenticateToken, AuthRequest } from "../middleware/auth";
import {
  getAIRecommendation,
  getAIRecommendationStream,
  UserProfileContext,
} from "../services/aiService";

const router = Router();

async function getUserProfile(userId?: number, bodyOverrides?: any): Promise<UserProfileContext> {
  if (!userId) {
    return {
      age: bodyOverrides?.age,
      name: bodyOverrides?.name,
      monthly_sip_budget: bodyOverrides?.monthly_sip_budget || bodyOverrides?.budget,
      risk_appetite: bodyOverrides?.risk_appetite,
      investment_goal: bodyOverrides?.investment_goal || bodyOverrides?.goal,
      investment_horizon: bodyOverrides?.investment_horizon || bodyOverrides?.horizon,
      occupation: bodyOverrides?.occupation,
    };
  }

  try {
    const result = await pool.query(
      `SELECT u.name AS user_name, pi.full_name, pi.age, pi.occupation, pi.monthly_sip_budget, pi.risk_appetite, pi.investment_goal,
              EXTRACT(YEAR FROM AGE(NOW(), pi.date_of_birth))::int AS computed_age
       FROM users u
       LEFT JOIN personal_information pi ON pi.user_id = u.id
       WHERE u.id = $1`,
      [userId]
    );

    const row = result.rows[0] || {};
    const effectiveAge = row.age ?? row.computed_age ?? bodyOverrides?.age ?? undefined;
    const effectiveName = row.full_name || row.user_name || bodyOverrides?.name || undefined;
    const effectiveBudget = row.monthly_sip_budget
      ? Number(row.monthly_sip_budget)
      : bodyOverrides?.monthly_sip_budget ?? bodyOverrides?.budget;
    const effectiveRisk = row.risk_appetite ?? bodyOverrides?.risk_appetite ?? undefined;
    const effectiveGoal = row.investment_goal ?? bodyOverrides?.investment_goal ?? bodyOverrides?.goal ?? undefined;
    const effectiveOccupation = row.occupation ?? bodyOverrides?.occupation ?? undefined;

    return {
      age: effectiveAge ? Number(effectiveAge) : undefined,
      name: effectiveName,
      monthly_sip_budget: effectiveBudget ? Number(effectiveBudget) : undefined,
      risk_appetite: effectiveRisk,
      investment_goal: effectiveGoal,
      investment_horizon: bodyOverrides?.investment_horizon ?? bodyOverrides?.horizon ?? undefined,
      occupation: effectiveOccupation,
    };
  } catch (err) {
    console.error("Failed to query user profile from DB:", err);
    return {};
  }
}

// ---------------------------------------------------------------------------
// POST /api/advisor/chat — Non-streaming
// ---------------------------------------------------------------------------
router.post("/chat", authenticateToken, async (req: AuthRequest, res) => {
  try {
    const { message, history } = req.body;
    if (!message || !message.trim()) {
      return res.status(400).json({ error: "message is required" });
    }

    const profile = await getUserProfile(req.userId, req.body);
    const aiResponse = await getAIRecommendation(message.trim(), profile, req.userId, history);

    res.json(aiResponse);
  } catch (err: any) {
    console.error("AI Advisor Error:", err.message);
    res.status(503).json({
      error: "MentraFiAI Neural Engine is currently offline.",
      detail: "Please start the Python LLM server with: python serve.py",
      reply: "I couldn't connect to the MentraFiAI Neural Engine. Please ensure `python serve.py` is running on port 8000."
    });
  }
});

// ---------------------------------------------------------------------------
// POST /api/advisor/chat/stream — Real-time Server-Sent Events (SSE)
// ---------------------------------------------------------------------------
router.post("/chat/stream", authenticateToken, async (req: AuthRequest, res) => {
  const { message, history } = req.body;
  if (!message || !message.trim()) {
    return res.status(400).json({ error: "message is required" });
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders();

  let ended = false;
  const abortController = new AbortController();

  // SSE Heartbeat every 8 seconds to prevent mobile socket drops
  const heartbeat = setInterval(() => {
    if (!ended) {
      try {
        res.write(": heartbeat\n\n");
        if (typeof (res as any).flush === "function") (res as any).flush();
      } catch {
        // Socket closed
      }
    }
  }, 8000);

  function endStream() {
    if (ended) return;
    ended = true;
    clearInterval(heartbeat);
    res.end();
  }

  res.on("close", () => {
    if (!ended) {
      ended = true;
      clearInterval(heartbeat);
      abortController.abort();
    }
  });

  try {
    const profile = await getUserProfile(req.userId, req.body);

    await getAIRecommendationStream(
      message.trim(),
      profile,
      req.userId,
      (chunk) => {
        res.write(`data: ${JSON.stringify({ chunk })}\n\n`);
        if (typeof (res as any).flush === "function") (res as any).flush();
      },
      (funds) => {
        res.write(`data: ${JSON.stringify({ recommendedFunds: funds })}\n\n`);
        if (typeof (res as any).flush === "function") (res as any).flush();
      },
      abortController.signal,
      history
    );

    res.write("data: [DONE]\n\n");
    endStream();
  } catch (err: any) {
    if (abortController.signal.aborted || res.destroyed) return;
    console.error("AI Advisor Stream Error:", err.message);
    res.write(`data: ${JSON.stringify({
      chunk: "I couldn't reach the MentraFiAI Neural Engine. Please ensure `python serve.py` is running on port 8000."
    })}\n\n`);
    res.write("data: [DONE]\n\n");
    endStream();
  }
});

export default router;
