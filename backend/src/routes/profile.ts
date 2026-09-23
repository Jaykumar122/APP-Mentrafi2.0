// src/routes/profile.ts
import { Router } from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import pool from "../config/db";
import { authenticateToken, AuthRequest } from "../middleware/auth";

const router = Router();

// ---------------------------------------------------------------------------
// Avatar upload setup
// ---------------------------------------------------------------------------
const uploadsDir = path.join(__dirname, "../../uploads");
if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadsDir),
  filename: (req: AuthRequest, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `avatar-${req.userId}-${Date.now()}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB max
  fileFilter: (_req, file, cb) => {
    if (!file.mimetype.startsWith("image/")) {
      return cb(new Error("Only image files are allowed"));
    }
    cb(null, true);
  },
});

// Helper: Calculate live portfolio and active SIP stats directly from DB
async function getProfileStats(userId?: number) {
  if (!userId) {
    return {
      portfolioValue: 0,
      portfolioReturnPercent: 0,
      activeSips: 0,
      monthlySipAmount: 0,
      fundsHeld: 0,
    };
  }
  try {
    const portfolioResult = await pool.query(
      `SELECT 
         COALESCE(SUM(up.units * COALESCE(f.nav, up.avg_purchase_nav, 0)), 0) AS total_value,
         COALESCE(SUM(up.invested_amount), 0) AS total_invested,
         COUNT(DISTINCT up.scheme_code) AS funds_held
       FROM user_portfolio up
       LEFT JOIN funds f ON f.scheme_code = up.scheme_code
       WHERE up.user_id = $1 AND up.units > 0`,
      [userId]
    );

    const pRow = portfolioResult.rows[0];
    const portfolioValue = Number(Number(pRow?.total_value || 0).toFixed(2));
    const totalInvested = Number(pRow?.total_invested || 0);
    const portfolioReturnPercent = totalInvested > 0
      ? Number((((portfolioValue - totalInvested) / totalInvested) * 100).toFixed(2))
      : 0;
    const fundsHeld = Number(pRow?.funds_held || 0);

    const sipResult = await pool.query(
      `SELECT 
         COUNT(*) AS active_sips,
         COALESCE(SUM(monthly_amount), 0) AS monthly_sip_amount
       FROM sips
       WHERE user_id = $1 AND LOWER(status) = 'active'`,
      [userId]
    );

    const sRow = sipResult.rows[0];
    const activeSips = Number(sRow?.active_sips || 0);
    const monthlySipAmount = Number(Number(sRow?.monthly_sip_amount || 0).toFixed(2));

    return {
      portfolioValue,
      portfolioReturnPercent,
      activeSips,
      monthlySipAmount,
      fundsHeld,
    };
  } catch (err) {
    console.error("Error calculating profile stats:", err);
    return {
      portfolioValue: 0,
      portfolioReturnPercent: 0,
      activeSips: 0,
      monthlySipAmount: 0,
      fundsHeld: 0,
    };
  }
}

// ---------------------------------------------------------------------------
// GET /api/profile — current user's info + portfolio stats
// ---------------------------------------------------------------------------
router.get("/", authenticateToken, async (req: AuthRequest, res) => {
  try {
    const userResult = await pool.query(
      `SELECT id, name, email, avatar_url, kyc_status, tier, created_at FROM users WHERE id = $1`,
      [req.userId]
    );
    if (userResult.rows.length === 0) {
      return res.status(404).json({ error: "User not found" });
    }
    const user = userResult.rows[0];

    const personalInformationResult = await pool.query(
      `SELECT full_name, phone, TO_CHAR(date_of_birth, 'YYYY-MM-DD') AS date_of_birth, gender, location, age, monthly_sip_budget, risk_appetite, investment_goal, occupation
       FROM personal_information WHERE user_id = $1`,
      [req.userId]
    );
    const personalInformation = personalInformationResult.rows[0] ?? null;

    const dateOfBirth = personalInformation?.date_of_birth || null;

    // Calculate live portfolio & SIP stats directly from DB
    const stats = await getProfileStats(req.userId);

    res.json({
      id: user.id,
      name: user.name,
      email: user.email,
      avatarUrl: user.avatar_url,
      kycStatus: user.kyc_status,
      tier: user.tier,
      personalInfo: {
        fullName: personalInformation?.full_name ?? user.name,
        phone: personalInformation?.phone ?? null,
        dateOfBirth,
        gender: personalInformation?.gender ?? null,
        location: personalInformation?.location ?? null,
        age: personalInformation?.age ?? null,
        monthlySipBudget: personalInformation?.monthly_sip_budget ?? null,
        riskAppetite: personalInformation?.risk_appetite ?? null,
        investmentGoal: personalInformation?.investment_goal ?? null,
        occupation: personalInformation?.occupation ?? null,
      },
      memberSince: user.created_at,
      stats,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch profile" });
  }
});

// ---------------------------------------------------------------------------
// PATCH /api/profile — update editable profile details
// ---------------------------------------------------------------------------
router.patch("/", authenticateToken, async (req: AuthRequest, res) => {
  const client = await pool.connect();
  try {
    const {
      name,
      phone,
      dateOfBirth,
      gender,
      location,
      avatarUrl,
      age,
      monthlySipBudget,
      riskAppetite,
      investmentGoal,
      occupation,
    } = req.body;

    // Calculate age from dateOfBirth if provided
    let finalAge = age !== null && age !== undefined ? parseInt(age, 10) : null;
    if (dateOfBirth) {
      const birth = new Date(dateOfBirth);
      if (!isNaN(birth.getTime())) {
        const today = new Date();
        let calculated = today.getFullYear() - birth.getFullYear();
        const m = today.getMonth() - birth.getMonth();
        if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) {
          calculated--;
        }
        if (calculated >= 0) {
          finalAge = calculated;
        }
      }
    }

    // Strict 18+ requirement
    if (finalAge !== null && finalAge !== undefined) {
      if (isNaN(finalAge) || finalAge < 18) {
        return res.status(400).json({ error: "Investor must be at least 18 years old to create an account." });
      }
      if (finalAge > 100) {
        return res.status(400).json({ error: "Please enter a valid date of birth." });
      }
    }

    // Validate monthly SIP budget if provided
    if (monthlySipBudget !== null && monthlySipBudget !== undefined) {
      const budgetNum = parseFloat(monthlySipBudget);
      if (isNaN(budgetNum) || budgetNum <= 0) {
        return res.status(400).json({ error: "Monthly SIP budget must be a positive number" });
      }
    }

    // Validate risk appetite if provided
    if (riskAppetite && !("Low" === riskAppetite || "Moderate" === riskAppetite || "High" === riskAppetite)) {
      return res.status(400).json({ error: "Risk appetite must be 'Low', 'Moderate', or 'High'" });
    }

    // Enforce unique phone number across accounts
    if (phone && typeof phone === "string" && phone.trim()) {
      const cleanPhone = phone.trim();
      const existingPhone = await pool.query(
        `SELECT user_id FROM personal_information WHERE phone = $1 AND user_id != $2`,
        [cleanPhone, req.userId]
      );
      if (existingPhone.rows.length > 0) {
        return res.status(400).json({
          error: "This phone number is already registered with another account. Please use a unique phone number.",
        });
      }
    }

    await client.query("BEGIN");

    try {
      const userResult = await client.query(
        `UPDATE users SET
         name = COALESCE($1, name),
         avatar_url = COALESCE($2, avatar_url)
         WHERE id = $3
         RETURNING id, name, email, avatar_url, kyc_status, tier, created_at`,
        [name, avatarUrl, req.userId]
      );

      const personalInformationResult = await client.query(
        `INSERT INTO personal_information (
           user_id,
           full_name,
           phone,
           date_of_birth,
           gender,
           location,
           age,
           monthly_sip_budget,
           risk_appetite,
           investment_goal,
           occupation,
           updated_at
         )
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
         ON CONFLICT (user_id) DO UPDATE SET
           full_name = COALESCE(EXCLUDED.full_name, personal_information.full_name),
           phone = COALESCE(EXCLUDED.phone, personal_information.phone),
           date_of_birth = COALESCE(EXCLUDED.date_of_birth, personal_information.date_of_birth),
           gender = COALESCE(EXCLUDED.gender, personal_information.gender),
           location = COALESCE(EXCLUDED.location, personal_information.location),
           age = COALESCE(EXCLUDED.age, personal_information.age),
           monthly_sip_budget = COALESCE(EXCLUDED.monthly_sip_budget, personal_information.monthly_sip_budget),
           risk_appetite = COALESCE(EXCLUDED.risk_appetite, personal_information.risk_appetite),
           investment_goal = COALESCE(EXCLUDED.investment_goal, personal_information.investment_goal),
           occupation = COALESCE(EXCLUDED.occupation, personal_information.occupation),
           updated_at = NOW()
         RETURNING full_name, phone, TO_CHAR(date_of_birth, 'YYYY-MM-DD') AS date_of_birth, gender, location, age, monthly_sip_budget, risk_appetite, investment_goal, occupation`,
        [
         req.userId,
         name,
         phone,
         dateOfBirth || null,
         gender,
         location,
         finalAge ?? null,
         monthlySipBudget ?? null,
         riskAppetite ?? null,
         investmentGoal ?? null,
         occupation ?? null,
        ]
      );

      await client.query("COMMIT");

      const personalInformation = personalInformationResult.rows[0] ?? null;
      const user = userResult.rows[0];

      const stats = await getProfileStats(req.userId);

      res.json({
        id: user.id,
        name: user.name,
        email: user.email,
        avatarUrl: user.avatar_url,
        kycStatus: user.kyc_status,
        tier: user.tier,
        memberSince: user.created_at,
        stats,
        personalInfo: {
         fullName: personalInformation?.full_name ?? user.name,
         phone: personalInformation?.phone ?? null,
         dateOfBirth: personalInformation?.date_of_birth || null,
         gender: personalInformation?.gender ?? null,
         location: personalInformation?.location ?? null,
         age: personalInformation?.age ?? null,
         monthlySipBudget: personalInformation?.monthly_sip_budget ?? null,
         riskAppetite: personalInformation?.risk_appetite ?? null,
         investmentGoal: personalInformation?.investment_goal ?? null,
         occupation: personalInformation?.occupation ?? null,
        },
      });
    } catch (error: any) {
      await client.query("ROLLBACK");
      if (error?.code === "23505" && (error?.constraint?.includes("phone") || error?.detail?.includes("phone"))) {
        return res.status(400).json({
          error: "This phone number is already registered with another account. Please use a unique phone number.",
        });
      }
      throw error;
    }
  } catch (err: any) {
    console.error(err);
    res.status(500).json({ error: err?.message || "Failed to update profile" });
  } finally {
    client.release();
  }
});

// ---------------------------------------------------------------------------
// POST /api/profile/avatar — upload a new profile picture
// ---------------------------------------------------------------------------
router.post("/avatar", authenticateToken, upload.single("avatar"), async (req: AuthRequest, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const avatarUrl = `/uploads/${req.file.filename}`;

    await pool.query(`UPDATE users SET avatar_url = $1 WHERE id = $2`, [avatarUrl, req.userId]);

    res.json({ avatarUrl });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to upload avatar" });
  }
});

export default router;