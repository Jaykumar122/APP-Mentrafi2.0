import { Router, Request, Response } from "express";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import pool from "../config/db";

const router = Router();

// Strict email format validator (no phone numbers)
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

// SIGNUP (Email only, strict 8-char password)
router.post("/signup", async (req: Request, res: Response) => {
  const client = await pool.connect();
  try {
    const { name, email, password } = req.body;

    if (!name || !email || !password) {
      return res.status(400).json({ error: "All fields are required" });
    }

    const cleanName = String(name).trim();
    const cleanEmail = String(email).trim().toLowerCase();

    if (!cleanName || cleanName.length < 2) {
      return res.status(400).json({ error: "Please enter your full name" });
    }

    if (!EMAIL_REGEX.test(cleanEmail)) {
      return res.status(400).json({ error: "Please provide a valid email address (e.g. name@example.com)" });
    }

    if (typeof password !== "string" || password.length < 8) {
      return res.status(400).json({ error: "Password must be at least 8 characters long" });
    }

    const existing = await pool.query(
      "SELECT id FROM users WHERE LOWER(email) = $1",
      [cleanEmail]
    );
    if (existing.rows.length > 0) {
      return res.status(400).json({ error: "An account with this email already exists" });
    }

    const hashedPassword = await bcrypt.hash(password, 10);

    await client.query("BEGIN");

    try {
      const result = await client.query(
        "INSERT INTO users (name, email, password) VALUES ($1, $2, $3) RETURNING id, name, email",
        [cleanName, cleanEmail, hashedPassword]
      );

      await client.query(
        `INSERT INTO personal_information (user_id, full_name)
         VALUES ($1, $2)
         ON CONFLICT (user_id) DO UPDATE SET
           full_name = EXCLUDED.full_name,
           updated_at = NOW()`,
        [result.rows[0].id, cleanName]
      );

      await client.query("COMMIT");

      return res.status(201).json({
        message: "Account created successfully",
        user: result.rows[0],
      });
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    }
  } catch (err) {
    console.error("Signup error:", err);
    return res.status(500).json({ error: "Server error during registration" });
  } finally {
    client.release();
  }
});

// SIGNIN (Email only, strict 8-char password)
router.post("/signin", async (req: Request, res: Response) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }

    const cleanEmail = String(email).trim().toLowerCase();

    if (!EMAIL_REGEX.test(cleanEmail)) {
      return res.status(400).json({ error: "Please enter a valid email address" });
    }

    if (typeof password !== "string" || password.length < 8) {
      return res.status(400).json({ error: "Password must be at least 8 characters long" });
    }

    const result = await pool.query(
      "SELECT * FROM users WHERE LOWER(email) = $1",
      [cleanEmail]
    );
    if (result.rows.length === 0) {
      return res.status(401).json({ error: "Invalid email or password" });
    }

    const user = result.rows[0];

    const isValid = await bcrypt.compare(password, user.password);
    if (!isValid) {
      return res.status(401).json({ error: "Invalid email or password" });
    }

    const jwtSecret = process.env.JWT_SECRET || "mentrafi-secure-jwt-secret-key-2026";
    const token = jwt.sign(
      { id: user.id, email: user.email },
      jwtSecret,
      { expiresIn: "7d" }
    );

    return res.status(200).json({
      message: "Signed in successfully",
      token,
      user: { id: user.id, name: user.name, email: user.email },
    });
  } catch (err) {
    console.error("Signin error:", err);
    return res.status(500).json({ error: "Server error during sign in" });
  }
});

export default router;