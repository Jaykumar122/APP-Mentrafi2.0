import pool from "./db";

async function migrate() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS users (
      id SERIAL PRIMARY KEY,
      name VARCHAR(100) NOT NULL,
      email VARCHAR(100) UNIQUE NOT NULL,
      password VARCHAR(255) NOT NULL,
      created_at TIMESTAMP DEFAULT NOW()
    )
  `);
  console.log("✅ Users table created");

  await pool.query(`
    CREATE TABLE IF NOT EXISTS funds (
      id SERIAL PRIMARY KEY,
      scheme_code INTEGER UNIQUE NOT NULL,
      name TEXT NOT NULL,
      category TEXT NOT NULL DEFAULT 'Equity',
      subcategory TEXT NOT NULL DEFAULT 'General',
      nav NUMERIC(12, 4),
      nav_date DATE,
      one_year_return NUMERIC(6, 2),
      rating NUMERIC(2, 1) DEFAULT 4.0,
      updated_at TIMESTAMP DEFAULT NOW()
    )
  `);
  console.log("✅ Funds table created");

  await pool.query(`
    ALTER TABLE funds
    ADD COLUMN IF NOT EXISTS plan_type TEXT DEFAULT 'Direct',
    ADD COLUMN IF NOT EXISTS option_type TEXT DEFAULT 'Growth'
  `);
  console.log("✅ Funds table extended with plan_type/option_type");

  await pool.query(`
    ALTER TABLE funds
    ADD COLUMN IF NOT EXISTS fund_house TEXT,
    ADD COLUMN IF NOT EXISTS launch_date DATE
  `);
  console.log("✅ Funds table extended with fund_house/launch_date");

  await pool.query(`
    ALTER TABLE users
    ADD COLUMN IF NOT EXISTS avatar_url TEXT,
    ADD COLUMN IF NOT EXISTS kyc_status VARCHAR(20) DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS tier VARCHAR(30) DEFAULT 'Standard Investor'
  `);
  console.log("✅ Users table extended with profile fields");

  await pool.query(`
    CREATE TABLE IF NOT EXISTS personal_information (
      id SERIAL PRIMARY KEY,
      user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
      full_name VARCHAR(100),
      phone VARCHAR(20),
      date_of_birth DATE,
      gender VARCHAR(20),
      location VARCHAR(120),
      created_at TIMESTAMP DEFAULT NOW(),
      updated_at TIMESTAMP DEFAULT NOW()
    )
  `);
  console.log("✅ Personal information table created");

  await pool.query(`
    ALTER TABLE personal_information
    ADD COLUMN IF NOT EXISTS full_name VARCHAR(100),
    ADD COLUMN IF NOT EXISTS age INTEGER,
    ADD COLUMN IF NOT EXISTS monthly_sip_budget NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS risk_appetite VARCHAR(50),
    ADD COLUMN IF NOT EXISTS investment_goal TEXT,
    ADD COLUMN IF NOT EXISTS occupation VARCHAR(80),
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW()
  `);
  console.log("✅ Personal information table extended with full set of profile fields");

  await pool.query(`
    CREATE UNIQUE INDEX IF NOT EXISTS personal_information_phone_unique_idx
    ON personal_information (phone)
    WHERE phone IS NOT NULL
  `);
  console.log("✅ Unique index on personal_information(phone) ensured");

  const legacyColumns = await pool.query(`
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'users'
      AND column_name IN ('phone', 'date_of_birth', 'gender', 'location')
  `);

  if (legacyColumns.rows.length > 0) {
    await pool.query(`
      INSERT INTO personal_information (user_id, full_name, phone, date_of_birth, gender, location)
      SELECT id, name, phone, date_of_birth, gender, location
      FROM users
      WHERE name IS NOT NULL
         OR phone IS NOT NULL
         OR date_of_birth IS NOT NULL
         OR gender IS NOT NULL
         OR location IS NOT NULL
      ON CONFLICT (user_id) DO UPDATE SET
        full_name = EXCLUDED.full_name,
        phone = EXCLUDED.phone,
        date_of_birth = EXCLUDED.date_of_birth,
        gender = EXCLUDED.gender,
        location = EXCLUDED.location,
        updated_at = NOW()
    `);
    console.log("✅ Personal information backfilled");
  }

  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_funds_category ON funds(category)
  `);
  console.log("✅ Funds indexes created");

  await pool.query(`
    CREATE TABLE IF NOT EXISTS fund_performance (
      id SERIAL PRIMARY KEY,
      scheme_code INTEGER UNIQUE NOT NULL REFERENCES funds(scheme_code) ON DELETE CASCADE,
      return_1m NUMERIC(6,2),
      return_3m NUMERIC(6,2),
      return_6m NUMERIC(6,2),
      return_1y NUMERIC(6,2),
      return_3y NUMERIC(6,2),
      return_5y NUMERIC(6,2),
      updated_at TIMESTAMP DEFAULT NOW()
    )
  `);
  console.log("✅ Fund performance table created");

  await pool.query(`
    ALTER TABLE fund_performance
    ADD COLUMN IF NOT EXISTS return_10y NUMERIC(6,2)
  `);
  console.log("✅ Fund performance table extended with return_10y");

  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_fund_performance_1y ON fund_performance(return_1y)
  `);
  console.log("✅ Fund performance indexes created");

  // -------------------------------------------------------------------------
  // nav_history — raw daily NAV series per fund. Populated from the same
  // MFapi history payload syncScheme() already fetches (no extra API calls).
  // -------------------------------------------------------------------------
  await pool.query(`
    CREATE TABLE IF NOT EXISTS nav_history (
      id SERIAL PRIMARY KEY,
      scheme_code INTEGER NOT NULL REFERENCES funds(scheme_code) ON DELETE CASCADE,
      nav NUMERIC(12,4) NOT NULL,
      nav_date DATE NOT NULL,
      UNIQUE(scheme_code, nav_date)
    )
  `);
  console.log("✅ NAV history table created");

  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_nav_history_scheme_date ON nav_history(scheme_code, nav_date DESC)
  `);
  console.log("✅ NAV history indexes created");

  // -------------------------------------------------------------------------
  // fund_scores — computed suitability score per fund per risk profile.
  // Populated later by the recommendation engine (not built yet).
  // -------------------------------------------------------------------------
  await pool.query(`
    CREATE TABLE IF NOT EXISTS fund_scores (
      id SERIAL PRIMARY KEY,
      scheme_code INTEGER NOT NULL REFERENCES funds(scheme_code) ON DELETE CASCADE,
      risk_profile TEXT NOT NULL,
      score NUMERIC(5,2) NOT NULL,
      computed_at TIMESTAMP DEFAULT NOW(),
      UNIQUE(scheme_code, risk_profile)
    )
  `);
  console.log("✅ Fund scores table created");

  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_fund_scores_profile ON fund_scores(risk_profile, score DESC)
  `);
  console.log("✅ Fund scores indexes created");

  // -------------------------------------------------------------------------
  // fund_descriptions — plain-English explanations per fund. MFapi doesn't
  // provide this; needs manual curation or AI-generated content later.
  // -------------------------------------------------------------------------
  await pool.query(`
    CREATE TABLE IF NOT EXISTS fund_descriptions (
      id SERIAL PRIMARY KEY,
      scheme_code INTEGER UNIQUE NOT NULL REFERENCES funds(scheme_code) ON DELETE CASCADE,
      summary TEXT,
      investment_strategy TEXT,
      suitable_for TEXT,
      updated_at TIMESTAMP DEFAULT NOW()
    )
  `);
  console.log("✅ Fund descriptions table created");

  // -------------------------------------------------------------------------
  // recommendation_history — log of what the AI advisor recommended to whom
  // and when. Populated once the AI advisor chat feature exists.
  // -------------------------------------------------------------------------
  await pool.query(`
    CREATE TABLE IF NOT EXISTS recommendation_history (
      id SERIAL PRIMARY KEY,
      user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      scheme_code INTEGER NOT NULL REFERENCES funds(scheme_code) ON DELETE CASCADE,
      reason TEXT,
      risk_profile_at_time TEXT,
      created_at TIMESTAMP DEFAULT NOW()
    )
  `);
  console.log("✅ Recommendation history table created");

  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_recommendation_history_user ON recommendation_history(user_id, created_at DESC)
  `);
  console.log("✅ Recommendation history indexes created");

  // -------------------------------------------------------------------------
  // user_portfolio — actual holdings: units owned, avg purchase NAV, amount
  // invested per user per fund. Backbone for the Portfolio screen (not built
  // yet).
  // -------------------------------------------------------------------------
  await pool.query(`
    CREATE TABLE IF NOT EXISTS user_portfolio (
      id SERIAL PRIMARY KEY,
      user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      scheme_code INTEGER NOT NULL REFERENCES funds(scheme_code) ON DELETE CASCADE,
      units NUMERIC(14,4) NOT NULL DEFAULT 0,
      avg_purchase_nav NUMERIC(12,4),
      invested_amount NUMERIC(14,2) NOT NULL DEFAULT 0,
      created_at TIMESTAMP DEFAULT NOW(),
      updated_at TIMESTAMP DEFAULT NOW(),
      UNIQUE(user_id, scheme_code)
    )
  `);
  console.log("✅ User portfolio table created");

  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_user_portfolio_user ON user_portfolio(user_id)
  `);
  console.log("✅ User portfolio indexes created");

  // -------------------------------------------------------------------------
  // sips — recurring SIP mandates the user has set up for a fund. Drives the
  // SIP dashboard screen (active/paused lists, progress, next installment).
  // -------------------------------------------------------------------------
  await pool.query(`
    CREATE TABLE IF NOT EXISTS sips (
      id SERIAL PRIMARY KEY,
      user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      scheme_code INTEGER NOT NULL REFERENCES funds(scheme_code) ON DELETE CASCADE,
      monthly_amount NUMERIC(12,2) NOT NULL,
      installment_day INTEGER NOT NULL DEFAULT 1,
      total_installments INTEGER NOT NULL DEFAULT 60,
      completed_installments INTEGER NOT NULL DEFAULT 0,
      status VARCHAR(20) NOT NULL DEFAULT 'active',
      start_date DATE NOT NULL DEFAULT CURRENT_DATE,
      next_installment_date DATE NOT NULL,
      created_at TIMESTAMP DEFAULT NOW(),
      updated_at TIMESTAMP DEFAULT NOW()
    )
  `);
  console.log("✅ SIPs table created");

  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_sips_user ON sips(user_id)
  `);
  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_sips_next_installment ON sips(status, next_installment_date)
  `);
  console.log("✅ SIPs indexes created");

  // -------------------------------------------------------------------------
  // sip_installments — log of each executed installment (real purchase),
  // used to compute a SIP's actual invested amount / current value / returns
  // instead of estimating them.
  // -------------------------------------------------------------------------
  await pool.query(`
    CREATE TABLE IF NOT EXISTS sip_installments (
      id SERIAL PRIMARY KEY,
      sip_id INTEGER NOT NULL REFERENCES sips(id) ON DELETE CASCADE,
      amount NUMERIC(12,2) NOT NULL,
      nav NUMERIC(12,4) NOT NULL,
      units NUMERIC(14,4) NOT NULL,
      installment_date DATE NOT NULL,
      created_at TIMESTAMP DEFAULT NOW()
    )
  `);
  console.log("✅ SIP installments table created");

  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_sip_installments_sip ON sip_installments(sip_id, installment_date DESC)
  `);
  console.log("✅ SIP installments indexes created");

  // -------------------------------------------------------------------------
  // transactions — records every buy, redeem (sell), and switch action
  // along with NAV, units, rupee amount, payment method, status, and target fund.
  // -------------------------------------------------------------------------
  await pool.query(`
    CREATE TABLE IF NOT EXISTS transactions (
      id SERIAL PRIMARY KEY,
      user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      scheme_code INTEGER NOT NULL REFERENCES funds(scheme_code) ON DELETE CASCADE,
      type VARCHAR(20) NOT NULL, -- 'BUY', 'REDEEM', 'SWITCH'
      amount NUMERIC(14,2) NOT NULL,
      units NUMERIC(14,4) NOT NULL,
      nav NUMERIC(12,4) NOT NULL,
      status VARCHAR(20) NOT NULL DEFAULT 'COMPLETED', -- 'PENDING', 'COMPLETED', 'FAILED'
      payment_method VARCHAR(50), -- 'UPI', 'NET_BANKING', 'DEBIT_CARD', 'BANK_TRANSFER', 'PORTFOLIO_SWITCH'
      target_scheme_code INTEGER REFERENCES funds(scheme_code) ON DELETE SET NULL,
      target_units NUMERIC(14,4),
      target_nav NUMERIC(12,4),
      created_at TIMESTAMP DEFAULT NOW()
    )
  `);
  console.log("✅ Transactions table created");

  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_transactions_user_date ON transactions(user_id, created_at DESC)
  `);
  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_transactions_scheme ON transactions(scheme_code)
  `);
  console.log("✅ Transactions indexes created");

  process.exit(0);
}

migrate();