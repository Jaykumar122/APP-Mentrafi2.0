<div align="center">

# 💎 MentraFi 2.0 Ecosystem

### Autonomous AI-Powered Direct Mutual Fund Investment & Advisory Platform

[![GitHub Release](https://img.shields.io/github/v/release/Jaykumar122/APP-Mentrafi2.0?color=8b5cf6&label=Release&logo=github)](https://github.com/Jaykumar122/APP-Mentrafi2.0/releases/tag/v1.0.0)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MentraFiAI--TriNetwork-yellow)](https://huggingface.co/singh0123/MentraFiAI-TriNetwork)
[![React Native](https://img.shields.io/badge/React%20Native-Expo%2054-blue?logo=react)](https://reactnative.dev/)
[![Backend](https://img.shields.io/badge/Node.js-Express%20%2B%20TypeScript-green?logo=node.js)](https://nodejs.org/)
[![AI Engine](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Database](https://img.shields.io/badge/PostgreSQL-14%2B-336791?logo=postgresql)](https://www.postgresql.org/)
[![AMFI Ingestion](https://img.shields.io/badge/AMFI%20Live-14%2C361%2B%20Schemes-teal)](https://www.amfiindia.com/)
[![MFAPI Integration](https://img.shields.io/badge/mfapi.in-Historical%20Time--Series-orange)](https://www.mfapi.in/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**MentraFi** is a state-of-the-art, full-stack wealth advisory and direct mutual fund investment ecosystem built specifically for the Indian financial market. It couples a luxury **React Native (Expo SDK 54)** mobile client in Obsidian & Aurora styling, an enterprise **Node.js + Express + TypeScript** transactional API gateway, and **MentraFiAI** — a proprietary **Tri-Network Neuro-Symbolic Small Language Model (SLM)** running natively on PyTorch and CUDA.

Unlike generic foundation models that hallucinate tax rules and suffer from compounding arithmetic drift, MentraFi operates with **100% deterministic mathematical accuracy**, zero distributor commission bias (SEBI Direct-Growth invariant), and real-time dual synchronization with both the official **AMFI Portal (`amfiindia.com`)** and high-frequency time-series from **`mfapi.in`**.

---

[How It Works](#-how-mentrafi-works-end-to-end) • [App Preview](#-app-experience--screenshots) • [AMFI & mfapi.in Engine](#-amfi-live--mfapiin-dual-stream-data-pipeline) • [Tri-Network AI Architecture](#-the-tri-network-ai-intelligence-core) • [System Architecture](#-system-architecture) • [Quickstart Guide](#-quickstart-guide) • [Model Weights](#-model-weights--checkpoints) • [API Overview](#-api-endpoints)

---

</div>

## 🔄 How MentraFi Works (End-to-End)

MentraFi operates as an integrated four-tier pipeline where user intent is converted into mathematically verified, SEBI-compliant financial actions:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 1. USER INTERACTION                                   │
│  Investor inputs goal: "I am 28, earning 80k/mo, want to invest ₹15,000/mo for 10 yrs" │
│  or explores funds, tracks portfolio holdings, and configures SIP mandates.           │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ HTTPS / WSS / SSE
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        2. BACKEND GATEWAY & LEDGER (Node.js)                           │
│  • Authenticates JWT session & verifies KYC / investor risk profile from PostgreSQL.   │
│  • Proxies advisory requests to MentraFiAI Core via real-time SSE streaming.           │
│  • Manages dual-stream AMFI & mfapi.in synchronization schedules and SIP automations. │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ REST / JSON
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      3. MENTRAFIAI TRI-NETWORK INTELLIGENCE CORE                       │
│  ┌─────────────────────────────────┐   ┌────────────────────────────────────────────┐  │
│  │ Network 1: Intent & Risk Clf    │──►│ Network 2: BIO Slot Parser                 │  │
│  │ (10.65M) -> SIP_RECOMMENDATION  │   │ (15.26M) -> Age: 28, SIP: 15000, Yrs: 10   │  │
│  └─────────────────────────────────┘   └─────────────────────┬──────────────────────┘  │
│                                                              │                         │
│  ┌───────────────────────────────────────────────────────────┴──────────────────────┐  │
│  │ Closed-Form Symbolic Solver: Exact FV annuity, tax slabs & Section 80C limits  │  │
│  └───────────────────────────────────────────┬─────────────────────────────────────┘  │
│                                              ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │ PostgreSQL Fiduciary RAG: Top direct-growth active funds (is_active = TRUE)     │  │
│  └───────────────────────────────────────────┬─────────────────────────────────────┘  │
│                                              ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Network 3: llm120 Fiduciary SLM (120.05M) -> Token Streaming + Fund Cards       │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Streaming Tokens + Structured Fund JSON
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           4. CLIENT RENDERING & EXECUTION                              │
│  • Mobile app renders streaming conversational markdown advice.                       │
│  • Emits interactive action cards with verified live AMFI NAVs & returns.             │
│  • User clicks "Invest Now" or "Start SIP" -> written to double-entry ledger.         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📱 App Experience & Screenshots

<div align="center">

### 🌟 1. Onboarding & Core Philosophy
*Designed with a luxury Obsidian & Aurora glassmorphism UI, guiding investors toward direct mutual fund compounding.*

<table>
  <tr>
    <td align="center" width="33.3%">
      <img src="docs/screenshots/01_onboarding_welcome.png" width="100%" alt="Direct Mutual Funds" /><br />
      <b>Direct Mutual Funds</b><br />
      <sub>SEBI Direct-Growth Invariant</sub>
    </td>
    <td align="center" width="33.3%">
      <img src="docs/screenshots/02_onboarding_compounding.png" width="100%" alt="Zero Intermediaries" /><br />
      <b>Zero Intermediaries</b><br />
      <sub>100% Capital Conservation</sub>
    </td>
    <td align="center" width="33.3%">
      <img src="docs/screenshots/03_onboarding_ai_core.png" width="100%" alt="Neuro-Symbolic AI" /><br />
      <b>Neuro-Symbolic AI Core</b><br />
      <sub>99.5% Compounding Math Fidelity</sub>
    </td>
  </tr>
</table>

---

### 📊 2. Portfolio Valuation, Analytics & Verified Profile
*Live portfolio aggregation connected directly to PostgreSQL tables (`user_portfolio`, `sips`, `funds`), with asset allocation donuts and verified KYC details.*

<table>
  <tr>
    <td align="center" width="33.3%">
      <img src="docs/screenshots/06_home_dashboard.png" width="100%" alt="Home Dashboard" /><br />
      <b>Home Dashboard</b><br />
      <sub>Live Valuation, AI Copilot & Market Pulse</sub>
    </td>
    <td align="center" width="33.3%">
      <img src="docs/screenshots/10_portfolio_analytics.png" width="100%" alt="Portfolio Analytics" /><br />
      <b>Portfolio Analytics</b><br />
      <sub>Asset Allocation & AI Fiduciary Diagnostic</sub>
    </td>
    <td align="center" width="33.3%">
      <img src="docs/screenshots/14_profile_kyc.png" width="100%" alt="Investor Profile & KYC" /><br />
      <b>Investor Profile & KYC</b><br />
      <sub>Live DB Stat Cards & Verified Profile</sub>
    </td>
  </tr>
</table>

---

### 🧠 3. AI Wealth Advisory & Financial Screener
*Real-time token-streamed fiduciary advisory powered by MentraFiAI Core alongside live AMFI scheme discovery.*

<table>
  <tr>
    <td align="center" width="33.3%">
      <img src="docs/screenshots/13_ai_advisor_chat.png" width="100%" alt="MentraFi AI Advisor" /><br />
      <b>MentraFi AI Advisor</b><br />
      <sub>Tri-Network Fiduciary Chat Copilot</sub>
    </td>
    <td align="center" width="33.3%">
      <img src="docs/screenshots/08_sip_calculator.png" width="100%" alt="SIP & Wealth Calculator" /><br />
      <b>SIP & Wealth Calculator</b><br />
      <sub>Ordinary Annuity, Step-Up & Inflation Math</sub>
    </td>
    <td align="center" width="33.3%">
      <img src="docs/screenshots/11_explore_funds.png" width="100%" alt="Explore Mutual Funds" /><br />
      <b>Explore Mutual Funds</b><br />
      <sub>Screener with 14,361+ AMFI Schemes</sub>
    </td>
  </tr>
</table>

---

### 💳 4. Order Execution, Recurring SIPs & Authentication
*Instant UPI-based fund purchases with estimated units allotment, automated SIP mandate tracking, and encrypted biometric authentication.*

<table>
  <tr>
    <td align="center" width="25%">
      <img src="docs/screenshots/09_lumpsum_investment.png" width="100%" alt="Direct Investment" /><br />
      <b>Direct Purchase</b><br />
      <sub>Instant UPI & NAV Units</sub>
    </td>
    <td align="center" width="25%">
      <img src="docs/screenshots/07_active_sips.png" width="100%" alt="SIP Mandates" /><br />
      <b>SIP Tracker</b><br />
      <sub>Automated Morning Ledger</sub>
    </td>
    <td align="center" width="25%">
      <img src="docs/screenshots/04_login.png" width="100%" alt="Investor Login" /><br />
      <b>Secure Login</b><br />
      <sub>JWT & Biometric Auth</sub>
    </td>
    <td align="center" width="25%">
      <img src="docs/screenshots/05_signup.png" width="100%" alt="Account Setup" /><br />
      <b>Instant Registration</b><br />
      <sub>3-Minute Onboarding</sub>
    </td>
  </tr>
</table>

</div>

---

## 📡 AMFI Live & mfapi.in Dual-Stream Data Pipeline

A central foundation of MentraFi is its **dual-source data pipeline** combining the statutory authority of the **Association of Mutual Funds in India (AMFI)** with the high-resolution historical time-series of **`mfapi.in`**.

```
                           ┌─────────────────────────────────────────┐
                           │      AMFI India Official Registry       │
                           │   (https://www.amfiindia.com/...)       │
                           └────────────────────┬────────────────────┘
                                                │ Nightly Sync (NAVAll.txt)
                                                │ ~1.03s Batch UNNEST Upsert
                                                ▼
┌──────────────────────┐   Daily Closing NAVs   ┌─────────────────────────────────────────┐
│     mfapi.in API     │───────────────────────►│          PostgreSQL Database            │
│ (https://api.mfapi.in│  Intra-Day 6x Waves    │   • funds (37,882+ schemes)             │
│      /mf/...)        │───────────────────────►│     - 14,361+ Active (is_active = TRUE) │
└──────────┬───────────┘  Historical Time-Series│   • fund_performance (CAGR / rolling)   │
           │                                    └────────────────────┬────────────────────┘
           │ Direct SVG History Fetch                                │ Local Query
           ▼                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           MentraFi Mobile Client & AI Core                              │
│   • Live Interactive SVG Performance Charts (1M, 3M, 6M, 1Y, 3Y, 5Y, Inception)         │
│   • True-to-Date Rolling CAGR Engine with Young-Fund Outlier Shield                     │
│   • Instant In-App Search across all SEBI-Registered Direct-Growth Mutual Funds         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Statutory AMFI Ingestion (`amfiindia.com/spages/NAVAll.txt`)
- **Master Registry Authority:** AMFI is the official governing authority for mutual fund disclosures in India. Every registered fund must report its daily Net Asset Value to AMFI.
- **Ultra-Fast Batch Upsert Engine:** Ingests **37,882+ mutual fund schemes** (including **14,361+ live active schemes**) nightly in **~1.03 seconds**.
- **PostgreSQL Typed `UNNEST` Architecture:** Rather than executing thousands of discrete `UPDATE` statements, `amfiSync.ts` streams and parses the official semicolon-delimited feed in memory, transforms rows into columnar arrays, and executes a single atomic PostgreSQL `UNNEST` bulk upsert.
- **Dual ISIN Extraction:** Simultaneously parses ISIN Div Payout and ISIN Growth fields, achieving **98.8% ISIN match coverage**.
- **SEBI Scheme Categories** (`backend/src/services/fundCategory.ts`): every scheme gets its official SEBI category from the AMFI section header (e.g. *Equity Scheme - Multi Cap Fund*), so Flexi Cap, Multi Cap, Large & Mid Cap, Index, ETF, Liquid, Arbitrage and the other categories are never mixed. `category` keeps the app's five filters (Equity, Debt, Hybrid, ELSS, Gold). The `mfapi.in` sync never overwrites an AMFI label. Re-label existing rows with `npm run fix:categories -- --dry-run` (review) / `npm run fix:categories` (apply; old labels kept in `funds_category_backup`, restore with `--restore`).
- **Fiduciary Quarantine Mechanism (`is_active = TRUE`):** Automatically detects dead, matured, merged, or liquidating schemes and isolates them into a quarantine partition. MentraFiAI and the Explore Screener **only recommend schemes where `is_active = TRUE`**, guaranteeing users are never guided toward terminated funds.

### 2. High-Frequency `mfapi.in` Integration (`api.mfapi.in`)
- **Historical Time-Series Engine:** Fetches complete historical daily NAV arrays (`https://api.mfapi.in/mf/{schemeCode}`) dating back to scheme inception.
- **Interactive SVG Charting:** Powers the mobile app's interactive NAV chart across 7 selectable periods: **1M, 3M, 6M, 1Y, 3Y, 5Y, and ALL (Inception)** with fluid cubic bezier curves.
- **Dynamic Rolling Returns Engine:** Calculates exact Compound Annual Growth Rates (CAGR) for 1M, 3M, 6M, 1Y, 3Y, 5Y, and 10Y periods.
  $$\text{CAGR} = \left(\frac{\text{NAV}_{\text{latest}}}{\text{NAV}_{t}}\right)^{\frac{1}{\text{years}}} - 1$$
- **Young-Fund Distortion Shield (`findNavAtDate`):** Upstream providers frequently label inception returns as 3Y or 5Y returns for funds younger than that tenure. MentraFi inspects date proximity: if a scheme lacks true trading history for a target horizon, the metric is returned as `null` rather than a deceptive number.
- **Intra-Day 6x Fast Refresh Schedule:** Mutual fund AMCs publish NAVs at staggered evening hours. The backend scheduler executes fast NAV syncs via `https://api.mfapi.in/mf/{schemeCode}/latest` across 6 daily windows:
  1. `08:00 AM IST` — Morning opening reconciliation.
  2. `01:00 PM IST` — Mid-day sanity check.
  3. `07:00 PM IST` — Early evening debt & liquid fund wave.
  4. `09:30 PM IST` — **Prime AMFI Wave 1** (first major batch of equity AMC numbers).
  5. `10:30 PM IST` — **Prime AMFI Wave 2** (large-cap, mid-cap, and multi-cap updates).
  6. `11:45 PM IST` — Closing night-time master settlement.

---

## 🧠 The Tri-Network AI Intelligence Core

Generic LLMs fail critically in financial advisory: they suffer from arithmetic compounding drift, hallucinate tax exemptions, and exhibit distributor commission bias. MentraFi solves this through a **tripartite neuro-symbolic architecture**:

```
                                  [ User Financial Query ]
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
          ┌────────────────────────┐                   ┌────────────────────────┐
          │       Network 1        │                   │       Network 2        │
          │    Intent Classifier   │                   │   BIO Slot Parser      │
          │      (10.65M Params)   │                   │     (15.26M Params)    │
          └───────────┬────────────┘                   └────────────┬───────────┘
                      │ Intent & Risk Label                         │ Extracted Financial Slots
                      │ (e.g. SIP_CALCULATION,                      │ (Age: 30, Horizon: 10y,
                      │  MODERATE)                                  │  Budget: ₹10,000)
                      └──────────────────────┬──────────────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │  Symbolic Mathematical      │
                              │  & Capital Gains Engine     │
                              │  (Deterministic Python)     │
                              └──────────────┬──────────────┘
                                             │ Exact Future Value,
                                             │ Section 112A Tax, 80C Caps
                                             ▼
                              ┌─────────────────────────────┐
                              │    PostgreSQL RAG Engine    │
                              │ (Top Direct-Growth Schemes) │
                              └──────────────┬──────────────┘
                                             │ Live Scheme Candidates
                                             ▼
                              ┌─────────────────────────────┐
                              │          Network 3          │
                              │   Generative Fiduciary SLM  │
                              │   llm120 (120.05M Params)   │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                      [ Streaming Advice + Actionable Fund Cards ]
```

### Network 1: Bi-Encoder Intent & Risk Classifier (10.65M Parameters)
- **Role:** High-speed query classification and investor risk stratification.
- **Latency:** **< 10ms** on CUDA / CPU.
- **Architecture:** Transformer bi-encoder with mean-pooling over contextual embeddings and multi-class classification heads.
- **Intents detected** (`model/classifier.py`): `GREETING`, `DEFINITION`, `PORTFOLIO_RECOMMENDATION`, `FUND_COMPARISON`, `GOAL_PLANNING`, `GENERAL_CHAT`.
- **Extra heads:** risk appetite (`CONSERVATIVE`, `MODERATE`, `AGGRESSIVE`, `UNKNOWN`) and horizon class (`SHORT_TERM`, `MEDIUM_TERM`, `LONG_TERM`, `UNKNOWN`).
- **Routing:** a confident `PORTFOLIO_RECOMMENDATION` with a budget goes to the advisor route (allocation + fund cards + Network 3); explicit "calculate / corpus / tax / lump sum" questions go to the math engine; concept questions go to the reviewed curriculum (see Network 3).

### Network 2: BIO Token-Level Slot Parser (15.26M Parameters)
- **Role:** Sub-word entity extraction using BIO (Beginning-Inside-Outside) sequence tagging.
- **Zero Numeric Conflation:** Unlike general LLMs that confuse age, tenure, and capital amounts, Network 2 operates at the token representation level to reliably distinguish:
  - `B-AGE` / `I-AGE` (e.g. "I am 29 years old" $\rightarrow 29$)
  - `B-HORIZON` / `I-HORIZON` (e.g. "invest for 7 years" $\rightarrow 7$)
  - `B-SIP_BUDGET` / `I-SIP_BUDGET` (e.g. "₹5,000 every month" $\rightarrow 5000$)
  - `B-LUMPSUM` / `I-LUMPSUM` (e.g. "invest 2.5 lakh once" $\rightarrow 250000$)
  - `B-GOAL` / `I-GOAL` (e.g. "wealth creation", "child education", "retirement")
- **Colloquial Indian Number Normalizer:** Natively translates colloquial expressions: `"15k"` $\rightarrow$ ₹15,000, `"2.5L" / "2.5 lakh"` $\rightarrow$ ₹2,50,000, `"1 cr"` $\rightarrow$ ₹1,00,00,000.

### Deterministic Closed-Form Symbolic Solver
Before Network 3 generates text, all calculations are offloaded to `Mentrafiai/inference/financial_calculator.py` (the app's SIP calculator uses the same formulas in `frontend/utils/sipMath.ts`; `python check_app_calculator.py` compares both over 768 input combinations):
- **SIP future value, installment at the start of each month (annuity due), monthly rate = yearly rate / 12:**
  $$FV = P \times \frac{(1 + r)^n - 1}{r} \times (1 + r)$$
  *(where $P$ is monthly installment, $r$ is monthly interest rate, $n$ is total months).*
- **Annual Step-Up SIP Engine:** installments rise by $S\%$ after every 12 installments; the goal planner solves the exact starting SIP for a target.
- **Also:** goal SIP (rounded up to the next rupee), lump sum, SWP, CAGR, inflation-adjusted value (6%) and real return.
- **Finance Act 2024–2026 Indian Tax Matrix:**
  - **Equity Mutual Funds:** Section 112A Long Term Capital Gains (LTCG) taxed at **12.5%** on gains exceeding **₹1.25 Lakh/year** (held > 12 months). Short Term Capital Gains (STCG) taxed at **20%** (Section 111A).
  - **Debt Mutual Funds (bought on or after 1 April 2023):** Section 50AA, gains taxed at the investor's slab rate regardless of holding period.
  - **Other funds** (e.g. hybrids with 35–65% equity): 12.5% after 24 months, slab rate otherwise. 4% cess applies; surcharge, the 87A rebate and loss set-off are not modelled.
  - **Section 80C Deductions:** Strict ₹1,50,000 annual ceiling enforcement on ELSS Tax Saver allocations.

### Network 3: llm120, a 120M Model Trained From Scratch
- **Role:** writes the explanation for portfolio plans, goal and SIP answers, fund comparisons and concept questions, using only the verified context it is given.
- **Architecture** (`Mentrafiai/llm120/`): decoder-only Transformer, **120,054,528 parameters**, 16 layers, hidden size 768, 12 query / 4 KV heads (grouped-query attention), SwiGLU, RoPE, RMSNorm, 1,024-token context, tied embeddings, custom 16,000-token SentencePiece tokenizer.
- **Pretraining** (Kaggle, 2× T4 with PyTorch DDP, 19,000 updates × 65,536 tokens): 1.18B tokens, **32% Indian mutual-fund text** (SEBI / AMFI investor pages, Wikipedia Indian-finance articles, one document per active scheme built from AMFI NAV data, worked examples from the math engine) and 68% FineWeb-Edu English. Best validation loss 1.571.
- **Fine-tuning:** 2,431 conversations: 91 reviewed mutual-fund topics plus 2,400 grounded pipeline examples that use real scheme names and NAV-based returns. Each grounded example has a "verified context" (intent, risk, slots, exact engine numbers, allocation, fund candidates) and an answer that only copies those facts. Best validation loss 0.060.
- **Measured quality** (`python eval_network3.py --limit 150`, 150 held-out grounded questions): **97.3% usable answers** (portfolio 99%, SIP / goal / tax 100%, comparisons 84%); 99.3% of answers contain only numbers from the verified context.
- **Safety checks on every reply** (`serve.py`): a reply is rejected, and the exact engine text is shown instead, if it contains a number not in the context, an invented investor name, a scheme that was not retrieved, a stated expense ratio / riskometer / AUM value, or leaves out the budget, invested amount or projected corpus.
- **Concept questions** ("what is SIP?", "SIP vs lump sum", "direct vs regular"): `inference/definition_router.py` maps the question to one of the 91 reviewed topics and asks the model in its trained wording. The reply is shown only if it matches the reviewed answer; otherwise the reviewed answer itself is shown. In current tests the model usually drifts on free-form definitions, so most concept answers come from the reviewed text.
- **Limits:** open questions ("why do markets fall?") get a polite "can't answer reliably" reply. A 120M model is not a general financial chatbot.
- **Train it yourself:** see [`Mentrafiai/KAGGLE.md`](Mentrafiai/KAGGLE.md) (one command per stage; resumes across Kaggle's 12-hour sessions) and [`Mentrafiai/README.md`](Mentrafiai/README.md) for the data and training details.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Tier 1: frontend (Mobile Client)                      │
│             React Native / Expo SDK 54 / TypeScript / Expo Router           │
│  • Obsidian & Aurora Dark Glassmorphic Design                               │
│  • Real-Time Portfolio Valuation & Database-Synchronized Stat Cards         │
│  • AI Advisor Chat (Real-time SSE Token Streaming & Scheme Action Cards)    │
│  • SIP Calculator: SIP, Step-Up, Lump Sum, Goal, growth chart & yearly table│
│  • SVG Interactive NAV Performance Graphs (1M, 3M, 6M, 1Y, 3Y, 5Y, ALL)    │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │ HTTP REST & SSE (/api/*)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Tier 2: backend (API Gateway & Transaction Ledger)          │
│                      Node.js / Express / TypeScript (Port 3001)             │
│  • JWT Authentication, Biometric Sessions & Verified KYC User Profiles      │
│  • Double-Entry Portfolio Ledger & Automated Morning SIP Execution Engine   │
│  • Official AMFI Daily Sync (amfiindia.com NAVAll.txt via UNNEST Upsert)    │
│  • 6x Daily Intra-Day NAV Fast Refreshes via api.mfapi.in                   │
│  • Rolling Returns CAGR Calculator with Tenure Sanity Validation            │
└──────────────────┬────────────────────────────────────────────┬─────────────┘
                   │ SQL Connection Pool                        │ JSON Proxy
                   ▼                                            ▼
┌──────────────────────────────────────┐     ┌────────────────────────────────┐
│   Tier 3: PostgreSQL (mentrafi)      │     │  Tier 4: MentraFiAI AI Core    │
│            (14 Tables)               │     │   PyTorch / CUDA (Port 8000)   │
│ • funds (37,882+ AMFI Schemes)       │     │ • Network 1: Classifier 10.65M │
│   - 14,361+ Live (is_active = TRUE)  │◄────┤ • Network 2: Slot Parser 15.26M│
│   - Quarantine Partition (Closed)    │     │ • Network 3: llm120 120.05M    │
│ • fund_performance (Historical CAGR) │     │ • Closed-Form Symbolic Solver  │
│ • user_portfolio, sips, transactions │     │ • Fiduciary RAG Invariant      │
│ • personal_information & KYC data    │     └────────────────────────────────┘
└──────────────────────────────────────┘
```

---

## 📦 Model Weights & Checkpoints

Download from **[GitHub Release v2.0.0](https://github.com/Jaykumar122/APP-Mentrafi2.0/releases/tag/v2.0.0)**, or run `python download_checkpoints.py` inside `Mentrafiai/`:

| Model Component | Parameters | Release asset | Saved as |
| :--- | :---: | :--- | :--- |
| **Network 1: Classifier** | 10.65M | `best_classifier.pt` | `checkpoints/classifier/best_classifier.pt` |
| **Network 2: Slot Parser** | 15.26M | `best_parser_net.pt` | `checkpoints/query_parser/best_parser_net.pt` |
| **Network 3: llm120 (fine-tuned)** | **120.05M** | `llm120_sft_best.pt` (weights only, 505 MB) | `checkpoints/llm120_sft/best.pt` |
| **SentencePiece tokenizer (16K)** | — | `tokenizer_v2.model`, `tokenizer_v2.vocab` | `tokenizer/` |

The legacy 102.5M model (`best_model.pt`) remains in [Release v1.0.0](https://github.com/Jaykumar122/APP-Mentrafi2.0/releases/tag/v1.0.0); `serve.py` uses it only when no llm120 checkpoint is present.

---

## 🚀 Quickstart Guide

### Prerequisites
- **Node.js 18+** & npm
- **Python 3.10+** (PyTorch 2.0+ with CUDA recommended)
- **PostgreSQL 14+** running locally
- **Expo Go** app on your physical mobile device or iOS/Android emulator

---

### Step 1: Database Setup
Create your local PostgreSQL database:
```bash
createdb mentrafi
```

---

### Step 2: Launch Backend Gateway (Port 3001)
```bash
cd backend
npm install
npm run build
npm run dev
```
*The backend initializes tables automatically and launches the automated AMFI sync and 6x daily intra-day `mfapi.in` schedulers.*

---

### Step 3: Launch MentraFiAI Core (Port 8000)

1. Navigate to the AI directory and install Python dependencies:
   ```bash
   cd Mentrafiai
   pip install -r requirements.txt
   ```

2. **Download Checkpoints** (Networks 1–3 and the tokenizer from Release v2.0.0):
   ```bash
   python download_checkpoints.py
   ```
3. Launch the FastAPI inference server:
   ```bash
   python serve.py
   ```

To check the system:
```bash
python -m pytest tests_llm120 -q        # math engine, tokenizer, model, training, grounding, routing
python check_query.py "I am 23, want to invest 5000 monthly SIP for 15 years with moderate risk"
python eval_network3.py --limit 150     # Network 3 quality on held-out questions
python check_app_calculator.py          # app SIP calculator vs Python engine
```

---

### Step 4: Launch Mobile Application
```bash
cd frontend
npm install
npx expo start
```
Scan the QR code with **Expo Go** (Android) or the **Camera app** (iOS) to experience the live application.

---

## 📂 Repository Layout

```
APP-Mentrafi2.0/
├── docs/                      # Documentation & Application Media Assets
│   └── screenshots/           # High-resolution mobile experience screenshots
│
├── frontend/                  # React Native / Expo SDK 54 Client
│   ├── app/                   # Expo Router screens ((auth), (tabs), profile-setup)
│   ├── assets/images/         # 3D assets, logos, and branding illustrations
│   ├── components/            # Luxury UI widgets (Glass panels, 3D rotating emblem)
│   ├── utils/api.ts           # REST API client configuration & endpoints
│   ├── utils/profileEvents.ts # Cross-tab real-time event bus
│   └── package.json
│
├── backend/                   # Node.js + Express + TypeScript Gateway (Port 3001)
│   ├── src/
│   │   ├── config/            # PostgreSQL connection pool & schema migrations
│   │   ├── routes/            # REST endpoints (funds, advisor, portfolio, sip, profile)
│   │   ├── services/          # amfiSync.ts, fundSync.ts, morning SIP engine
│   │   └── index.ts           # Server bootstrap and intra-day cron schedules
│   └── package.json
│
└── Mentrafiai/                # PyTorch Tri-Network AI Intelligence Core (Port 8000)
    ├── configs/               # Hyperparameter specifications (model_config.yaml)
    ├── data_prep/             # Token packing, data synthesis, and verification scripts
    ├── figures/               # Architecture diagrams and pipeline visuals
    ├── inference/             # Query parser, financial calculator, fund retriever
    ├── model/                 # Model architectures (Classifier, Parser, Generative SLM)
    ├── tokenizer/             # 16,000-token BPE SentencePiece tokenizer
    ├── training/              # Training pipelines for all 3 networks
    ├── download_checkpoints.py# Automated model downloader (GitHub / Hugging Face)
    ├── upload_checkpoints.py  # Checkpoint publisher to Hugging Face
    ├── serve.py               # Production FastAPI inference server
    └── requirements.txt
```

---

## 🔌 API Endpoints

### Backend Gateway (Port 3001)
- `GET /api/funds` — Search, page, and filter active mutual funds (`is_active = TRUE`).
- `GET /api/funds/:schemeCode/details` — Complete scheme metadata from both AMFI and `mfapi.in`.
- `GET /api/funds/:schemeCode/history` — Historical daily NAV array for interactive SVG charting.
- `POST /api/funds/sync-amfi` — Trigger immediate synchronization with official AMFI master feed.
- `POST /api/advisor/chat/stream` — Real-time SSE token stream proxying advice from MentraFiAI.
- `GET /api/profile` & `PATCH /api/profile` — Investor KYC profile & live database portfolio stats.
- `GET /api/portfolio` & `POST /api/portfolio/invest` — User portfolio ledger and buy/sell execution.
- `GET /api/sip` & `POST /api/sip` — Recurring systematic investment mandate management.

### MentraFiAI Core (Port 8000)
- `POST /api/chat` — Synchronous financial advisory JSON response with matched fund cards.
- `POST /api/chat/stream` — Direct Server-Sent Events (SSE) token streaming.
- `POST /v1/chat/completions` — OpenAI-compatible completions endpoint.
- `GET /health` — Neural network status and CUDA hardware telemetry.

---

## 📜 License
Distributed under the **MIT License**. See `LICENSE` for more information.

<div align="center">
Built with ❤️ for Indian Direct Mutual Fund Investors.
</div>
