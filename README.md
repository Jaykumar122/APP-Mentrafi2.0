<div align="center">

# 💎 MentraFi 2.0 Ecosystem

### Autonomous AI-Powered Direct Mutual Fund Investment & Advisory Platform

[![GitHub Release](https://img.shields.io/github/v/release/Jaykumar122/APP-Mentrafi2.0?color=8b5cf6&label=Release&logo=github)](https://github.com/Jaykumar122/APP-Mentrafi2.0/releases/tag/v1.0.0)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MentraFiAI--TriNetwork-yellow)](https://huggingface.co/singh0123/MentraFiAI-TriNetwork)
[![React Native](https://img.shields.io/badge/React%20Native-Expo%2054-blue?logo=react)](https://reactnative.dev/)
[![Backend](https://img.shields.io/badge/Node.js-Express%20%2B%20TypeScript-green?logo=node.js)](https://nodejs.org/)
[![AI Engine](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Database](https://img.shields.io/badge/PostgreSQL-14%2B-336791?logo=postgresql)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**MentraFi** is a state-of-the-art, full-stack wealth advisory and mutual fund investment ecosystem built specifically for the Indian financial market. It couples a luxury **React Native (Expo SDK 54)** mobile client in Obsidian & Aurora styling, an enterprise **Node.js + Express + TypeScript** transactional API gateway, and **MentraFiAI** — a proprietary **Tri-Network Neuro-Symbolic Small Language Model (SLM)** running natively on PyTorch and CUDA.

Unlike generic foundation models that hallucinate tax rules and suffer from compounding arithmetic drift, MentraFi operates with **100% deterministic mathematical accuracy**, zero distributor commission bias (SEBI Direct-Growth invariant), and real-time synchronization with the official **AMFI Portal (`amfiindia.com`)**.

---

[Key Features](#-key-features) • [System Architecture](#-system-architecture) • [Repository Layout](#-repository-layout) • [Quickstart Guide](#-quickstart-guide) • [AI Tri-Network](#-the-tri-network-ai-engine) • [API Overview](#-api-endpoints)

---

</div>

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Tier 1: frontend (Mobile Client)                      │
│             React Native / Expo SDK 54 / TypeScript / Expo Router           │
│  • Obsidian & Aurora Design Language • Real-Time Portfolio Valuation        │
│  • AI Advisor Chat (Real-time SSE Token Streaming & Scheme Cards)           │
│  • Interactive Compounding SIP Calculators • One-Tap Direct Execution       │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │ HTTP REST & SSE (/api/*)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Tier 2: backend (API Gateway & Transaction Ledger)          │
│                      Node.js / Express / TypeScript (Port 3001)             │
│  • JWT Authentication, Biometric Tokens & KYC User Profiles                 │
│  • Double-Entry Portfolio Ledger & Automated Morning SIP Execution Engine   │
│  • Official AMFI Daily Sync (amfiindia.com NAVAll.txt via UNNEST Upsert)    │
│  • 6x Daily Intra-Day NAV Fast Refreshes & Weekly Rolling Returns Engine    │
└──────────────────┬────────────────────────────────────────────┬─────────────┘
                   │ SQL Connection Pool                        │ JSON Proxy
                   ▼                                            ▼
┌──────────────────────────────────────┐     ┌────────────────────────────────┐
│   Tier 3: PostgreSQL (mentrafi)      │     │  Tier 4: MentraFiAI AI Core    │
│            (14 Tables)               │     │   PyTorch / CUDA (Port 8000)   │
│ • funds (37,882 AMFI Schemes)        │     │ • Network 1: Classifier 10.65M │
│   - 8,853 Live (is_active = TRUE)    │◄────┤ • Network 2: Parser 15.26M     │
│   - 29,029 Quarantine Archive        │     │ • Network 3: Generative 102.5M │
│ • fund_performance (37,764 Rows)     │     │ • Closed-Form Symbolic Solver  │
│ • fund_holdings & stock overlaps     │     │ • Fiduciary RAG Invariant      │
│ • Users, Portfolios, SIP Mandates    │     └────────────────────────────────┘
└──────────────────────────────────────┘
```

---

## ✨ Key Features

- **🧠 Proprietary Tri-Network AI Advisor:**
  - **Network 1 (10.65M):** Instant financial intent routing and risk appetite classification (<10ms).
  - **Network 2 (15.26M):** Token-level BIO entity extractor parsing investor age, horizon, and monthly budget without numeric conflation.
  - **Network 3 (102.5M):** Autoregressive language model producing fiduciary, SEBI-compliant advice grounded in Finance Act 2024–2026 taxation laws.
  - **Symbolic Solver:** Deterministic ordinary annuity compounder ($FV = P \times \frac{(1+r)^n - 1}{r}$) guaranteeing zero arithmetic leakage.

- **📊 Live AMFI Mutual Fund Ingestion:**
  - Ingests **14,361+ live schemes** nightly from the official AMFI master feed (`amfiindia.com/spages/NAVAll.txt`) in **~1.03s** using batch PostgreSQL typed `UNNEST` upserts.
  - Dual-column ISIN extraction providing **98.8% ISIN coverage**.
  - **Fiduciary Quarantine (`is_active = TRUE`):** Filters out closed/matured schemes so users are never recommended dead investments.

- **💎 Luxury Obsidian & Aurora Mobile App:**
  - Modern dark-mode interface with glassmorphism, glowing gradients, and fluid haptic feedback.
  - Interactive multi-mode SIP calculator (Regular, Annual Step-Up, Lumpsum, and Goal Target).
  - Live interactive SVG NAV performance graphs with period selection (1M, 3M, 6M, 1Y, 3Y, 5Y, ALL).

- **🛡️ Statutory Indian Financial Grounding:**
  - Native benchmarks for PPF (7.1%), Sukanya Samriddhi SSY (8.2%), NSC (7.7%), Sovereign Gold Bonds (2.5%), Senior Citizen Savings Scheme (8.2%), and RBI Floating Rate Bonds.
  - Exact capital gains calculation: Equity LTCG (12.5% over ₹1.25L), STCG (20%), and Section 50AA debt marginal slab rates.

---

## 📂 Repository Layout

```
APP-Mentrafi2.0/
├── frontend/                  # React Native / Expo Mobile Client
│   ├── app/                   # Expo Router navigation (Auth, Tabs, Onboarding)
│   ├── assets/images/         # 3D assets, logos, and branding
│   ├── components/            # Luxury UI widgets (Glass panels, 3D rotating emblem)
│   ├── utils/api.ts           # REST API client configuration
│   └── package.json
│
├── backend/                   # Node.js + Express + TypeScript Gateway (Port 3001)
│   ├── src/
│   │   ├── config/            # PostgreSQL connection pool & migrations
│   │   ├── routes/            # REST endpoints (funds, advisor, portfolio, sip, auth)
│   │   ├── services/          # AMFI sync, fund syncing, SIP scheduler
│   │   └── index.ts           # Server bootstrap and cron jobs
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

## 🚀 Quickstart Guide

### Prerequisites
- **Node.js 18+** & npm
- **Python 3.10+** (with PyTorch and CUDA recommended)
- **PostgreSQL 14+** running locally with database `mentrafi`
- **Expo Go** app on your iOS / Android device or an emulator

---

### Step 1: Database Setup
Create and configure your PostgreSQL database:
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

---

### Step 3: Launch MentraFiAI Core (Port 8000)

1. Navigate to the AI directory and install dependencies:
   ```bash
   cd Mentrafiai
   pip install -r requirements.txt
   ```

2. **Download Model Checkpoints:**
   Download the 3 pre-trained networks automatically from **[GitHub Releases v1.0.0](https://github.com/Jaykumar122/APP-Mentrafi2.0/releases/tag/v1.0.0)** or **[Hugging Face](https://huggingface.co/singh0123/MentraFiAI-TriNetwork)**:
   ```bash
   python download_checkpoints.py
   ```

3. Launch the inference server:
   ```bash
   python serve.py
   ```

---

### Step 4: Launch Mobile Application
```bash
cd frontend
npm install
npx expo start
```
Scan the QR code with **Expo Go** on Android or the **Camera app** on iOS to launch the application.

---

## 🧠 The Tri-Network AI Engine

| Network | Parameter Size | Architecture | Checkpoint File | Download Source |
| :--- | :---: | :--- | :--- | :--- |
| **Network 1** | **10.65M** | Bi-Encoder Intent & Risk Classifier | `checkpoints/classifier/best_classifier.pt` | [Release v1.0.0](https://github.com/Jaykumar122/APP-Mentrafi2.0/releases/tag/v1.0.0) / [Hugging Face](https://huggingface.co/singh0123/MentraFiAI-TriNetwork) |
| **Network 2** | **15.26M** | BIO Token-Level Slot Parser | `checkpoints/query_parser/best_parser_net.pt` | [Release v1.0.0](https://github.com/Jaykumar122/APP-Mentrafi2.0/releases/tag/v1.0.0) / [Hugging Face](https://huggingface.co/singh0123/MentraFiAI-TriNetwork) |
| **Network 3** | **102.5M** | Autoregressive Financial SLM | `checkpoints/finetune_v5/best_model.pt` | [Release v1.0.0](https://github.com/Jaykumar122/APP-Mentrafi2.0/releases/tag/v1.0.0) / [Hugging Face](https://huggingface.co/singh0123/MentraFiAI-TriNetwork) |

To run isolated tests on Networks 1 and 2:
```bash
cd Mentrafiai
python test_networks_1_2.py
```

---

## 🔌 API Endpoints

### Backend Gateway (Port 3001)
- `GET /api/funds` — Search, page, and filter active mutual funds (`is_active = TRUE`).
- `GET /api/funds/:schemeCode/details` — Scheme details (ISIN, NAV, returns, expense ratio, AUM, holdings).
- `POST /api/funds/sync-amfi` — Trigger immediate synchronization with official AMFI master feed.
- `POST /api/advisor/chat/stream` — Real-time SSE token stream proxying advice from MentraFiAI.
- `GET /api/portfolio` & `POST /api/portfolio/invest` — User portfolio ledger and buy/sell execution.
- `GET /api/sip` & `POST /api/sip` — Recurring systematic investment mandate management.

### MentraFiAI Core (Port 8000)
- `POST /api/chat` — Synchronous financial advisory JSON response with matched fund cards.
- `POST /api/chat/stream` — Direct Server-Sent Events (SSE) token streaming.
- `GET /health` — Neural network status and CUDA hardware telemetry.

---

## 📜 License
Distributed under the **MIT License**. See `LICENSE` for more information.

<div align="center">
Built with ❤️ for Indian Direct Mutual Fund Investors.
</div>
