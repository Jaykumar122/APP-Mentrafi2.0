"""
MentraFiAI API Inference Server
Serves the proprietary 3-network stack (Classifier + Query Parser + 102.5M Generative SLM + PostgreSQL RAG).
Provides REST and SSE endpoints for Mentrafi-api backend and Mentrafi mobile app.
"""

import sys
import os
import re
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from model import ModelConfig, MentraFiAI
from tokenizer.tokenizer_utils import Tokenizer
from training.utils import load_config
from inference.query_parser import FinancialQueryParser
from inference.db_fund_retriever import FundDatabase, format_star_rating
from inference.financial_calculator import (
    detect_and_calculate,
    calculate_sip,
    calculate_goal_sip,
    calculate_lumpsum,
    calculate_step_up_sip,
    format_inr,
    parse_currency_amount,
)
from inference.generate import (
    is_greeting,
    handle_greeting,
    is_informational,
    informational_reply,
    _find_glossary_entry,
    FINANCIAL_GLOSSARY,
)

# Fix console encoding for Windows
if sys.platform == "win32":
    os.system("")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("mentrafiai-server")

# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

class UserProfileContext(BaseModel):
    age: Optional[int] = None
    name: Optional[str] = None
    monthly_sip_budget: Optional[float] = None
    risk_appetite: Optional[str] = None
    investment_goal: Optional[str] = None
    investment_horizon: Optional[int] = None
    occupation: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    user_id: Optional[int] = None
    profile: Optional[UserProfileContext] = None
    stream: Optional[bool] = False

class FundCardResponse(BaseModel):
    id: str
    schemeCode: Optional[int] = None
    name: str
    category: str
    subcategory: str
    rating: float
    oneYearReturn: Optional[float] = None
    fiveYearReturn: Optional[float] = None
    nav: Optional[float] = None
    monthlyAmount: Optional[int] = None
    allocationPercent: Optional[int] = None
    planType: str = "Direct Growth"

class ChatResponse(BaseModel):
    reply: str
    recommendedFunds: List[FundCardResponse] = Field(default_factory=list)
    intent: str = "CONVERSATIONAL"
    risk: str = "MODERATE"
    latencyMs: Optional[float] = None

# OpenAI-compatible completions request
class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAICompletionsRequest(BaseModel):
    model: Optional[str] = "mentrafiai-v5"
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 350
    stream: Optional[bool] = False

# ─────────────────────────────────────────────────────────────────────────────
# MentraFiAI Engine Core
# ─────────────────────────────────────────────────────────────────────────────

class MentraFiAIEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer: Optional[Tokenizer] = None
        self.model: Optional[MentraFiAI] = None
        self.query_parser: Optional[FinancialQueryParser] = None
        self.db: Optional[FundDatabase] = None
        self.checkpoint_name = "finetune_v5"

    def load(self):
        logger.info(f"Loading MentraFiAI Core on device: {self.device}...")

        # 1. Tokenizer
        tok_path = str(ROOT_DIR / "tokenizer" / "tokenizer_v2.model")
        logger.info(f"Loading SentencePiece Tokenizer: {tok_path}")
        self.tokenizer = Tokenizer(tok_path)

        # 2. Network 3: Generative LLM (102.5M)
        cfg_path = str(ROOT_DIR / "configs" / "model_config.yaml")
        cfg_dict = load_config(cfg_path)
        cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in ModelConfig.__dataclass_fields__})
        self.model = MentraFiAI(cfg).to(self.device)

        ckpt_candidates = [
            ROOT_DIR / "checkpoints" / "finetune_v5" / "best_model.pt",
            ROOT_DIR / "checkpoints" / "finetune_v4" / "best_model.pt",
            ROOT_DIR / "checkpoints" / "finetune_v3" / "best_model.pt",
            ROOT_DIR / "checkpoints" / "pretrain_v2" / "best_model.pt",
        ]
        chosen_ckpt = None
        for ck in ckpt_candidates:
            if ck.exists():
                chosen_ckpt = ck
                break

        if not chosen_ckpt:
            raise FileNotFoundError("No valid MentraFiAI checkpoint found in checkpoints/!")

        logger.info(f"Loading Network 3 weights from: {chosen_ckpt}")
        ckpt = torch.load(chosen_ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.checkpoint_name = chosen_ckpt.parent.name
        logger.info(f"Network 3 ready ({self.checkpoint_name}, step {ckpt.get('step', 0):,})")

        # 3. Network 1 (10M Classifier) & Network 2 (15M Query Parser)
        clf_path = ROOT_DIR / "checkpoints" / "classifier" / "best_classifier.pt"
        parser_path = ROOT_DIR / "checkpoints" / "query_parser" / "best_parser_net.pt"
        self.query_parser = FinancialQueryParser(
            classifier_ckpt=str(clf_path) if clf_path.exists() else None,
            parser_ckpt=str(parser_path) if parser_path.exists() else None,
            tokenizer_path=tok_path,
            device=self.device
        )
        logger.info("Networks 1 & 2 (Classifier & Slot Parser) initialized.")

        # 4. PostgreSQL Database
        self.db = FundDatabase()
        if self.db.is_connected:
            count = self.db.get_fund_count()
            logger.info(f"PostgreSQL Scheme DB connected: {count:,} mutual funds active.")
        else:
            logger.warning("PostgreSQL Scheme DB offline; using deterministic category templates.")

    def clean_financial_prompt(self, text: str) -> str:
        text = text.strip("' `")
        text = re.sub(r"\bgood\s+mori?ng\b", "good morning", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhelo+\b", "hello", text, flags=re.IGNORECASE)
        text = re.sub(r"\bnamste\b", "namaste", text, flags=re.IGNORECASE)
        text = text.replace("₹", "Rs ")
        text = re.sub(r"\b(\d+)\s*[--]?\s*years?\s*[--]?\s*old\b", r"age \1", text, flags=re.IGNORECASE)
        text = re.sub(r"\bI am (\d+)\b(?!\s*(?:lakh|crore|k|thousand))", r"age \1", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\d+(?:\.\d+)?)\s*(?:crore|crores|cr)\b", lambda m: f"Rs {int(float(m.group(1))*10000000):,}", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs|l)\b", lambda m: f"Rs {int(float(m.group(1))*100000):,}", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\d+)k\b", lambda m: f"Rs {int(m.group(1))*1000:,}", text, flags=re.IGNORECASE)
        text = re.sub(r"\bsip\b", "SIP", text, flags=re.IGNORECASE)
        text = re.sub(r"\belss\b", "ELSS", text, flags=re.IGNORECASE)
        text = re.sub(r"\bnav\b", "NAV", text, flags=re.IGNORECASE)
        text = re.sub(r"\bltcg\b", "LTCG", text, flags=re.IGNORECASE)
        text = re.sub(r"\bstcg\b", "STCG", text, flags=re.IGNORECASE)
        text = re.sub(r"Rs\s+Rs\s*", "Rs ", text)
        return re.sub(r"\s+", " ", text).strip()

    def sip_compounding(self, monthly_sip: float, annual_cagr: float, years: int) -> float:
        r = (annual_cagr / 100.0) / 12.0
        n = years * 12
        return monthly_sip * (((1.0 + r)**n - 1.0) / r)

    def generate_portfolio_plan(
        self,
        budget: float,
        risk: str,
        horizon: int,
        name: Optional[str] = None,
        age: Optional[int] = None,
        occupation: Optional[str] = None,
        goal: Optional[str] = None,
    ):
        risk_lower = (risk or "moderate").lower()
        goal_lower = (goal or "").lower()
        occ_lower = (occupation or "").lower()

        special_notes: List[str] = []

        # ── 1. Determine Asset Allocation Matrix by Risk, Goal & Occupation ──
        # Strict Risk Profile Guardrail: Low Risk investors must never receive aggressive equity
        if "conservative" in risk_lower or "low" in risk_lower:
            strategy_title = "Conservative Capital Preservation & Low Volatility"
            splits = [
                ("Conservative Hybrid Fund", "conservative_hybrid", 55),
                ("Corporate Bond / Short Duration Debt", "debt", 35),
                ("Large Cap Bluechip Anchor", "large_cap", 10),
            ]
            cagr = 8.2
            special_notes.append("• **Fiduciary Risk Shield:** Pure high-volatility equities (Flexi Cap, Mid Cap, Small Cap) are excluded to respect your Low Risk profile.")
            special_notes.append("• **Capital Stability:** 90% allocated to conservative hybrid and AAA corporate debt ensures steady compounding above inflation without capital stress.")
        elif "tax" in goal_lower or "80c" in goal_lower or "elss" in goal_lower:
            strategy_title = "Tax-Saving Wealth Accumulator (Section 80C + High Growth)"
            splits = [
                ("ELSS Tax Saver (3-Yr Lock-in)", "elss", 40),
                ("Flexi Cap Multi-Cap Equity", "flexi_cap", 35),
                ("Large Cap Bluechip", "large_cap", 25),
            ]
            cagr = 13.0
            special_notes.append("• **Section 80C Benefit:** Up to ₹1.5 Lakh/year eligible for tax deduction under the Old Tax Regime.")
            special_notes.append("• **Shortest Lock-In:** ELSS has only a 3-year lock-in (compared to 15 years for PPF and 5 years for Tax FDs), compounding superior equity wealth.")
        elif "retire" in goal_lower:
            if (age and age >= 52) or "conservative" in risk_lower or "low" in risk_lower:
                strategy_title = "Retirement Capital Security & SWP Income Engine"
                splits = [
                    ("Conservative Hybrid / Equity Savings", "hybrid", 45),
                    ("Short Duration AAA Debt", "debt", 35),
                    ("Large Cap Bluechip", "large_cap", 20),
                ]
                cagr = 9.2
                special_notes.append("• **Downside Protection:** Heavy allocation to fixed income and arbitrage cushions market volatility.")
                special_notes.append("• **SWP Ready:** Set up a Systematic Withdrawal Plan for predictable, tax-efficient monthly income.")
            else:
                strategy_title = "Long-Horizon Retirement Wealth Engine"
                splits = [
                    ("Flexi Cap Multi-Cap Equity", "flexi_cap", 45),
                    ("Mid Cap Emerging Bluechip", "mid_cap", 35),
                    ("Large Cap Bluechip", "large_cap", 20),
                ]
                cagr = 13.5
                special_notes.append("• **Retirement Runway:** Multi-cap equity engine designed to aggressively compound corpus before transitioning to debt.")
        elif "education" in goal_lower or "child" in goal_lower:
            strategy_title = "Child Higher Education Future Fund"
            splits = [
                ("Flexi Cap Multi-Cap Equity", "flexi_cap", 45),
                ("Mid Cap Growth Engine", "mid_cap", 35),
                ("Balanced Advantage / Hybrid", "hybrid", 20),
            ]
            cagr = 13.2
            special_notes.append("• **Beating Education Inflation:** Higher education in India inflates at ~10% annually; this equity allocation outpaces education inflation.")
        elif "house" in goal_lower or "home" in goal_lower:
            strategy_title = "Home Purchase & Down Payment Accumulator"
            splits = [
                ("Balanced Advantage / Dynamic Asset Allocation", "hybrid", 40),
                ("Flexi Cap Equity", "flexi_cap", 35),
                ("Short Duration Debt", "debt", 25),
            ]
            cagr = 11.5
            special_notes.append("• **Capital Shield near Purchase:** Dynamic asset allocation protects capital against sudden market corrections near down-payment year.")
        elif "emergency" in goal_lower:
            strategy_title = "Emergency Contingency & High Liquidity Buffer"
            splits = [
                ("Arbitrage / Liquid Fund", "debt", 60),
                ("Conservative Hybrid / Equity Savings", "hybrid", 40),
            ]
            cagr = 7.5
            special_notes.append("• **Instant Liquidity:** Zero-lockin capital buffer accessible within 24–48 hours without exit penalties.")
        # If no explicit goal, tailor by OCCUPATION:
        elif "student" in occ_lower:
            strategy_title = "Student Wealth Multiplier (Maximum Compounding Runway)"
            splits = [
                ("Flexi Cap Multi-Cap Equity", "flexi_cap", 45),
                ("Mid Cap Growth Engine", "mid_cap", 35),
                ("Small Cap High Alpha", "small_cap", 20),
            ]
            cagr = 14.5
            special_notes.append("• **Your Superpower is Time:** Starting in college gives you 15–25 years of compounding runway. Even ₹1,000–₹3,000/month compounds into Crores.")
            special_notes.append("• **High Volatility Tolerance:** Market dips are beneficial because your monthly SIP acquires units at bargain valuations.")
        elif "retired" in occ_lower:
            strategy_title = "Retirement Capital Preservation & Low Volatility"
            splits = [
                ("Conservative Hybrid / Equity Savings", "hybrid", 45),
                ("Short Duration AAA Debt", "debt", 35),
                ("Large Cap Bluechip", "large_cap", 20),
            ]
            cagr = 9.0
            special_notes.append("• **Zero Capital Stress:** Heavy fixed-income and hedged equity protects accumulated life savings.")
            special_notes.append("• **Inflation Beat:** 20% bluechip equity ensures purchasing power does not erode over time.")
        elif "business" in occ_lower or "self employed" in occ_lower or "freelance" in occ_lower:
            strategy_title = "Business Cash-Flow Resilience & Wealth Engine"
            splits = [
                ("Balanced Advantage Dynamic Asset Allocation", "hybrid", 40),
                ("Flexi Cap Multi-Sector Equity", "flexi_cap", 35),
                ("Arbitrage / Short Duration Debt", "debt", 25),
            ]
            cagr = 11.8
            special_notes.append("• **Business Income Buffer:** Balanced Advantage funds automatically rebalance, buying equities low during market crashes and locking in profits when frothy.")
            special_notes.append("• **Flexible SIP:** Pause or step-up monthly installments anytime without penalties.")
        elif "job" in occ_lower or "salaried" in occ_lower:
            if "conservative" in risk_lower or "low" in risk_lower:
                strategy_title = "Salaried Investor Capital Security"
                splits = [
                    ("Balanced Advantage / Hybrid", "hybrid", 50),
                    ("Large Cap Equity", "large_cap", 30),
                    ("Short Duration Debt", "debt", 20),
                ]
                cagr = 9.5
                special_notes.append("• **Capital Stability:** Tailored for risk-averse salaried professionals seeking predictable steady gains above bank FDs.")
            elif "aggressive" in risk_lower or "high" in risk_lower:
                strategy_title = "Salaried High-Alpha Wealth Maximizer"
                splits = [
                    ("Flexi Cap Equity", "flexi_cap", 40),
                    ("Mid Cap Equity", "mid_cap", 35),
                    ("Small Cap Equity", "small_cap", 25),
                ]
                cagr = 14.2
                special_notes.append("• **Maximum Long-Term Alpha:** Aggressive equity allocation leveraging your steady salary cash flows.")
            else:
                strategy_title = "Salaried Wealth Builder (Disciplined Monthly Compounding)"
                splits = [
                    ("Flexi Cap Multi-Cap Equity", "flexi_cap", 45),
                    ("Large & Mid Cap Bluechip", "large_cap", 35),
                    ("Mid Cap Growth Engine", "mid_cap", 20),
                ]
                cagr = 12.8
                special_notes.append("• **Salary-Day Auto Debit:** Set SIP debit on the 2nd to 5th of each month right after salary credit.")
                special_notes.append("• **Step-Up SIP Ready:** Boost installment by 10% whenever you receive an annual salary appraisal.")
        elif "homemaker" in occ_lower:
            strategy_title = "Homemaker Financial Independence & Wealth Shield"
            splits = [
                ("Large Cap Bluechip", "large_cap", 40),
                ("Balanced Advantage / Hybrid", "hybrid", 40),
                ("Flexi Cap Equity", "flexi_cap", 20),
            ]
            cagr = 11.5
            special_notes.append("• **Steady Compounding:** Builds independent personal wealth with low volatility and maximum peace of mind.")
        else:
            if "conservative" in risk_lower or "low" in risk_lower:
                strategy_title = "Conservative (Capital Preservation & Low Volatility)"
                splits = [("Conservative Hybrid", "conservative_hybrid", 55), ("Corporate Bond / Short Duration Debt", "debt", 35), ("Large Cap Bluechip Anchor", "large_cap", 10)]
                cagr = 8.2
            elif "aggressive" in risk_lower or "high" in risk_lower:
                strategy_title = "Aggressive (High Alpha Growth)"
                splits = [("Mid Cap Equity", "mid_cap", 40), ("Small Cap Equity", "small_cap", 35), ("Flexi Cap Equity", "flexi_cap", 25)]
                cagr = 14.5
            else:
                strategy_title = "Moderate (Balanced Growth)"
                splits = [("Flexi Cap Equity", "flexi_cap", 50), ("Large Cap Equity", "large_cap", 30), ("Mid Cap Equity", "mid_cap", 20)]
                cagr = 12.5

        # ── 2. Zero Mathematical Drift Exact Integer Rupee Allocation ──
        allocs = []
        accum = 0
        recommended_funds: List[FundCardResponse] = []

        for i, (display_cat, cat_key, target_pct) in enumerate(splits):
            if i == len(splits) - 1:
                amt = int(budget - accum)
            else:
                amt = int(round(budget * target_pct / 100))
                accum += amt

            fund_data = self.db.get_top_fund(cat_key) if (self.db and self.db.is_connected) else None
            allocs.append((display_cat, cat_key, amt, target_pct, fund_data))

            if fund_data:
                recommended_funds.append(FundCardResponse(
                    id=str(fund_data["scheme_code"]),
                    schemeCode=fund_data["scheme_code"],
                    name=fund_data["name"],
                    category=fund_data.get("category") or "Debt",
                    subcategory=fund_data.get("subcategory") or display_cat,
                    rating=float(fund_data.get("rating") or 4.0),
                    oneYearReturn=float(fund_data.get("return_1y") or 0.0),
                    fiveYearReturn=float(fund_data.get("return_5y") or 0.0),
                    nav=float(fund_data.get("nav") or 0.0),
                    monthlyAmount=amt,
                    allocationPercent=target_pct,
                    planType=f"{fund_data.get('plan_type', 'Direct')} {fund_data.get('option_type', 'Growth')}"
                ))

        # ── 3. Deterministic Compounding Mathematics ──
        invested = budget * horizon * 12
        corpus = self.sip_compounding(budget, cagr, horizon)
        real_corpus = corpus / (1.06 ** horizon)

        # 10% Step-Up SIP projection (compounding at month end)
        step_up_corpus = 0.0
        step_cur_sip = budget
        r_step = (cagr / 100.0) / 12.0
        total_step_invested = 0.0
        for y in range(horizon):
            for m in range(12):
                total_step_invested += step_cur_sip
                step_up_corpus = step_up_corpus * (1.0 + r_step) + step_cur_sip
            step_cur_sip *= 1.10

        greeting = f"Hello {name}!" if name else "Welcome to your personalized wealth strategy!"
        profile_pills = []
        if occupation:
            profile_pills.append(f"💼 {occupation}")
        if goal:
            profile_pills.append(f"🎯 {goal.replace('_', ' ').title()}")
        if age:
            profile_pills.append(f"🎂 Age {age}")
        profile_str = f" ({' • '.join(profile_pills)})" if profile_pills else ""

        lines = [
            f"{greeting}",
            f"Here is your fiduciary mutual fund portfolio tailored for a **₹{budget:,.0f}/month SIP** over a **{horizon}-year horizon**{profile_str}.\n",
            f"### 🛡️ Advisory Strategy: **{strategy_title}**\n",
            f"### 🎯 Strategic Asset Allocation (Zero-Drift):",
        ]

        for display_cat, _, amt, pct, fund in allocs:
            fund_name = fund["name"] if fund else f"Direct-Growth {display_cat} Fund"
            nav_txt = f" (NAV ₹{fund['nav']:.2f})" if fund and fund.get("nav") else ""
            lines.append(f"• **{display_cat} ({pct}% — ₹{amt:,.0f}/mo):** {fund_name}{nav_txt}")

        lines.extend([
            f"\n💰 **Deterministic SIP Compounding Projections:**",
            f"• Monthly Commitment: **₹{budget:,.0f} / month**",
            f"• Total Principal Invested ({horizon} Years): **{format_inr(invested)}**",
            f"• **Estimated Maturity Corpus (@ ~{cagr:.1f}% CAGR):** **{format_inr(corpus)}**",
            f"• Inflation-Adjusted Purchasing Power (@ 6% inflation): **{format_inr(real_corpus)}**\n",
            f"🚀 **Power of 10% Annual Step-Up SIP:**",
            f"• Total Invested: **{format_inr(total_step_invested)}**",
            f"• **Estimated Step-Up Corpus:** **{format_inr(step_up_corpus)}** *(+{format_inr(step_up_corpus - corpus)} extra wealth!)*\n",
            f"> *Returns are illustrative estimates, not guaranteed. Actual returns depend on market performance.*\n",
        ])

        if special_notes:
            lines.append("💡 **Fiduciary Insights for Your Profile:**")
            lines.extend(special_notes)
            lines.append("")

        lines.extend([
            f"⚖️ **Current 2026 Indian Taxation Rule:**",
            f"Long-Term Capital Gains (LTCG) above **₹1.25 Lakh** in a financial year are taxed at **12.5%** without indexation (STCG at 20.0%). All recommendations strictly utilize Direct-Growth plans to eliminate distributor commissions."
        ])

        return "\n".join(lines), recommended_funds

    def _fund_dict_to_response(self, fund_data: Dict[str, Any], amt: int = 5000, pct: int = 100) -> FundCardResponse:
        return FundCardResponse(
            id=str(fund_data["scheme_code"]),
            schemeCode=fund_data["scheme_code"],
            name=fund_data["name"],
            category=fund_data.get("category") or "Equity",
            subcategory=fund_data.get("subcategory") or "Direct-Growth",
            rating=float(fund_data.get("rating") or 4.0),
            oneYearReturn=float(fund_data.get("return_1y") or 0.0),
            fiveYearReturn=float(fund_data.get("return_5y") or 0.0),
            nav=float(fund_data.get("nav") or 0.0),
            monthlyAmount=amt,
            allocationPercent=pct,
            planType=f"{fund_data.get('plan_type', 'Direct')} {fund_data.get('option_type', 'Growth')}"
        )

    def generate_llm_reply(self, cleaned_query: str, max_tokens: int = 300, temperature: float = 0.3) -> str:
        prefix_ids = [self.tokenizer.bos_id, self.tokenizer.user_id] + self.tokenizer.encode(cleaned_query) + [self.tokenizer.assistant_id]
        idx = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        prompt_len = len(prefix_ids)

        with torch.inference_mode():
            out = self.model.generate(
                idx,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
            )
            raw_ids = out[0].tolist()[prompt_len:]

            stop_ids = {self.tokenizer.eos_id, self.tokenizer.user_id, self.tokenizer.assistant_id}
            cleaned_ids = []
            for tok_id in raw_ids:
                if tok_id in stop_ids:
                    break
                cleaned_ids.append(tok_id)

            reply = self.tokenizer.decode(cleaned_ids).strip()

            for tag in ["<assistant>", "<user>", "<eos>", "<bos>", "<pad>", "User:", "Assistant:"]:
                if tag in reply:
                    reply = reply.split(tag)[0].strip()

            # Clean up squashed text and formatting
            reply = re.sub(r'(\d+\.\s+[A-Z])', r'\n\n\1', reply)
            reply = re.sub(r'([^\n])\s*(•|\-)\s+', r'\1\n\n• ', reply)
            reply = re.sub(r'Rs\.?\s*5[,.]008\b', '₹5,000', reply)
            reply = re.sub(r'Rs\.?\s*2i\b', '₹200', reply)
            reply = re.sub(r'\n{3,}', '\n\n', reply).strip()

            if not reply:
                reply = (
                    "MentraFi AI analyzes Indian mutual funds with zero mathematical drift. "
                    "You can ask me about SIP planning, ELSS tax savings under Section 80C, or compare Direct vs Regular plans."
                )

            return reply

    def process_query(
        self,
        query: str,
        profile: Optional[UserProfileContext] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ):
        clean = self.clean_financial_prompt(query)
        parsed = self.query_parser.parse(query)
        intent = parsed.intent
        risk = (profile.risk_appetite if profile and profile.risk_appetite else parsed.risk) or "MODERATE"
        budget = parsed.amount or (profile.monthly_sip_budget if profile else None)
        horizon = parsed.horizon or (profile.investment_horizon if profile else 10)
        age = (profile.age if (profile and profile.age) else parsed.age)
        name = (profile.name if (profile and profile.name) else parsed.name)
        occupation = (profile.occupation if (profile and profile.occupation) else parsed.occupation)
        goal = (profile.investment_goal if (profile and profile.investment_goal) else parsed.goal)

        # 0. Multi-Turn Context Memory Backfill from conversation history
        if history:
            for turn in reversed(history):
                prev_text = turn.get("content", "")
                if turn.get("role") == "user" and prev_text:
                    prev_parsed = self.query_parser.parse(prev_text)
                    if not budget and prev_parsed.amount:
                        budget = prev_parsed.amount
                    if not age and prev_parsed.age:
                        age = prev_parsed.age
                    if not name and prev_parsed.name:
                        name = prev_parsed.name
                    if not parsed.horizon and prev_parsed.horizon:
                        horizon = prev_parsed.horizon
                    if not occupation and prev_parsed.occupation:
                        occupation = prev_parsed.occupation
                    if not goal and prev_parsed.goal:
                        goal = prev_parsed.goal

        clean_lower = clean.lower()
        has_fund_keywords = any(w in clean_lower for w in [
            "fund", "funds", "sip", "cap", "elss", "nav", "portfolio", "invest", "lumpsum", "debt", "equity", "hybrid", "tax", "recommend", "suggest", "best"
        ])

        # 1. Greetings: Only trigger if genuine greeting and NOT discussing funds/investing
        is_pure_greeting = (is_greeting(clean) or is_greeting(query) or intent == "GREETING") and not has_fund_keywords
        if is_pure_greeting:
            reply = self.generate_llm_reply(clean, max_tokens=150, temperature=0.6)
            if not reply or len(reply.strip()) < 15:
                user_name = f", {name}" if name else ""
                reply = (
                    f"Hello{user_name}! 👋 Welcome to MentraFi AI, your intelligent Indian mutual fund wealth advisor.\n\n"
                    "I evaluate 37,882 Direct-Growth schemes (8,853 active in 2026), structure goal-based SIP portfolios, and apply current 2026 taxation rules. How can I help you invest today?"
                )
            return reply, [], "GREETING", risk

        # 2. Deterministic Financial Calculator Engine (Zero Math Hallucination)
        calc_reply = detect_and_calculate(query, default_years=int(horizon or 10), default_cagr=12.0, risk_profile=risk)
        if calc_reply:
            recommended_funds: List[FundCardResponse] = []
            if self.db and self.db.is_connected:
                risk_lower = (risk or "").lower()
                is_low_risk = "low" in risk_lower or "conservative" in risk_lower or any(w in clean_lower for w in ["low risk", "conservative", "safe", "capital protect", "capital preservation", "risk averse"])
                if is_low_risk:
                    suitable_funds = self.db.get_suitable_low_risk_funds(limit=2)
                    for f in suitable_funds:
                        rec_amt = int(round((budget or 5000) / max(1, len(suitable_funds))))
                        rec_pct = int(round(100 / max(1, len(suitable_funds))))
                        recommended_funds.append(self._fund_dict_to_response(f, amt=rec_amt, pct=rec_pct))
                else:
                    cat_choice = "flexi_cap" if (horizon and horizon >= 7) else "hybrid"
                    top_fund = self.db.get_top_fund(cat_choice)
                    if top_fund:
                        recommended_funds.append(self._fund_dict_to_response(top_fund, amt=int(budget or 5000), pct=100))
            return calc_reply, recommended_funds, "CALCULATION", risk

        # 3. Dedicated Fund Category & Keyword Search (Real Database Schemes)
        cat_keywords = {
            "large cap": "large_cap", "bluechip": "large_cap", "top 100": "large_cap",
            "mid cap": "mid_cap", "emerging equity": "mid_cap",
            "small cap": "small_cap",
            "flexi cap": "flexi_cap", "multi cap": "flexi_cap",
            "elss": "elss", "tax saver": "elss", "tax saving": "elss",
            "hybrid": "hybrid", "balanced advantage": "hybrid",
            "debt": "debt", "short duration": "debt", "liquid fund": "debt",
            "index fund": "index", "nifty 50": "index", "sensex": "index"
        }

        matched_cat = None
        for kw, cat_key in cat_keywords.items():
            if kw in clean_lower:
                matched_cat = (kw, cat_key)
                break

        is_search_intent = any(w in clean_lower for w in ["recommend", "best", "top", "suggest", "which fund", "list", "show me", "good fund"])
        if matched_cat and (is_search_intent or intent in ("FUND_RECOMMENDATION", "PORTFOLIO_RECOMMENDATION")):
            kw_name, cat_key = matched_cat
            db_funds = self.db.search_funds(kw_name, limit=3) if (self.db and self.db.is_connected) else []
            if not db_funds and self.db and self.db.is_connected:
                single_f = self.db.get_top_fund(cat_key)
                if single_f:
                    db_funds = [single_f]

            if db_funds:
                fund_cards = [self._fund_dict_to_response(f, amt=int(budget or 5000), pct=int(100 // len(db_funds))) for f in db_funds]
                lines = [
                    f"### 🏆 Top Direct-Growth {kw_name.title()} Mutual Funds\n",
                    f"Based on 3-year risk-adjusted rolling performance and verified SEBI compliance ratings, here are top-rated schemes in the **{kw_name.title()}** category:\n"
                ]
                for idx, f in enumerate(db_funds, 1):
                    nav_str = f"₹{f['nav']:.2f}" if f.get("nav") else "N/A"
                    r1_str = f"{f['return_1y']:.2f}%" if f.get("return_1y") is not None else "N/A"
                    r3_str = f"{f['return_3y']:.2f}%" if f.get("return_3y") is not None else "N/A"
                    lines.append(f"**{idx}. {f['name']}**")
                    lines.append(f"• Rating: **{format_star_rating(f.get('rating'))}** | Current NAV: **{nav_str}**")
                    lines.append(f"• 1-Year Return: **{r1_str}** | 3-Year Return: **{r3_str}**")
                    lines.append(f"• Category: {f.get('category', 'Equity')} ({f.get('subcategory', kw_name.title())})\n")

                lines.extend([
                    "⚖️ **Advisory Note:**",
                    "All funds selected strictly utilize **Direct-Growth plans** to eliminate distributor commissions. Invest through regular monthly SIPs to take advantage of rupee-cost averaging."
                ])
                return "\n".join(lines), fund_cards, "CATEGORY_SEARCH", risk

        # 4. Informational & Financial Concept Queries (SIP, NAV, ELSS, SWP, LTCG, etc.)
        # NOTE: is_informational() already applies the recommend-cue exclusion and
        # definition-cue requirement internally. Do NOT OR in a raw glossary keyword
        # match here - that bypasses those guards and hijacks unrelated questions
        # (e.g. "what happens to my SIP if the fund house shuts down?") into a
        # generic canned definition just because the word "SIP" appears in them.
        is_info = is_informational(clean) or is_informational(query)
        has_explicit_budget = bool(budget) and intent == "PORTFOLIO_RECOMMENDATION"

        if is_info and not has_explicit_budget:
            info_reply = informational_reply(clean) or informational_reply(query)
            if info_reply:
                return info_reply, [], "INFORMATIONAL", risk

        # 5. Portfolio Recommendation (Triggered if budget is available, even from past conversation memory)
        is_rec_prompt = any(w in clean_lower for w in ["recommend", "best funds", "suggest", "for me", "portfolio", "plan", "start", "invest"])
        if (intent == "PORTFOLIO_RECOMMENDATION" or parsed.amount or is_rec_prompt) and budget:
            reply, funds = self.generate_portfolio_plan(
                budget=float(budget),
                risk=risk,
                horizon=int(horizon or 10),
                name=name,
                age=age,
                occupation=occupation,
                goal=goal,
            )
            return reply, funds, "PORTFOLIO_RECOMMENDATION", risk

        # 6. Fallback: Direct / General Questions via Generative Model
        reply = self.generate_llm_reply(clean, max_tokens=350, temperature=0.3)
        return reply, [], intent, risk


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App Setup
# ─────────────────────────────────────────────────────────────────────────────

engine = MentraFiAIEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()
    yield

app = FastAPI(
    title="MentraFiAI Local Inference Server",
    description="Triple Neural Network System for Indian Mutual Funds (SEBI Grounded)",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "MentraFiAI Neural Engine",
        "device": engine.device,
        "checkpoint": engine.checkpoint_name,
        "databaseConnected": engine.db.is_connected if engine.db else False,
        "schemesCount": engine.db.get_fund_count() if engine.db and engine.db.is_connected else 0
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")

    reply, funds, intent, risk = engine.process_query(req.message, req.profile, req.history)
    return ChatResponse(
        reply=reply,
        recommendedFunds=funds,
        intent=intent,
        risk=risk
    )

@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")

    reply, funds, intent, risk = engine.process_query(req.message, req.profile, req.history)

    async def event_generator() -> AsyncGenerator[str, None]:
        # 1. Yield recommended funds metadata card first if present
        if funds:
            funds_payload = [f.model_dump() for f in funds]
            yield f"data: {json.dumps({'recommendedFunds': funds_payload})}\n\n"
            await asyncio.sleep(0.02)

        # 2. Token/word streaming emulation for ultra-smooth UI typewriter effect
        words = reply.split(" ")
        chunk_size = 3
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if i + chunk_size < len(words):
                chunk += " "
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            await asyncio.sleep(0.015)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/v1/chat/completions")
async def openai_completions(req: OpenAICompletionsRequest):
    """OpenAI compatible chat completions endpoint"""
    user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    reply, funds, intent, risk = engine.process_query(user_msg)

    if req.stream:
        async def openai_stream():
            words = reply.split(" ")
            chunk_size = 3
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size]) + " "
                yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"

        return StreamingResponse(openai_stream(), media_type="text/event-stream")

    return {
        "id": f"chatcmpl-{int(asyncio.get_event_loop().time()*1000)}",
        "object": "chat.completion",
        "model": req.model or "mentrafiai-v5",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply
                },
                "finish_reason": "stop"
            }
        ]
    }

@app.get("/api/funds/{scheme_code}/details")
async def get_fund_details_endpoint(scheme_code: int):
    """Fetches complete scheme details including AMFI, MFAPI, and SEBI metrics."""
    if not engine.db or not engine.db.is_connected:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    data = engine.db.get_fund_details(scheme_code)
    if not data:
        raise HTTPException(status_code=404, detail="Fund scheme code not found")
    return data

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Launching MentraFiAI Local Inference Gateway on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
