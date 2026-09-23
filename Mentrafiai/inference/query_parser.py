"""
MentraFiAI — Financial Query Parser & 10M Neural Intent Engine
==============================================================
Dual-Engine System:
1. 10M Parameter Neural Intent & Parameter Classifier (MentraIntentClassifier):
   - High-speed Transformer encoder analyzing query semantics
   - Multi-head outputs: Intent, Neural Risk Appetite, Neural Horizon Class
2. High-Precision Entity Extraction:
   - Age: Exact identification ("I am 28", "28 years old", "age: 35", "30 yo")
   - Amount: Indian notation parser (₹, Rs, 10k, 1.5 Lakh, 2.5 Crore)
   - Mode: SIP (monthly/recurring) vs LUMPSUM (one-time)
   - Horizon: Exact year extraction ("10 years", "5-year horizon", "in 15 yrs")
   - Risk: Neural head + nuanced financial risk profile matching
   - AI Persona: Replies like an elite financial advisor, not a basic chatbot
   - Natural Conversation: Dynamic, warm, time-aware greetings ("good morning", "good night", "hello", "hey")
"""

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure safe UTF-8 encoding on Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from model.classifier import MentraIntentClassifier, ClassifierConfig, INTENTS, RISK_CLASSES, HORIZON_CLASSES
from model.parser_network import MentraQueryParserNet, ParserNetConfig, load_parser_network
from tokenizer.tokenizer_utils import Tokenizer


@dataclass
class FinancialQuery:
    raw_query: str
    cleaned_query: str
    intent: str
    intent_confidence: float
    age: Optional[int] = None
    amount: Optional[float] = None
    amount_formatted: Optional[str] = None
    mode: str = "SIP"  # "SIP" or "LUMPSUM"
    risk: str = "moderate"
    horizon: int = 10  # default in years
    horizon_class: str = "LONG_TERM"
    name: Optional[str] = None
    goal: Optional[str] = None
    occupation: Optional[str] = None
    funds: List[str] = field(default_factory=list)
    is_complete: bool = False
    missing_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": f"{self.intent_confidence * 100:.1f}%",
            "age": self.age,
            "amount": self.amount,
            "amount_formatted": self.amount_formatted,
            "mode": self.mode,
            "risk": self.risk,
            "horizon_years": self.horizon,
            "horizon_class": self.horizon_class,
            "name": self.name,
            "goal": self.goal,
            "occupation": self.occupation,
            "funds": self.funds,
            "is_complete_for_portfolio": self.is_complete,
            "missing_fields": self.missing_fields,
        }


class FinancialQueryParser:
    """Production Multi-Network Financial Query Engine:
    - Network 1: 6.6M Intent & Risk Classifier
    - Network 2: 10M Neural Slot & Entity Extraction Parser
    """

    def __init__(
        self,
        classifier_ckpt: Optional[str] = None,
        parser_ckpt: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Tokenizer
        tok_file = tokenizer_path or str(BASE_DIR / "tokenizer" / "tokenizer_v2.model")
        if not os.path.exists(tok_file):
            tok_file = str(BASE_DIR / "tokenizer" / "tokenizer.model")
        self.tokenizer = Tokenizer(tok_file)

        # 2. Load Network 1: 10M Neural Intent Classifier
        ckpt_file = classifier_ckpt or str(BASE_DIR / "checkpoints" / "classifier" / "best_classifier.pt")
        self.classifier = None
        if os.path.exists(ckpt_file):
            try:
                ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
                cfg = ckpt.get("cfg", ClassifierConfig())
                self.classifier = MentraIntentClassifier(cfg).to(self.device)
                self.classifier.load_state_dict(ckpt["model_state_dict"])
                self.classifier.eval()
                print(f"[FinancialQueryParser] Network 1 (10M Intent Classifier) loaded successfully.")
            except Exception as e:
                print(f"[WARN] Could not load classifier checkpoint: {e}")
        else:
            print(f"[INFO] Classifier checkpoint not found at {ckpt_file}. Using rule fallback.")

        # 3. Load Network 2: 15M Neural Financial Query Parser
        parser_file = parser_ckpt or str(BASE_DIR / "checkpoints" / "query_parser" / "best_parser_net.pt")
        self.parser_net = None
        if os.path.exists(parser_file):
            try:
                self.parser_net = load_parser_network(parser_file, device=self.device)
                print(f"[FinancialQueryParser] Network 2 (15M Query Parser) loaded successfully.")
            except Exception as e:
                print(f"[WARN] Could not load 15M parser network checkpoint: {e}")
        else:
            print(f"[INFO] 15M Parser network checkpoint not found at {parser_file}. Using rule-based slot extraction.")

    def clean_text(self, text: str) -> str:
        t = text.strip("\"'""'' `")
        t = t.replace("₹", "Rs ")
        t = t.replace("INR", "Rs ")
        # Convert "k" notation: e.g. "10k" -> "10000"
        t = re.sub(r"\b(\d+)\s*k\b", lambda m: f"{int(m.group(1)) * 1000}", t, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", t).strip()

    def extract_age(self, text: str) -> Optional[int]:
        patterns = [
            r"\bage\s*(?:is|of|[:=])?\s*(\d{2})\b",
            r"\b(\d{2})\s*(?:years?\s*old|y/?o|-year-old)\b",
            r"\bI\s+am\s+(\d{2})\b(?!\s*(?:lakh|crore|k|thousand|rupees|rs|%))",
            r"\b(\d{2})\s*(?:m|f)\b",
            r"\bturned\s+(\d{2})\b",
            r"\bat\s+(\d{2})\s*years\b",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 18 <= val <= 95:
                    return val
        return None

    def extract_amount(self, text: str) -> tuple[Optional[float], Optional[str]]:
        # 1. Lakhs
        m_lakh = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:lakhs?|lacs?|lac|l)\b", text, re.IGNORECASE)
        if m_lakh:
            val = float(m_lakh.group(1)) * 100000.0
            return val, f"Rs {val:,.0f}"

        # 2. Crores
        m_cr = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:crores?|cr)\b", text, re.IGNORECASE)
        if m_cr:
            val = float(m_cr.group(1)) * 10000000.0
            return val, f"Rs {val:,.0f}"

        # 3. Explicit Currency "Rs 10,000"
        m_curr = re.search(r"(?:Rs\.?\s*)\s*([\d,]+)", text, re.IGNORECASE)
        if m_curr:
            raw = m_curr.group(1).replace(",", "")
            if raw.isdigit():
                val = float(raw)
                if val >= 100:
                    return val, f"Rs {val:,.0f}"

        # 4. SIP / Budget contextual number
        m_ctx = re.search(r"(?:sip|budget|invest|save|amount)\s*(?:is|of|:)?\s*([\d,]{3,7})\b", text, re.IGNORECASE)
        if m_ctx:
            raw = m_ctx.group(1).replace(",", "")
            if raw.isdigit():
                val = float(raw)
                if val >= 100:
                    return val, f"Rs {val:,.0f}"

        # 5. Monthly frequency trailing: e.g. "10000/month", "5000 monthly"
        m_month = re.search(r"\b([\d,]{3,7})\s*(?:/month|monthly|per\s*month)\b", text, re.IGNORECASE)
        if m_month:
            raw = m_month.group(1).replace(",", "")
            if raw.isdigit():
                val = float(raw)
                if val >= 100:
                    return val, f"Rs {val:,.0f}"

        return None, None

    def extract_mode(self, text: str) -> str:
        low = text.lower()
        lumpsum_cues = ["lumpsum", "lump sum", "one time", "once", "single investment", "fd into mutual fund"]
        if any(c in low for c in lumpsum_cues):
            return "LUMPSUM"
        return "SIP"

    def extract_horizon(self, text: str, neural_class: str = "LONG_TERM") -> tuple[int, str]:
        patterns = [
            r"\b(\d+)\s*[-–]?\s*years?(?:\s*horizon|\s*period|\s*duration|\s*time|\s*goal)?\b(?!\s*old)",
            r"for\s+(\d+)\s+years?\b(?!\s*old)",
            r"in\s+(\d+)\s+years?\b(?!\s*old)",
            r"\b(\d+)\s*yr?s?\s*horizon\b",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 1 <= val <= 35:
                    h_class = "SHORT_TERM" if val <= 3 else "MEDIUM_TERM" if val <= 7 else "LONG_TERM"
                    return val, h_class

        if neural_class == "SHORT_TERM":
            return 3, "SHORT_TERM"
        elif neural_class == "MEDIUM_TERM":
            return 5, "MEDIUM_TERM"
        else:
            return 10, "LONG_TERM"

    def extract_risk(self, text: str, neural_risk: str = "MODERATE") -> str:
        low = text.lower()
        if re.search(r"\b(moderately aggressive|moderate aggressive|aggressive growth)\b", low):
            return "moderately aggressive"
        if re.search(r"\b(aggressive|high risk|maximum return|crypto investor|high growth)\b", low):
            return "aggressive"
        if re.search(r"\b(conservative|low risk|safe|capital protection|fd alternative|risk averse|no risk)\b", low):
            return "conservative"
        if re.search(r"\b(moderate|balanced|medium risk|steady|normal risk)\b", low):
            return "moderate"

        if neural_risk == "CONSERVATIVE":
            return "conservative"
        elif neural_risk == "AGGRESSIVE":
            return "aggressive"
        return "moderate"

    def extract_name(self, text: str) -> Optional[str]:
        m = re.search(r"(?:My name is|I am|I\'m|Hi, I am|This is)\s+([A-Z][a-z]+)\b", text)
        if m:
            cand = m.group(1)
            if cand.lower() not in ["investor", "here", "looking", "interested", "planning", "working"]:
                return cand
        return None

    def extract_occupation(self, text: str) -> Optional[str]:
        low = text.lower()
        if re.search(r"\b(student|college|university|studying|undergrad|postgrad|intern)\b", low):
            return "Student"
        if re.search(r"\b(salaried|job|employee|working professional|software engineer|developer|manager|corporate|doing job|service|salary)\b", low):
            return "Doing Job / Salaried"
        if re.search(r"\b(retired|pensioner|senior citizen|retirement age|ex-serviceman)\b", low):
            return "Retired Person"
        if re.search(r"\b(business|businessman|entrepreneur|self employed|shopkeeper|trader|firm owner)\b", low):
            return "Self Employed / Business"
        if re.search(r"\b(freelance|freelancer|consultant|gig worker|creator|contractor)\b", low):
            return "Freelancer / Consultant"
        if re.search(r"\b(homemaker|housewife|stay at home)\b", low):
            return "Homemaker"
        return None

    def extract_goal(self, text: str) -> Optional[str]:
        low = text.lower()
        goals = {
            "retirement": ["retire", "retirement", "pension", "post retirement", "old age fund"],
            "child_education": ["child", "daughter", "son", "kid", "education", "college", "school", "higher study"],
            "house_purchase": ["house", "home", "flat", "apartment", "down payment", "property", "buy home"],
            "wealth_creation": ["wealth", "rich", "crorepati", "financial freedom", "growth", "corpus", "wealth creation", "1 crore", "compounding"],
            "tax_saving": ["tax", "80c", "elss", "save tax", "tax saving", "deduction"],
            "emergency_fund": ["emergency", "contingency", "rainy day", "safe buffer", "liquid fund"],
        }
        for goal_name, cues in goals.items():
            if any(c in low for c in cues):
                return goal_name
        return None

    def parse(self, query: str) -> FinancialQuery:
        clean = self.clean_text(query)
        
        intent = "GENERAL_CHAT"
        conf = 0.85
        neural_risk = "MODERATE"
        neural_hor = "LONG_TERM"

        if self.classifier is not None:
            ids = self.tokenizer.encode(clean)
            if len(ids) > 0:
                idx = torch.tensor([ids[:256]], dtype=torch.long, device=self.device)
                pred = self.classifier.predict(idx)
                intent = pred["intent"]
                conf = pred["intent_confidence"]
                neural_risk = pred["predicted_risk"]
                neural_hor = pred["predicted_horizon"]

        low = clean.lower()
        if re.search(r"^\s*(hi+|hello+|hey+|good\s*(?:morning|afternoon|evening|night)|namaste|bye|tata)\b", low):
            if not re.search(r"(?:sip|invest|rs\.?|portfolio|fund|cagr|lakh)", low):
                intent = "GREETING"

        # Entity & Slot Extraction (Network 2 Neural Parser + Occupation & Goal Detection)
        funds = []
        occupation = self.extract_occupation(clean)

        if self.parser_net is not None:
            net_res = self.parser_net.parse_query(clean, self.tokenizer)
            age = net_res.get("age") or self.extract_age(clean)
            amount = net_res.get("amount")
            if amount is None:
                amount, _ = self.extract_amount(clean)
            mode = net_res.get("mode") or self.extract_mode(clean)
            horizon = net_res.get("horizon")
            if not horizon or horizon == 10:
                horizon, hor_class = self.extract_horizon(clean, neural_hor)
            else:
                hor_class = "SHORT_TERM" if horizon <= 3 else ("MEDIUM_TERM" if horizon <= 7 else "LONG_TERM")
            risk = net_res.get("risk") or self.extract_risk(clean, neural_risk)
            goal = net_res.get("goal") or self.extract_goal(clean)
            funds = net_res.get("funds", [])
        else:
            age = self.extract_age(clean)
            amount, _ = self.extract_amount(clean)
            mode = self.extract_mode(clean)
            horizon, hor_class = self.extract_horizon(clean, neural_hor)
            risk = self.extract_risk(clean, neural_risk)
            goal = self.extract_goal(clean)

        # Fiduciary adjustments based on Occupation & Goal
        if occupation == "Retired Person":
            if not any(k in low for k in ["aggressive", "high risk"]):
                risk = "conservative"
            if horizon > 8:
                horizon = 7
                hor_class = "MEDIUM_TERM"
        elif occupation == "Student":
            if horizon < 10:
                horizon = 15
                hor_class = "LONG_TERM"

        if goal == "tax_saving":
            # ELSS has a mandatory 3-year lock-in
            horizon = max(horizon, 3)

        amt_fmt = f"Rs {amount:,.0f}" if amount is not None else None
        name = self.extract_name(clean)

        missing = []
        if amount is None:
            missing.append("monthly_sip_amount")
        if age is None:
            missing.append("investor_age")

        is_complete = len(missing) == 0

        if amount is not None and intent in ["GENERAL_CHAT", "GREETING"]:
            intent = "PORTFOLIO_RECOMMENDATION"

        return FinancialQuery(
            raw_query=query,
            cleaned_query=clean,
            intent=intent,
            intent_confidence=conf,
            age=age,
            amount=amount,
            amount_formatted=amt_fmt,
            mode=mode,
            risk=risk,
            horizon=horizon,
            horizon_class=hor_class,
            name=name,
            goal=goal,
            occupation=occupation,
            funds=funds,
            is_complete=is_complete,
            missing_fields=missing,
        )

    def handle_conversation(self, parsed: FinancialQuery) -> str:
        """Replies naturally like an AI Financial Advisor, not a chatbot."""
        t = parsed.cleaned_query.lower()
        user_name = f", {parsed.name}" if parsed.name else ""

        if re.search(r"\b(good\s*morning|good\s*moring|gm|good\s*mrng|mrng)\b", t):
            return (
                f"Good morning{user_name}! 🌅 I hope you have a productive day ahead.\n\n"
                f"Markets are constantly in motion, but disciplined SIP investing remains the most proven path to long-term wealth. "
                f"Whether you would like to analyze a mutual fund, calculate SIP compounding, or design a customized portfolio, "
                f"I'm ready to assist. How can I help you today?"
            )

        if re.search(r"\b(good\s*afternoon|aftrnun)\b", t):
            return (
                f"Good afternoon{user_name}! ☀️ Hope your day is going well.\n\n"
                f"Indian equity markets are currently active. Let me know if you would like "
                f"to evaluate fund performance, plan financial goals, or review current 2026 tax-efficient investment strategies!"
            )

        if re.search(r"\b(good\s*evening|ge|evng)\b", t):
            return (
                f"Good evening{user_name}! 🌆 As the market day concludes and AMCs compute daily NAVs,\n\n"
                f"it's an ideal time to review your wealth roadmap or plan new SIP allocations. What financial goals would you like to explore?"
            )

        if re.search(r"\b(good\s*night|gn|sleep\s*well)\b", t):
            return (
                f"Good night{user_name}! 🌙 Rest well.\n\n"
                f"The beauty of disciplined compounding is that your money works for you around the clock, even while you sleep. "
                f"Whenever you're ready to continue building your portfolio, I'll be here. Have a restful night!"
            )

        if re.search(r"\b(namaste|namaskar|pranam|radhe|vanakkam|sat\s*sri\s*akal|adaab)\b", t):
            return (
                f"Namaste{user_name}! 🙏 Welcome to MentraFiAI.\n\n"
                f"I am your personal AI financial co-pilot, providing unbiased, data-grounded guidance across 37,700+ Indian mutual funds "
                f"without distributor commission bias. How can I assist your wealth creation journey today?"
            )

        if re.search(r"\b(bye|goodbye|see\s*you|cya|tata|alvida)\b", t):
            return (
                f"Goodbye{user_name}! 👋 Thank you for consulting MentraFiAI.\n\n"
                f"Stay committed to your monthly SIPs, stay calm through market cycles, and let compounding do the heavy lifting. Have a wonderful day!"
            )

        if re.search(r"\b(thank\s*you|thanks|dhanyawad|shukriya|great\s*job)\b", t):
            return (
                f"You're very welcome{user_name}! 🤝 It is a pleasure assisting you.\n\n"
                f"Wealth creation is a marathon, not a sprint. Whenever you have questions on fund selection, "
                f"SEBI regulations, or tax rules, I'm always here to guide you."
            )

        return (
            f"Hello{user_name}! 👋 I am **MentraFiAI**, your AI financial co-pilot for Indian mutual funds.\n\n"
            f"Here is how I can assist your wealth creation journey in 2026:\n"
            f"• **Personalized Portfolios:** Tell me your age, monthly budget (e.g. *₹10,000/month*), and risk appetite.\n"
            f"• **Goal Planning:** Calculate corpus projections for retirement, child education, or home buying.\n"
            f"• **Fund Comparisons:** Analyze active vs index funds, expense ratios (TER), and Sharpe ratios.\n"
            f"• **Taxation Guidance:** Current 2026 Indian tax rules (12.5% LTCG with ₹1.25L annual exemption & 20% STCG).\n\n"
            f"What financial goal or question would you like to explore today?"
        )
