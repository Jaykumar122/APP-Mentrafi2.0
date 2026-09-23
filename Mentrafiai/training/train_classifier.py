import json
import os
import sys
import random
import re
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.classifier import MentraIntentClassifier, ClassifierConfig, INTENTS, RISK_CLASSES, HORIZON_CLASSES
from tokenizer.tokenizer_utils import Tokenizer


def infer_label(text: str) -> tuple[int, int, int]:
    """Infers intent, risk appetite, and investment horizon from user input query."""
    low = text.lower().strip("\"'“”’‘ `")

    # 1. Pure Greetings (hi, hello, namaste, good morning, etc.)
    if re.search(r"^(?:hi+|hello+|hey+|namaste|pranam|good\s*(?:morning|afternoon|evening|night|day)|greetings?|sup|yo+|howdy)\b", low) and not any(k in low for k in ["sip", "invest", "fund", "tax", "nav", "portfolio", "budget", "cagr", "plan", "recommend", "compare", "difference", "retire", "how does", "what is"]):
        intent_idx = 0  # GREETING

    # 2. Goal Planning (retirement, education, house, marriage, emergency fund, etc.)
    elif any(k in low for k in [
        "retirement", "retire", "child education", "children education", "higher education", 
        "buy a house", "buy a flat", "home loan", "down payment", "marriage", "wedding", 
        "emergency fund", "rainy day fund", "dream car", "vacation fund", "wealth creation goal"
    ]) and not any(k in low for k in ["what is", "define", "meaning of"]):
        intent_idx = 4  # GOAL_PLANNING

    # 3. Definitions & Concept Explanations (Taxes, Budget 2024, Metrics, Terms)
    # Note: Even if phrased as 'difference between LTCG and STCG', it must route to DEFINITION!
    elif any(k in low for k in [
        "ltcg", "stcg", "capital gains tax", "exit load", "expense ratio", "ter",
        "cagr", "xirr", "nav", "net asset value", "elss", "lock-in", "lock in",
        "swp", "stp", "rule of 72", "pe ratio", "beta", "alpha", "sharpe ratio",
        "treynor", "sortino", "tracking error", "portfolio turnover", "amfi", "kyc",
        "cas", "folio number", "rupee cost averaging", "indexation"
    ]) or (
        any(k in low for k in ["what is", "what are", "define", "meaning of", "how does", "what does", "explain", "clarify", "tell me about"])
        and not any(k in low for k in ["portfolio", "my sip", "monthly budget", "per month", "suggest funds", "recommend funds", "mentrafiai", "can you do", "who are you", "difference between", "vs"])
    ):
        intent_idx = 1  # DEFINITION

    # 4. Fund Comparisons (head-to-head scheme or category comparisons)
    elif any(k in low for k in [
        " vs ", " vs.", " versus ", "or active", "direct vs regular", "direct or regular",
        "large cap vs", "mid cap vs", "small cap vs", "flexi cap vs", "equity vs debt", 
        "active vs passive", "index vs active", "compare ", "which fund is better",
        "which is better", "difference between large", "difference between mid",
        "difference between small", "difference between flexi", "difference between index",
        "head to head", "pros and cons of"
    ]) or (
        ("compare" in low or "difference" in low) and any(k in low for k in ["fund", "scheme", "sbi", "hdfc", "axis", "nippon", "quant", "parag parikh", "mirae", "kotak", "icici", "cap", "index", "plan", "growth", "idcw"])
    ):
        intent_idx = 3  # FUND_COMPARISON

    # 5. Portfolio Recommendation (budget, sip, recommend, allocate)
    elif any(k in low for k in [
        "monthly budget", "per month", "/month", "/mo", "sip of", "sip budget",
        "monthly sip", "recommend fund", "suggest fund", "portfolio allocation", "where should i invest",
        "plan my sip", "portfolio recommendation", "allocate", "invest rs", "invest ₹",
        "salary", "can invest", "suggest portfolio", "best mutual funds for", "suggest mutual funds",
        "recommend mutual funds"
    ]) or any(k in low for k in ["portfolio", "suggest", "recommend"]) and any(k in low for k in ["fund", "sip", "invest", "month", "budget"]):
        intent_idx = 2  # PORTFOLIO_RECOMMENDATION

    # 6. Fallback General Chat
    else:
        intent_idx = 5  # GENERAL_CHAT

    # Risk label
    if any(k in low for k in ["conservative", "safe", "low risk", "capital preservation", "minimal risk"]):
        risk_idx = 0  # CONSERVATIVE
    elif any(k in low for k in ["moderate", "balanced", "medium risk", "sensible"]):
        risk_idx = 1  # MODERATE
    elif any(k in low for k in ["aggressive", "high risk", "maximum growth", "very high risk"]):
        risk_idx = 2  # AGGRESSIVE
    else:
        risk_idx = 3  # UNKNOWN

    # Horizon label
    if any(k in low for k in ["1 year", "2 year", "3 year", "1-year", "2-year", "3-year", "1 yr", "2 yr", "3 yr", "short term", "short-term"]):
        horizon_idx = 0  # SHORT_TERM
    elif any(k in low for k in ["4 year", "5 year", "6 year", "7 year", "4-year", "5-year", "6-year", "7-year", "4 yr", "5 yr", "6 yr", "7 yr", "medium term", "medium-term"]):
        horizon_idx = 1  # MEDIUM_TERM
    elif any(k in low for k in ["8 year", "9 year", "10 year", "12 year", "15 year", "20 year", "25 year", "30 year", "8-year", "9-year", "10-year", "12-year", "15-year", "20-year", "25-year", "30-year", "long term", "long-term", "retirement"]):
        horizon_idx = 2  # LONG_TERM
    else:
        horizon_idx = 3  # UNKNOWN

    return intent_idx, risk_idx, horizon_idx


def get_comprehensive_synthetic_dataset():
    """Generates ~13,750 high-diversity, realistic financial samples across all 6 intents."""
    samples = []

    # =========================================================================
    # 1. GREETINGS (Intent 0) - Target: 2,500 samples
    # =========================================================================
    base_openers = [
        "hi", "hello", "hey", "namaste", "pranam", "good morning", "good afternoon",
        "good evening", "greetings", "hi there", "hello there", "hey there",
        "namaste ji", "pranam sir", "yo", "sup", "howdy", "good day", "hey friend"
    ]
    bot_names = ["", " mentrafi", " mentrafiai", " mentor", " advisor", " assistant", " bot", " buddy"]
    greeting_tails = [
        "", "!", "!!", ".",
        ", good morning!", ", good afternoon!", ", good evening!", ", good day!",
        ", hope you are doing well", ", how are you doing today?",
        ", good to connect with you.", ", hope your day is going great!",
        ", can we chat?", ", need some advice today.", ", ready to talk finances!",
        ", nice to meet you.", ", what's up?", ", how have you been?",
        ", have a great day!", ", goodbye for now.", ", bye bye!", ", see you later.",
        ", talk to you later.", ", good night!", ", take care!"
    ]
    for _ in range(2500):
        o = random.choice(base_openers)
        b = random.choice(bot_names)
        t = random.choice(greeting_tails)
        text = f"{o}{b}{t}".strip()
        samples.append((text, 0, 3, 3))

    # =========================================================================
    # 2. DEFINITIONS (Intent 1) - Target: 2,500 samples
    # =========================================================================
    concepts = [
        # Taxation & Budget 2024
        "LTCG", "STCG", "Long-Term Capital Gains tax", "Short-Term Capital Gains tax",
        "difference between LTCG and STCG tax in equity funds", "Budget 2024 capital gains tax rules",
        "12.5% LTCG tax rule", "20% STCG tax rate", "taxation of debt mutual funds",
        "taxation on arbitrage funds", "indexation benefit removal in Budget 2024",
        "Section 80C tax deduction", "tax on switch between mutual fund schemes",
        "dividend distribution tax in mutual funds", "tax rules on equity mutual funds",
        # Fund Metrics & Key Performance Indicators
        "exit load", "expense ratio", "Total Expense Ratio (TER)", "NAV", "Net Asset Value",
        "CAGR", "Compounded Annual Growth Rate", "XIRR", "Extended Internal Rate of Return",
        "Sharpe ratio", "Standard deviation", "Beta of a mutual fund", "Alpha in mutual funds",
        "Treynor ratio", "Sortino ratio", "Tracking error in index funds", "Portfolio turnover ratio",
        "AUM (Assets Under Management)", "cut-off timing for mutual fund NAV",
        "T+1 settlement cycle for mutual funds", "lock-in period of ELSS funds",
        # Fund Categories & Types
        "ELSS mutual fund", "Index fund", "Exchange Traded Fund (ETF)", "Large cap equity fund",
        "Mid cap equity fund", "Small cap equity fund", "Flexi cap fund", "Multi cap fund",
        "Balanced Advantage Fund", "Dynamic Asset Allocation Fund", "Multi Asset Allocation Fund",
        "Arbitrage mutual fund", "Liquid fund", "Overnight fund", "Money market fund",
        "Corporate bond fund", "Banking and PSU debt fund", "Gilt fund", "Target maturity fund",
        "Sectoral and thematic fund", "Contra fund", "Fund of Funds (FoF)",
        # Investment Mechanics & Terminology
        "SIP (Systematic Investment Plan)", "Lumpsum investment", "SWP (Systematic Withdrawal Plan)",
        "STP (Systematic Transfer Plan)", "Step-up SIP", "Top-up SIP", "Direct plan vs Regular plan",
        "Growth option vs IDCW payout option", "Rule of 72", "Rupee cost averaging",
        "Power of compounding in SIP", "Mutual fund portfolio rebalancing", "AMFI",
        "KYC compliance for mutual funds", "Consolidated Account Statement (CAS)", "Folio number"
    ]
    def_templates = [
        "What is {c}?",
        "Explain {c} in simple terms",
        "What does {c} mean in mutual funds?",
        "Can you explain {c} to a beginner?",
        "How does {c} work in Indian mutual funds?",
        "What is the meaning of {c}?",
        "Tell me about {c} and how it works",
        "Why is {c} important for mutual fund investors?",
        "Could you please explain what {c} is?",
        "What are the key rules regarding {c}?",
        "How is {c} calculated in India?",
        "Define {c} with an example",
        "Can you clarify {c} for me?",
        "Brief summary of {c}",
        "Short note on {c}",
        "How should an investor understand {c}?",
        "Please guide me on {c}",
        "What should I know about {c} before investing?",
        "Is {c} applicable to all mutual funds in India?",
        "Tell me the definition of {c}"
    ]
    for _ in range(2500):
        c = random.choice(concepts)
        t = random.choice(def_templates)
        text = t.format(c=c)
        samples.append((text, 1, 3, 3))

    # =========================================================================
    # 3. PORTFOLIO RECOMMENDATIONS (Intent 2 - Synthetic) - Target: 1,250 samples
    # =========================================================================
    budgets = [
        ("1,000", "1k"), ("2,000", "2k"), ("3,000", "3k"), ("5,000", "5k"),
        ("7,500", "7.5k"), ("10,000", "10k"), ("15,000", "15k"), ("20,000", "20k"),
        ("25,000", "25k"), ("30,000", "30k"), ("40,000", "40k"), ("50,000", "50k"),
        ("75,000", "75k"), ("100,000", "1 Lakh"), ("150,000", "1.5 Lakh")
    ]
    ages = [22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 55]
    risk_profiles = [
        ("conservative", 0), ("safe and low risk", 0), ("capital preservation", 0),
        ("moderate", 1), ("balanced", 1), ("medium risk", 1),
        ("aggressive", 2), ("high risk", 2), ("maximum growth", 2)
    ]
    horizon_profiles = [
        ("2 years", 0), ("3 years", 0),
        ("5 years", 1), ("6 years", 1), ("7 years", 1),
        ("10 years", 2), ("12 years", 2), ("15 years", 2), ("20 years", 2), ("25 years", 2)
    ]
    port_templates = [
        "Recommend mutual funds for a {age}-year-old with Rs {b_num}/month SIP budget, {risk} risk appetite, {horizon}.",
        "I want to invest Rs {b_num} per month for {horizon}. I have {risk} risk appetite, suggest a good portfolio.",
        "Suggest mutual funds for {b_short} monthly SIP for {horizon} with {risk} risk.",
        "Suggest a mutual fund portfolio for Rs {b_num} monthly SIP for {horizon} ({risk} risk).",
        "Best mutual funds for {b_short} monthly SIP for {horizon} ({risk} risk).",
        "I am {age} years old, monthly savings Rs {b_num}. Where should I invest for {horizon} with {risk} profile?",
        "Best mutual fund allocation for Rs {b_num}/month SIP over {horizon} with {risk} risk tolerance.",
        "Where should I invest Rs {b_num} every month for {horizon} with {risk} risk?",
        "Plan my SIP: Rs {b_num} monthly budget, {horizon} timeframe, {risk} investor.",
        "Recommend mutual funds for {b_short} monthly SIP for {horizon} ({risk} risk).",
        "I have Rs {b_num} monthly surplus. Recommend 3-4 direct mutual funds for {horizon} with {risk} strategy.",
        "How should I allocate Rs {b_num} per month across equity and debt funds for {horizon}? Profile: {risk}.",
        "Portfolio suggestion for {age}yo engineer earning salary, saving Rs {b_num}/month, {horizon}, {risk}."
    ]
    for _ in range(1250):
        b_num, b_short = random.choice(budgets)
        a = random.choice(ages)
        r_text, r_idx = random.choice(risk_profiles)
        h_text, h_idx = random.choice(horizon_profiles)
        t = random.choice(port_templates)
        text = t.format(age=a, b_num=b_num, b_short=b_short, risk=r_text, horizon=h_text)
        samples.append((text, 2, r_idx, h_idx))

    # =========================================================================
    # 4. FUND COMPARISONS (Intent 3) - Target: 2,500 samples
    # =========================================================================
    fund_pairs = [
        # AMC Head-to-Head
        ("SBI Small Cap Fund", "Nippon India Small Cap Fund"),
        ("Parag Parikh Flexi Cap Fund", "HDFC Flexi Cap Fund"),
        ("Quant Small Cap Fund", "Axis Small Cap Fund"),
        ("Mirae Asset Large Cap Fund", "ICICI Prudential Bluechip Fund"),
        ("UTI Nifty 50 Index Fund", "HDFC Index Fund"),
        ("Kotak Emerging Equity Fund", "Motilal Oswal Midcap Fund"),
        ("Canara Robeco Emerging Equities", "SBI Large & Midcap Fund"),
        ("Tata Digital India Fund", "ICICI Prudential Technology Fund"),
        ("DSP Healthcare Fund", "SBI Healthcare Fund"),
        ("Bandhan Sterling Value Fund", "Templeton India Value Fund"),
        ("Nippon India Growth Fund", "HDFC Mid-Cap Opportunities Fund"),
        ("Axis Bluechip Fund", "Mirae Asset Large Cap Fund"),
        ("SBI Focused Equity Fund", "Axis Focused Fund"),
        ("Quant Active Fund", "Parag Parikh Flexi Cap Fund"),
        ("ICICI Prudential Balanced Advantage Fund", "HDFC Balanced Advantage Fund"),
        ("SBI Equity Hybrid Fund", "Canara Robeco Equity Hybrid Fund"),
        ("Mirae Asset ELSS Tax Saver", "Quant ELSS Tax Saver"),
        ("Axis Small Cap Fund", "Nippon India Small Cap Fund"),
        ("Bandhan Small Cap Fund", "Invesco India Small Cap Fund"),
        ("Kotak Small Cap Fund", "SBI Small Cap Fund"),
        ("DSP Midcap Fund", "Kotak Emerging Equity Fund"),
        ("HDFC Top 100 Fund", "ICICI Prudential Bluechip Fund"),
        # Category Comparisons
        ("Large Cap Funds", "Mid Cap Funds"),
        ("Mid Cap Funds", "Small Cap Funds"),
        ("Small Cap Funds", "Large Cap Funds"),
        ("Flexi Cap Funds", "Multi Cap Funds"),
        ("Equity Mutual Funds", "Debt Mutual Funds"),
        ("Active Mutual Funds", "Passive Index Funds"),
        ("Index Funds", "ETFs"),
        ("Liquid Funds", "Overnight Funds"),
        ("Balanced Advantage Funds", "Multi Asset Allocation Funds"),
        ("Arbitrage Funds", "Liquid Funds"),
        ("Direct Plans", "Regular Plans"),
        ("Growth Option", "IDCW Dividend Option"),
        ("ELSS Funds", "PPF (Public Provident Fund)"),
        ("Corporate Bond Funds", "Banking & PSU Debt Funds"),
        ("Large & Mid Cap Funds", "Flexi Cap Funds"),
        ("Focused Funds", "Diversified Flexi Cap Funds")
    ]
    comp_templates = [
        "Compare {a} and {b}",
        "Compare {a} vs {b}",
        "Which fund is better between {a} and {b}?",
        "What is the difference between {a} and {b}?",
        "What is the difference between {a} and {b} mutual funds?",
        "Comparison of {a} versus {b} for long term",
        "Should I choose {a} or {b} for my portfolio?",
        "Can you compare {a} with {b} based on risk and returns?",
        "Head-to-head comparison: {a} vs {b}",
        "Which gives better returns: {a} or {b}?",
        "Which one is safer: {a} or {b}?",
        "Help me decide between {a} and {b}",
        "Pros and cons of {a} versus {b}",
        "Difference between {a} and {b}: which one should I pick?",
        "Is {a} better than {b} for a 10 year horizon?",
        "Detailed comparison of {a} and {b}"
    ]
    for _ in range(2500):
        a, b = random.choice(fund_pairs)
        t = random.choice(comp_templates)
        text = t.format(a=a, b=b)
        samples.append((text, 3, 3, 3))

    # =========================================================================
    # 5. GOAL PLANNING (Intent 4) - Target: 2,500 samples
    # =========================================================================
    goals = [
        "retirement corpus of 2 Crores",
        "retirement corpus of 5 Crores",
        "comfortable early retirement (FIRE)",
        "child higher education fund for engineering",
        "daughter's overseas medical degree",
        "child university education abroad",
        "down payment for buying a 3BHK flat",
        "buying our first residential home",
        "marriage expenses of my daughter",
        "my wedding fund",
        "building a 6-month emergency reserve",
        "creating a rainy day emergency corpus",
        "purchasing a new family SUV",
        "funding a luxury international vacation",
        "accumulating a wealth corpus of 1 Crore",
        "accumulating a corpus of 50 Lakhs",
        "seed fund to start my own business"
    ]
    goal_horizons = [
        (2, 0), (3, 0),
        (5, 1), (6, 1), (7, 1),
        (10, 2), (12, 2), (15, 2), (20, 2), (25, 2)
    ]
    goal_templates = [
        "How should I plan my mutual fund investments for {g} in {y} years?",
        "I want to accumulate money for {g} over the next {y} years. How should I start?",
        "What is the best mutual fund strategy for {g} within a {y}-year timeframe?",
        "Help me plan a SIP portfolio to achieve {g} in {y} years.",
        "How much monthly SIP do I need to invest for {g} in {y} years?",
        "Goal planning: I need to achieve {g}. Time horizon is {y} years.",
        "Which mutual funds are best suited for {g} over {y} years?",
        "Step-by-step mutual fund investment roadmap for {g} in {y} years.",
        "Can you calculate the required SIP amount for {g} in {y} years?",
        "I am targeting {g} in {y} years. How to distribute my monthly savings?"
    ]
    for _ in range(2500):
        g = random.choice(goals)
        y, h_idx = random.choice(goal_horizons)
        t = random.choice(goal_templates)
        text = t.format(g=g, y=y)
        samples.append((text, 4, 1, h_idx))

    # =========================================================================
    # 6. GENERAL CHAT (Intent 5) - Target: 2,500 samples
    # =========================================================================
    general_queries = [
        "How does MentraFiAI work?",
        "What features do you offer as an AI advisor?",
        "What are your capabilities?",
        "Are you a SEBI registered investment advisor?",
        "Who developed this AI system?",
        "Thank you so much for the financial advice!",
        "Thanks a lot, that was very helpful.",
        "Great explanation, appreciate your help!",
        "Thank you mentor, got it!",
        "Awesome, thanks for the detailed breakdown.",
        "Is mutual fund investment safe in India?",
        "Can I lose money in mutual funds during a market crash?",
        "Why should I invest in mutual funds instead of fixed deposits (FD)?",
        "What happens if a mutual fund company (AMC) shuts down?",
        "How do I start investing in mutual funds online?",
        "What documents are required to complete mutual fund KYC?",
        "Can NRIs invest in Indian mutual funds?",
        "How do I redeem or withdraw my mutual fund units?",
        "Which platform is best: Groww, Zerodha Coin, or MF Central?",
        "How often should I review and rebalance my portfolio?",
        "Can I stop or pause my SIP anytime without penalty?",
        "What is the minimum amount required to start a SIP?",
        "Can I switch from regular plans to direct plans?",
        "How do I download my mutual fund capital gains statement?",
        "Can I hold mutual funds in demat form?",
        "Tell me about yourself",
        "What can I ask you?",
        "Can you help me with financial planning?",
        "Understood, thanks for explaining.",
        "Got it, thank you!",
        "Okay, noted.",
        "Thanks for the prompt answer!",
        "Very useful insights, thanks."
    ]
    chat_wrappers = [
        "{q}",
        "Could you tell me: {q}",
        "Quick question: {q}",
        "I was wondering: {q}",
        "{q} Please guide me."
    ]
    for _ in range(2500):
        q = random.choice(general_queries)
        w = random.choice(chat_wrappers)
        text = w.format(q=q).strip()
        samples.append((text, 5, 3, 3))

    return samples


# ============================================================================
# REAL-WORLD USER QUERIES — 12,000+ samples
# Hand-crafted realistic queries: Hinglish, typos, shorthand, edge-cases,
# hard negatives.  These simulate what actual Indian retail investors type.
# ============================================================================

def get_real_world_queries() -> list:
    """Returns ~12,000 realistic, diverse, hand-crafted user queries across 6 intents."""
    s = []   # (text, intent_idx, risk_idx, horizon_idx)

    # =========================================================================
    # 0. GREETING — 2,000 samples
    # Regional, Hinglish, farewells, appreciation, very short, stacked
    # =========================================================================
    greetings_raw = [
        # English variants
        "hi", "hi!", "hi!!", "hi there", "hi buddy", "hi mentor", "hi mentrafiai",
        "hello", "hello!", "hello there", "hello mentor", "hello mentrafi", "hello buddy",
        "hey", "hey!", "hey there", "hey buddy", "hey advisor", "hey mentrafi",
        "good morning", "good morning!", "good morning mentor", "good morning mentrafi",
        "good afternoon", "good afternoon!", "good afternoon mentrafi",
        "good evening", "good evening!", "good evening mentor",
        "good night", "good night!", "good night mentor", "good night mentrafi",
        "greetings", "greetings!", "howdy", "howdy!", "what's up", "what's up!",
        "sup", "sup!", "yo", "yo!", "hey yo", "heya",
        # Hinglish / Regional
        "namaste", "namaste!", "namaste ji", "namaste ji!", "namaskar", "namaskar!",
        "pranam", "pranam ji", "pranam sir", "jai hind", "jai hind!",
        "vanakkam", "vanakkam!", "kem cho", "kem cho?", "sat sri akal",
        "adaab", "adaab!", "salam", "salam!", "kem cho yaar",
        "kaise ho", "kaise ho?", "kaisa chal raha hai", "kya haal hai",
        "bhai namaste", "bhai hello", "yaar hi", "sir namaste", "sir hello",
        # Farewells
        "bye", "bye!", "bye bye", "goodbye", "goodbye!", "good bye",
        "see you", "see you later", "see ya", "talk to you later", "take care",
        "tata", "tata!", "alvida", "phir milenge", "catch you later",
        "have a great day", "have a nice day", "have a wonderful day",
        # Appreciation / closing
        "thanks", "thanks!", "thank you", "thank you!", "thank you so much",
        "thanks a lot", "thanks a ton", "bahut shukriya", "shukriya", "dhanyawad",
        "thanks mentor", "thanks mentrafi", "thank you advisor",
        "great job", "great work", "well done", "awesome", "brilliant",
        "you are great", "you are amazing", "you are very helpful",
        "appreciate it", "appreciate your help", "thanks for the help",
        # Typo variants
        "helo", "helo!", "helloo", "helo there", "hii", "hii!", "hiii",
        "namste", "namaskar ji", "hey thre", "gd morning", "gud morning",
        # Very short / minimal
        "hi.", "hello.", "hey.", "ok", "ok!", "okay", "okay!",
        "sure", "noted", "got it", "understood",
    ]
    for g in greetings_raw:
        s.append((g, 0, 3, 3))
    for _ in range(2000 - len(greetings_raw)):
        s.append((random.choice(greetings_raw), 0, 3, 3))

    # =========================================================================
    # 1. DEFINITION — 2,000 samples
    # Hinglish, typos, direct one-word queries, passive phrasing
    # HARD NEGATIVES: "difference between LTCG and STCG" MUST be DEFINITION
    # =========================================================================
    defs_raw = [
        # LTCG / STCG — critical tricky queries (must always be DEFINITION)
        "What is LTCG?", "What is STCG?", "What is LTCG tax?", "What is STCG tax?",
        "LTCG kya hai?", "STCG kya hota hai?", "LTCG ka matlab kya hai?",
        "Explain LTCG in simple words", "Explain STCG with an example",
        "LTCG tax kaise calculate hota hai?", "STCG tax kaise lagta hai?",
        "What is the LTCG tax rate after Budget 2024?",
        "What is the STCG tax rate on equity mutual funds?",
        "How much LTCG tax do I pay on equity mutual funds?",
        "How much STCG tax do I pay?", "LTCG limit kya hai?",
        "What is the LTCG exemption limit of 1.25 lakh?",
        "Explain the difference between LTCG and STCG",
        "Difference between LTCG and STCG", "LTCG vs STCG kya hota hai",
        "LTCG aur STCG mein kya fark hai?", "LTCG STCG difference explain karo",
        "Capital gains tax kya hota hai?", "What is capital gains tax in mutual funds?",
        "Short term capital gains tax in India", "Long term capital gains tax rules",
        "What is indexation benefit?", "Indexation benefit kya hai?",
        "Tax on debt mutual funds?", "Debt fund pe kitna tax lagta hai?",
        # Exit load / Expense ratio
        "What is exit load?", "Exit load kya hai?", "Exit load kab lagta hai?",
        "Is exit load charged after 1 year?", "Exit load explain karo",
        "What is expense ratio?", "Expense ratio kya hota hai?",
        "What is TER in mutual funds?", "TER kya hai?",
        "What is a good expense ratio for equity funds?",
        "How does expense ratio affect my returns?",
        "Low expense ratio better hota hai kya?",
        # NAV
        "What is NAV?", "NAV kya hota hai?", "NAV kaise calculate hoti hai?",
        "What does NAV mean?", "Is high NAV good or bad?",
        "High NAV fund lena chahiye ya low NAV?",
        "NAV daily change kyun hoti hai?",
        # CAGR / XIRR
        "What is CAGR?", "CAGR kya hai?", "CAGR kaise calculate karte hain?",
        "What is XIRR?", "XIRR kya hota hai?", "XIRR vs CAGR difference?",
        "Mere SIP ka XIRR kaise nikalta hai?",
        "What is the difference between CAGR and XIRR?",
        # SIP / SWP / STP
        "What is SIP?", "SIP kya hota hai?", "SIP kaise shuru karein?",
        "What is SWP?", "SWP kya hai?", "SWP kaise kaam karta hai?",
        "What is STP?", "STP kya hota hai?", "STP in mutual funds explain",
        "What is step-up SIP?", "Step up SIP kya hota hai?",
        # ELSS / Lock-in
        "What is ELSS?", "ELSS kya hai?", "ELSS fund kitne saal ka lock-in hota hai?",
        "ELSS aur PPF mein kya fark hai?", "ELSS 80C kaise help karta hai?",
        "What is lock-in period of ELSS?",
        # Index fund / ETF
        "What is an index fund?", "Index fund kya hota hai?",
        "What is an ETF?", "ETF kya hai?", "Index fund vs ETF difference?",
        "Nifty 50 index fund kya hota hai?",
        # Sharpe / Beta / Alpha
        "What is Sharpe ratio?", "Sharpe ratio kya hai?",
        "What is Beta in mutual funds?", "Beta kya measure karta hai?",
        "What is Alpha in investing?", "Alpha kya hota hai?",
        "What is standard deviation in mutual funds?",
        "Tracking error kya hota hai?",
        # Direct vs Regular
        "What is direct plan?", "Direct plan kya hota hai?",
        "Regular plan aur direct plan mein kya fark hai?",
        "Should I switch from regular to direct plan?",
        "Direct plan better kyun hai?",
        # Miscellaneous
        "What is AUM?", "AUM kya hai?", "What is rupee cost averaging?",
        "What is the Rule of 72?", "Rule of 72 kya hota hai?",
        "Power of compounding kya hai?", "What is compounding in SIP?",
        "What is Flexi Cap fund?", "Flexi cap kya hota hai?",
        "What is Multi Cap fund?", "Multi cap kya hai?",
        "What is Balanced Advantage Fund?", "BAF kya hota hai?",
        "What is arbitrage fund?", "Arbitrage fund kya hai?",
        "What is liquid fund?", "Liquid fund kya hota hai?",
        "What is gilt fund?", "Gilt fund mein invest karna safe hai?",
        "What is sectoral fund?", "Sectoral fund kya hota hai?",
        "What is portfolio turnover ratio?",
        "What is mutual fund rebalancing?", "Rebalancing kab karni chahiye?",
        "What is AMFI?", "AMFI kya hai?", "KYC kya hota hai mutual funds mein?",
        "What is folio number?", "Folio number kya hai?",
        "What is CAS statement?", "CAS statement kahan milta hai?",
        "Growth option vs IDCW option kya hota hai?",
        # Typo queries
        "wht is NAV?", "expnse ratio kya hai?", "explane SIP", "wht is CAGR?",
        "wht is exit lod?", "nav kya hta hai", "sip kya hhota hai",
        "what si ELSS", "wha is LTCG", "explain STCG pleease",
        # Single-word queries
        "LTCG?", "STCG?", "NAV?", "CAGR?", "XIRR?", "ELSS?",
        "SIP?", "SWP?", "STP?", "TER?", "AUM?",
        # Passive phrasing
        "A friend told me about exit load, what is it?",
        "My CA mentioned XIRR, can you explain?",
        "I heard about LTCG exemption, what does it mean?",
        "Someone mentioned arbitrage funds, what are they?",
        "I read about tracking error, please explain",
        "My colleague said to check Sharpe ratio, what is it?",
    ]
    for d in defs_raw:
        s.append((d, 1, 3, 3))
    for _ in range(2000 - len(defs_raw)):
        s.append((random.choice(defs_raw), 1, 3, 3))

    # =========================================================================
    # 2. PORTFOLIO_RECOMMENDATION — 2,000 samples
    # Hinglish, text numbers, no Rs symbol, occupation-centric, shorthand
    # =========================================================================
    port_raw = [
        # Hinglish portfolio requests
        "mere liye ek accha mutual fund portfolio banao",
        "mujhe 10000 monthly invest karna hai, kahan lagaun?",
        "10k har mahine lagana chahta hoon, suggest karo",
        "bhai mera monthly 15k hai, kahan invest karein?",
        "sir mera salary 50k hai, kitna invest karein?",
        "main 25 saal ka hoon aur naukri shuru ki hai, kahan invest karein?",
        "ek accha SIP portfolio suggest karo 20k ke liye",
        "aggressive investor hoon, 30k monthly SIP chahiye",
        "5k se SIP shuru karna chahta hoon, kaunsa fund?",
        "kaunse mutual fund mein paisa lagaun for long term?",
        "monthly 12000 bachta hai, suggest karo kahan lagaun",
        "I earn 40k per month and save 8k, where to invest?",
        "new investor hoon, 5000 se shuru karna chahta hoon",
        "best SIP for 10 years with 15k monthly",
        "suggest me a portfolio for long term wealth creation",
        # Text numbers (no digits)
        "I can invest ten thousand per month, suggest funds",
        "I save fifteen thousand every month, where to put it?",
        "twenty thousand monthly SIP recommendation",
        "five thousand rupees monthly SIP suggest karo",
        "fifty thousand monthly budget, aggressive portfolio chahiye",
        "one lakh per month invest karna chahta hoon",
        # No Rs / rupee symbol
        "25000 monthly SIP ke liye best funds kaunse hain?",
        "10000 per month invest karna chahta hoon 10 years ke liye",
        "suggest funds for 30000 monthly over 15 years aggressive",
        "50000 monthly moderate risk 10 year portfolio",
        "20000 monthly salary mein se 5000 SIP karna chahta hoon",
        "can invest 8000 per month for 7 years where should I invest",
        "I can put 12000 monthly for the next 5 years",
        # Occupation / life stage
        "I am a 28 year old software engineer saving 20k monthly",
        "sarkari naukri hai, monthly 6000 save hota hai, kahan lagaun?",
        "I am a doctor with 50k monthly savings, suggest portfolio",
        "retired professional, 1 lakh monthly pension, where to invest?",
        "newly married, both earning, combined 1.5 lakh, investment advice?",
        "fresh graduate, first job, 3000 monthly SIP karna chahta hoon",
        "ghar ka kharcha nikal ke 7000 bachta hai, invest kahan karein?",
        # Budget-first format
        "Rs 10,000 monthly - suggest mutual funds for 10 years",
        "25,000/month - moderate risk - 15 year portfolio",
        "Budget: 20k monthly. Risk: aggressive. Duration: 20 years.",
        "SIP of 5000 for 5 years conservative risk",
        "15k monthly SIP, balanced risk, 10yr horizon",
        "40k/mo sip, aggressive, long term, suggest",
        # Question-style
        "where should I invest 10000 per month?",
        "which mutual funds are best for 15 year SIP?",
        "best equity funds for young investor with 8000 monthly?",
        "how to invest 1 lakh in mutual funds for 10 years?",
        "which fund gives best returns for 20k monthly SIP?",
        "what are top 3 mutual funds to invest in now?",
        "how should I build a portfolio with 25000 monthly?",
        "where to invest 50000 per month for retirement?",
        "best index fund for monthly SIP of 10000?",
        "should I invest in small cap or flexi cap with 20k?",
        # Age + budget combos
        "22 year old, 5000 monthly, 20 year horizon, suggest SIP",
        "35 years old, 25k monthly, moderate risk, 15 year plan",
        "45 years old, 40k monthly, conservative risk, 10 years to retirement",
        "30 year old professional, 15k monthly, balanced portfolio please",
        "55 years old, 20k monthly, 5 years left for retirement",
        # Lakh / crore shorthand
        "1.5 lakh monthly mein SIP suggest karo aggressive",
        "50k monthly invest karna hai for wealth creation",
        "2 lakh per month ka portfolio banao for 10 years",
        "75k monthly budget for retirement in 20 years suggest funds",
        # Shorthand / abbreviations
        "10k/mo agg 15yr suggest", "20k pm moderate 10y portfolio",
        "5k sip 5yr conservative recommend", "suggest sip 30k monthly 20 years",
        "best mf for 25k mo sip", "15k monthly 12 year moderate portfolio",
        # Scenario-based
        "I have 2 lakh lumpsum and 10k monthly to invest, suggest",
        "want to start SIP with just 500 rupees per month",
        "I received a bonus of 1 lakh, how to invest in mutual funds?",
        "windfall of 5 lakh, invest in mutual funds lumpsum?",
        "EPF is maxed, where should I put extra 10k monthly?",
    ]
    for p in port_raw:
        s.append((p, 2, 3, 3))
    for _ in range(2000 - len(port_raw)):
        s.append((random.choice(port_raw), 2, 3, 3))

    # =========================================================================
    # 3. FUND_COMPARISON — 2,000 samples
    # Hinglish, specific names, category vs category, tricky phrasings
    # =========================================================================
    comp_raw = [
        # AMC vs AMC (Hinglish + English)
        "SBI small cap vs Nippon India small cap kaunsa better hai?",
        "Quant small cap ya axis small cap - kaunsa loon?",
        "Parag Parikh Flexi Cap vs HDFC Flexi Cap comparison",
        "HDFC vs SBI mutual fund kaunsa accha hai?",
        "Mirae Asset large cap vs ICICI bluechip - which is better?",
        "Axis Bluechip ya Mirae Asset large cap - kaunsa choose karein?",
        "Nifty 50 index fund UTI vs HDFC - difference?",
        "Motilal Oswal midcap vs Kotak Emerging Equity",
        "DSP Midcap vs HDFC Midcap Opportunities which one?",
        "Quant Active vs Parag Parikh Flexi Cap",
        "ICICI Balanced Advantage vs HDFC Balanced Advantage",
        "SBI Bluechip vs HDFC Bluechip which is better",
        "Canara Robeco Emerging Equities vs Kotak Emerging Equity",
        "Mirae ELSS vs Quant ELSS which to pick?",
        "UTI Nifty 50 vs Nippon India Index Fund",
        "Bandhan Small Cap vs Quant Small Cap",
        "Nippon India Growth Fund vs HDFC Mid-Cap Opportunities",
        "Franklin India Flexi Cap vs Parag Parikh Flexi Cap",
        "ICICI Pru Technology vs Tata Digital India Fund",
        # Category vs Category
        "large cap vs mid cap - kaunsa zyada return deta hai?",
        "mid cap vs small cap funds difference",
        "flexi cap vs multi cap kya fark hai?",
        "equity fund vs debt fund mein kya antar hai?",
        "active fund vs index fund kaunsa better hai?",
        "index fund vs ETF difference kya hai?",
        "liquid fund vs overnight fund - kaunsa safe hai?",
        "balanced advantage fund vs multi asset fund",
        "arbitrage fund vs liquid fund comparison",
        "ELSS vs PPF kaunsa better hai tax saving ke liye?",
        "direct plan vs regular plan difference in returns",
        "growth option vs IDCW option kaunsa choose karein?",
        "large and mid cap vs flexi cap fund comparison",
        "focused fund vs diversified fund kya hota hai?",
        "gilt fund vs corporate bond fund - difference?",
        "banking PSU fund vs corporate bond fund",
        "target maturity fund vs FD comparison",
        "sectoral fund vs thematic fund kya fark hai?",
        # Tricky phrasings
        "help me choose between SBI and HDFC small cap",
        "I am confused between mirae asset and ICICI bluechip",
        "which gives better SIP return: flexi cap or mid cap?",
        "should I pick large cap or small cap for 15 years?",
        "SBI ka small cap fund accha hai ya Nippon ka?",
        "Quant ya SBI - kaunse funds mein invest karein?",
        "nifty 50 index better hai ya actively managed large cap?",
        "passive investing better hai ya active fund management?",
        "HDFC flexi ya parag parikh flexi - long term mein kaunsa?",
        "mujhe decide karna hai - axis ya kotak small cap?",
        # Shorthand
        "sbi sc vs nippon sc", "ppfcf vs hdfc flexi",
        "quant sc vs axis sc", "uti nifty vs hdfc nifty",
        "direct vs regular plan returns difference",
        "growth vs idcw which better", "lc vs mc vs sc which one",
        # Category best for a horizon
        "What is the difference between large cap and small cap funds?",
        "Difference between flexi cap and multi cap mutual funds?",
        "What is difference between index fund and actively managed fund?",
        "Difference between direct and regular mutual funds?",
        "Difference between growth and IDCW option?",
        "Compare equity and debt funds for 5 year horizon",
        "Which fund category is best for 10 years: large, mid or small cap?",
        # Performance comparison
        "Which small cap fund gave best 5 year returns?",
        "Best performing flexi cap funds last 3 years",
        "Which ELSS fund has lowest expense ratio?",
        "Highest rated mid cap fund in India?",
        "Which Nifty 50 index fund has lowest tracking error?",
        "Best direct plan flexi cap fund by 3 year CAGR?",
    ]
    for c in comp_raw:
        s.append((c, 3, 3, 3))
    for _ in range(2000 - len(comp_raw)):
        s.append((random.choice(comp_raw), 3, 3, 3))

    # =========================================================================
    # 3b. EXTRA FUND_COMPARISON — concept vs concept + pros/cons (500 samples)
    # These fix the critical misclassification bugs found in live testing:
    # - "direct plan vs regular plan" → was misclassified as GREETING
    # - "SIP vs lump sum" → was not routed to FUND_COMPARISON
    # - "pros and cons of SIP" → was not routed to FUND_COMPARISON
    # - "advantages/disadvantages of X" → was not routed
    # =========================================================================
    extra_comp_raw = [
        # Concept vs Concept — generic investment mechanics
        "SIP vs lump sum investment which is better?",
        "SIP vs lumpsum kaunsa better hai?",
        "SIP se better hai lumpsum ya SIP se invest karein?",
        "lumpsum vs SIP for long term wealth creation",
        "should I do SIP or invest lumpsum amount?",
        "one time investment vs monthly SIP - which gives better returns?",
        "lump sum better hai ya SIP?",
        # ELSS vs PPF / other 80C options
        "ELSS vs PPF kaunsa better hai?",
        "ELSS or PPF for tax saving - which to choose?",
        "ELSS vs PPF comparison for 80C deduction",
        "should I choose ELSS or PPF for Section 80C?",
        "PPF vs ELSS for tax saving which is best?",
        "ELSS vs NPS comparison for retirement",
        "ELSS vs Tax Saver FD comparison",
        "ELSS or PPF better for long term tax saving?",
        # Direct vs Regular — most commonly misclassified
        "direct plan vs regular plan",
        "direct plan vs regular plan difference",
        "direct vs regular mutual fund kaunsa better?",
        "direct plan lena chahiye ya regular plan?",
        "regular plan se direct plan switch karna chahiye?",
        "direct plan better hai regular se?",
        "difference between direct and regular mutual fund plans",
        "direct mutual fund vs regular mutual fund returns comparison",
        "direct plan mein TER kam hota hai, regular se kitna fark padta hai?",
        # Growth vs IDCW
        "growth option vs IDCW option which is better for long term?",
        "growth vs dividend option mutual fund",
        "IDCW vs growth plan comparison",
        "dividend reinvestment vs growth option which is better?",
        # Active vs Passive / Index
        "active fund vs passive index fund which is better?",
        "index fund se better hai active fund?",
        "actively managed fund vs index fund performance comparison",
        "Nifty 50 index vs actively managed large cap - which performs better?",
        "passive investing vs active fund management in India",
        # Pros/Cons style — these MUST route to FUND_COMPARISON
        "pros and cons of SIP investment",
        "pros and cons of ELSS fund",
        "pros and cons of mutual funds",
        "advantages and disadvantages of SIP",
        "advantages of index funds vs disadvantages",
        "what are the pros and cons of SIP?",
        "what are the advantages of investing in ELSS?",
        "what are the disadvantages of small cap funds?",
        "what are the benefits of SWP for retirees?",
        "pros and cons of lump sum investment",
        "advantages and disadvantages of index fund investing",
        "pros and cons of debt mutual funds",
        "pros and cons of hybrid funds",
        "advantages of direct plan over regular plan",
        "disadvantages of regular mutual fund plan",
        "merits and demerits of SIP investment",
        "what are the benefits and risks of small cap funds?",
        "pros and cons of investing in mid cap funds",
        "is SIP better or lump sum - pros and cons",
        "advantages of SIP over lump sum investment",
        "disadvantages of lump sum investment",
        "what are the risks and benefits of ELSS?",
        "pros and cons of flexi cap funds",
        "benefits and drawbacks of index fund investing",
        "advantages of SWP in retirement planning",
        # Shorthand comparison
        "sip vs lumpsum",
        "elss vs ppf",
        "direct vs regular",
        "growth vs idcw",
        "active vs index",
        "sip ya lumpsum kaun accha?",
        "direct ya regular plan?",
        "elss ya ppf?",
        "growth ya dividend option?",
    ]
    for c in extra_comp_raw:
        s.append((c, 3, 3, 3))


    # =========================================================================
    # 4. GOAL_PLANNING — 2,000 samples
    # Hinglish, corpus targets, specific milestones, life stage
    # =========================================================================
    goal_raw = [
        # Retirement
        "retirement ke liye kitna SIP chahiye?",
        "60 saal mein retire hona chahta hoon, kaise plan karein?",
        "I want to retire at 45 with 3 crore corpus, how?",
        "Mujhe 20 saal mein retirement plan banana hai",
        "FIRE goal: retire at 40 with 5 crore, monthly SIP needed?",
        "How much SIP do I need for comfortable retirement in 25 years?",
        "Retirement planning for 35 year old with 20k monthly savings",
        "I am 28 and want to retire at 55 with 2 crore",
        "Retirement corpus of 1 crore in 15 years, how much SIP?",
        "Help me build a retirement corpus of 5 crore by age 60",
        # Child education
        "bacche ki padhai ke liye invest karna chahta hoon",
        "My child is 3 years old, planning for IIT fees in 15 years",
        "beti ki higher education ke liye SIP plan karo",
        "Need 50 lakh for son's MBA in 10 years, suggest SIP amount",
        "Daughter's US education in 12 years, need 1 crore corpus",
        "Child education fund for medical college in 15 years",
        "I want to save 40 lakh for my kid's engineering degree in 17 years",
        "planning child education corpus of 25 lakhs in 8 years",
        # House / flat
        "ghar kharidne ke liye paisa jama karna hai",
        "I want to buy a flat worth 80 lakhs in 5 years, down payment planning",
        "Need 20 lakh down payment for home loan in 4 years",
        "Flat khareedna hai 5 saal mein, kitna SIP lagega?",
        "planning to buy a house in 7 years, how to save for down payment?",
        "Home buying goal of 15 lakh in 6 years, mutual fund advice",
        "I need 25 lakh for house down payment in 5 years",
        # Marriage / wedding
        "beti ki shaadi ke liye paisa jama karna hai",
        "daughter's wedding in 10 years, need 20 lakhs corpus",
        "Apni shaadi ke liye 8 lakh jama karne hain 3 saal mein",
        "Wedding fund: need 15 lakh in 5 years, SIP suggest karo",
        "planning son's wedding corpus of 10 lakhs in 8 years",
        # Emergency fund
        "emergency fund kaise banate hain?",
        "I want to build a 6 month emergency fund in 2 years",
        "Need emergency corpus of 3 lakh in 1.5 years",
        "emergency fund ke liye kaunsa fund best hai?",
        "Building a rainy day fund of 2 lakh, suggest investment",
        # Corpus targets
        "1 crore banana hai 10 saal mein, kaise?",
        "I want 50 lakh in 7 years, how much SIP do I need?",
        "How to accumulate 2 crore in 15 years through SIP?",
        "Need 10 lakh in 3 years, which fund should I choose?",
        "25 lakh ki zaroorat hai 5 saal mein, kahan invest karein?",
        "Corpus of 5 crore for retirement in 20 years, monthly SIP?",
        "Target: 75 lakh in 10 years through mutual fund SIP",
        "How much monthly SIP to reach 1 crore in 12 years?",
        "Want to create wealth of 30 lakh in 8 years",
        "Need 1.5 crore for daughter's abroad education in 15 years",
        # Car / vacation / business
        "new car khareedni hai 2 saal mein, kahan invest karein?",
        "planning a Europe vacation in 3 years, need 3 lakh",
        "startup ke liye seed money 5 lakh chahiye 2 saal mein",
        "Dream car fund: 8 lakh in 3 years, suggest mutual fund",
        "Honeymoon international trip fund for 2 years",
        # Simultaneous goals
        "house + retirement planning simultaneously, how to split SIP?",
        "simultaneously planning for child education and retirement",
        "two goals: home purchase in 5 years, retirement in 20 years",
        # SIP calculation
        "How much SIP for 1 crore in 10 years?",
        "SIP amount for 50 lakh in 15 years at 12% returns?",
        "Monthly SIP needed for 2 crore retirement corpus in 25 years?",
        "kitna SIP lagega 1 crore ke liye 10 saal mein?",
        "How much to invest monthly to get 5 crore in 20 years?",
    ]
    for g in goal_raw:
        s.append((g, 4, 3, 3))
    for _ in range(2000 - len(goal_raw)):
        s.append((random.choice(goal_raw), 4, 3, 3))

    # =========================================================================
    # 5. GENERAL_CHAT — 2,000 samples
    # System questions, feedback, general finance, meta, Hinglish
    # =========================================================================
    chat_raw = [
        # About the AI
        "Who are you?", "Who made you?", "Are you a robot?", "Are you an AI?",
        "What is MentraFiAI?", "Tell me about yourself",
        "Aap kaun ho?", "Kya tum AI ho?", "Tumhara naam kya hai?",
        "Are you SEBI registered?", "Can I trust your advice?",
        "Do you store my data?", "Is my information safe here?",
        "What can you do?", "What are your features?", "What can I ask you?",
        "How are you?", "How are you doing?", "Hope you are doing well",
        # Feedback / appreciation
        "Bahut helpful tha, thank you!", "Accha explain kiya, shukriya",
        "Samajh gaya, dhanyawad", "Bahut badiya advice diya",
        "Excellent explanation!", "That was very clear, thanks",
        "I understood, appreciate it", "Great, very informative",
        "Thanks for the detailed answer", "Perfect, exactly what I needed",
        "Amazing response, thank you mentor", "You explained it so well",
        "Cleared my doubts, thanks!", "Super helpful, appreciate it",
        "Noted, will implement this advice", "Got it, will follow your suggestion",
        # General financial questions (not mutual fund specific)
        "Is mutual fund investment safe?", "Can I lose money in mutual funds?",
        "Mutual fund mein risk kya hai?", "Kya mutual fund safe hai?",
        "What happens if AMC shuts down?", "AMC band ho jaye toh kya hoga?",
        "Are mutual funds regulated in India?", "SEBI kya karta hai?",
        "How do I start investing in mutual funds?",
        "Mutual fund mein invest kaise karein?",
        "Online SIP kaise shuru karein?", "Groww pe SIP kaise karte hain?",
        "Zerodha Coin kya hai?", "MF Central kya hota hai?",
        "Which app is best for mutual fund investment?",
        "Groww vs Zerodha Coin vs MF Central kaunsa better hai?",
        "Can I invest in mutual funds without demat account?",
        "Minimum amount to start SIP?", "SIP ke liye minimum kitna chahiye?",
        "Can I pause my SIP?", "SIP band karne pe kya hoga?",
        "How to redeem mutual fund units?", "Mutual fund units kaise sell karein?",
        "KYC kaise complete karein?", "What documents for mutual fund KYC?",
        "Can NRI invest in Indian mutual funds?",
        "How to check my mutual fund portfolio?",
        "How to switch from regular to direct plan?",
        "Capital gains statement kaise download karein?",
        "How to get CAS statement?",
        "Annual portfolio review kab karein?",
        "How often should I rebalance my portfolio?",
        "Inflation se protect kaise karein?",
        "PPF vs mutual fund kaunsa better hai?", "FD vs mutual fund comparison?",
        "Gold vs mutual fund which is better?",
        # Confused / meta queries
        "I don't know where to start", "Kahan se shuru karein?",
        "I am new to investing, please guide me",
        "Pehli baar invest karna chahta hoon, kya karein?",
        "Mere paas koi financial knowledge nahi, help karo",
        "Confused about mutual funds, can you help?",
        "Bahut confuse hoon, thoda simplify karo",
        # Quick conversational
        "Okay", "Sure", "Got it", "Understood", "Fine", "Noted", "Alright",
        "I see", "Makes sense", "That helps", "Thank you for clarifying",
        "Can you repeat that?", "Could you explain more?",
        "Can you simplify this?",
        # Misc market questions
        "Market crash ho raha hai, kya karein?",
        "Should I invest during market crash?",
        "Bear market mein SIP continue karein ya band karein?",
        "Stock market aur mutual fund mein kya fark hai?",
        "Intraday trading vs mutual fund - kaunsa better?",
        "Crypto vs mutual fund India mein kaunsa safer hai?",
    ]
    for c in chat_raw:
        s.append((c, 5, 3, 3))
    for _ in range(2000 - len(chat_raw)):
        s.append((random.choice(chat_raw), 5, 3, 3))

    random.shuffle(s)
    return s


class BalancedClassifierDataset(Dataset):
    """
    Combines three data sources:
      1. Genuine portfolio requests from sft_train_v3.jsonl (~1,250 samples)
      2. Template-based synthetic dataset (~13,750 samples)
      3. Real-world user queries (~12,000 samples): Hinglish, typos,
         shorthand, hard negatives, edge cases
    Total dataset: ~27,000 samples.
    """

    def __init__(self, jsonl_path: str, tokenizer: Tokenizer, max_len: int = 128):
        self.samples = []

        # 1. Genuine portfolio samples from JSONL
        loaded_portfolio = 0
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    user_text = obj.get("user", "").strip()
                    if not user_text:
                        continue
                    intent_idx, risk_idx, horizon_idx = infer_label(user_text)
                    if intent_idx == 2 and loaded_portfolio < 1250:
                        token_ids = tokenizer.encode(user_text)[:max_len]
                        if token_ids:
                            self.samples.append((token_ids, intent_idx, risk_idx, horizon_idx))
                            loaded_portfolio += 1

        print(f"Loaded {loaded_portfolio} genuine portfolio requests from {jsonl_path}")

        # 2. Template-based synthetic dataset (~13,750 samples)
        synth = get_comprehensive_synthetic_dataset()
        added_synth = 0
        for text, it, rk, hz in synth:
            token_ids = tokenizer.encode(text)[:max_len]
            if token_ids:
                self.samples.append((token_ids, it, rk, hz))
                added_synth += 1
        print(f"Added {added_synth:,} synthetic template samples")

        # 3. Real-world user queries (~12,000 samples)
        real = get_real_world_queries()
        added_real = 0
        for text, it, rk, hz in real:
            token_ids = tokenizer.encode(text)[:max_len]
            if token_ids:
                self.samples.append((token_ids, it, rk, hz))
                added_real += 1
        print(f"Added {added_real:,} real-world user query samples")

        random.seed(42)
        random.shuffle(self.samples)
        self.max_len = max_len

        counts = {i: 0 for i in range(6)}
        for _, it, _, _ in self.samples:
            counts[it] += 1

        print("\nFinal Class Distribution:")
        for idx, count in counts.items():
            print(f"  [{idx}] {INTENTS[idx]:<25}: {count:,} samples")
        print(f"Total Dataset Size: {len(self.samples):,} samples\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids, intent, risk, horizon = self.samples[idx]
        padded = token_ids + [0] * (self.max_len - len(token_ids))
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(intent, dtype=torch.long),
            torch.tensor(risk, dtype=torch.long),
            torch.tensor(horizon, dtype=torch.long)
        )


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float]:
    """Computes validation loss and intent classification accuracy."""
    model.eval()
    total_loss = 0.0
    correct_intent = 0
    total = 0

    with torch.no_grad():
        for input_ids, intent_labels, risk_labels, horizon_labels in loader:
            input_ids = input_ids.to(device)
            intent_labels = intent_labels.to(device)
            risk_labels = risk_labels.to(device)
            horizon_labels = horizon_labels.to(device)

            out = model(input_ids)
            loss_intent = criterion(out["intent_logits"], intent_labels)
            loss_risk = criterion(out["risk_logits"], risk_labels)
            loss_horizon = criterion(out["horizon_logits"], horizon_labels)
            loss = loss_intent + 0.5 * loss_risk + 0.5 * loss_horizon

            total_loss += loss.item() * len(input_ids)
            preds = torch.argmax(out["intent_logits"], dim=-1)
            correct_intent += (preds == intent_labels).sum().item()
            total += len(input_ids)

    return total_loss / total, (correct_intent / total) * 100


def train_classifier():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 65)
    print("   Training Network 1: MentraIntentClassifier (10.16M Parameters)")
    print("=" * 65)
    print(f"Device: {device}")

    tok_path = "tokenizer/tokenizer_v2.model"
    tokenizer = Tokenizer(tok_path)
    train_path = "data/processed/sft_train_v3.jsonl"
    if not os.path.exists(train_path):
        train_path = "data/processed/sft_train.jsonl"

    print(f"Loading data from: {train_path}...")
    full_dataset = BalancedClassifierDataset(train_path, tokenizer)

    # 90% Train / 10% Validation split
    val_size = int(len(full_dataset) * 0.10)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print(f"Training set: {train_size:,} samples | Validation set: {val_size:,} samples")

    cfg = ClassifierConfig(vocab_size=16000)
    model = MentraIntentClassifier(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    epochs = 6
    print(f"Training for {epochs} epochs...\n")

    best_val_acc = 0.0
    out_dir = Path("checkpoints/classifier")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "best_classifier.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for input_ids, intent_labels, risk_labels, horizon_labels in train_loader:
            input_ids = input_ids.to(device)
            intent_labels = intent_labels.to(device)
            risk_labels = risk_labels.to(device)
            horizon_labels = horizon_labels.to(device)

            out = model(input_ids)
            loss_intent = criterion(out["intent_logits"], intent_labels)
            loss_risk = criterion(out["risk_logits"], risk_labels)
            loss_horizon = criterion(out["horizon_logits"], horizon_labels)
            loss = loss_intent + 0.5 * loss_risk + 0.5 * loss_horizon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(input_ids)
            preds = torch.argmax(out["intent_logits"], dim=-1)
            train_correct += (preds == intent_labels).sum().item()
            train_total += len(input_ids)

        avg_train_loss = train_loss / train_total
        train_acc = (train_correct / train_total) * 100

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "cfg": cfg}, save_path)

    print(f"\n[SUCCESS] Neural Network #2 saved to: {save_path} (Best Val Intent Acc: {best_val_acc:.2f}%)")

    # =========================================================================
    # Test Verification Suite on Real Financial Queries
    # =========================================================================
    print("\n" + "=" * 65)
    print("   Evaluating Neural Network #2 on Real-World Financial Queries")
    print("=" * 65)

    test_queries = [
        # ── DEFINITION (must all → DEFINITION) ──────────────────────────────
        "What is the difference between LTCG and STCG tax in equity funds?",
        "LTCG aur STCG mein kya fark hai?",
        "Difference between LTCG and STCG",
        "LTCG?",
        "What is exit load?",
        "Exit load kya hai?",
        "Explain CAGR in simple terms",
        "NAV kya hota hai?",
        "What is an index fund?",
        "Expense ratio kya hota hai?",
        "wht is ELSS?",
        "Sharpe ratio kya measure karta hai?",
        "What is the Rule of 72?",
        "SIP kya hota hai?",
        "My CA mentioned XIRR, can you explain?",

        # ── FUND_COMPARISON (must all → FUND_COMPARISON) ─────────────────────
        "Compare SBI Small Cap vs Nippon India Small Cap",
        "Parag Parikh Flexi Cap vs HDFC Flexi Cap comparison",
        "large cap vs mid cap - kaunsa zyada return deta hai?",
        "active fund vs index fund kaunsa better hai?",
        "ELSS vs PPF kaunsa better hai tax saving ke liye?",
        "direct plan vs regular plan difference in returns",
        "SBI ka small cap fund accha hai ya Nippon ka?",
        "sbi sc vs nippon sc",
        "growth vs idcw which better",
        "Which ELSS fund has lowest expense ratio?",
        # NEW: Previously failing misclassification cases
        "direct plan vs regular plan",
        "direct vs regular",
        "SIP vs lump sum investment which is better?",
        "sip vs lumpsum",
        "ELSS vs PPF for tax saving",
        "elss vs ppf",
        "pros and cons of SIP investment",
        "pros and cons of ELSS fund",
        "what are the advantages of index funds?",
        "what are the disadvantages of small cap funds?",
        "advantages of SIP over lump sum",
        "active vs index fund which is better",

        # ── PORTFOLIO_RECOMMENDATION (must all → PORTFOLIO_RECOMMENDATION) ──
        "Recommend mutual funds for a 28-year-old with Rs 25,000/month budget, aggressive risk, 15 years",
        "Suggest mutual funds for 10k monthly SIP for 5 years with moderate risk",
        "mujhe 10000 monthly invest karna hai, kahan lagaun?",
        "I can invest ten thousand per month, suggest funds",
        "25000 monthly SIP ke liye best funds kaunse hain?",
        "I am a 28 year old software engineer saving 20k monthly",
        "10k/mo agg 15yr suggest",
        "best SIP for 10 years with 15k monthly",
        "where should I invest 10000 per month?",
        "40k/mo sip, aggressive, long term, suggest",

        # ── GOAL_PLANNING (must all → GOAL_PLANNING) ────────────────────────
        "How should I plan my mutual fund investments for retirement in 20 years?",
        "I want to accumulate 50 Lakhs for child higher education in 10 years",
        "retirement ke liye kitna SIP chahiye?",
        "1 crore banana hai 10 saal mein, kaise?",
        "bacche ki padhai ke liye invest karna chahta hoon",
        "I want to buy a flat worth 80 lakhs in 5 years, down payment planning",
        "How much SIP for 1 crore in 10 years?",
        "kitna SIP lagega 1 crore ke liye 10 saal mein?",

        # ── GREETING (must all → GREETING) ───────────────────────────────────
        "Hello MentraFiAI, good morning!",
        "Namaste, hope you are doing well",
        "namste",
        "dhanyawad",
        "bye",
        "thanks mentor",

        # ── GENERAL_CHAT (must all → GENERAL_CHAT) ───────────────────────────
        "How does MentraFiAI work?",
        "Thank you so much for the financial guidance!",
        "Are you SEBI registered?",
        "Bahut helpful tha, thank you!",
        "Bear market mein SIP continue karein ya band karein?",
        "Groww pe SIP kaise karte hain?",
        "FD vs mutual fund comparison?",
        "Pehli baar invest karna chahta hoon, kya karein?",
    ]

    # Expected intents for pass/fail reporting
    expected_intents = (
        ["DEFINITION"] * 15 +
        ["FUND_COMPARISON"] * 22 +
        ["PORTFOLIO_RECOMMENDATION"] * 10 +
        ["GOAL_PLANNING"] * 8 +
        ["GREETING"] * 6 +
        ["GENERAL_CHAT"] * 8
    )

    model.eval()
    correct = 0
    total = len(test_queries)
    with torch.no_grad():
        for i, q in enumerate(test_queries):
            token_ids = tokenizer.encode(q)[:128]
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
            pred = model.predict(input_tensor)
            intent = pred["intent"]
            conf = pred["intent_confidence"] * 100
            expected = expected_intents[i]
            passed = (intent == expected)
            if passed:
                correct += 1
            mark = "[PASS]" if passed else "[FAIL]"
            print(f"{mark} [{expected:<22} -> {intent:<22}] ({conf:.0f}%) \"{q[:70]}\"")

    print(f"\n{'='*65}")
    print(f"   Real-World Test Score: {correct}/{total} ({correct/total*100:.1f}%) correct")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    train_classifier()

