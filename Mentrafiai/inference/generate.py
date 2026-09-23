"""Sampling utilities re-exported for convenience; the actual logic lives in
MentraFiAI.generate() (model/architecture.py) so training and inference share
one implementation."""

import re

import torch


GREETING_REPLY = (
    "Hi! I'm MentraFiAI, your personal mutual fund advisor. \U0001f44b\n"
    "I can help you:\n"
    "  \u2022 Find the right mutual funds for your age, income, and risk level\n"
    "  \u2022 Calculate SIP returns and plan for goals like retirement or education\n"
    "  \u2022 Explain concepts like SIP, NAV, ELSS, expense ratio\n\n"
    "Just tell me: your age, monthly budget, risk appetite, and investment horizon \u2014 "
    "and I'll recommend the best funds for you!"
)
GREETING_RE = re.compile(
    r"^\s*(?:hi+|hello+|hey+|hlo+|hlw+|helo+|howdy|yo+|sup|"
    r"(?:good|gud)\s*(?:mori?ng|mrng|morning|afternoon|aftrnun|evening|evng|day|night)|"
    r"gm|ge|gn|"
    r"what'?s?\s+up|greetings?|namaste|namaskar|namste|pranam|ram\s*ram|radhe\s*radhe|"
    r"kem\s*cho|sat\s*sri\s*akal|vanakkam|adaab|kasa\s*kay|"
    r"thanks?|thank\s+you|dhanyawad|shukriya|"
    r"bye+|goodbye+|cya|see\s+you|tata|alvida)\s*"
    r"[!.,?]*\s*(?:there|you|buddy|pal|sir|madam|bro|friend|all|team|mentor|mentra|mentrafiai)?\s*[!.,?]*\s*$",
    re.IGNORECASE,
)


def is_greeting(text: str) -> bool:
    if not text:
        return False
    return bool(GREETING_RE.fullmatch(text.strip()))


def handle_greeting(text: str) -> str:
    """Returns an instant, context-aware, polished greeting without LLM hallucination."""
    t = text.lower().strip()
    
    # 1. Farewells
    if re.search(r"\b(?:bye|goodbye|cya|see\s+you|tata|alvida)\b", t):
        return (
            "Goodbye! Thank you for consulting MentraFiAI. "
            "Stay disciplined with your SIPs and have a wonderful day! 🚀"
        )
        
    # 2. Appreciation & Thanks
    if re.search(r"\b(?:thanks|thank\s+you|dhanyawad|shukriya)\b", t):
        return (
            "You're very welcome! Glad I could help. "
            "Feel free to ask anytime if you want to explore new funds, calculate goals, or compare schemes! 🤝"
        )
        
    # 3. Dynamic Salutation based on input or time
    salutation = "Hello!"
    if re.search(r"\b(?:good|gud)\s*(?:mori?ng|mrng|morning)|gm\b", t):
        salutation = "Good morning!"
    elif re.search(r"\b(?:good|gud)\s*(?:afternoon|aftrnun)\b", t):
        salutation = "Good afternoon!"
    elif re.search(r"\b(?:good|gud)\s*(?:evening|evng)|ge\b", t):
        salutation = "Good evening!"
    elif re.search(r"\b(?:namaste|namaskar|namste|pranam)\b", t):
        salutation = "Namaste!"
    elif re.search(r"\b(?:sat\s*sri\s*akal|kem\s*cho|vanakkam|adaab)\b", t):
        salutation = "Namaste & Warm Greetings!"

    return (
        f"{salutation} Welcome to MentraFiAI, your intelligent mutual fund and wealth advisory assistant. 🤝\n\n"
        f"I'm here to provide data-grounded, unbiased financial guidance based on 37,700+ live schemes. Here is what I can do for you:\n\n"
        f"• **Portfolio Recommendation:** e.g., *'Suggest a mutual fund portfolio for ₹10,000/month for 10 years, moderate risk'*\n"
        f"• **Goal Planning:** e.g., *'I want to accumulate ₹1 crore in 12 years, how much SIP do I need?'*\n"
        f"• **Fund Comparison:** e.g., *'Compare SBI Small Cap vs Nippon India Small Cap'*\n"
        f"• **Taxation & Concepts:** e.g., *'What is LTCG tax in equity funds?'* or *'Explain expense ratio'*\n"
        f"• **Fund Search & NAV:** e.g., *'NAV of Parag Parikh Flexi Cap'*\n\n"
        f"How can I help you plan your investments today?"
    )


# ---------------------------------------------------------------------------
# Informational / definitional intent
# ---------------------------------------------------------------------------
# General "what is X" questions must NOT be routed through the recommendation
# pipeline (which would build a fund panel and let the model hallucinate a fake
# profile). We answer common terms from a curated glossary instead of the model.

# Ordered so multi-word / more specific keys are checked before generic ones
# (e.g. "expense ratio" before "ratio", "index fund" before "fund").
FINANCIAL_GLOSSARY: list[tuple[tuple[str, ...], str]] = [
    (("ltcg", "stcg", "capital gains tax", "taxation", "tax on mutual fund", "tax in equity fund", "difference between ltcg and stcg"),
     "Under Indian tax rules (current 2026 regime under Union Budget 2024 amendments):\n\n"
     "• Short-Term Capital Gains (STCG): If equity mutual fund units are held for 12 months or less, realized profits are taxed at a flat 20%.\n"
     "• Long-Term Capital Gains (LTCG): If equity mutual fund units are held for more than 12 months, profits up to ₹1.25 Lakh per financial year are 100% Tax-Free. Profits exceeding ₹1.25 Lakh are taxed at a flat 12.5%.\n"
     "• Debt Funds: Capital gains are added to your annual income and taxed as per your personal income tax slab rate, regardless of holding period."),
    (("direct vs regular", "direct plan", "regular plan"),
     "Every mutual fund scheme in India is offered in two variants:\n\n"
     "• Direct Plan: You invest directly with the Asset Management Company (AMC) without intermediaries. Zero broker commissions are deducted from your investment. The Expense Ratio (TER) is 0.5% to 1.5% lower, delivering significantly higher wealth compounding over time.\n"
     "• Regular Plan: You invest through a distributor, broker, or bank. The AMC pays an ongoing annual commission to the broker by deducting it from your daily NAV, reducing your long-term returns.\n\n"
     "Recommendation: Always choose Direct Plans via platforms like MFCentral, Zerodha Coin, or Groww."),
    (("sip vs lumpsum", "lumpsum vs sip", "sip or lumpsum", "lump sum vs sip", "lumpsum", "lump sum"),
     "Comparison: SIP vs Lumpsum Mutual Fund Investing:\n\n"
     "• SIP (Systematic Investment Plan):\n"
     "  - Invest a fixed amount periodically (e.g. ₹5,000 every month).\n"
     "  - Best for salaried individuals and continuous wealth creation.\n"
     "  - Benefits from Rupee-Cost Averaging across market volatility.\n"
     "  - Zero market timing stress.\n\n"
     "• Lumpsum Investment:\n"
     "  - Invest a single large sum of capital all at once.\n"
     "  - Best for windfalls, bonuses, or property sale proceeds.\n"
     "  - Delivers superior returns during the onset of strong bull markets.\n\n"
     "Smart Advisory Tip: If investing a large lumpsum into equity, park the money in a Liquid Fund and execute a 6–12 month STP (Systematic Transfer Plan) to avoid market peak risk."),
    (("sip", "sips", "systematic investment plan", "sip kya hai", "sip plan", "sip investment", "about sip", "what is a sip"),
     "A SIP (Systematic Investment Plan) is a disciplined method of investing a fixed sum of money into mutual funds at regular intervals (typically monthly or quarterly).\n\n"
     "Key Advantages:\n"
     "• Rupee-Cost Averaging: When market prices drop, your fixed installment automatically buys more units; when prices rise, you buy fewer units. You never have to time the market.\n"
     "• Power of Compounding: Small, consistent contributions compound exponentially over 5, 10, or 15+ years into substantial wealth.\n"
     "• Low Minimum Entry: Start investing with as little as ₹500/month via automated bank mandate (UPI/eNACH).\n"
     "• Complete Flexibility: You can pause, increase (Step-up), or stop your SIP anytime with zero penalties (for open-ended non-ELSS funds).\n"
     "• Financial Discipline: Automates wealth accumulation on your monthly salary date before discretionary expenses."),
    (("swp", "systematic withdrawal plan"),
     "A SWP (Systematic Withdrawal Plan) allows you to withdraw a fixed predetermined amount from your mutual fund at regular intervals (e.g. monthly) while the remaining capital stays invested and compounding.\n\n"
     "Key Advantages:\n"
     "• Predictable Cash Flow: Excellent for retirees or anyone seeking regular monthly income.\n"
     "• High Tax Efficiency vs FD: Only the capital gains component of each withdrawal is taxable, making it far more tax-efficient than Fixed Deposit interest."),
    (("stp", "systematic transfer plan"),
     "A STP (Systematic Transfer Plan) allows you to automatically transfer a fixed amount from one mutual fund scheme (usually a safe Liquid or Overnight fund) to an equity fund at regular intervals.\n\n"
     "Key Advantage:\n"
     "• Lump-Sum Risk Mitigation: Protects you from investing a large sum at a market peak by averaging your entry over 6 to 12 months while earning ~6.5-7% on the unallocated cash."),
    (("step up sip", "top up sip", "step-up sip"),
     "A Step-Up SIP (or Top-Up SIP) allows you to automatically increase your monthly investment by a fixed percentage (e.g. 10%) or fixed amount (e.g. ₹1,000) every year in line with your salary increments.\n\n"
     "Compounding Multiplier: Increasing your SIP by just 10% annually can almost double your final accumulated corpus over a 15-year horizon!"),
    (("elss", "tax saving fund", "tax saver fund", "equity linked savings scheme"),
     "An ELSS (Equity Linked Savings Scheme) is a diversified equity mutual fund designed for tax savings under Section 80C of the Income Tax Act.\n\n"
     "Key Features:\n"
     "• Tax Deductions: Invest up to ₹1.5 Lakh per financial year to reduce taxable income under Section 80C (Old Tax Regime).\n"
     "• Shortest Lock-in: Only 3 years — the shortest lock-in among all 80C instruments (PPF is 15 years, Tax-Saver FD is 5 years, NPS is up to age 60).\n"
     "• Equity Wealth Creation: Invests predominantly in equity shares (minimum 80%), offering superior long-term inflation-beating potential.\n"
     "• Taxation: Gains above ₹1.25 Lakh per financial year are taxed as equity LTCG at 12.5% upon redemption."),
    (("index fund", "index funds", "passive fund", "passive investing"),
     "An Index Fund is a low-cost passive mutual fund that replicates a specific market benchmark (such as the Nifty 50 or BSE Sensex).\n\n"
     "Key Features:\n"
     "• Ultra-Low Expense Ratio: No active stock picking means Total Expense Ratios (TER) are typically 0.1% to 0.4% in Direct Plans.\n"
     "• Zero Fund Manager Bias: Returns mirror the broader market without the risk of fund manager underperformance.\n"
     "• Ideal Core Foundation: Outstanding low-cost choice for beginner and long-term passive investors."),
    (("nav", "net asset value", "nav rate", "nav price", "nav kya hai", "nav kya hota hai", "what is nav"),
     "NAV (Net Asset Value) represents the per-unit market price of a mutual fund scheme, calculated at the end of every business day (around 9:00 PM IST).\n\n"
     "How It Is Calculated:\n"
     "NAV = (Total Scheme Assets - Total Liabilities & Operating Expenses) / Total Outstanding Units\n\n"
     "Important Myth Buster:\n"
     "A higher NAV (e.g. ₹200) does NOT mean a fund is 'expensive' compared to one with an NAV of ₹20. Mutual fund returns depend strictly on the percentage growth of the underlying portfolio, not the absolute NAV. A 15% gain on a ₹20 NAV fund and a 15% gain on a ₹200 NAV fund produce the exact same rupee profit."),
    (("expense ratio", "ter", "total expense ratio"),
     "The Total Expense Ratio (TER) is the annual percentage fee charged by the Asset Management Company (AMC) to manage the fund, covering management fees, administrative costs, and regulatory compliance.\n\n"
     "Rule of Thumb: Lower is always better! In Direct Plans, equity funds typically charge 0.5% - 1.0%, while index funds charge 0.1% - 0.3%."),
    (("exit load",),
     "An Exit Load is a fractional fee (typically 1.0%) charged by the mutual fund if you redeem (sell) units before a specified minimum holding period — typically 365 days for equity funds. If you redeem after 1 year, the exit load is 0% (free). Liquid and overnight funds have exit loads lasting only a few days or zero."),
    (("cagr", "xirr"),
     "• CAGR (Compound Annual Growth Rate): Measures the annualized smoothed return of a one-time lump sum investment over multiple years.\n"
     "• XIRR (Extended Internal Rate of Return): Specifically used for SIPs and multiple cash inflows/outflows on irregular dates to accurately measure your annualized return."),
    (("rule of 72", "72 rule"),
     "The Rule of 72 is a quick mental math formula to estimate how many years it takes for your investment to double in value:\n\n"
     "Years to Double ≈ 72 ÷ Expected Annual Return (CAGR)\n\n"
     "Examples:\n"
     "• At 12% CAGR (typical Equity/Flexi Cap): 72 ÷ 12 = 6 Years to double\n"
     "• At 14.4% CAGR (aggressive Mid/Small Cap): 72 ÷ 14.4 = 5 Years to double\n"
     "• At 7.2% CAGR (FD / Debt Fund): 72 ÷ 7.2 = 10 Years to double"),
    (("large cap", "large cap fund"),
     "Large Cap mutual funds invest at least 80% of their assets in India's top 100 companies by market capitalization (e.g. Reliance, HDFC Bank, TCS, Infosys). They offer bedrock stability, steady earnings, and lower drawdowns during market corrections."),
    (("mid cap", "mid cap fund"),
     "Mid Cap mutual funds invest at least 65% in companies ranked 101st to 250th by market cap. These are agile, fast-growing companies that offer higher growth potential than large caps, balanced with moderate volatility over a 5+ year horizon."),
    (("small cap", "small cap fund"),
     "Small Cap mutual funds invest at least 65% in companies ranked 251st and beyond by market cap. They offer high wealth creation potential during economic expansions, but carry substantial short-term volatility. Strictly suitable for 7+ year horizons."),
    (("flexi cap", "flexi cap fund", "multi cap"),
     "Flexi Cap mutual funds invest across large, mid, and small-cap stocks without statutory allocation limits. The fund manager has complete dynamic freedom to shift allocations based on market valuations, making them an ideal core, all-weather fund."),
    (("nfo", "new fund offer"),
     "An NFO (New Fund Offer) is the initial launch of a new mutual fund scheme by an AMC, typically priced at ₹10 per unit.\n\n"
     "Advisory Note: Unlike stock IPOs, an NFO unit price of ₹10 is NOT a cheap valuation or discount. NFOs have zero historical track record. Investors are almost always better off choosing existing funds with 5+ years of verified performance."),
    (("mutual fund", "mutual funds"),
     "A mutual fund pools money from multiple investors to invest in a professionally managed, diversified portfolio of equities, debt instruments, or other securities. Investors own units proportional to their share, and the fund value tracks the underlying NAV daily."),
    (("etf", "exchange traded fund"),
     "An ETF (Exchange-Traded Fund) is an index-tracking fund that trades live on stock exchanges (NSE/BSE) like an individual share. It requires a Demat and trading account to purchase and features real-time intra-day pricing and low expense ratios."),
    (("risk tolerance", "risk appetite"),
     "Risk tolerance reflects your emotional and financial ability to withstand market fluctuations. Higher risk appetite suits equity investments for 7+ years; lower risk appetite suits hybrid, arbitrage, or debt funds."),
    (("investment horizon", "time horizon"),
     "Investment horizon is the total duration you intend to keep your capital invested before withdrawal. Longer horizons (5-15+ years) allow equity compounding to smooth out short-term market volatility."),
    (("compounding", "compound interest"),
     "Compounding is the process where returns earned on your principal themselves generate further returns over time. Over horizons of 10–20 years, compounding accounts for the vast majority of total accumulated wealth."),
    (("sharpe ratio", "sharpe"),
     "The Sharpe Ratio measures risk-adjusted performance by calculating excess return earned per unit of portfolio volatility. A Sharpe Ratio greater than 1.0 indicates good risk-adjusted returns; higher is always better."),
    (("fd vs mutual fund", "fixed deposit vs", "mutual fund vs fd"),
     "• Fixed Deposit (FD): Offers guaranteed interest (~6.5% - 7.5%), but returns are fully taxable at your personal income tax slab. Real returns after inflation and taxes are often near zero or negative.\n"
     "• Mutual Funds: Market-linked equity funds historically deliver 12% - 15% CAGR over long horizons, with ₹1.25 Lakh annual LTCG tax exemption and lower 12.5% capital gains tax."),
    (("bear market", "market crash"),
     "In a market crash or bear market, the golden rule of mutual funds is: **Never stop your SIP!** Continuing your SIP allows you to accumulate significantly more units at discounted NAVs (rupee-cost averaging). Bull markets generate paper gains, but bear markets build lasting wealth if you stay disciplined."),
    (("sebi", "sebi registered"),
     "MentraFiAI is an AI-powered mutual fund advisory and analytical system. All scheme data is sourced from SEBI-registered Asset Management Companies in India. Mutual fund investments are subject to market risks; please read scheme-related documents carefully."),
    (("how does mentrafiai work", "what is mentrafiai", "who are you"),
     "MentraFiAI is an elite triple neural network wealth intelligence system. It combines a 10M Intent Classifier, a 15M Query Parser, a PostgreSQL database of 37,700+ live schemes, and a 102M Generative LLM to deliver mathematically exact, un-hallucinated mutual fund portfolios and advisory."),
]


# Definitional phrasing that signals the user wants an explanation, not a pick.
_DEFINITION_RE = re.compile(
    r"\b(what\s+is|what\s+are|what'?s|what\s+does|whats|define|explain|meaning\s+of|"
    r"how\s+does|how\s+do|how\s+is|tell\s+me\s+about|info\s+on|details\s+about|guide\s+on|difference\s+between|"
    r"kya\s+hai|kya\s+hota\s+hai|kya\s+fark\s+hai|samjhao)\b",
    re.IGNORECASE,
)
# Cues that this is really a recommendation request despite "what is" phrasing
# (e.g. "what is the best fund for me").
_RECOMMEND_CUE_RE = re.compile(
    r"\b(best|which|recommend|suggest|should\s+i|for\s+me|invest\s+in|pick|choose|"
    r"top\s+fund|good\s+fund|portfolio|banao|kaunsa)\b",
    re.IGNORECASE,
)


def _find_glossary_entry(text: str) -> str | None:
    low = text.lower().strip()
    matched_entries = []
    for keys, definition in FINANCIAL_GLOSSARY:
        for k in keys:
            pattern = r"\b" + re.escape(k) + r"\b"
            if re.search(pattern, low):
                matched_entries.append((len(k), definition))
    if matched_entries:
        # Longest / most specific matching phrase takes precedence
        matched_entries.sort(key=lambda x: x[0], reverse=True)
        return matched_entries[0][1]
    return None


def is_informational(text: str) -> bool:
    """True for general/definitional questions ('what is SIP', 'how does a SIP
    work', 'what does expense ratio mean', 'NAV kya hota hai', 'LTCG?') that
    should be answered with an explanation rather than a fund recommendation panel."""
    if not text:
        return False
        
    # If the user provides a budget (e.g., Rs 10000, ₹5,000, 10k/month), it is a portfolio request, NOT a definition!
    if re.search(r"(?:rs\.?|₹)\s*\d+|\b\d+k\b|\b\d+\s*(?:lakh|crore)\b|\b\d{3,7}\s*(?:/month|monthly|per\s*month|budget)", text, re.IGNORECASE):
        return False

    if _RECOMMEND_CUE_RE.search(text):
        return False  # "what is the best fund for me" → recommendation, not a definition
        
    has_def_cue = bool(_DEFINITION_RE.search(text))
    glossary_match = _find_glossary_entry(text) is not None
    
    if has_def_cue and glossary_match:
        return True
        
    # If no definitional cue, match only if the query is a concise lookup (e.g. "LTCG?", "exit load", "XIRR")
    if glossary_match and len(text.strip().split()) <= 4:
        return True
        
    return False


def informational_reply(text: str) -> str:
    """Return a curated definition for a recognized term (see is_informational)."""
    definition = _find_glossary_entry(text)
    return definition or (
        "I can explain mutual fund concepts like SIP, NAV, expense ratio, ELSS, "
        "index funds, and risk. Ask me about a specific term, or tell me your age, "
        "monthly amount, risk tolerance, and horizon for a fund recommendation."
    )


def _truncate_on_repeat(text: str, max_repeat: int = 2) -> str:
    """Backstop for residual degeneration the sampler didn't prevent.

    Cuts the reply the moment a short phrase (2-6 words) starts repeating more
    than `max_repeat` times back-to-back, and collapses an immediately repeated
    single token/word run. Only trims trailing loop garbage; coherent prose that
    never loops is returned unchanged.
    """
    if not text:
        return text
    words = text.split()
    # Detect a repeating window of length w (words) that recurs consecutively.
    for w in range(2, 7):
        i = 0
        while i + w * (max_repeat + 1) <= len(words):
            window = words[i:i + w]
            reps = 1
            j = i + w
            while words[j:j + w] == window and j + w <= len(words):
                reps += 1
                j += w
            if reps > max_repeat:
                return " ".join(words[:i + w]).rstrip(",;: ") + ("." if not words[:i + w][-1].endswith(('.', '!', '?')) else "")
            i += 1
    # Collapse a single word/phrase repeated many times in a row (e.g. "0.7-year
    # horizon, 0.7-year horizon, ...") that the windowed check above may miss.
    collapsed, prev, run = [], None, 0
    for tok in words:
        if tok == prev:
            run += 1
            if run >= max_repeat:
                continue
        else:
            run = 0
        collapsed.append(tok); prev = tok
    return " ".join(collapsed)


def generate_text(model, tokenizer, prompt: str, device,
                   max_new_tokens=200, temperature=0.7, top_k=50, top_p=0.9,
                   repetition_penalty=1.2, no_repeat_ngram_size=4):
    """CRITICAL FIX: Improved default sampling parameters to reduce degeneration"""
    ids = tokenizer.encode(prompt, add_bos=True)
    prompt_len = len(ids)  # Fix 1: remember prompt length so we can drop it from the output
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_k=top_k, top_p=top_p,
                          repetition_penalty=repetition_penalty,
                          no_repeat_ngram_size=no_repeat_ngram_size)
    # Decode ONLY the newly generated tokens, never the echoed prompt.
    return _truncate_on_repeat(tokenizer.decode(out[0].tolist()[prompt_len:]))


def generate_chat_reply(model, tokenizer, user_text: str, device,
                         max_new_tokens=200, temperature=0.7, top_k=50, top_p=0.9,
                         repetition_penalty=1.2, no_repeat_ngram_size=4):
    """Generate dynamic contextual reply directly from the fine-tuned LLM."""
    prefix_ids = [tokenizer.bos_id, tokenizer.user_id] + tokenizer.encode(user_text) + [tokenizer.assistant_id]
    prompt_len = len(prefix_ids)
    idx = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_k=top_k, top_p=top_p,
                          repetition_penalty=repetition_penalty,
                          no_repeat_ngram_size=no_repeat_ngram_size)
    raw_ids = out[0].tolist()[prompt_len:]
    stop_ids = {tokenizer.eos_id, tokenizer.user_id, tokenizer.assistant_id}
    cleaned_ids = []
    for tok_id in raw_ids:
        if tok_id in stop_ids:
            break
        cleaned_ids.append(tok_id)
    return _truncate_on_repeat(tokenizer.decode(cleaned_ids).strip())
