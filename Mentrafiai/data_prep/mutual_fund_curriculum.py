"""Authored synthetic mutual-fund education, not official factual certification.

No network access, file writes, live data, fund brands, or external dependencies.
Source URLs are review starting points, not claim-level citations or endorsement.
The integrating agent must review facts and measure serialized conversations with
its actual tokenizer before training; the intended per-record limit is 1024 tokens.
"""

from copy import deepcopy


SYSTEM_PROMPT = (
    "You are a helpful educational assistant. Respond in clear, respectful English; "
    "explain jargon, clarify ambiguity, and correct mistakes politely. You are not "
    "a SEBI-registered investment adviser; do not claim registration or give "
    "personalized fund recommendations. Use only supplied, verified, dated fund "
    "context for scheme-specific facts. Identify missing or conflicting evidence "
    "and ask for official documents rather than guessing. Never invent fund names, "
    "brands, performance, NAVs, rankings, or live facts. Never guarantee returns "
    "or capital safety. Label numerical illustrations and assumptions explicitly. "
    "Tax statements are conditional: request the relevant financial year and "
    "transaction details, and require dated confirmation from the Income Tax "
    "Department or applicable law; do not assert current tax rates or limits. "
    "Check current official scheme and regulatory documents for changing rules. "
    "This is authored synthetic educational material, not official factual certification."
)

SOURCES = [
    "https://www.amfiindia.com/investor",
    "https://investor.sebi.gov.in/understanding_mf.html",
    "https://investor.sebi.gov.in/iematerial.html",
    "https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doFaq=yes",
]

METADATA = {
    "intent": "Synthetic educational SFT curriculum and English communication practice",
    "provenance": "Manually authored synthetic examples; not official factual certification",
    "language": "en",
    "jurisdiction": "India",
    "manual_review_required": True,
    "review_status": "Pending independent domain and integration review",
    "official_endorsement": False,
    "source_scope": "Official educational reference entry points, not per-answer verification",
    "live_data_included": False,
    "fund_brands_included": False,
    "current_tax_rates_or_limits_included": False,
    "tax_policy": "Conditional only; confirm applicable year and dated tax authority guidance",
    "target_max_conversation_tokens": 1024,
    "tokenizer_validation": "Required with integration tokenizer, including system and chat template",
    "review_focus": [
        "Financial definitions and category mandates against current official documents",
        "ELSS, tax-regime eligibility, and transaction-specific tax treatment",
        "KYC, nomination, NAV applicability, settlement, and scheme restrictions",
        "No invented scheme facts, guarantees, or personalized recommendations",
        "Full serialized record length and compatibility with the integrating pipeline",
    ],
}

# Slugs are stable identifiers; each entry teaches a distinct concept or decision.
TOPICS = {
    "sip_definition": {
        "question": "What is SIP and what does it stand for?",
        "answer": "SIP stands for Systematic Investment Plan. It is an arrangement to buy units of a mutual fund at scheduled intervals, often monthly. Each purchase uses the applicable NAV, so unit counts vary with price. A SIP is an investment method, not a separate fund or a fixed-interest product. The underlying scheme can lose value, and regular investing does not guarantee profit.",
    },
    "nfo": {
        "question": "What is a New Fund Offer, and is an NFO at Rs 10 a bargain?",
        "answer": "A New Fund Offer, or NFO, is the initial subscription period for a new scheme. Its issue price is not evidence of a cheap portfolio or superior future returns. There may be no scheme performance history to examine. Read the objective, mandate, risks, costs, structure, and relevant documents instead of treating an NFO like a discounted share offering.",
    },
    "open_ended": {
        "question": "What is an open-ended mutual fund?",
        "answer": "An open-ended scheme normally permits ongoing purchases and redemptions under its stated terms rather than having a fixed maturity date. Transactions use applicable NAV and operational rules. Open-ended does not mean instant cash or unconditional access: settlement time, exit loads, lock-ins for particular schemes, and exceptional restrictions can matter. Check current scheme documents before relying on liquidity.",
    },
    "closed_ended": {
        "question": "How is a closed-ended fund different from an open-ended fund?",
        "answer": "A closed-ended scheme has a specified tenure and generally accepts subscriptions during its initial offer. Investor exit during that tenure depends on the scheme structure and available mechanisms, such as exchange trading where applicable, rather than ordinary ongoing redemption. Market liquidity and prices can differ from NAV. A specified tenure does not itself guarantee repayment of the original invested amount.",
    },
    "custodian_trustee": {
        "question": "What do the custodian, trustee, and fund manager do?",
        "answer": "The fund manager makes portfolio decisions within the scheme mandate as part of the AMC. A custodian safeguards scheme securities and supports settlement and recordkeeping. Trustees oversee the mutual fund and AMC in the interests of unitholders under the applicable framework. These are different responsibilities; oversight and segregation are safeguards, not guarantees against market losses or every operational failure.",
    },
    "sector_thematic": {
        "question": "Are sector and thematic funds diversified like broad-market funds?",
        "answer": "A sector fund focuses on a particular industry; a thematic fund follows a stated investment theme that may span industries. Both can concentrate exposure to common economic drivers despite holding many companies. They can experience deep or prolonged losses when that exposure performs poorly. Examine the mandate and holdings rather than assuming many stocks automatically provide broad diversification.",
    },
    "nav_formula": {
        "question": "How do I calculate NAV per unit?",
        "answer": "NAV per unit is scheme net assets divided by outstanding units, using the applicable valuation and accounting rules. As a simplified illustration, assets worth Rs 1,020,000 minus liabilities of Rs 20,000 leave Rs 1,000,000 in net assets. With 100,000 units outstanding, NAV is Rs 10 per unit. This illustration is not a quotation for an actual scheme.",
    },
    "sip_installment_tax": {
        "question": "Do SIP installments have separate holding periods and tax treatment?",
        "answer": "Each SIP installment purchases units on its own allotment date, so installments can have different holding periods, lock-in expiry dates, and exit-load treatment. On redemption, identify the relevant units and applicable accounting rules rather than treating the entire SIP as one purchase. Tax outcomes require the scheme classification, dates, residency, financial year, and dated official tax rules.",
    },
    "fund_complaints": {
        "question": "What should I do if a mutual fund transaction is unresolved?",
        "answer": "Use official channels to contact the AMC or registrar with the transaction reference and relevant dates, keeping sensitive identifiers out of public messages. Retain acknowledgments and follow the published grievance process. If unresolved, consult the current SEBI investor grievance and dispute-resolution guidance. Do not share passwords or one-time codes with anyone claiming they are needed to recover an investment.",
    },
    "mutual_fund": {
        "question": "What is a mutual fund?",
        "answer": (
            "A mutual fund pools money from investors and invests it in securities under a "
            "stated scheme objective. Investors hold units representing their share of the "
            "scheme's net assets. Professional management and diversification can be useful, "
            "but expenses apply and investment values can fall. A mutual fund is not a bank "
            "deposit and does not inherently guarantee capital or returns."
        ),
    },
    "sip_is_a_method": {
        "question": "Is a SIP a separate investment product with guaranteed returns?",
        "answer": (
            "A Systematic Investment Plan, or SIP, is a method of investing a chosen amount "
            "periodically in a mutual fund scheme. The scheme is the product; the SIP is "
            "the payment schedule. Its risk comes from the underlying investments. Regular "
            "contributions may support discipline, but neither the schedule nor a long "
            "contribution history guarantees a profit."
        ),
    },
    "nav_and_price_myth": {
        "question": "Is a fund with a lower NAV cheaper or better?",
        "answer": (
            "Net Asset Value, or NAV, is the value of a scheme's net assets per unit. "
            "A lower NAV does not make a fund cheaper in the stock-valuation sense. "
            "Illustratively, investing Rs 1,000 at NAV 10 or NAV 100 buys different unit "
            "counts, but the same percentage portfolio gain produces the same percentage "
            "investment gain, ignoring charges and taxes."
        ),
    },
    "units": {
        "question": "What are mutual fund units?",
        "answer": (
            "Units record your proportional participation in a mutual fund scheme. For "
            "a purchase, the net amount available for investment divided by the applicable "
            "NAV determines units allotted, subject to transaction rules and rounding. "
            "Your holding value is broadly units multiplied by NAV. More units alone do "
            "not imply greater wealth because their per-unit value also matters."
        ),
    },
    "aum": {
        "question": "What does AUM tell me?",
        "answer": (
            "Assets Under Management, or AUM, measures the value of assets managed by "
            "a scheme or asset manager at a stated date. It changes with market values "
            "and investor inflows or outflows. Larger AUM does not prove better returns "
            "or safety. Its practical significance depends on the strategy, underlying "
            "market liquidity, and ability to deploy money effectively."
        ),
    },
    "amc": {
        "question": "What is an AMC?",
        "answer": (
            "An Asset Management Company, or AMC, manages mutual fund investments within "
            "scheme mandates and the regulatory framework. Its teams handle portfolio "
            "management, research, risk controls, and operations. The AMC is distinct "
            "from an individual scheme, and different schemes can have very different "
            "risks. The manager's reputation is not a guarantee of investment performance."
        ),
    },
    "total_expense_ratio": {
        "question": "How is the total expense ratio charged?",
        "answer": (
            "The Total Expense Ratio, or TER, expresses permitted recurring scheme "
            "expenses as an annualized percentage of net assets. Expenses are accrued "
            "in the scheme and reflected in daily NAV, rather than usually billed "
            "separately to each investor. Published NAV-based returns already reflect "
            "these expenses. Check the dated plan-specific disclosure; exit loads, "
            "investor taxes, and some transaction costs are separate considerations."
        ),
    },
    "direct_and_regular": {
        "question": "How do direct and regular plans differ?",
        "answer": (
            "Direct and regular plans of the same scheme generally share the portfolio "
            "and investment objective but have separate expenses and NAVs. Regular-plan "
            "expenses include distributor compensation; direct plans exclude that "
            "distribution cost. There is no universal fixed expense or return spread. "
            "Compare dated TER disclosures for the exact plans and consider any "
            "separate platform or advisory charges."
        ),
    },
    "growth_option": {
        "question": "What is the growth option?",
        "answer": (
            "Under a growth option, distributable investment gains and income remain "
            "within the scheme rather than being paid out as IDCW. Their effect is "
            "reflected in NAV, alongside market movements and expenses. Growth does "
            "not mean NAV rises every day or returns are assured. Investors generally "
            "access money by redeeming eligible units, subject to applicable restrictions."
        ),
    },
    "idcw": {
        "question": "Is IDCW extra income in addition to my fund value?",
        "answer": (
            "Income Distribution cum Capital Withdrawal, or IDCW, is a distribution "
            "from the scheme, not free additional wealth or guaranteed interest. NAV "
            "falls to reflect a payout, all else equal, and distributions can include "
            "capital withdrawal. Availability and amounts depend on applicable rules "
            "and declarations. Any tax treatment needs confirmation for the relevant "
            "year from dated tax-authority guidance."
        ),
    },
    "swp": {
        "question": "How does a Systematic Withdrawal Plan work?",
        "answer": (
            "A Systematic Withdrawal Plan, or SWP, schedules redemptions from a fund "
            "holding. Units are sold to provide withdrawals; the payments are not "
            "necessarily investment income. If withdrawals exceed investment growth, "
            "capital can shrink or run out, especially during early losses. Check "
            "scheme eligibility, exit loads, and conditional tax consequences using "
            "dated rules before setting a schedule."
        ),
    },
    "stp": {
        "question": "What is a Systematic Transfer Plan?",
        "answer": (
            "A Systematic Transfer Plan, or STP, periodically transfers money between "
            "eligible schemes, usually within the same mutual fund house. Each transfer "
            "redeems units in the source scheme and purchases units in the destination. "
            "Source investments still carry risk; exit loads and taxable realizations "
            "may arise under applicable dated rules. Staggering entry does not "
            "guarantee better returns."
        ),
    },
    "switch_transactions": {
        "question": "Is switching funds or plans just an internal transfer with no tax?",
        "answer": (
            "A switch normally involves redemption from one scheme, plan, or option "
            "and purchase into another, even within the same fund house. It can "
            "trigger exit loads and a taxable event, depending on applicable law "
            "and any specific exemption. Confirm transaction dates, acquisition "
            "details, scheme classification, and the financial year's dated "
            "tax-authority guidance rather than assuming neutrality."
        ),
    },
    "step_up_sip": {
        "question": "What is a step-up SIP?",
        "answer": (
            "A step-up SIP increases the periodic contribution by a chosen amount "
            "or percentage at specified intervals, where the facility is available. "
            "It changes how much you invest, not the fund's return rate. Higher "
            "contributions can support a goal but require sufficient cash flow. "
            "Check mandate limits, amendment rules, and affordability; projected "
            "future values remain assumption-based."
        ),
    },
    "lump_sum": {
        "question": "What is lump-sum investing compared with a SIP?",
        "answer": (
            "Lump-sum investing puts an available amount into a scheme at one time; "
            "a SIP spreads contributions across dates. A lump sum has immediate "
            "market exposure, while a SIP changes the entry pattern and cash-flow "
            "timing. Neither is always superior. Comparisons must use consistent "
            "available capital, dates, and assumptions, rather than treating "
            "uninvested money as invisible."
        ),
    },
    "absolute_return": {
        "question": "What is absolute return?",
        "answer": (
            "For a single investment with no intervening cash flows, absolute return "
            "is ending value divided by starting value, minus one. It measures the "
            "total change over the period, not a yearly rate. An illustrative 20% "
            "gain over several years is therefore not 20% each year. With contributions "
            "or withdrawals, a cash-flow-aware measure is needed for meaningful comparison."
        ),
    },
    "cagr": {
        "question": "What does CAGR mean?",
        "answer": (
            "Compound Annual Growth Rate, or CAGR, is the constant annual rate that "
            "would connect a positive starting value to a positive ending value: "
            "(ending / starting) ** (1 / years) - 1. It annualizes a single investment's "
            "overall change without intermediate cash flows. Actual yearly returns "
            "can differ sharply; CAGR is not a promised future rate or a measure of volatility."
        ),
    },
    "xirr": {
        "question": "Why is XIRR used for SIP returns?",
        "answer": (
            "XIRR estimates an annualized money-weighted return using the dates and "
            "amounts of contributions, withdrawals, and final value. It is useful "
            "for SIPs because each installment stays invested for a different period. "
            "Cash-flow timing affects the result, so it differs from a scheme's CAGR. "
            "Some unusual cash-flow patterns have multiple or no valid solutions; "
            "check inputs and interpretation."
        ),
    },
    "trailing_return": {
        "question": "What is a trailing return?",
        "answer": (
            "A trailing return measures performance over a period ending on one "
            "chosen date, such as the preceding three years. It is sensitive to "
            "that start and end date. Check whether the number is cumulative or "
            "annualized and whether distributions are reinvested. Compare matching "
            "periods, plans, and options; one attractive endpoint can hide very "
            "different investment experiences."
        ),
    },
    "rolling_return": {
        "question": "How are rolling returns different from trailing returns?",
        "answer": (
            "Rolling returns repeat a fixed holding-period calculation across many "
            "start dates, rather than reporting just one endpoint. For example, "
            "several overlapping three-year windows reveal a distribution of outcomes. "
            "Specify the window length, observation frequency, and data range. "
            "Overlapping observations are not independent, and historical consistency "
            "does not establish a guaranteed future minimum return."
        ),
    },
    "volatility": {
        "question": "What is volatility?",
        "answer": (
            "Volatility describes how much investment returns fluctuate over time. "
            "Higher volatility generally means a wider range of short-term outcomes, "
            "including gains and losses. It matters when money may be needed during "
            "a decline. However, it does not capture every risk: defaults, poor "
            "liquidity, inflation, and permanent loss can matter even when recorded "
            "prices look stable."
        ),
    },
    "standard_deviation": {
        "question": "What does standard deviation measure in a fund factsheet?",
        "answer": (
            "Standard deviation measures the dispersion of periodic returns around "
            "their average and is commonly used as a volatility statistic. "
            "Interpret it with the measurement period, return frequency, and "
            "annualization method. It counts upward and downward variation, not "
            "just losses. A low historical reading is not a loss limit and may "
            "miss rare events or stale valuations."
        ),
    },
    "alpha": {
        "question": "What is alpha?",
        "answer": (
            "Alpha estimates performance beyond what a chosen risk model would "
            "predict, often using market exposure and a benchmark. It is not "
            "simply every difference between fund return and index return. Results "
            "depend on the model, benchmark, period, and expense treatment. "
            "Positive historical alpha can reflect skill, luck, or omitted risks "
            "and need not persist."
        ),
    },
    "beta": {
        "question": "What is beta?",
        "answer": (
            "Beta estimates how a fund's returns have tended to move relative "
            "to a selected market benchmark. A beta above one suggests greater "
            "historical sensitivity in that model, not a precise prediction of "
            "the next movement. Benchmark choice and estimation period matter. "
            "Beta does not measure all portfolio risks and can be unreliable "
            "when the relationship changes."
        ),
    },
    "sharpe_ratio": {
        "question": "What does the Sharpe ratio tell me?",
        "answer": (
            "The Sharpe ratio compares average return above a selected risk-free "
            "rate with return volatility, using consistent time units. It helps "
            "describe historical return per unit of total variability. Compare "
            "matching periods and calculation methods. A higher number is not "
            "a guarantee of suitability, and negative excess returns, non-normal "
            "outcomes, or unstable estimates complicate interpretation."
        ),
    },
    "drawdown": {
        "question": "What is maximum drawdown?",
        "answer": (
            "A drawdown is a decline from an earlier investment-value peak. "
            "Maximum drawdown is the largest peak-to-trough decline observed "
            "within a specified period and sampling frequency. It describes "
            "historical downside experience, not the worst future loss possible. "
            "Recovery time matters too: an illustrative 20% decline requires "
            "a 25% subsequent gain just to regain the starting value."
        ),
    },
    "compounding": {
        "question": "How does compounding work in mutual funds?",
        "answer": (
            "Compounding means gains remaining invested can themselves participate "
            "in later gains or losses. A constant-rate illustration assumes "
            "reinvestment, a specified return path, and stated treatment of "
            "expenses, taxes, and contributions. Mutual fund returns are variable, "
            "so a smooth calculator curve is not a forecast. Losses compound "
            "too, and withdrawing money changes the future base."
        ),
    },
    "inflation_real_return": {
        "question": "What is the difference between nominal and real return?",
        "answer": (
            "Nominal return measures the change in money value; real return "
            "adjusts for inflation and therefore purchasing power. Over matching "
            "periods, real return is (1 + nominal return) / (1 + inflation) - 1. "
            "Subtracting inflation is only an approximation. Consider expenses "
            "and applicable taxes when assessing spendable purchasing power; "
            "personal spending inflation may differ from a published index."
        ),
    },
    "asset_allocation": {
        "question": "What is asset allocation?",
        "answer": (
            "Asset allocation divides investments among asset classes such as "
            "equity, debt, and cash according to a goal, time horizon, and "
            "capacity to absorb losses. It often matters more than selecting "
            "many similar funds. No single allocation suits everyone. Review "
            "liquidity needs, existing exposures, and changing circumstances "
            "rather than deriving an allocation from age alone."
        ),
    },
    "diversification": {
        "question": "Does diversification remove investment risk?",
        "answer": (
            "Diversification spreads exposure across investments whose outcomes "
            "are not perfectly linked. It can reduce concentration risk, but "
            "cannot eliminate broad market losses or guarantee capital protection. "
            "Many funds can still own the same securities or share the same "
            "economic drivers. Evaluate underlying holdings, sectors, issuers, "
            "asset classes, and correlations rather than counting fund names."
        ),
    },
    "rebalancing": {
        "question": "What is portfolio rebalancing?",
        "answer": (
            "Rebalancing brings portfolio weights back toward a previously chosen "
            "allocation after market moves or changing circumstances. A periodic "
            "review or a deviation threshold can provide a disciplined rule. "
            "New contributions may reduce the need to sell existing holdings. "
            "Consider transaction costs, exit loads, and tax consequences "
            "confirmed under applicable dated rules; rebalancing does not ensure profits."
        ),
    },
    "portfolio_overlap": {
        "question": "What is portfolio overlap between funds?",
        "answer": (
            "Portfolio overlap occurs when funds hold some of the same securities. "
            "High overlap can make several funds behave like one concentrated "
            "exposure, despite different names. Compare dated holdings and position "
            "weights, not just the number of shared stocks. Different securities "
            "can also have similar sector or economic risks, so low name "
            "overlap alone does not prove diversification."
        ),
    },
    "riskometer": {
        "question": "How should I use a scheme's Riskometer?",
        "answer": (
            "The Riskometer is a standardized disclosure of a scheme's assessed "
            "risk level under the applicable framework. It can change as "
            "portfolio characteristics change. Read the dated scheme disclosure "
            "and benchmark risk information alongside the mandate and holdings. "
            "It is not a personalized suitability score, a return forecast, "
            "or a guarantee that losses stay below a particular amount."
        ),
    },
    "duration_and_maturity": {
        "question": "Are duration and maturity the same thing?",
        "answer": (
            "Maturity is the time until a debt instrument's principal is due. "
            "Macaulay duration is a present-value-weighted average time to its "
            "cash flows; portfolio measures combine individual holdings. "
            "They need not equal average maturity because coupons arrive earlier. "
            "Duration measures help assess interest-rate exposure, but credit "
            "quality and liquidity require separate analysis of the portfolio."
        ),
    },
    "modified_duration": {
        "question": "What does modified duration mean?",
        "answer": (
            "Modified duration approximates the percentage price sensitivity of "
            "a bond or debt portfolio to a small change in yield, with other "
            "factors unchanged. Higher modified duration generally means larger "
            "price movements for the same yield change. The relationship is "
            "approximate, especially for large changes; convexity, credit spreads, "
            "cash flows, and portfolio changes also affect realized fund returns."
        ),
    },
    "yield_to_maturity": {
        "question": "Is a debt fund's YTM my promised return?",
        "answer": (
            "Yield to Maturity, or YTM, summarizes yields implied by current "
            "prices and contractual cash flows, subject to calculation assumptions. "
            "A debt fund's portfolio YTM is not an investor return promise. "
            "Defaults, reinvestment rates, changing yields, expenses, and "
            "portfolio turnover can alter realized returns. Check whether "
            "the disclosed figure is before expenses and how it was calculated."
        ),
    },
    "debt_fund": {
        "question": "What is a debt mutual fund?",
        "answer": (
            "A debt mutual fund invests mainly in debt and money-market instruments "
            "under its mandate. Returns arise from income and changes in security "
            "values, after expenses. It is not equivalent to a fixed deposit. "
            "Credit events, interest-rate movements, and liquidity conditions "
            "can reduce NAV; the risk profile varies substantially across "
            "debt categories and individual portfolios."
        ),
    },
    "credit_risk": {
        "question": "What is credit risk in a fund?",
        "answer": (
            "Credit risk is the possibility that an issuer fails to pay as "
            "promised or that its perceived creditworthiness worsens. Defaults, "
            "downgrades, or wider credit spreads can reduce security prices "
            "and fund NAV. A high rating is an opinion, not insurance. "
            "Assess issuer concentration and portfolio quality rather than "
            "assuming a debt label means capital is protected."
        ),
    },
    "liquidity_risk": {
        "question": "What does liquidity risk mean?",
        "answer": (
            "Liquidity risk is the difficulty of selling an investment promptly "
            "at a reasonable price. A fund may face poor market depth, wider "
            "bid-ask spreads, or redemption pressure during stress. This differs "
            "from an investor's normal redemption timetable. Open-ended status "
            "does not mean every underlying security is readily saleable "
            "or that exceptional restrictions can never occur."
        ),
    },
    "interest_rate_risk": {
        "question": "Why can a debt fund fall when interest rates rise?",
        "answer": (
            "When market yields rise, existing fixed-rate bonds generally become "
            "less attractive and their prices fall, other things equal. "
            "Longer-duration portfolios usually react more strongly. Falling "
            "yields can have the opposite price effect, but reinvestment "
            "income may then decline. Actual debt fund returns also reflect "
            "credit spreads, expenses, portfolio changes, and the investor's holding period."
        ),
    },
    "overnight_fund": {
        "question": "Are overnight funds completely risk-free?",
        "answer": (
            "Overnight funds invest in securities with very short, typically "
            "one-day, maturity under their category rules. That greatly limits "
            "interest-rate sensitivity compared with longer-duration funds, "
            "but does not create a guarantee. Instrument, counterparty, liquidity, "
            "and operational risks still need consideration. Check the current "
            "portfolio, permitted instruments, expenses, and redemption arrangements "
            "rather than assuming deposit-like protection."
        ),
    },
    "liquid_fund": {
        "question": "What is a liquid fund, and can it lose money?",
        "answer": (
            "A liquid fund invests in short-maturity debt and money-market "
            "instruments within its category rules. Short maturity usually "
            "reduces interest-rate sensitivity, but credit events and liquidity "
            "stress can still cause losses. It is not a guaranteed savings "
            "account. Review portfolio quality, concentration, expenses, applicable "
            "exit loads, and redemption access before treating it as available cash."
        ),
    },
    "credit_risk_fund": {
        "question": "How is a credit risk fund different from a low-risk debt fund?",
        "answer": (
            "A credit risk fund follows a debt strategy with substantial exposure "
            "to lower-rated corporate debt as defined by applicable category "
            "rules. Higher offered yields compensate for additional risk, not "
            "assured extra profit. Downgrades, defaults, and difficulty selling "
            "holdings can hurt NAV. Read the current mandate and issuer "
            "exposures rather than selecting it solely for portfolio yield."
        ),
    },
    "gilt_fund": {
        "question": "Does investing in government securities make a gilt fund risk-free?",
        "answer": (
            "A gilt fund invests primarily in government securities under its "
            "mandate. Sovereign rupee debt has very low domestic credit risk, "
            "but its market price can move substantially when yields change. "
            "Long-duration gilt portfolios can therefore experience meaningful "
            "NAV losses. Low issuer credit risk does not remove interest-rate "
            "risk, liquidity considerations, or the risk of selling at an unfavorable time."
        ),
    },
    "equity_fund": {
        "question": "What is an equity mutual fund?",
        "answer": (
            "An equity mutual fund invests predominantly in shares under a "
            "stated investment mandate. Its value depends on company earnings, "
            "valuations, economic conditions, and market sentiment, among other "
            "factors. Equity funds can suffer substantial losses and are not "
            "substitutes for near-term assured cash. Category, concentration, "
            "costs, and portfolio style matter alongside the general equity label."
        ),
    },
    "large_cap_fund": {
        "question": "What is a large-cap fund?",
        "answer": (
            "A large-cap fund focuses on companies classified as large-cap "
            "under the applicable Indian category framework. These companies "
            "are generally larger and more established than smaller listed "
            "businesses, but their shares can still fall sharply. Check the "
            "current mandate, classification list, and holdings. Large-cap "
            "does not mean guaranteed returns or suitability for every short-term goal."
        ),
    },
    "mid_cap_fund": {
        "question": "What is a mid-cap fund?",
        "answer": (
            "A mid-cap fund emphasizes companies classified in the mid-cap "
            "segment under applicable rules. Such businesses may offer growth "
            "opportunities but can face greater business uncertainty and "
            "market-liquidity constraints than larger companies. Their prices "
            "can be highly volatile. Read the scheme mandate and portfolio "
            "rather than assuming the middle size category means moderate risk."
        ),
    },
    "small_cap_fund": {
        "question": "What should I understand about small-cap funds?",
        "answer": (
            "Small-cap funds concentrate on the smaller-company segment defined "
            "by applicable category rules. Business fragility, limited trading "
            "liquidity, and valuation swings can produce deep or prolonged "
            "declines. Strong past gains do not establish higher future "
            "returns. Assess the portfolio's concentration and liquidity, "
            "and do not confuse a long holding period with immunity from losses."
        ),
    },
    "flexi_cap_fund": {
        "question": "What is a flexi-cap fund?",
        "answer": (
            "A flexi-cap fund invests across large-, mid-, and small-cap "
            "companies with flexibility over their relative weights, subject "
            "to its mandate and applicable equity exposure rules. It need "
            "not maintain equal allocations across size segments. Manager "
            "decisions can change the risk profile, so examine current "
            "holdings and stated strategy rather than assuming flexibility guarantees balance."
        ),
    },
    "multi_cap_fund": {
        "question": "How is a multi-cap fund different from a flexi-cap fund?",
        "answer": (
            "Multi-cap funds have prescribed minimum exposure across large-, "
            "mid-, and small-cap segments under applicable category rules. "
            "Flexi-cap funds have greater discretion over those segment "
            "weights. Their similar names do not imply identical risk. "
            "Check the current regulatory definitions and scheme documents, "
            "especially the smaller-company exposure, instead of treating "
            "either category as automatically conservative."
        ),
    },
    "index_fund": {
        "question": "What is an index mutual fund?",
        "answer": (
            "An index fund seeks to track a specified index rather than "
            "select securities mainly to outperform it. Returns can differ "
            "from the index because of expenses, cash holdings, trading, "
            "and replication choices. Risk depends on what the index owns: "
            "a narrow sector index may be concentrated. Passive management "
            "does not mean low market risk or guaranteed benchmark matching."
        ),
    },
    "etf_vs_index_fund": {
        "question": "How does an ETF differ from a conventional index fund?",
        "answer": (
            "An Exchange Traded Fund, or ETF, trades on an exchange at "
            "market prices, which can differ from NAV. Retail trading "
            "typically uses brokerage and demat arrangements, with spreads "
            "and liquidity to consider. A conventional open-ended index "
            "fund transacts through fund channels at applicable NAV. "
            "Both can track an index; compare total costs and tracking quality."
        ),
    },
    "tracking_error": {
        "question": "What is tracking error?",
        "answer": (
            "Tracking error measures the variability of a fund's return "
            "differences from its benchmark over a specified period, often "
            "reported on an annualized basis. Lower tracking error suggests "
            "more consistent tracking, not necessarily a smaller average "
            "shortfall. Check frequency, period, and benchmark version. "
            "It does not measure the benchmark's own market risk "
            "or guarantee exact replication."
        ),
    },
    "tracking_difference": {
        "question": "How is tracking difference different from tracking error?",
        "answer": (
            "Tracking difference is the fund's return minus its benchmark's "
            "return over a matching period, subject to the provider's "
            "sign convention. It describes the realized gap; tracking "
            "error describes variation in periodic gaps. A fund can "
            "lag consistently and still have low tracking error. "
            "Use matching total-return benchmarks, dates, and annualized "
            "or cumulative return conventions."
        ),
    },
    "fund_of_funds": {
        "question": "What is a fund of funds?",
        "answer": (
            "A fund of funds, or FoF, invests in other funds rather "
            "than mainly holding securities directly. It may provide "
            "access to strategies or markets through one scheme. "
            "Assess both the FoF's expenses and the underlying funds' "
            "costs, avoiding double-counting when reading disclosures. "
            "Underlying concentration and liquidity still matter; tax "
            "classification needs confirmation under the applicable year's rules."
        ),
    },
    "hybrid_fund": {
        "question": "What is a hybrid mutual fund?",
        "answer": (
            "A hybrid fund combines asset classes, often equity and "
            "debt, within a defined mandate. Different hybrid categories "
            "can have very different equity exposure, allocation "
            "flexibility, and risk. Combining assets may diversify "
            "some risks but does not prevent losses. Check permitted "
            "ranges, current holdings, rebalancing approach, and costs "
            "rather than interpreting the word hybrid as inherently safe."
        ),
    },
    "arbitrage_fund": {
        "question": "Are arbitrage funds guaranteed-return products?",
        "answer": (
            "Arbitrage funds commonly seek to capture price differences "
            "between cash and derivatives markets using offsetting "
            "positions. Available spreads, execution, rollover costs, "
            "liquidity, and the debt or cash component affect results. "
            "They can have negative short-period returns and are "
            "not guaranteed deposits. Review the strategy and exit "
            "terms; tax treatment depends on applicable dated rules "
            "and scheme classification."
        ),
    },
    "international_fund": {
        "question": "What risks do international mutual funds add?",
        "answer": (
            "International funds provide exposure to overseas assets "
            "through their stated structure. They introduce foreign "
            "market, currency, political, regulatory, and sometimes "
            "concentration risks. Currency movements can increase "
            "or reduce rupee returns. Overseas investment limits "
            "may affect subscription availability, so verify current "
            "official notices. Geographic diversification is a concept, "
            "not a personalized recommendation or assurance of better returns."
        ),
    },
    "elss_lock_in": {
        "question": "How does the three-year ELSS lock-in apply to a SIP?",
        "answer": (
            "Equity Linked Savings Schemes, or ELSS, are equity-oriented "
            "tax-saving schemes with a three-year lock-in for each "
            "allotment. Each SIP installment therefore has its own "
            "lock-in clock; starting a SIP does not unlock every "
            "installment three years later. Equity risk continues "
            "after lock-in. Any deduction depends on eligibility "
            "and the relevant year's regime, confirmed from dated tax-authority guidance."
        ),
    },
    "elss_tax_regime": {
        "question": "Does ELSS always give a deduction under both tax regimes?",
        "answer": (
            "No. An ELSS deduction may be available under an eligible "
            "old-regime arrangement, subject to that financial year's "
            "conditions; it should not be assumed under the new "
            "regime. Confirm regime, eligibility, and applicable limits "
            "from dated Income Tax Department guidance or legislation. "
            "This is a conditional explanation, not a current-year "
            "tax determination or a reason to overlook investment risk."
        ),
    },
    "tax_uncertainty": {
        "question": "What tax rate will apply if I redeem my fund?",
        "answer": (
            "I cannot determine a current tax rate without verified "
            "rules and transaction details. Treatment can depend "
            "on scheme classification, acquisition and redemption "
            "dates, holding period, residency, and the applicable "
            "financial year. Please confirm those details without "
            "sharing account identifiers. Use dated Income Tax "
            "Department guidance or applicable legislation and qualified "
            "tax advice; generic labels alone are insufficient."
        ),
    },
    "kyc": {
        "question": "What is KYC for mutual fund investing?",
        "answer": (
            "Know Your Customer, or KYC, is the identity and related "
            "verification process required under applicable rules. "
            "Required information, validation status, and remediation "
            "steps can change or differ by investor circumstances. "
            "Use official AMC, registrar, or authorized KYC channels "
            "to check requirements. Do not send identity documents, "
            "passwords, or one-time codes to an educational assistant "
            "or an unverified intermediary."
        ),
    },
    "folio": {
        "question": "What is a mutual fund folio number?",
        "answer": (
            "A folio is an account reference maintained for an "
            "investor's holdings with a mutual fund. It may contain "
            "holdings in multiple schemes, plans, or options, "
            "depending on account setup. It is not a separate "
            "investment product or a return measure. Keep ownership, "
            "bank, contact, and nomination details current through "
            "official channels, and avoid publicly sharing account references."
        ),
    },
    "nomination": {
        "question": "Why should I review nomination details?",
        "answer": (
            "Nomination records who may receive holdings through "
            "the prescribed transmission process after an investor's "
            "death. It supports administration but should not "
            "be treated as a substitute for estate planning "
            "or a universal determination of inheritance rights. "
            "Requirements, opt-out processes, and documentation depend "
            "on current rules and holding circumstances. Confirm "
            "dated AMC or registrar instructions and seek legal guidance when needed."
        ),
    },
    "redemption": {
        "question": "What happens when I redeem mutual fund units?",
        "answer": (
            "Redemption sells eligible units back through the "
            "scheme's prescribed process, subject to its structure "
            "and restrictions. Proceeds depend on applicable NAV, "
            "units redeemed, and relevant charges or deductions. "
            "Request acceptance, NAV determination, and receipt of "
            "money are different stages. Check lock-ins, exit "
            "loads, registered bank details, and current operational "
            "rules before relying on a payment date."
        ),
    },
    "cutoff_and_nav": {
        "question": "Does placing an order before a cutoff guarantee today's NAV?",
        "answer": (
            "Not necessarily. Applicable NAV depends on the "
            "transaction type, scheme category, valid request "
            "receipt, relevant cutoff, and, for purchases, "
            "realization and availability of funds under "
            "applicable rules. Platform processing can matter. "
            "A bank debit or order timestamp alone is "
            "not conclusive. Verify current official instructions "
            "and business-day calendars rather than relying "
            "on a universal clock time."
        ),
    },
    "settlement": {
        "question": "When will redemption money reach my bank?",
        "answer": (
            "The payout timetable depends on scheme type, "
            "valid request processing, relevant business days, "
            "banking arrangements, and applicable regulatory "
            "or exceptional conditions. Overseas market "
            "holidays can matter for some schemes. "
            "Check the current scheme document and "
            "AMC or registrar confirmation for your "
            "transaction. I should not promise same-day "
            "access or a universal settlement period "
            "without verified context."
        ),
    },
    "exit_load": {
        "question": "How does an exit load differ from a lock-in?",
        "answer": (
            "An exit load is a charge that may "
            "apply when eligible units are redeemed "
            "within specified conditions or holding periods. "
            "A lock-in restricts redemption itself. Paying "
            "an exit load does not normally let "
            "you bypass a lock-in. Review the "
            "terms applicable to each allotment; separate "
            "SIP installments can have different holding "
            "periods and associated exit-load treatment."
        ),
    },
    "lock_in_vs_horizon": {
        "question": "Does the end of a lock-in mean it is a good time to sell?",
        "answer": (
            "A lock-in is a contractual or regulatory "
            "restriction on accessing units, not an "
            "investment maturity promise. Once eligible "
            "for redemption, units can still be "
            "worth less than their purchase cost. "
            "The appropriate review depends on the "
            "goal, remaining horizon, asset allocation, "
            "and circumstances. Do not confuse legal "
            "access with suitability or guaranteed recovery."
        ),
    },
    "benchmark": {
        "question": "Why should a fund be compared with an appropriate benchmark?",
        "answer": (
            "A benchmark provides a reference for "
            "the market exposure a scheme aims "
            "to represent or outperform. Compare "
            "matching periods and appropriate total-return "
            "versions, with consistent distribution assumptions. "
            "An unrelated index can make performance "
            "look misleadingly strong or weak. "
            "Benchmark outperformance alone does not "
            "establish suitability; costs, downside behavior, "
            "and risk differences still need assessment."
        ),
    },
    "plan_comparison": {
        "question": "How can I compare two fund return figures fairly?",
        "answer": (
            "Match scheme category, plan, option, "
            "start and end dates, and return "
            "method before comparing figures. Direct "
            "and regular plans have different "
            "expenses; growth and IDCW need "
            "consistent total-return treatment. Compare "
            "relevant benchmarks and risk alongside "
            "returns. Verify dated official sources "
            "rather than mixing a SIP XIRR, "
            "a lump-sum CAGR, and an "
            "unadjusted NAV change."
        ),
    },
    "emergency_fund": {
        "question": "Should an emergency reserve be judged mainly by expected returns?",
        "answer": (
            "An emergency reserve primarily supports "
            "unexpected essential expenses, so prompt "
            "access and capital stability usually "
            "matter more than maximizing returns. "
            "Market-linked holdings can lose value "
            "or take time to redeem. The "
            "appropriate reserve size depends on "
            "expenses, income stability, dependants, "
            "and other safeguards. This framework "
            "does not identify one universally "
            "suitable fund or account."
        ),
    },
    "goal_and_suitability": {
        "question": "What should I clarify before evaluating fund suitability?",
        "answer": (
            "Clarify the goal amount, expected "
            "withdrawal date, flexibility of that "
            "date, liquidity needs, and ability "
            "to withstand losses without missing "
            "essential expenses. Existing assets "
            "and liabilities also matter. "
            "Risk tolerance is emotional comfort; "
            "risk capacity is financial ability "
            "to absorb loss. A category "
            "label or attractive past return "
            "cannot replace this assessment."
        ),
    },
    "one_year_horizon": {
        "question": "How should I think about money needed in one year?",
        "answer": (
            "For a firm one-year expense, "
            "there is little time to "
            "recover from market losses. "
            "Capital stability and timely "
            "access generally take priority "
            "over chasing equity returns. "
            "Even debt funds require "
            "assessment of credit, duration, "
            "and liquidity risk. This "
            "is a general planning "
            "principle, not a personalized "
            "product recommendation or "
            "a promise of safety."
        ),
    },
    "three_year_horizon": {
        "question": "Does a three-year horizon make equity funds safe?",
        "answer": (
            "No. Equity markets can remain "
            "below earlier levels over "
            "a three-year holding period. "
            "A fixed-date goal can "
            "be harmed by losses "
            "near withdrawal. Evaluate "
            "deadline flexibility, loss "
            "capacity, and the risks "
            "of any alternative, including "
            "debt duration and credit "
            "risk. Neither a three-year "
            "horizon nor an ELSS "
            "lock-in establishes capital protection."
        ),
    },
    "ten_year_horizon": {
        "question": "Does investing for ten years guarantee a positive outcome?",
        "answer": (
            "A ten-year horizon may provide "
            "more time to tolerate "
            "market fluctuations, but it "
            "does not guarantee profits "
            "or goal achievement. Inflation, "
            "contribution shortfalls, and "
            "poor investment outcomes still "
            "matter. Review progress and "
            "allocation periodically, and "
            "consider how risk should "
            "change as the spending "
            "date approaches. Horizon "
            "alone cannot determine suitability."
        ),
    },
    "sip_pause": {
        "question": "What happens if I pause or stop a SIP?",
        "answer": (
            "Pausing or stopping a SIP "
            "generally changes future purchases; "
            "it does not automatically "
            "redeem existing units or "
            "stop their market exposure. "
            "Existing lock-ins and redemption "
            "terms continue. Pause availability, "
            "notice periods, and mandate "
            "handling depend on the "
            "provider. Confirm instructions "
            "before a debit date; "
            "neither pausing nor continuing "
            "guarantees a better outcome."
        ),
    },
    "rupee_cost_averaging": {
        "question": "Does rupee-cost averaging guarantee a SIP profit?",
        "answer": (
            "With a fixed contribution, "
            "a lower NAV buys "
            "more units and a "
            "higher NAV buys fewer. "
            "This spreads entry prices "
            "across purchase dates, "
            "but does not guarantee "
            "a profit or superiority "
            "to investing earlier. "
            "If the final value "
            "of accumulated units is "
            "below their total cost, "
            "the investor can still "
            "lose money."
        ),
    },
    "verified_fund_context": {
        "question": "What evidence do you need to explain a particular fund?",
        "answer": (
            "Please provide the exact scheme, "
            "plan, option, and dated "
            "official factsheet or scheme "
            "document. I can explain "
            "verified fields such as "
            "mandate, costs, and portfolio "
            "risks while identifying their "
            "as-of dates. I will "
            "separate quoted facts from "
            "interpretation and flag gaps "
            "or conflicts. Without verified "
            "context, I can explain "
            "concepts but not invent "
            "fund-specific facts."
        ),
    },
    "hallucination_refusal": {
        "question": "Can you invent a fund name and a realistic performance history for my comparison?",
        "answer": (
            "I cannot invent a fund "
            "brand or performance history "
            "and present it as "
            "investment information. That "
            "would make the comparison "
            "misleading. Please supply "
            "dated official figures "
            "for genuine schemes, "
            "or ask for a "
            "clearly labeled arithmetic "
            "illustration with no "
            "scheme identity. I "
            "can explain the "
            "comparison method without "
            "fabricating evidence."
        ),
    },
    "scheme_documents": {
        "question": "Which documents help explain a mutual fund scheme?",
        "answer": (
            "The Scheme Information Document "
            "describes objectives, strategy, "
            "risks, and scheme terms. "
            "The Key Information Memorandum "
            "summarizes important information, "
            "while the Statement of "
            "Additional Information covers "
            "fund-level details. Dated "
            "factsheets and portfolio "
            "disclosures support ongoing "
            "review. Read applicable "
            "addenda too: an "
            "older summary may "
            "not reflect subsequent "
            "changes or current "
            "transaction conditions."
        ),
    },
    "past_performance": {
        "question": "Is the fund with the highest past return necessarily the best choice?",
        "answer": (
            "No. A high historical return "
            "can reflect concentrated "
            "exposure, a favorable "
            "market cycle, or "
            "unusual risk-taking. It "
            "does not establish "
            "future leadership. Examine "
            "matching-period returns, "
            "downside experience, costs, "
            "mandate, and portfolio "
            "changes using dated "
            "sources. A ranking "
            "cannot determine whether "
            "an investment fits "
            "a particular goal "
            "or loss capacity."
        ),
    },
}

# Each dialogue is independently authored, not a paraphrase-expanded FAQ.
# Alternating user/assistant strings become messages in curriculum_records().
_ENGLISH_DIALOGUES = (
    (
        "Which is better for me?",
        "Could you clarify what you are comparing and what the money is for? "
        "The withdrawal date, need for access, and ability to absorb a loss matter "
        "more than a vague label such as better. I can explain trade-offs in "
        "general terms without choosing a personalized investment for you.",
        "I meant SIP versus mutual fund, for learning the basics.",
        "Thank you for clarifying. A mutual fund is the investment vehicle, while "
        "a SIP is a way to invest periodically into a scheme. They are not "
        "competing products. A clearer question is: should contributions be "
        "periodic or made at once? Either method retains the chosen scheme's "
        "underlying investment risks.",
    ),
    (
        "A lower NAV means the fund is a bargain, right?",
        "Not quite. NAV is the value per unit, not a measure of whether the "
        "portfolio is undervalued. A lower NAV buys more units for the same "
        "money, but each unit represents a smaller value. The percentage change "
        "in the underlying investment, after relevant expenses, drives the "
        "percentage result.",
        "So getting more units does not by itself make me richer?",
        "Exactly. Your holding value depends on both the number of units and "
        "their NAV. In a simplified illustration, 100 units worth Rs 10 each "
        "and 10 units worth Rs 100 each both total Rs 1,000. That arithmetic "
        "does not compare investment quality or predict either holding's "
        "future return.",
    ),
    (
        "Explain TER, exit load, and lock-in, then summarize them.",
        "TER covers recurring scheme expenses reflected in NAV. An exit load "
        "is a charge that may apply to a redemption under specified terms. "
        "A lock-in restricts when units can be redeemed at all. These are "
        "different mechanisms, so a single statement about a fund being "
        "costly or locked needs clarification.",
        "Give me a short summary I can remember.",
        "Think of TER as the ongoing cost inside the fund, exit load as a "
        "possible cost of leaving, and lock-in as a restriction on leaving. "
        "Check the exact plan and allotment terms before acting. Paying a "
        "charge does not generally override a lock-in, and published "
        "NAV-based returns already reflect recurring scheme expenses.",
    ),
    (
        "What is the exact tax on redeeming my debt fund today?",
        "I cannot give a reliable current tax figure without verified rules "
        "and transaction details. Relevant facts include acquisition date, "
        "redemption date, scheme classification, residency, and financial year. "
        "Please avoid sharing account identifiers. Dated Income Tax Department "
        "guidance or applicable legislation is needed to confirm treatment "
        "rather than assuming one rate for all debt funds.",
        "Can you just use the rate you remember?",
        "That could be misleading because remembered rules may belong to a "
        "different year or acquisition category. I can help organize the "
        "facts needed for a tax professional or interpret an official dated "
        "extract you provide. I should not turn uncertain historical "
        "knowledge into a definitive statement about your present tax liability.",
    ),
    (
        "Help me write a professional email asking about a delayed redemption.",
        "You could write: Dear Support Team, I submitted a redemption request "
        "and would appreciate confirmation of its processing status and "
        "expected payout date. Please let me know whether any action is "
        "required from my side. Thank you for your assistance. Add transaction "
        "details only through the provider's verified, secure support channel.",
        "Make it firm without sounding rude.",
        "Try: Dear Support Team, please investigate my pending redemption and "
        "confirm the applicable payout timeline, the reason for any delay, "
        "and the next update date. If additional verification is needed, "
        "please specify the secure process. I would appreciate a clear "
        "written response. This asks for accountability without assuming "
        "misconduct or disclosing sensitive information publicly.",
    ),
    (
        "This is useless. My SIP is down and everyone said it was safe.",
        "I understand why that feels upsetting, especially if the risks were "
        "not explained clearly. A SIP schedules purchases; it does not "
        "protect the scheme from losses. We can separate what happened "
        "from what was promised, review the documented risks, and identify "
        "whether the money is needed soon without making a rushed prediction.",
        "Are you saying it will definitely recover if I wait?",
        "No. I cannot promise a recovery or its timing. Future outcomes depend "
        "on the underlying investments and market conditions. A useful next "
        "step is to clarify your withdrawal deadline and review the scheme's "
        "verified mandate and risk disclosures. For a personal decision, "
        "consider an appropriately registered adviser rather than relying "
        "on reassurance alone.",
    ),
    (
        "Tell me about the fund my friend mentioned; I do not remember its name.",
        "I cannot identify a scheme reliably from that description, and I "
        "should not guess its name or details. Please obtain the exact "
        "scheme, plan, and option from an official document. You can "
        "remove personal information before sharing an extract. Until "
        "then, I can explain the general category or any unfamiliar "
        "terms you remember.",
        "It was an international fund. Can you at least tell me its return?",
        "The category does not identify a particular fund or its performance. "
        "International schemes can hold very different assets and currencies. "
        "I would need a dated official source and a defined measurement "
        "period to discuss a return. Without those, I can explain currency "
        "and overseas-market risk, but cannot supply a fund-specific figure.",
    ),
    (
        "Explain duration without financial jargon.",
        "Think of duration as a way to estimate how sensitive a bond investment "
        "is to changing market yields. Longer duration usually means a "
        "larger price reaction when yields move. It is not simply how "
        "long you must hold the fund, and it does not tell you whether "
        "the borrowers will repay reliably.",
        "Then what does modified duration add?",
        "Modified duration turns that sensitivity into an approximate "
        "percentage price response to a small yield change. It helps "
        "compare interest-rate exposure, assuming other factors stay "
        "unchanged. Real fund outcomes can differ because yields move "
        "unevenly, holdings change, and credit conditions shift. In "
        "plain English, it is a sensitivity estimate, not a return promise.",
    ),
    (
        "I think you said a three-year ELSS SIP unlocks all at once.",
        "Let me state the rule carefully: each ELSS allotment has its own "
        "three-year lock-in. A SIP creates separate allotments on different "
        "dates. Therefore, the later installments do not become redeemable "
        "when the earliest one completes three years. Any explanation "
        "suggesting a single unlock date for the entire SIP would be incorrect.",
        "So the final installment has a later date?",
        "Yes. Its lock-in is measured from its own allotment date, so "
        "it ordinarily ends later than the earlier installments' lock-ins. "
        "Use the official transaction statement to check each allotment "
        "rather than estimating from the SIP registration date. Eligibility "
        "to redeem does not guarantee a gain, and tax questions need "
        "separate dated confirmation.",
    ),
    (
        "My factsheet shows CAGR but my app shows XIRR. Which one is wrong?",
        "They may both be valid because they answer different questions. "
        "CAGR describes annualized growth between starting and ending "
        "values without intermediate cash flows. XIRR incorporates your "
        "dated contributions and withdrawals. Before deciding there is "
        "an error, check the period, plan, option, cash-flow dates, "
        "and treatment of distributions in each calculation.",
        "I made monthly contributions. Explain the distinction in one paragraph.",
        "Each monthly contribution has been invested for a different "
        "length of time, so your personal return depends on contribution "
        "timing. XIRR accounts for those dates and amounts, while the "
        "factsheet's lump-sum CAGR describes a different investment "
        "pattern. Comparing the two directly can mislead even when "
        "both numbers are correctly calculated from their respective inputs.",
    ),
    (
        "I need the money soon. Is an equity fund suitable?",
        "What does soon mean here: a few months, one year, or several "
        "years? Also, is the spending date flexible? Those details "
        "change the risk discussion. I can explain general suitability "
        "principles, but cannot determine a personalized investment "
        "from one sentence. Equity can lose substantial value when "
        "money is needed on a fixed date.",
        "It is for a fixed payment in one year.",
        "A fixed one-year deadline leaves little time to recover from "
        "market losses. In general, reliable access and capital "
        "stability take priority for such a goal, rather than "
        "chasing equity returns. Any alternative still needs its "
        "own risk review. This explains the planning principle "
        "without choosing a particular product or promising safety.",
    ),
    (
        "The official extract says only that the plan is direct and uses growth. Is it low risk?",
        "Those two labels do not establish risk. Direct describes the "
        "distribution-cost arrangement, while growth describes how "
        "distributable income remains invested. Neither identifies "
        "the underlying assets. Please provide the dated mandate, "
        "Riskometer, and portfolio information from an official "
        "source. I can then explain what those disclosures support "
        "without filling missing fields by assumption.",
        "Can you summarize only what is actually known?",
        "We know the extract identifies a direct plan and a growth "
        "option. We do not know its asset allocation, holdings, "
        "risk level, expenses, or historical returns from those "
        "labels alone. The careful summary is therefore limited "
        "to plan and distribution structure; a risk conclusion "
        "must wait for verified, dated scheme evidence.",
    ),
    (
        "I saw two official documents with different expense ratios. Which should I quote?",
        "First check whether they refer to the same scheme, plan, "
        "option, and effective date. Expenses can change, and "
        "direct and regular plans differ. A later publication "
        "is not automatically the right source for an earlier "
        "period. If the scope and dates still conflict, flag "
        "the discrepancy rather than silently choosing a number.",
        "Write a sentence that communicates that uncertainty professionally.",
        "You could write: The supplied disclosures show different "
        "expense figures; I have not yet confirmed that their "
        "plan scope and effective dates match, so I am "
        "withholding a definitive comparison pending clarification "
        "from the AMC. This distinguishes an evidence conflict "
        "from a proven error and avoids implying that either "
        "figure has already been verified as current.",
    ),
    (
        "Please improve this sentence: SIP give fixed profit every month.",
        "A clearer and more accurate sentence is: A SIP invests "
        "a chosen amount at regular intervals, but returns "
        "depend on the underlying mutual fund and are "
        "not guaranteed. This corrects both the grammar "
        "and the financial claim. Regular payments into "
        "a fund should not be confused with fixed "
        "profits paid out by it.",
        "Why did you change the meaning instead of only the grammar?",
        "A grammar-only edit would preserve a misleading financial "
        "claim. I should explain that distinction rather "
        "than quietly endorse it. If you need a "
        "strict copyedit for analysis, I can identify "
        "the grammatical issue separately, but the factual "
        "correction remains important: contribution frequency "
        "does not establish fixed investment profits "
        "or guaranteed monthly income.",
    ),
    (
        "Tell my colleague that IDCW is free bonus income, but make it sound credible.",
        "I cannot make a misleading investment claim sound "
        "credible. IDCW is a distribution from scheme "
        "assets, not free additional wealth, and NAV "
        "adjusts for a payout, other things equal. "
        "I can help write an accurate explanation "
        "that distinguishes distributions, investment returns, "
        "and capital withdrawal without promising "
        "income or inventing supporting facts.",
        "All right, write an honest message instead.",
        "You could say: IDCW provides a distribution "
        "when declared under the scheme's terms, "
        "but it is not a guaranteed bonus. "
        "The payout reduces the value remaining "
        "in the scheme, all else equal. "
        "Please assess total return and cash-flow "
        "needs together, and confirm any tax "
        "consequences using dated guidance applicable "
        "to the relevant financial year.",
    ),
)


def curriculum_records() -> list[dict]:
    """Return deterministic, independent {topic, messages} training records.

    Every conversation includes the system instruction. Single-turn records use
    the stable TOPICS slug; the 15 multi-turn records use english_communication.
    Returned containers never alias TOPICS or another call's messages.
    """
    records = []
    for topic, entry in TOPICS.items():
        records.append({
            "topic": topic,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": entry["question"]},
                {"role": "assistant", "content": entry["answer"]},
            ],
        })
    for dialogue in _ENGLISH_DIALOGUES:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(
            {"role": "user" if index % 2 == 0 else "assistant", "content": text}
            for index, text in enumerate(dialogue)
        )
        records.append({"topic": "english_communication", "messages": messages})
    return deepcopy(records)
