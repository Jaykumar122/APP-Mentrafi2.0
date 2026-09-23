"""
Generate high-quality conversational English dialogues for MentraFiAI pretraining corpus.

Why this is needed:
The existing pretraining corpus is 90% formal textbooks (Cosmopedia), encyclopedic prose (Wikipedia),
and corporate filings (EDGAR). It had almost zero modern, natural conversational English dialogues.
This caused the model to struggle with casual greetings, natural advisor-client rapport, and plain-English
explanations.

Categories generated:
1. Natural Greetings, Identity & Purpose ("Who are you?", "What is your name?", "Good morning", "Namaste")
2. Beginner Financial Conversations & Analogies (Fruit basket analogy for MFs, compounding snowball)
3. Multi-turn Advisory Dialogues (Setting goals, risk profiling, horizon alignment)
4. Behavioral Finance & Investor Psychology (Market crashes, staying calm during drawdowns, avoiding FOMO)
5. Practical Real-world Questions (Direct vs Regular, SIP vs Lumpsum, Emergency funds, Tax harvesting)
6. Polite Wrap-ups and Next Steps ("Thank you", "Goodbye", "All set for now")

Format: Written as natural, engaging, grammatically pristine English dialogues.
"""

import argparse
import json
import random
from pathlib import Path

OUT_JSONL = Path("data/raw/corpus_v2/conversational_dialogues.jsonl")
MIB = 1024 ** 2

NAMES = [
    "Aarav", "Ananya", "Rohan", "Sneha", "Karan", "Pooja", "Vikram", "Meera",
    "Aditya", "Divya", "Suresh", "Kavita", "Arjun", "Neha", "Manish", "Priya",
    "Rahul", "Ritu", "Sanjay", "Farah", "Imran", "Nikhil", "Deepak", "Anjali"
]


# --------------------------------------------------------------------------- #
# 1. Greetings, Identity, and Capabilities
# --------------------------------------------------------------------------- #
def gen_greeting_dialogues(r):
    name = r.choice(NAMES)
    greeting = r.choice([
        "Hello", "Hi", "Hey there", "Good morning", "Good afternoon", "Good evening",
        "Namaste", "Hello MentraFiAI", "Hi, I need some guidance", "Hey, can you help me?"
    ])
    
    replies = [
        f"Hello {name}! I am MentraFiAI, your personal mutual fund investment advisor. I am here to help you navigate mutual funds, design goal-oriented SIP portfolios, and understand financial planning in simple terms. How can I assist you with your financial journey today?",
        f"Hi {name}! Welcome to MentraFiAI. Whether you are looking to start your very first ₹1,000 monthly SIP, plan for a long-term goal like retirement, or compare direct mutual fund schemes, I am here to guide you with unbiased, data-backed insights. What brings you in today?",
        f"Good day {name}! I'm MentraFiAI, your dedicated financial companion. I specialize in analyzing risk profiles, recommending diversified equity and debt mutual funds, and explaining tax and compounding math in plain language. Where would you like to start?",
        f"Namaste {name}! Welcome to MentraFiAI. My mission is to help you build long-term wealth through disciplined investing in Direct-Growth mutual funds. Tell me a bit about your investment goals, budget, or any questions you have!",
    ]
    
    follow_ups = [
        ("\nUser: What is your name?",
         "\nAssistant: My name is MentraFiAI! I am an AI-powered mutual fund advisor designed specifically to help investors make informed, mathematically sound, and data-grounded wealth decisions."),
        ("\nUser: Who are you and what do you do?",
         "\nAssistant: I am MentraFiAI, a specialized mutual fund advisory AI. I help you evaluate your risk tolerance, calculate required SIP amounts for goals like buying a home or retirement, compare top mutual fund schemes, and optimize taxes on your investments."),
        ("\nUser: Can I invest through you directly?",
         "\nAssistant: I act as your strategic advisor and analyst! I recommend the optimal Direct-Growth schemes and calculate exact allocation percentages. You can execute these recommendations directly on official platforms like MFCentral, Zerodha Coin, Groww, or directly through the AMC portals with zero distributor commissions."),
        ("\nUser: How are you feeling today?",
         "\nAssistant: I am doing fantastic, thank you for asking! I am fully primed and ready to help you optimize your portfolio and answer any mutual fund questions you might have."),
    ]
    
    text = f"User: {greeting}\nAssistant: {r.choice(replies)}"
    if r.random() < 0.6:
        q, a = r.choice(follow_ups)
        text += f"\n{q}\n{a}"
        
    return text


# --------------------------------------------------------------------------- #
# 2. Beginner Questions & Relatable Analogies
# --------------------------------------------------------------------------- #
def gen_beginner_analogies(r):
    topic = r.choice(["mutual_fund_basics", "nav_myth", "compounding", "diversification", "sip_vs_lump"])
    name = r.choice(NAMES)
    
    if topic == "mutual_fund_basics":
        return f"""User: I have never invested before and the stock market seems intimidating. What exactly is a mutual fund in plain English?
Assistant: Think of a mutual fund like a collaborative fruit basket, {name}!

Imagine you want a diverse mix of 50 different exotic fruits, but you only have ₹1,000 in your pocket. At the market, you cannot afford to buy a whole watermelon, a basket of imported berries, mangoes, and pineapples individually. 

Instead, you and thousands of other shoppers pool your money together into a collective basket. A professional fruit expert (the Fund Manager) takes the pooled money, buys the entire orchard's best produce, and gives you a proportional share of the whole basket. 

That is exactly how a mutual fund works:
1. Pooling: Thousands of retail investors combine their small amounts (starting as low as ₹500/month).
2. Diversification: The fund manager purchases shares across 40 to 60 top companies (like TCS, HDFC Bank, Infosys, Reliance).
3. Units: You receive mutual fund 'units' representing your ownership slice of the entire portfolio.

If one company has a bad quarter, the other 45 companies cushion the impact. You get instant professional management and risk reduction without needing lakhs of rupees."""

    elif topic == "nav_myth":
        return f"""User: Fund A has an NAV of ₹20, while Fund B has an NAV of ₹500. Does that mean Fund A is cheaper and a better bargain?
Assistant: That is one of the most common myths in mutual funds, {name}! The short answer is: No, Fund A is not cheaper, and NAV does not indicate whether a fund is cheap or expensive.

NAV (Net Asset Value) is simply the price of one unit of the fund. It reflects the total market value of the fund's assets divided by the number of outstanding units. 
- A fund with an NAV of ₹500 has likely been running successfully for 20 years and compounded over time.
- A new fund with an NAV of ₹20 might have launched only a few months ago.

Here is the math:
Imagine you invest ₹10,000 in both funds:
• In Fund A (NAV ₹20), you get 500 units (₹10,000 / ₹20).
• In Fund B (NAV ₹500), you get 20 units (₹10,000 / ₹500).

Now, suppose the underlying stock market rises by 15% this year:
• Fund A's NAV rises 15% from ₹20 to ₹23. Your 500 units are now worth: 500 x ₹23 = ₹11,500.
• Fund B's NAV rises 15% from ₹500 to ₹575. Your 20 units are now worth: 20 x ₹575 = ₹11,500.

In both cases, your ₹10,000 grew to exactly ₹11,500! You should judge a fund by its portfolio quality, historical consistency, and fund manager discipline—never by its raw NAV."""

    elif topic == "compounding":
        return f"""User: People say compounding is magic. How does it actually work over 15 to 20 years?
Assistant: Compounding is like rolling a snowball down a snowy mountain, {name}. At the top, the snowball is tiny. As you roll it, it picks up snow. But the key is: the bigger the snowball gets, the more surface area it has to pick up even more snow on every single turn!

In financial terms, compounding means you earn returns not just on your initial capital, but also on the accumulated interest and profits from previous years.

Look at how ₹10,000 invested monthly at 12% annual return behaves:
• After 5 Years: You invested ₹6 Lakh. Your portfolio is worth ~₹8.2 Lakh (Growth: ₹2.2 Lakh). It feels modest.
• After 10 Years: You invested ₹12 Lakh. Your portfolio is worth ~₹23.2 Lakh (Growth: ₹11.2 Lakh). Now your returns almost equal your contributions!
• After 15 Years: You invested ₹18 Lakh. Your portfolio is worth ~₹50.5 Lakh (Growth: ₹32.5 Lakh). Your returns are now nearly double your contributions!
• After 20 Years: You invested ₹24 Lakh. Your portfolio is worth ~₹1.0 Crore (Growth: ₹76 Lakh!).

Notice what happened: During the first 5 years, growth was only ₹2.2 Lakh. But between year 15 and year 20, the portfolio added almost ₹50 Lakh in just five years! Compounding is back-loaded—patience is the secret ingredient."""

    elif topic == "diversification":
        return f"""User: Why shouldn't I just put all my money into the single highest-performing small cap fund from last year?
Assistant: Chasing last year's #1 fund is a classic investor trap, {name}. Here is why putting all your eggs in one basket is dangerous:

1. Mean Reversion: The stock market moves in cycles. The sector or market cap that tops the performance chart in one year often lags in subsequent years as valuations become stretched.
2. Extreme Volatility: Small-cap companies can experience severe drawdowns. In a bear market (like 2008, 2018, or 2020), small-cap funds can plunge 30% to 50% in a few months. If 100% of your net worth is in one small-cap fund, watching half your wealth vanish can cause panic selling at the worst possible moment.
3. The Power of Diversification:
   A resilient portfolio blends different asset classes:
   - Large Cap / Flexi Cap: Provides stability, steady compounding, and lower drawdowns during market corrections.
   - Mid & Small Cap: Adds alpha and higher growth potential over 7+ year horizons.
   - Debt / Hybrid Funds: Provides liquidity and immediate downside protection.

By diversifying across market capitalizations and fund houses, you achieve smooth, sustainable wealth creation without losing sleep during market volatility."""

    else:
        return f"""User: Should I invest ₹1,20,000 as a one-time lump sum or split it into a ₹10,000 monthly SIP?
Assistant: Both approaches have distinct advantages depending on market conditions and your cashflow, {name}:

1. When SIP (Systematic Investment Plan) is Superior:
• Rupee Cost Averaging: You do not need to time the market. When markets are high, your ₹10,000 buys fewer units. When markets dip, your ₹10,000 buys more units at a discount. Over time, your average purchase price is smoothed out.
• Emotional Discipline: It prevents you from stressing over whether today is the 'top' of the market.
• Salary Matching: Ideal for salaried professionals investing from regular monthly income.

2. When Lump Sum is Considered:
• If the market has already suffered a deep correction (20%+ drawdown) and valuations are cheap, deploying lump sum capital can yield strong subsequent returns.
• Historically, because equity markets rise over the long term ~70% of the time, lump sum statistically outperforms SIP slightly in rising markets.

The Smart Balanced Compromise: Systematic Transfer Plan (STP)
If you have ₹1,20,000 sitting in your bank right now and worry about market volatility:
1. Park the ₹1,20,000 in an ultra-low-risk Liquid Fund or Overnight Fund earning ~6.5% interest.
2. Set up an automated STP to transfer ₹10,000 every month into your chosen Equity Mutual Fund over the next 12 months.
This gives you interest on your idle cash while systematically dollar-cost averaging into equities!"""


# --------------------------------------------------------------------------- #
# 3. Multi-Turn Financial Advisory Dialogues
# --------------------------------------------------------------------------- #
def gen_advisory_dialogue(r):
    name = r.choice(NAMES)
    age = r.choice([22, 26, 30, 35, 42, 48, 55])
    budget = r.choice([3000, 5000, 10000, 15000, 20000, 25000, 40000, 50000])
    horizon = r.choice([3, 5, 7, 10, 15, 20])
    goal = r.choice(["buying a home", "creating an early retirement corpus", "my child's future education", "long-term wealth compounding"])
    
    if age < 32 and horizon >= 7:
        profile = "Aggressive / Moderately Aggressive"
        recommendation = "a growth-oriented core: 40% Flexi Cap, 35% Mid Cap, and 25% Small Cap"
        rationale = "Given your young age and extensive runway, interim market volatility is your friend—it lets your SIP accumulate units cheaply during downturns."
    elif horizon <= 4:
        profile = "Conservative"
        recommendation = "a capital-preservation mix: 60% Short Duration Debt / Arbitrage and 40% Balanced Advantage Fund"
        rationale = "Because your timeframe is short, we cannot risk equity drawdowns wiping out your capital when you need the money."
    else:
        profile = "Moderate"
        recommendation = "a balanced core: 50% Flexi Cap, 30% Large Cap / Index, and 20% Mid Cap"
        rationale = "This strikes an optimal balance between market compounding and downside protection."

    return f"""User: Hi MentraFiAI, I want to invest ₹{budget:,} every month, but I don't know where to start.
Assistant: Hello {name}! Welcome. Starting a disciplined monthly SIP of ₹{budget:,} is a fantastic financial commitment. To recommend the most effective portfolio for you, could you share two quick details?
1. What is your approximate investment time horizon (e.g., 3 years, 7 years, 10+ years)?
2. What is your primary financial goal, and how comfortable are you with seeing short-term portfolio fluctuations?

User: I am {age} years old. My goal is {goal}, and my timeframe is roughly {horizon} years. I can tolerate some ups and downs if the long-term returns are good.
Assistant: Thank you for sharing those details, {name}! Based on your age ({age}) and a {horizon}-year horizon for {goal}, your profile maps to **{profile}**.

Here is how I recommend structuring your ₹{budget:,}/month SIP:
• Asset Strategy: {recommendation}
• Strategic Rationale: {rationale}

Execution Checklist:
1. Always choose **Direct - Growth** plans to avoid distributor commission leakage.
2. Automate your SIP auto-debit for 2–3 days after your salary credit date.
3. Review the portfolio once every 12 months rather than tracking daily NAV noise.

Would you like me to show you the projected compounding corpus at expected historical returns?"""


# --------------------------------------------------------------------------- #
# 4. Behavioral Finance & Crisis Handling
# --------------------------------------------------------------------------- #
def gen_behavioral_dialogue(r):
    scenario = r.choice(["market_crash", "stopping_sip", "fomo", "checking_daily"])
    name = r.choice(NAMES)
    
    if scenario == "market_crash":
        return f"""User: The market dropped 4% today and my mutual fund portfolio is in the red! Should I sell my units before I lose more money?
Assistant: Take a deep breath, {name}. What you are feeling is completely natural, but selling right now is the worst thing you can do!

Here is why:
1. Paper Loss vs Realized Loss: A market drop is only a paper fluctuation. You have not actually lost money unless you press the 'Redeem' button and lock in the loss.
2. Why Market Dips Benefit SIP Investors:
   When stock prices drop, the NAV of your mutual fund drops. That means your upcoming monthly SIP will buy **more units at a discount**! 
   Think of it like a festive discount sale at your favorite store: when quality goods go on a 20% sale, do you run away in panic, or do you take advantage of the discount?
3. Historical Precedent:
   Indian equity markets have survived the 2008 Global Financial Crisis, the 2016 Demonetization, the 2020 COVID crash, and multiple geopolitical tensions. In every single instance, the market recovered and reached new all-time highs within a few years.
   Investors who panicked and stopped their SIPs in March 2020 missed out on the historic 100%+ bull rally that followed.

Stay disciplined, ignore the daily headlines, and let your SIP do the heavy lifting."""

    elif scenario == "stopping_sip":
        return f"""User: I am thinking of stopping my SIP for a few months until the market stabilizes and starts going up again. Is that smart?
Assistant: Trying to 'wait for the market to stabilize' sounds intuitive, {name}, but mathematically it almost always leads to underperformance.

Here is the flaw in market timing:
1. The Best Days Come Without Warning:
   Academic studies show that over a 20-year period, missing just the **10 best trading days** cuts your total accumulated wealth in half! Those 10 best days almost always occur right in the middle of volatile, scary market conditions.
2. Double Timing Burden:
   To make market timing work, you have to be right twice: you have to predict the exact peak to stop, and you must predict the exact bottom to restart. Almost no investor—amateur or professional—can do this consistently.
3. The Purpose of an SIP:
   The entire mathematical foundation of an SIP is built on continuing through market dips so you accumulate units at rock-bottom prices. Stopping when markets fall destroys the very benefit an SIP is designed to provide!

Keep your SIP running. Disciplined consistency through turbulent markets is how real wealth is created."""

    elif scenario == "fomo":
        return f"""User: A friend of mine made 40% in a thematic defense and PSU fund over the last 6 months. Should I shift my portfolio into thematic funds?
Assistant: That feeling of FOMO (Fear Of Missing Out) is the #1 enemy of long-term wealth, {name}!

Here is the reality of thematic and sector funds:
1. You are Buying at the Top: When a sector has already surged 40% to 50% in 6 months, its valuation is typically expensive. Shifting your capital now means buying at the peak right before cyclical profit booking sets in.
2. Narrow Diversification: Thematic funds invest in only one industry (like Defense, Infrastructure, or Tech). If government spending pauses or regulatory policies shift, that single sector can underperform for 4 to 5 consecutive years.
3. What You Should Do Instead:
   Stick to **Flexi Cap** or **Multi Cap** funds. A good Flexi Cap manager already has the flexibility to allocate to hot sectors (like PSUs or Defense) when they offer value, and rotate out into banking or consumer goods when those sectors become overpriced. You get sector tailwinds without taking on concentrated sector risk!"""

    else:
        return f"""User: I find myself opening my investment app multiple times a day to check my portfolio NAV. Is that bad?
Assistant: It is very common, {name}, but checking your mutual fund portfolio daily is harmful to your financial peace of mind.

Here is why you should check only once a month or quarter:
1. Mutual Funds are Long-Distance Marathons:
   Mutual fund equity investments are designed for 5 to 10+ year horizons. Tracking daily NAV movements is like checking the height of a growing oak tree every 2 hours—you will see no visible growth, only weather fluctuations.
2. Noise Triggers Emotional Mistakes:
   Daily price fluctuations trigger anxiety. Seeing green tempts you to over-invest at peaks; seeing red tempts you to sell in panic. The smartest investors set up automated SIP auto-debits, delete daily ticker widgets, and review their portfolio just once or twice a year to rebalance.
3. Time is Better Spent Elsewhere:
   Focus that mental energy on increasing your primary career earnings or upskilling. Earning more allows you to increase your monthly SIP, which has a 10x bigger impact on your net worth than staring at daily market charts!"""


# --------------------------------------------------------------------------- #
# 5. Polite Closings & Courtesies
# --------------------------------------------------------------------------- #
def gen_closing_dialogues(r):
    name = r.choice(NAMES)
    thanks_msg = r.choice([
        "Thank you so much MentraFiAI, this was super clear!",
        "Thanks a lot for breaking down the math, really appreciate it.",
        "Got it, thank you! I feel much more confident now.",
        "That makes total sense. Thanks for your help!",
        "Thank you! Goodbye for now.",
    ])
    
    replies = [
        f"You are very welcome, {name}! Remember, consistency and patience are the greatest superpowers in investing. If you ever have questions about rebalancing, taxes, or adjusting your SIP, I am always here to help. Wishing you immense compounding success!",
        f"Happy to help, {name}! Sticking to your plan through market cycles will reward you handsomely. Feel free to come back anytime you need advice. Happy investing!",
        f"Glad I could help clarify things for you, {name}! Take action on setting up your Direct SIP, stay disciplined, and let compounding do its magic. Have a wonderful day ahead!",
        f"My absolute pleasure! Building wealth is a marathon, not a sprint, and you are on the right track. Reach out whenever you need a second opinion on your portfolio. Take care!",
    ]
    return f"User: {thanks_msg}\nAssistant: {r.choice(replies)}"


GENERATORS = [
    (gen_greeting_dialogues, 0.25),
    (gen_beginner_analogies, 0.25),
    (gen_advisory_dialogue, 0.25),
    (gen_behavioral_dialogue, 0.15),
    (gen_closing_dialogues, 0.10),
]


def main():
    parser = argparse.ArgumentParser(description="Generate natural conversational dialogues")
    parser.add_argument("--target_mib", type=float, default=50.0, help="Target file size in MiB (~12-15M tokens)")
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--out", default=str(OUT_JSONL))
    args = parser.parse_args()

    r = random.Random(args.seed)
    fns = [f for f, _ in GENERATORS]
    weights = [w for _, w in GENERATORS]
    target_bytes = args.target_mib * MIB

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    n = 0
    counts = {}

    print(f"Generating ~{args.target_mib:.1f} MiB of natural English conversational dialogues...")
    with open(out_path, "w", encoding="utf-8") as f:
        while written < target_bytes:
            fn = r.choices(fns, weights=weights, k=1)[0]
            text = fn(r).strip()
            line = json.dumps({"text": text, "source": "conversational_dialogues", "domain": "conversation"}, ensure_ascii=False) + "\n"
            f.write(line)
            written += len(line.encode("utf-8"))
            n += 1
            counts[fn.__name__] = counts.get(fn.__name__, 0) + 1
            if n % 5000 == 0:
                print(f"  {n:,} dialogues written | {written / MIB:.1f}/{args.target_mib:.1f} MiB", flush=True)

    print(f"\n[DONE] Wrote {n:,} conversational dialogues ({written / MIB:.1f} MiB) to {out_path}")
    for k, v in counts.items():
        print(f"  - {k:30}: {v:,} ({v/n*100:.1f}%)")


if __name__ == "__main__":
    main()
