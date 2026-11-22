"""Fetch real Indian mutual-fund / market text for pretraining (polite, robots-aware).

Sources:
  * Wikipedia (CC BY-SA): seed articles + selected Indian finance categories,
    plain text via the MediaWiki API.
  * SEBI investor site and AMFI investor pages: same-site crawl from seed URLs,
    respecting robots.txt, 1 request/second, page caps.

Output: data/raw/mf_corpus/web_<source>.jsonl  ({"text", "url", "source", "license"})
Run: python -m data_prep.fetch_mf_web
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.robotparser
from collections import deque
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "mf_corpus"
UA = "MentraFi-corpus-builder/1.0 (educational research; low rate)"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = UA

WIKI_SEEDS = [
    "Mutual fund", "Mutual funds in India", "Securities and Exchange Board of India",
    "Association of Mutual Funds in India", "Systematic investment plan", "Net asset value",
    "Exchange-traded fund", "Index fund", "Expense ratio", "Open-end fund", "Closed-end fund",
    "Fund of funds", "Money market fund", "Bond fund", "Balanced fund", "Equity-linked savings scheme",
    "Unit Trust of India", "BSE SENSEX", "NIFTY 50", "Bombay Stock Exchange", "National Stock Exchange of India",
    "Reserve Bank of India", "Indian rupee", "Income tax in India", "Public Provident Fund (India)",
    "National Pension System", "Employees' Provident Fund Organisation", "Securities market in India",
    "Stock market in India", "Economy of India", "Inflation in India", "Banking in India",
    "Government of India bonds", "Treasury bill", "Asset allocation", "Diversification (finance)",
    "Compound interest", "Dollar cost averaging", "Sharpe ratio", "Alpha (finance)", "Beta (finance)",
    "Bond duration", "Yield to maturity", "Credit risk", "Interest rate risk", "Liquidity risk",
    "Inflation", "Real interest rate", "Rebalancing investments", "Active management", "Passive management",
    "Market capitalization", "Large-cap", "Mid-cap stock", "Small-cap stock", "Value investing",
    "Growth investing", "Benchmark (finance)", "Tracking error", "Total return index", "Annuity",
    "Retirement planning", "Financial literacy", "Know your customer", "Depository participant",
    "National Securities Depository Limited", "Central Depository Services", "Demat account",
    "Capital gains tax", "Dividend", "Stock split", "Bull market", "Market trend", "Volatility (finance)",
    "Standard deviation", "Risk–return spectrum", "Modern portfolio theory", "Gold exchange-traded fund",
    "Sovereign Gold Bond", "Kisan Vikas Patra", "National Savings Certificate", "Sukanya Samriddhi Account",
    "Fixed deposit", "Recurring deposit", "Corporate bond", "Government bond", "Credit rating agency",
    "CRISIL", "ICRA Limited", "CARE Ratings", "Insurance Regulatory and Development Authority",
    "Pension Fund Regulatory and Development Authority", "Goods and Services Tax (India)",
    "Securities Appellate Tribunal", "SEBI Act, 1992", "Harshad Mehta", "2008 financial crisis",
    "COVID-19 recession", "Demonetisation in India", "Financial planner", "Wealth management",
]
WIKI_CATEGORIES = [
    "Category:Mutual funds of India", "Category:Investment management companies of India",
    "Category:Stock exchanges in India", "Category:Securities and Exchange Board of India",
    "Category:Stock market indices of India", "Category:Taxation in India", "Category:Banking in India",
    "Category:Financial regulatory authorities of India", "Category:Personal finance",
    "Category:Investment funds", "Category:Financial markets in India", "Category:Indian stock market",
    "Category:Pensions in India", "Category:Savings schemes of India",
]
SITES = {
    "sebi": {"seeds": ["https://investor.sebi.gov.in/", "https://investor.sebi.gov.in/understanding_mf.html",
                       "https://investor.sebi.gov.in/iematerial.html"],
             "allow": r"^https://investor\.sebi\.gov\.in/", "max_pages": 400},
    "amfi": {"seeds": ["https://www.amfiindia.com/investor", "https://www.amfiindia.com/"],
             "allow": r"^https://www\.amfiindia\.com/(investor|investor-corner|knowledge|mutual-fund)", "max_pages": 300},
    "mfsahihai": {"seeds": ["https://mutualfundssahihai.com/en"],
                  "allow": r"^https://mutualfundssahihai\.com/en", "max_pages": 300},
}
MF_WORDS = re.compile(r"\b(mutual funds?|sip|nav|scheme|investor|sebi|amfi|equity|debt|portfolio|returns?|"
                      r"redemption|kyc|units?|fund|market|tax|risk)\b", re.I)


def wiki_api(**params):
    params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
    for attempt in range(8):
        try:
            response = SESSION.get("https://en.wikipedia.org/w/api.php", params=params, timeout=30)
        except requests.RequestException:
            time.sleep(5 + attempt * 5)
            continue
        if response.status_code == 200 and "error" not in response.json():
            return response.json()
        wait = int(response.headers.get("Retry-After", "0") or 0)
        time.sleep(max(wait, 5 + attempt * 10))
    raise RuntimeError(f"Wikipedia API kept failing for {params.get('titles') or params.get('cmtitle')}")


def wiki_titles() -> list[str]:
    titles = set(WIKI_SEEDS)
    for category in WIKI_CATEGORIES:
        queue = deque([(category, 0)])
        seen = {category}
        while queue:
            name, depth = queue.popleft()
            cont = {}
            while True:
                try:
                    data = wiki_api(action="query", list="categorymembers", cmtitle=name, cmlimit="500", **cont)
                except RuntimeError as exc:
                    print(f"  skip category: {exc}", flush=True)
                    break
                for member in data["query"]["categorymembers"]:
                    if member["ns"] == 0:
                        titles.add(member["title"])
                    elif member["ns"] == 14 and depth < 1 and member["title"] not in seen:
                        seen.add(member["title"])
                        queue.append((member["title"], depth + 1))
                if "continue" not in data:
                    break
                cont = {"cmcontinue": data["continue"]["cmcontinue"]}
                time.sleep(1.0)
            time.sleep(1.0)
    return sorted(titles)


def clean_wiki(text: str) -> str:
    cut = re.split(r"\n== (?:See also|References|External links|Further reading|Notes|Bibliography) ==", text)[0]
    cut = re.sub(r"\n=+ *([^=\n]+?) *=+\n", r"\n\n\1\n", cut)
    return re.sub(r"\n{3,}", "\n\n", cut).strip()


def fetch_wikipedia() -> int:
    titles = wiki_titles()
    print(f"wikipedia: {len(titles)} titles", flush=True)
    count = 0
    with (OUT / "web_wikipedia.jsonl").open("w", encoding="utf-8") as out:
        for index, title in enumerate(titles):
            try:
                data = wiki_api(action="query", prop="extracts", explaintext="1", redirects="1", titles=title)
                page = data["query"]["pages"][0]
                text = clean_wiki(page.get("extract") or "")
            except Exception as exc:  # network hiccup: skip the page, keep going
                print(f"  skip {title}: {exc}", flush=True)
                continue
            if len(text.split()) >= 120:
                out.write(json.dumps({"text": f"{page['title']}\n\n{text}", "source": "wikipedia",
                                      "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page['title'])}",
                                      "license": "CC BY-SA 4.0"}, ensure_ascii=False) + "\n")
                count += 1
            if index % 200 == 0:
                print(f"  wikipedia {index}/{len(titles)} kept {count}", flush=True)
            time.sleep(0.6)
    return count


def page_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "noscript", "iframe", "svg", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    blocks = []
    for node in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        text = " ".join(node.get_text(" ", strip=True).split())
        if len(text) >= 25 and not re.search(r"cookie|javascript|copyright ©|all rights reserved", text, re.I):
            blocks.append(text)
    deduped, seen = [], set()
    for block in blocks:
        if block not in seen:
            seen.add(block)
            deduped.append(block)
    return "\n\n".join(deduped)


def crawl(name: str, config: dict) -> int:
    allow = re.compile(config["allow"])
    robots = {}
    queue = deque(config["seeds"])
    seen = set(queue)
    kept = fetched = 0
    with (OUT / f"web_{name}.jsonl").open("w", encoding="utf-8") as out:
        while queue and fetched < config["max_pages"]:
            url = queue.popleft()
            host = urllib.parse.urlsplit(url)
            base = f"{host.scheme}://{host.netloc}"
            if base not in robots:
                parser = urllib.robotparser.RobotFileParser(base + "/robots.txt")
                try:
                    parser.read()
                except Exception:
                    parser = None
                robots[base] = parser
            if robots[base] is not None and not robots[base].can_fetch(UA, url):
                continue
            try:
                response = SESSION.get(url, timeout=30)
            except Exception:
                continue
            fetched += 1
            time.sleep(1.0)
            if response.status_code != 200 or "html" not in response.headers.get("content-type", ""):
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a", href=True):
                target = urllib.parse.urljoin(url, link["href"]).split("#")[0]
                if allow.match(target) and target not in seen and not re.search(r"\.(pdf|jpg|png|zip|xls|xlsx|doc|docx|mp4)$", target, re.I):
                    seen.add(target)
                    queue.append(target)
            text = page_text(response.text)
            words = len(text.split())
            if words >= 150 and len(MF_WORDS.findall(text)) >= 8:
                out.write(json.dumps({"text": text, "source": name, "url": url,
                                      "license": "public website text; internal training use"}, ensure_ascii=False) + "\n")
                kept += 1
            if fetched % 50 == 0:
                print(f"  {name}: fetched {fetched}, kept {kept}, queue {len(queue)}", flush=True)
    return kept


def main() -> None:
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    only_wiki = "--wikipedia-only" in sys.argv
    for name, config in ({} if only_wiki else SITES).items():
        try:
            summary[name] = crawl(name, config)
        except Exception as exc:
            print(f"{name} failed: {exc}", flush=True)
            summary[name] = 0
        print(f"{name}: kept {summary[name]}", flush=True)
    summary["wikipedia"] = fetch_wikipedia()
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
