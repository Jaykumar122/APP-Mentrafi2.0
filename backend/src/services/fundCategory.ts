// One fund classifier for every sync job.
//
// `category` keeps the app's five filter values (Equity, Debt, Hybrid, ELSS, Gold).
// `subcategory` is the SEBI scheme category (Oct 2017 circular, updated 2020/2023):
// "Multi Cap", "Flexi Cap" and "Large & Mid Cap" are different categories and must
// never be merged. The AMFI NAVAll.txt section header (e.g. "Open Ended Schemes
// (Equity Scheme - Multi Cap Fund)") is the official source; the scheme name is used
// only when no header is available (mfapi.in sync, closed / legacy schemes).
//
// Keep the subcategory strings in sync with inference/db_fund_retriever.py, which
// filters on "Flexi Cap", "Large Cap", "Mid Cap", "Small Cap" and "Tax Saving".

export type FundCategory = "Equity" | "Debt" | "Hybrid" | "ELSS" | "Gold";

export interface FundClass {
  category: FundCategory;
  subcategory: string;
}

type Rule = [RegExp, FundCategory, string];

// Ordered: the first match wins, so more specific patterns come first.
const RULES: Rule[] = [
  // Commodities
  [/\bsilver\b/, "Gold", "Silver"],
  [/\bgold\b/, "Gold", "Gold"],
  // ELSS / tax saver
  [/\belss\b|tax ?saver|tax saving|\blong term equity\b/, "ELSS", "Tax Saving"],
  // Hybrid
  [/\barbitrage\b/, "Hybrid", "Arbitrage"],
  [/\bequity savings?\b/, "Hybrid", "Equity Savings"],
  [/balanced advantage|dynamic asset allocation/, "Hybrid", "Balanced Advantage"],
  [/multi[- ]?asset/, "Hybrid", "Multi Asset Allocation"],
  [/conservative hybrid|debt hybrid|monthly income plan|\bmip\b|regular savings/, "Hybrid", "Conservative Hybrid"],
  [/balanced hybrid/, "Hybrid", "Balanced Hybrid"],
  [/aggressive hybrid|equity hybrid|hybrid equity|\bequity (?:&|and) debt\b/, "Hybrid", "Aggressive Hybrid"],
  [/life ?cycle/, "Hybrid", "Life Cycle"],
  [/\bhybrid\b|\bbalanced\b/, "Hybrid", "Hybrid"],
  // Debt
  [/\bovernight\b/, "Debt", "Overnight"],
  [/\bliquid\b/, "Debt", "Liquid"],
  [/ultra short/, "Debt", "Ultra Short Duration"],
  [/\blow duration\b/, "Debt", "Low Duration"],
  [/money market/, "Debt", "Money Market"],
  [/medium to long/, "Debt", "Medium to Long Duration"],
  [/\bmedium (?:duration|term)\b/, "Debt", "Medium Duration"],
  [/\blong duration\b|\blong term (?:debt|bond|income|gilt|fund)\b/, "Debt", "Long Duration"],
  [/\bshort (?:duration|term)\b/, "Debt", "Short Duration"],
  [/dynamic (?:bond|term|debt)/, "Debt", "Dynamic Bond"],
  [/corporate bond/, "Debt", "Corporate Bond"],
  [/credit risk/, "Debt", "Credit Risk"],
  [/banking (?:&|and) psu/, "Debt", "Banking & PSU"],
  [/\bgilt\b|constant maturity|\bg-?sec\b/, "Debt", "Gilt"],
  [/\bfloat(?:er|ing)\b/, "Debt", "Floater"],
  [/fixed maturity|\bfmp\b|fixed term|fixed tenure|fixed horizon|dual advantage|\bftif\b|\binterval\b|capital protection/, "Debt", "Fixed Maturity / Interval"],
  [/\b(?:debt|bond|income|treasury|savings fund|crisil[- ]ibx|sdl|psu bond)\b/, "Debt", "Debt"],
  // Overseas exposure first (a "Nasdaq 100 ETF" is international), then passive equity:
  // a "Nifty Midcap 150 Index Fund" is an Index Fund, not a Mid Cap fund.
  [/overseas|international|\bglobal\b|\bus\b|u\.s\.|nasdaq|\bworld\b|china|japan|europe|emerging markets|\basia|hang seng|s&p 500|\bnyse\b|fang\+?/, "Equity", "International"],
  [/\b(?:etf|exchange traded)\b/, "Equity", "ETF"],
  [/\bindex\b|\bnifty\b|\bsensex\b|\bbse \d/, "Equity", "Index Fund"],
  // Active equity, by SEBI category
  [/large (?:&|and) mid ?cap|large ?& ?midcap|largemidcap/, "Equity", "Large & Mid Cap"],
  [/multi ?cap/, "Equity", "Multi Cap"],
  [/flexi ?cap/, "Equity", "Flexi Cap"],
  [/small ?cap/, "Equity", "Small Cap"],
  [/mid ?cap|\bemerging\b/, "Equity", "Mid Cap"],
  [/large ?cap|blue ?chip|\btop 100\b/, "Equity", "Large Cap"],
  [/\bfocused\b/, "Equity", "Focused"],
  [/\bcontra\b/, "Equity", "Contra"],
  [/\bvalue\b/, "Equity", "Value"],
  [/dividend yield/, "Equity", "Dividend Yield"],
  [/\bretirement\b|\bpension\b/, "Equity", "Retirement"],
  [/\bchild(?:ren)?'?s?\b/, "Equity", "Children's"],
  [/fund of funds?|\bfof\b/, "Equity", "Fund of Funds"],
  [/banking|financial services|pharma|healthcare|infra|technology|digital|consumption|manufacturing|\bpsu\b|energy|\bauto\b|fmcg|innovation|\besg\b|thematic|sector|business cycle|special situations|transportation|defen[cs]e|housing|\bmnc\b|momentum|rural|commodities/,
    "Equity", "Sectoral / Thematic"],
];

// Map the AMFI section header's text inside the brackets to the same subcategory names.
function fromHeader(header: string, name: string): FundClass | null {
  const inner = (header.match(/\((.*)\)\s*$/)?.[1] ?? header).toLowerCase();
  // Old generic headers ("Growth", "Income", "ELSS", "Gilt") carry no SEBI category.
  if (!inner.includes(" - ")) return null;
  const [group, detailRaw] = inner.split(/\s+-\s+/, 2);
  const detail = detailRaw ?? "";
  if (/sectoral|thematic/.test(detail) && /equity/.test(group)) return { category: "Equity", subcategory: "Sectoral / Thematic" };
  // Passive funds: the header says "index fund" / "ETF"; the asset class comes from the
  // header when it is explicit, otherwise from the name (generic "Other Scheme - Index
  // Funds" and "Other ETFs" contain debt index funds and liquid ETFs).
  const passive = /index funds/.test(group) || /\bindex funds\b/.test(detail) ? "Index Fund"
    : /exchange traded|etf/.test(group) || /\betfs?\b/.test(detail) ? "ETF" : null;
  if (passive) {
    if (/silver/.test(detail)) return { category: "Gold", subcategory: "Silver" };
    if (/gold/.test(detail)) return { category: "Gold", subcategory: "Gold" };
    if (/debt/.test(detail)) return { category: "Debt", subcategory: passive };
    if (/hybrid/.test(detail)) return { category: "Hybrid", subcategory: passive };
    if (/overseas/.test(detail)) return { category: "Equity", subcategory: "International" };
    if (/equity/.test(detail)) return { category: "Equity", subcategory: passive };
    const byName = matchRules(name);
    if (byName && byName.category !== "Equity" && byName.category !== "ELSS") {
      return { category: byName.category, subcategory: byName.category === "Gold" ? byName.subcategory : passive };
    }
    return { category: "Equity", subcategory: passive };
  }
  if (/fund of funds|fof/.test(group + " " + detail)) return null; // name decides (gold FoF, debt FoF, overseas FoF)
  if (/solution oriented|children/.test(group)) {
    const byName = matchRules(name);
    const category = byName && (byName.category === "Debt" || byName.category === "Hybrid") ? byName.category : "Equity";
    return { category, subcategory: /retirement/.test(detail) ? "Retirement" : "Children's" };
  }
  if (/life cycle/.test(group)) return { category: "Hybrid", subcategory: "Life Cycle" };
  // Equity / Hybrid / Debt headers: reuse the name rules on the official category text.
  const prefix = /hybrid/.test(group) ? "hybrid " : /debt|income/.test(group) ? "debt " : "";
  const match = matchRules(prefix + detail);
  if (!match) return null;
  // The header's group is authoritative for the top-level category.
  if (/equity/.test(group) && match.category !== "ELSS") return { category: "Equity", subcategory: match.subcategory };
  if (/hybrid/.test(group)) return { category: "Hybrid", subcategory: match.category === "Hybrid" ? match.subcategory : "Hybrid" };
  if (/debt|income/.test(group)) return { category: "Debt", subcategory: match.category === "Debt" ? match.subcategory : "Debt" };
  return match;
}

// Plan / option wording says nothing about the portfolio ("Income Distribution cum
// Capital Withdrawal" is the IDCW option, not a debt fund).
const PLAN_WORDS = /income distribution cum capital withdrawal|\bidcw\b|\b(?:direct|regular) plan\b|\bgrowth option\b|\bdividend (?:payout|reinvestment|option)\b|\bbonus\b/g;

function matchRules(text: string): FundClass | null {
  const t = text.toLowerCase().replace(PLAN_WORDS, " ").replace(/\s+/g, " ");
  for (const [pattern, category, subcategory] of RULES) {
    if (pattern.test(t)) return { category, subcategory };
  }
  return null;
}

/** Classify a scheme from its AMFI section header (preferred) or its name. */
export function classifyFund(name: string, amfiHeader?: string | null): FundClass {
  if (amfiHeader) {
    const official = fromHeader(amfiHeader, name);
    if (official) return official;
  }
  const byName = matchRules(name);
  if (byName) return byName;
  if (amfiHeader && /income|debt|gilt/i.test(amfiHeader)) return { category: "Debt", subcategory: "Debt" };
  return { category: "Equity", subcategory: "Diversified" };
}
