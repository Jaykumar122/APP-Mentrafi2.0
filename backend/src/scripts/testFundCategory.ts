// Checks for services/fundCategory.ts:  npx tsx src/scripts/testFundCategory.ts
// 1. Hand-picked names must get the right SEBI category.
// 2. Every scheme in today's AMFI feed: name-only classification vs. the official header.
import { classifyFund } from "../services/fundCategory";

const CASES: [string, string | null, string, string][] = [
  ["HSBC Multi Cap Fund - Direct Plan - Growth", null, "Equity", "Multi Cap"],
  ["ITI Multi Cap Fund - Direct Plan - Growth Option", null, "Equity", "Multi Cap"],
  ["Parag Parikh Flexi Cap Fund - Direct Plan - Growth", null, "Equity", "Flexi Cap"],
  ["HDFC Large & Mid Cap Fund - Direct Plan - Growth Option", null, "Equity", "Large & Mid Cap"],
  ["Motilal Oswal Large and Midcap Fund - Direct Plan - Growth", null, "Equity", "Large & Mid Cap"],
  ["Sundaram Mid Cap Fund - Direct Plan - GROWTH", null, "Equity", "Mid Cap"],
  ["quant Large Cap Fund - Direct Plan - Growth Option", null, "Equity", "Large Cap"],
  ["Kotak Flexicap Fund - Monthly Payout of Income Distribution cum capital withdrawal option", null, "Equity", "Flexi Cap"],
  ["Motilal Oswal Nifty Midcap 150 Index Fund - Direct Plan", null, "Equity", "Index Fund"],
  ["SBI Long Term Equity Fund - Direct Plan - Growth", null, "ELSS", "Tax Saving"],
  ["Axis ELSS Tax Saver Fund - Direct Growth", null, "ELSS", "Tax Saving"],
  ["ICICI Prudential Equity Arbitrage Fund - Direct Plan - Growth", null, "Hybrid", "Arbitrage"],
  ["Kotak Conservative Hybrid Fund - Direct Plan - Growth", null, "Hybrid", "Conservative Hybrid"],
  ["HDFC Balanced Advantage Fund - Direct Plan - Growth", null, "Hybrid", "Balanced Advantage"],
  ["SBI Liquid Fund - Direct Plan - Growth", null, "Debt", "Liquid"],
  ["AXIS Corporate Bond Fund - Direct Plan - Growth Option", null, "Debt", "Corporate Bond"],
  ["SBI Gold Fund - Direct Plan - Growth", null, "Gold", "Gold"],
  ["Kotak Silver ETF Fund of Fund - Direct Plan - Growth", null, "Gold", "Silver"],
  ["Motilal Oswal Nasdaq 100 FOF - Direct Plan", null, "Equity", "International"],
  ["ICICI Prudential Value Fund - Direct Plan - Growth", null, "Equity", "Value"],
  ["Invesco India Contra Fund - Direct Plan - Growth", null, "Equity", "Contra"],
  ["Nippon India Pharma Fund - Direct Plan - Growth", null, "Equity", "Sectoral / Thematic"],
  ["Mirae Asset NYSE FANG+ ETF", null, "Equity", "International"],
  ["Reliance Dual Advantage Fixed Tenure Fund - Plan A", null, "Debt", "Fixed Maturity / Interval"],
  ["Some Fund", "Open Ended Schemes(Equity Scheme - Multi Cap Fund)", "Equity", "Multi Cap"],
  ["Some Fund", "Open Ended Schemes(Equity Schemes - Large & Mid Cap Fund)", "Equity", "Large & Mid Cap"],
  ["Some Fund", "Open Ended Schemes(Equity Scheme - ELSS)", "ELSS", "Tax Saving"],
  ["Some Fund", "Open Ended Schemes(Hybrid Scheme - Arbitrage Fund)", "Hybrid", "Arbitrage"],
  ["Some Fund", "Open Ended Schemes(Debt Scheme - Medium to Long Duration Fund)", "Debt", "Medium to Long Duration"],
  ["Some Fund", "Open Ended Schemes(Income/Debt Oriented Schemes - Floating Interest Rates Fund)", "Debt", "Floater"],
  ["Some Fund", "Open Ended Schemes(Exchange Traded Funds (ETFs) - Silver ETF)", "Gold", "Silver"],
  ["Some Fund", "Open Ended Schemes(Index Funds - Debt Funds)", "Debt", "Index Fund"],
  ["Some Fund", "Open Ended Schemes(Equity Schemes - Thematic Fund)", "Equity", "Sectoral / Thematic"],
  ["Nippon India Gold Savings Fund", "Open Ended Schemes(Fund of Funds Scheme (Domestic) - Fund of Funds Scheme (Domestic))", "Gold", "Gold"],
];

async function main() {
  let failed = 0;
  for (const [name, header, category, subcategory] of CASES) {
    const got = classifyFund(name, header);
    if (got.category !== category || got.subcategory !== subcategory) {
      failed++;
      console.log(`FAIL ${name} | ${header ?? "-"} -> ${got.category}/${got.subcategory}, want ${category}/${subcategory}`);
    }
  }
  console.log(`hand-picked cases: ${CASES.length - failed}/${CASES.length} pass`);

  const text = await (await fetch("https://www.amfiindia.com/spages/NAVAll.txt")).text();
  let header: string | null = null;
  let total = 0, agree = 0, unknownHeader = 0;
  const confusions = new Map<string, number>();
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (/^(Open Ended|Close Ended|Interval Fund)/.test(line)) { header = line; continue; }
    const parts = line.split(";");
    if (parts.length < 8 || !/^\d+$/.test(parts[0]) || !header) continue;
    const official = classifyFund(parts[3], header);
    const byName = classifyFund(parts[3]);
    if (official.subcategory === "Diversified") unknownHeader++;
    total++;
    if (official.category === byName.category && official.subcategory === byName.subcategory) agree++;
    else {
      const key = `${official.category}/${official.subcategory} <- name says ${byName.category}/${byName.subcategory}`;
      confusions.set(key, (confusions.get(key) ?? 0) + 1);
    }
  }
  console.log(`AMFI feed: ${total} schemes; name-only agrees with official header for ${agree} (${(100 * agree / total).toFixed(1)}%); "Diversified" after header: ${unknownHeader}`);
  [...confusions.entries()].sort((a, b) => b[1] - a[1]).slice(0, 15).forEach(([k, v]) => console.log(`  ${v}  ${k}`));
  process.exit(failed ? 1 : 0);
}

main();
