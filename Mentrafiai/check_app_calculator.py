"""Compare the app's SIP calculator (frontend/utils/sipMath.ts) with the Python math engine.

    python check_app_calculator.py

Runs the TypeScript code itself with Node (no copy of the formulas), over a grid of
inputs for all four modes, and also checks a few published reference values.
"""
import itertools
import json
import subprocess
from pathlib import Path

from inference import financial_calculator as fc

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
RUNNER = """
import { computeSip } from "./utils/sipMath.ts";
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(cases.map((c) => computeSip(c))));
"""

amounts, rates, years_list = [500, 5000, 12345, 100000], [0, 1, 6, 8, 12, 12.5, 15, 30], [1, 3, 10, 15, 25, 40]
cases = []
for mode in ("Regular", "Step-Up", "Lumpsum", "Goal Target"):
    for amount, rate, years in itertools.product(amounts, rates, years_list):
        cases.append({"mode": mode, "monthlyInvestment": amount, "lumpsumInvestment": amount * 20,
                      "targetAmount": amount * 200, "expectedReturn": rate, "timePeriod": years,
                      "stepUpRate": 10 if mode != "Step-Up" else (5 + years % 3 * 5)})

runner = FRONTEND / "sipMath.runner.cts"
runner.write_text(RUNNER.replace('import { computeSip } from "./utils/sipMath.ts";',
                                 'const { computeSip } = require("./utils/sipMath.ts");'), encoding="utf-8")
try:
    out = subprocess.run(["node", str(runner)], input=json.dumps(cases), capture_output=True, text=True,
                         cwd=FRONTEND, check=True).stdout
finally:
    runner.unlink(missing_ok=True)
results = json.loads(out)


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


failures = 0
for case, got in zip(cases, results):
    mode, rate, years = case["mode"], case["expectedReturn"], case["timePeriod"]
    if mode == "Regular":
        exp = fc.calculate_sip(case["monthlyInvestment"], years, rate)
        pairs = [("totalValue", exp["corpus"]), ("investedAmount", exp["invested"]),
                 ("realPurchasingPower", exp["real_corpus"])]
    elif mode == "Step-Up":
        exp = fc.calculate_step_up_sip(case["monthlyInvestment"], years, rate, case["stepUpRate"])
        pairs = [("totalValue", exp["corpus"]), ("investedAmount", exp["invested"]),
                 ("lastInstallment", exp["final_sip"]), ("realPurchasingPower", exp["real_corpus"])]
    elif mode == "Lumpsum":
        exp = fc.calculate_lumpsum(case["lumpsumInvestment"], years, rate)
        pairs = [("totalValue", exp["corpus"]), ("investedAmount", exp["invested"]),
                 ("realPurchasingPower", exp["real_corpus"])]
    else:
        exp = fc.calculate_goal_sip(case["targetAmount"], years, rate)
        pairs = [("requiredSIP", exp["required_sip"]), ("investedAmount", exp["total_invested"]),
                 ("totalValue", exp["projected_corpus"])]
    for key, want in pairs:
        if not close(got[key], want):
            failures += 1
            if failures <= 10:
                print(f"MISMATCH {mode} {case}: {key} app={got[key]} engine={want}")
    if mode == "Step-Up":
        pairs = [("fixedSipValue", fc.calculate_sip(case["monthlyInvestment"], years, rate)["corpus"])]
    elif mode == "Goal Target":
        pairs = [("goalStepUpStart", fc.calculate_goal_step_up_sip(case["targetAmount"], years, rate, 10)["starting_sip"])]
    else:
        pairs = []
    last = got["yearly"][-1] if got["yearly"] else None
    pairs += [("yearly[-1].value", got["totalValue"]), ("yearly[-1].invested", got["investedAmount"])]
    for key, want in pairs:
        value = (last["value"] if key.endswith("value") else last["invested"]) if key.startswith("yearly") else got[key]
        if last is None or not close(value, want):
            failures += 1
            if failures <= 10:
                print(f"MISMATCH {mode} {case}: {key} app={value} engine={want}")
    if last is not None and (len(got["yearly"]) != -(-years // 1) or abs(last["year"] - years) > 1e-9):
        failures += 1
        print(f"YEARLY SERIES {case}: {len(got['yearly'])} points, last year {last['year']}")
    if mode == "Goal Target" and got["totalValue"] + 1e-6 < case["targetAmount"]:
        failures += 1
        print(f"GOAL NOT REACHED {case}: {got['totalValue']}")
    if mode == "Regular":
        want_years = [y for y in (5, 10, 15, 20, 25, 30) if y <= years]
        if [m["year"] for m in got["milestones"]] != want_years:
            failures += 1
            print(f"MILESTONES {case}: {got['milestones']}")

print(f"compared {len(cases)} input combinations across 4 modes: {failures} mismatches")

# Published reference: Rs 10,000/month, 10 years, 12% (start-of-month) = Rs 23,23,391
ref = json.loads(subprocess.run(
    ["node", "-e", "const {computeSip}=require('./utils/sipMath.ts');"
     "console.log(JSON.stringify([computeSip({mode:'Regular',monthlyInvestment:10000,lumpsumInvestment:0,"
     "targetAmount:0,expectedReturn:12,timePeriod:10,stepUpRate:0}),computeSip({mode:'Goal Target',"
     "monthlyInvestment:0,lumpsumInvestment:0,targetAmount:10000000,expectedReturn:12,timePeriod:15,stepUpRate:0})]))"],
    capture_output=True, text=True, cwd=FRONTEND, check=True).stdout)
print(f"reference 10,000 x 10y @12%: app Rs {ref[0]['totalValue']:,.0f} (expected 23,23,391)")
print(f"reference 1 crore in 15y @12%: app Rs {ref[1]['requiredSIP']:,.0f}/month (engine 19,819)")
