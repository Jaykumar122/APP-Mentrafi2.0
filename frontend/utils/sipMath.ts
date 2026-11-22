// SIP calculator math, shared by the SIP Calculator screen and its test.
// Same conventions as the backend engine (Mentrafiai/inference/financial_calculator.py):
// monthly rate = annual / 12, SIP installment at the START of each month (annuity due),
// step-up applied after every 12 installments, lump sum compounded yearly,
// "today's money" = value / 1.06^years.

export type SipMode = "Regular" | "Step-Up" | "Lumpsum" | "Goal Target";

export interface SipInputs {
  mode: SipMode;
  monthlyInvestment: number;
  lumpsumInvestment: number;
  targetAmount: number;
  expectedReturn: number; // % a year
  timePeriod: number; // years
  stepUpRate: number; // % a year
}

export interface SipResult {
  investedAmount: number;
  estimatedReturns: number;
  totalValue: number;
  investedSharePct: number; // invested as a % of the final value (donut chart)
  realPurchasingPower: number;
  requiredSIP: number; // Goal Target: rupees a month, rounded up; otherwise the monthly input
  lastInstallment: number; // Step-Up: the final month's installment
  milestones: { year: number; value: number }[]; // Regular: value after 5/10/15... years
  yearly: YearPoint[]; // end of each year (last point may be a part year)
  fixedSipValue: number; // Step-Up: same starting SIP without step-up, for comparison
  goalStepUpStart: number; // Goal Target: starting SIP if raised 10% every year
}

export interface YearPoint {
  year: number;
  invested: number;
  value: number;
}

export const INFLATION_PCT = 6;

export function sipFactor(months: number, monthlyRate: number): number {
  if (months <= 0) return 0;
  if (monthlyRate === 0) return months;
  return ((Math.pow(1 + monthlyRate, months) - 1) / monthlyRate) * (1 + monthlyRate);
}

const finite = (value: number) => (Number.isFinite(value) ? value : 0);

export function computeSip(input: SipInputs): SipResult {
  const r = input.expectedReturn / 100 / 12;
  const years = Math.max(0, input.timePeriod);
  const n = Math.round(years * 12);
  let investedAmount = 0;
  let totalValue = 0;
  let requiredSIP = input.monthlyInvestment;
  let lastInstallment = input.monthlyInvestment;
  const milestones: { year: number; value: number }[] = [];
  const yearly: YearPoint[] = [];
  let fixedSipValue = 0;
  let goalStepUpStart = 0;
  // Month-by-month path for any monthly SIP (start-of-month installments).
  const sipPath = (start: number, stepPct: number) => {
    const points: YearPoint[] = [];
    let installment = start, balance = 0, paid = 0;
    for (let month = 0; month < n; month++) {
      if (month > 0 && month % 12 === 0) installment *= 1 + stepPct / 100;
      paid += installment;
      balance = (balance + installment) * (1 + r);
      if ((month + 1) % 12 === 0 || month === n - 1) points.push({ year: (month + 1) / 12, invested: paid, value: balance });
    }
    return { points, balance, paid, installment };
  };

  if (input.mode === "Regular" && input.monthlyInvestment > 0 && n > 0) {
    totalValue = input.monthlyInvestment * sipFactor(n, r);
    investedAmount = input.monthlyInvestment * n;
    for (const year of [5, 10, 15, 20, 25, 30]) {
      if (year <= years) milestones.push({ year, value: input.monthlyInvestment * sipFactor(year * 12, r) });
    }
    yearly.push(...sipPath(input.monthlyInvestment, 0).points);
  } else if (input.mode === "Step-Up" && input.monthlyInvestment > 0 && n > 0) {
    let installment = input.monthlyInvestment;
    for (let month = 0; month < n; month++) {
      if (month > 0 && month % 12 === 0) installment *= 1 + input.stepUpRate / 100;
      investedAmount += installment;
      totalValue = (totalValue + installment) * (1 + r);
    }
    lastInstallment = installment;
    yearly.push(...sipPath(input.monthlyInvestment, input.stepUpRate).points);
    fixedSipValue = input.monthlyInvestment * sipFactor(n, r);
  } else if (input.mode === "Lumpsum" && input.lumpsumInvestment > 0 && years > 0) {
    investedAmount = input.lumpsumInvestment;
    totalValue = input.lumpsumInvestment * Math.pow(1 + input.expectedReturn / 100, years);
    for (let y = 1; y <= Math.ceil(years); y++) {
      const t = Math.min(y, years);
      yearly.push({ year: t, invested: input.lumpsumInvestment,
                    value: input.lumpsumInvestment * Math.pow(1 + input.expectedReturn / 100, t) });
    }
  } else if (input.mode === "Goal Target" && input.targetAmount > 0 && n > 0) {
    requiredSIP = Math.ceil(input.targetAmount / sipFactor(n, r) - 1e-9);
    investedAmount = requiredSIP * n;
    totalValue = requiredSIP * sipFactor(n, r);
    yearly.push(...sipPath(requiredSIP, 0).points);
    // The step-up corpus is linear in the starting SIP: solve with one ₹1 path.
    const unit = sipPath(1, 10).balance;
    goalStepUpStart = unit > 0 ? Math.ceil(input.targetAmount / unit - 1e-9) : 0;
  }

  return {
    investedAmount: finite(investedAmount),
    estimatedReturns: finite(totalValue - investedAmount),
    totalValue: finite(totalValue),
    investedSharePct: totalValue > 0 ? finite((investedAmount / totalValue) * 100) : 0,
    realPurchasingPower: finite(totalValue / Math.pow(1 + INFLATION_PCT / 100, years)),
    requiredSIP: finite(requiredSIP),
    lastInstallment: finite(lastInstallment),
    milestones,
    yearly,
    fixedSipValue: finite(fixedSipValue),
    goalStepUpStart: finite(goalStepUpStart),
  };
}
