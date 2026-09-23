import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import { ArrowLeft, TrendingUp, Sparkles, Zap, ShieldAlert, Info, Target, Coins } from "lucide-react-native";
import { useMemo, useState } from "react";
import { Dimensions, Text, TouchableOpacity, View, ScrollView, TextInput } from "react-native";
import Svg, { Circle, Defs, LinearGradient as SvgGrad, Path, Stop } from "react-native-svg";
import { AuthBackground, C, depthShadow, GlowBackdrop } from "../(auth)/login";

const { width } = Dimensions.get("window");
const GREEN = "#4ade80";

type Mode = "Regular" | "Step-Up" | "Lumpsum" | "Goal Target";
const MODES: Mode[] = ["Regular", "Step-Up", "Lumpsum", "Goal Target"];

function GrowthIcon() {
  return (
    <Svg width="100" height="100" viewBox="0 0 150 150">
      <Defs>
        <SvgGrad id="barTop1" x1="0" y1="0" x2="1" y2="1">
          <Stop offset="0%" stopColor="#c9b3ff" />
          <Stop offset="100%" stopColor={C.violet} />
        </SvgGrad>
        <SvgGrad id="barSide1" x1="0" y1="0" x2="0" y2="1">
          <Stop offset="0%" stopColor="#5a34a8" />
          <Stop offset="100%" stopColor="#3c2170" />
        </SvgGrad>
        <SvgGrad id="barTop2" x1="0" y1="0" x2="1" y2="1">
          <Stop offset="0%" stopColor="#ff9dbc" />
          <Stop offset="100%" stopColor={C.pink} />
        </SvgGrad>
        <SvgGrad id="barSide2" x1="0" y1="0" x2="0" y2="1">
          <Stop offset="0%" stopColor="#b23360" />
          <Stop offset="100%" stopColor="#7a1f42" />
        </SvgGrad>
        <SvgGrad id="barTop3" x1="0" y1="0" x2="1" y2="1">
          <Stop offset="0%" stopColor="#ffe9a8" />
          <Stop offset="100%" stopColor="#ffcf6e" />
        </SvgGrad>
        <SvgGrad id="barSide3" x1="0" y1="0" x2="0" y2="1">
          <Stop offset="0%" stopColor="#c9932e" />
          <Stop offset="100%" stopColor="#8a5e18" />
        </SvgGrad>
      </Defs>
      <Path d="M34,108 L34,86 L54,74 L54,96 Z" fill="url(#barTop1)" />
      <Path d="M54,96 L54,74 L64,80 L64,102 Z" fill="url(#barSide1)" />
      <Path d="M60,108 L60,68 L80,56 L80,96 Z" fill="url(#barTop2)" />
      <Path d="M80,96 L80,56 L90,62 L90,102 Z" fill="url(#barSide2)" />
      <Path d="M86,108 L86,46 L106,34 L106,96 Z" fill="url(#barTop3)" />
      <Path d="M106,96 L106,34 L116,40 L116,102 Z" fill="url(#barSide3)" />
      <Path d="M28 108 h94 v6 h-94 Z" fill="#000" opacity="0.25" />
      <Path d="M40 90 L66 72 L92 58 L112 42" stroke="#fff" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round" opacity="0.85" />
      <Path d="M104 42 L112 42 L112 50" stroke="#fff" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round" opacity="0.85" />
      <Path d="M86 48 L104 36 L110 40 L92 52 Z" fill="#fff" opacity="0.3" />
    </Svg>
  );
}

function DonutChart({
  percentage,
  totalValue,
}: {
  percentage: number;
  totalValue: number;
}) {
  const size = 160;
  const strokeWidth = 14;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const validPercentage = isNaN(percentage) ? 0 : Math.min(percentage, 100);
  const offset = circumference - (validPercentage / 100) * circumference;

  const formatValue = (val: number) => {
    if (val >= 10000000) return `₹${(val / 10000000).toFixed(2)}Cr`;
    if (val >= 100000) return `₹${(val / 100000).toFixed(2)}L`;
    return `₹${val.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
  };

  return (
    <View style={{ alignItems: "center", justifyContent: "center", width: size, height: size }}>
      <Svg
        width={size}
        height={size}
        style={{ position: "absolute", transform: [{ rotate: "-90deg" }] }}
      >
        <Defs>
          <SvgGrad id="donutGrad" x1="0" y1="0" x2="1" y2="1">
            <Stop offset="0%" stopColor={C.pink} />
            <Stop offset="100%" stopColor={C.violet} />
          </SvgGrad>
        </Defs>
        <Circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke={C.inputBorder} strokeWidth={strokeWidth} />
        <Circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="url(#donutGrad)"
          strokeWidth={strokeWidth}
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={offset}
          strokeLinecap="round"
        />
      </Svg>
      <View style={{ alignItems: "center" }}>
        <Text style={{ color: C.textFaint, fontSize: 10, letterSpacing: 1 }}>MATURITY</Text>
        <Text style={{ color: "#fff", fontSize: 16, fontWeight: "800", marginTop: 4 }}>
          {formatValue(totalValue)}
        </Text>
      </View>
    </View>
  );
}

function SliderInput({
  value,
  min,
  max,
  step,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (val: number) => void;
}) {
  const sliderWidth = width - 80;
  const percentage = ((value - min) / (max - min)) * 100;
  const thumbPos = (percentage / 100) * sliderWidth;

  const handleTouch = (e: any) => {
    const x = e.nativeEvent.locationX;
    const ratio = Math.max(0, Math.min(1, x / sliderWidth));
    const raw = min + ratio * (max - min);
    const stepped = Math.round(raw / step) * step;
    onChange(Math.max(min, Math.min(max, stepped)));
  };

  return (
    <View style={{ height: 36, justifyContent: "center" }} onTouchStart={handleTouch} onTouchMove={handleTouch}>
      <View style={{ height: 5, backgroundColor: C.inputBorder, borderRadius: 3, width: sliderWidth }}>
        <LinearGradient
          colors={[C.pink, C.violet]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
          style={{ height: 5, borderRadius: 3, width: `${Math.max(0, Math.min(100, percentage))}%` }}
        />
      </View>
      <View
        style={{
          position: "absolute",
          left: Math.max(0, Math.min(sliderWidth, thumbPos)) - 11,
          width: 22,
          height: 22,
          borderRadius: 11,
          backgroundColor: "white",
          borderWidth: 3,
          borderColor: C.pink,
          ...depthShadow("sm"),
        }}
      />
    </View>
  );
}

export default function SIPCalculator() {
  const router = useRouter();
  
  const [activeMode, setActiveMode] = useState<Mode>("Regular");

  const [monthlyInvestment, setMonthlyInvestment] = useState(5000);
  const [lumpsumInvestment, setLumpsumInvestment] = useState(100000);
  const [targetAmount, setTargetAmount] = useState(10000000);
  
  const [expectedReturn, setExpectedReturn] = useState(12);
  const [timePeriod, setTimePeriod] = useState(15);
  const [stepUpRate, setStepUpRate] = useState(10);

  const calc = useMemo(() => {
    let investedAmount = 0;
    let estimatedReturns = 0;
    let totalValue = 0;
    let percentage = 0;
    
    const r = expectedReturn / 100 / 12;
    const n = timePeriod * 12;
    const annualR = expectedReturn / 100;

    if (activeMode === "Regular") {
      const P = monthlyInvestment;
      if (P > 0 && n > 0) {
        totalValue = P * ((Math.pow(1 + r, n) - 1) / r);
        investedAmount = P * n;
        estimatedReturns = totalValue - investedAmount;
        percentage = (investedAmount / totalValue) * 100;
      }
    } else if (activeMode === "Step-Up") {
      const P = monthlyInvestment;
      if (P > 0 && n > 0) {
        let curP = P;
        for (let y = 0; y < timePeriod; y++) {
          for (let m = 0; m < 12; m++) {
            investedAmount += curP;
            totalValue = totalValue * (1 + r) + curP;
          }
          curP *= (1 + stepUpRate / 100);
        }
        estimatedReturns = totalValue - investedAmount;
        percentage = (investedAmount / totalValue) * 100;
      }
    } else if (activeMode === "Lumpsum") {
      const P = lumpsumInvestment;
      if (P > 0 && timePeriod > 0) {
        totalValue = P * Math.pow(1 + annualR, timePeriod);
        investedAmount = P;
        estimatedReturns = totalValue - investedAmount;
        percentage = (investedAmount / totalValue) * 100;
      }
    } else if (activeMode === "Goal Target") {
      const FV = targetAmount;
      if (FV > 0 && n > 0 && r > 0) {
        const requiredSIP = FV / ((Math.pow(1 + r, n) - 1) / r);
        totalValue = FV;
        investedAmount = requiredSIP * n;
        estimatedReturns = totalValue - investedAmount;
        percentage = (investedAmount / totalValue) * 100;
      }
    }

    const realPurchasingPower = totalValue / Math.pow(1.06, timePeriod);
    
    // Milestones for timeline
    let m5 = 0, m10 = 0, m15 = 0;
    if (activeMode === "Regular") {
      const p = monthlyInvestment;
      m5 = p * ((Math.pow(1 + r, 5 * 12) - 1) / r);
      m10 = p * ((Math.pow(1 + r, 10 * 12) - 1) / r);
      m15 = p * ((Math.pow(1 + r, 15 * 12) - 1) / r);
    }

    return {
      investedAmount,
      estimatedReturns,
      totalValue,
      percentage,
      realPurchasingPower,
      requiredSIP: activeMode === "Goal Target" ? (investedAmount / (timePeriod * 12)) : monthlyInvestment,
      m5, m10, m15
    };
  }, [activeMode, monthlyInvestment, lumpsumInvestment, targetAmount, expectedReturn, timePeriod, stepUpRate]);

  const formatINR = (val: number) =>
    `₹${val.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;

  const PRESET_RATES = [
    { label: "6% Debt", rate: 6 },
    { label: "10% Hybrid", rate: 10 },
    { label: "12% Nifty 50", rate: 12 },
    { label: "15% Mid Cap", rate: 15 },
  ];
  
  const SIP_CHIPS = [1000, 2500, 5000, 10000];
  const YEARS_CHIPS = [5, 10, 15, 20, 25];
  const STEPUP_CHIPS = [5, 10, 15, 20];

  return (
    <View style={{ flex: 1, backgroundColor: C.bgBottom }}>
      <AuthBackground />

      {/* Header */}
      <LinearGradient
        colors={[C.card, "#050508"]}
        start={{ x: 0.2, y: 0 }}
        end={{ x: 0.8, y: 1 }}
        style={{
          paddingHorizontal: 20,
          paddingTop: 56,
          paddingBottom: 20,
          borderBottomLeftRadius: 32,
          borderBottomRightRadius: 32,
          borderBottomWidth: 1,
          borderColor: C.cardEdge,
        }}
      >
        <View style={{ flexDirection: "row", alignItems: "center", gap: 14 }}>
          <TouchableOpacity
            onPress={() => router.back()}
            style={{
              width: 36,
              height: 36,
              borderRadius: 18,
              backgroundColor: C.input,
              borderWidth: 1,
              borderColor: C.inputBorder,
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <ArrowLeft color="#fff" size={18} />
          </TouchableOpacity>
          <View>
            <Text style={{ color: "#fff", fontSize: 20, fontWeight: "700" }}>SIP & Wealth Calculator</Text>
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: 4 }}>
               <ShieldAlert size={10} color={C.textFaint} />
               <Text style={{ color: C.textFaint, fontSize: 11 }}>SEBI Compliant Ordinary Annuity Math</Text>
            </View>
          </View>
        </View>

        {/* Mode Switcher */}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginTop: 20 }} contentContainerStyle={{ gap: 8 }}>
          {MODES.map((m) => (
            <TouchableOpacity
              key={m}
              onPress={() => setActiveMode(m)}
              style={{
                paddingVertical: 8,
                paddingHorizontal: 16,
                borderRadius: 20,
                backgroundColor: activeMode === m ? "rgba(236,72,153,0.15)" : C.input,
                borderWidth: 1,
                borderColor: activeMode === m ? C.pink : C.inputBorder,
              }}
            >
              <Text style={{ color: activeMode === m ? C.pink : C.textMuted, fontSize: 13, fontWeight: "600" }}>
                {m}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </LinearGradient>

      <ScrollView contentContainerStyle={{ paddingHorizontal: 20, paddingTop: 10, paddingBottom: 100 }} showsVerticalScrollIndicator={false}>
        
        {/* Master Valuation Bento Card */}
        <LinearGradient
          colors={[C.input, "rgba(236, 72, 153, 0.05)"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={{
            borderRadius: 24,
            padding: 20,
            borderWidth: 1,
            borderColor: C.inputBorder,
            marginBottom: 20,
            alignItems: 'center',
          }}
        >
          <View style={{ position: "absolute", top: -10 }}>
            <GlowBackdrop from={C.pink} to={C.violet} />
          </View>
          <Text style={{ color: C.textFaint, fontSize: 12, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
            {activeMode === "Goal Target" ? "Target Achieved With SIP Of" : "Expected Maturity Corpus"}
          </Text>
          <Text style={{ color: "#fff", fontSize: 32, fontWeight: "800", marginBottom: 6 }}>
             {activeMode === "Goal Target" ? formatINR(calc.requiredSIP) + "/mo" : formatINR(calc.totalValue)}
          </Text>
          
          <View style={{ flexDirection: 'row', alignItems: 'center', backgroundColor: 'rgba(74,222,128,0.15)', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12, marginBottom: 20 }}>
            <Sparkles size={12} color={GREEN} />
            <Text style={{ color: GREEN, fontSize: 11, fontWeight: '700', marginLeft: 6 }}>
              {activeMode !== "Goal Target" && calc.investedAmount > 0 ? `🚀 ${(calc.totalValue / calc.investedAmount).toFixed(2)}x Capital Growth` : 'Wealth Generator Active'}
            </Text>
          </View>

          <View style={{ flexDirection: 'row', width: '100%', gap: 12, marginBottom: 20 }}>
            <View style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.2)', padding: 12, borderRadius: 16 }}>
              <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 4 }}>Total Invested</Text>
              <Text style={{ color: "#fff", fontSize: 14, fontWeight: "700" }}>{formatINR(calc.investedAmount)}</Text>
            </View>
            <View style={{ flex: 1, backgroundColor: 'rgba(74,222,128,0.1)', padding: 12, borderRadius: 16 }}>
              <Text style={{ color: GREEN, fontSize: 11, marginBottom: 4 }}>Estimated Gain</Text>
              <Text style={{ color: GREEN, fontSize: 14, fontWeight: "800" }}>+{formatINR(calc.estimatedReturns)}</Text>
            </View>
          </View>

          <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", width: '100%', backgroundColor: 'rgba(255,255,255,0.03)', padding: 12, borderRadius: 12 }}>
            <View>
              <Text style={{ color: C.textMuted, fontSize: 11, fontWeight: "600" }}>Purchasing Power (6% Inflation)</Text>
            </View>
            <Text style={{ color: "#ffcf6e", fontSize: 13, fontWeight: "700" }}>{formatINR(calc.realPurchasingPower)}</Text>
          </View>
        </LinearGradient>

        {/* Visualizer Row */}
        <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24, paddingHorizontal: 10 }}>
           <View style={{ alignItems: "center" }}>
             <DonutChart percentage={calc.percentage} totalValue={calc.totalValue} />
           </View>
           <View style={{ height: 140, justifyContent: 'center' }}>
             <GrowthIcon />
           </View>
        </View>

        {/* Dynamic Inputs Based on Mode */}
        <View style={{ gap: 20, marginBottom: 24 }}>
          {/* Amount Slider */}
          {(activeMode === "Regular" || activeMode === "Step-Up") && (
            <View>
              <View style={{ flexDirection: "row", justifyContent: "space-between", marginBottom: 8 }}>
                <Text style={{ color: "#fff", fontSize: 13, fontWeight: "500" }}>Monthly Investment</Text>
                <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }}>{formatINR(monthlyInvestment)}</Text>
              </View>
              <SliderInput value={monthlyInvestment} min={500} max={100000} step={500} onChange={setMonthlyInvestment} />
              <View style={{ flexDirection: 'row', gap: 6, marginTop: 12 }}>
                {SIP_CHIPS.map(val => (
                   <TouchableOpacity key={val} onPress={() => setMonthlyInvestment(prev => Math.min(100000, prev + val))} style={{ backgroundColor: C.input, paddingVertical: 4, paddingHorizontal: 8, borderRadius: 8, borderWidth: 1, borderColor: C.inputBorder }}>
                     <Text style={{ color: C.textMuted, fontSize: 10 }}>+{formatINR(val)}</Text>
                   </TouchableOpacity>
                ))}
              </View>
            </View>
          )}

          {activeMode === "Lumpsum" && (
            <View>
              <View style={{ flexDirection: "row", justifyContent: "space-between", marginBottom: 8 }}>
                <Text style={{ color: "#fff", fontSize: 13, fontWeight: "500" }}>One-Time Investment</Text>
                <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }}>{formatINR(lumpsumInvestment)}</Text>
              </View>
              <SliderInput value={lumpsumInvestment} min={5000} max={10000000} step={5000} onChange={setLumpsumInvestment} />
            </View>
          )}

          {activeMode === "Goal Target" && (
            <View>
              <View style={{ flexDirection: "row", justifyContent: "space-between", marginBottom: 8 }}>
                <Text style={{ color: "#fff", fontSize: 13, fontWeight: "500" }}>Target Corpus</Text>
                <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }}>{formatINR(targetAmount)}</Text>
              </View>
              <SliderInput value={targetAmount} min={100000} max={100000000} step={100000} onChange={setTargetAmount} />
            </View>
          )}

          {/* Expected Return */}
          <View>
            <View style={{ flexDirection: "row", justifyContent: "space-between", marginBottom: 8 }}>
              <Text style={{ color: "#fff", fontSize: 13, fontWeight: "500" }}>Expected Return Rate</Text>
              <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }}>{expectedReturn}%</Text>
            </View>
            <SliderInput value={expectedReturn} min={1} max={30} step={0.5} onChange={setExpectedReturn} />
            <View style={{ flexDirection: 'row', gap: 6, marginTop: 12 }}>
              {PRESET_RATES.map((p) => (
                <TouchableOpacity key={p.rate} onPress={() => setExpectedReturn(p.rate)} style={{ backgroundColor: expectedReturn === p.rate ? 'rgba(236,72,153,0.2)' : C.input, paddingVertical: 4, paddingHorizontal: 8, borderRadius: 8, borderWidth: 1, borderColor: expectedReturn === p.rate ? C.pink : C.inputBorder }}>
                  <Text style={{ color: expectedReturn === p.rate ? '#fff' : C.textMuted, fontSize: 10 }}>{p.label}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Time Period */}
          <View>
            <View style={{ flexDirection: "row", justifyContent: "space-between", marginBottom: 8 }}>
              <Text style={{ color: "#fff", fontSize: 13, fontWeight: "500" }}>Time Period</Text>
              <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }}>{timePeriod} Years</Text>
            </View>
            <SliderInput value={timePeriod} min={1} max={40} step={1} onChange={setTimePeriod} />
            <View style={{ flexDirection: 'row', gap: 6, marginTop: 12 }}>
              {YEARS_CHIPS.map((y) => (
                <TouchableOpacity key={y} onPress={() => setTimePeriod(y)} style={{ backgroundColor: timePeriod === y ? 'rgba(236,72,153,0.2)' : C.input, paddingVertical: 4, paddingHorizontal: 8, borderRadius: 8, borderWidth: 1, borderColor: timePeriod === y ? C.pink : C.inputBorder }}>
                  <Text style={{ color: timePeriod === y ? '#fff' : C.textMuted, fontSize: 10 }}>{y}Y</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Step-Up Rate */}
          {activeMode === "Step-Up" && (
            <View>
              <View style={{ flexDirection: "row", justifyContent: "space-between", marginBottom: 8 }}>
                <Text style={{ color: "#fff", fontSize: 13, fontWeight: "500" }}>Annual Step-Up Increment</Text>
                <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }}>{stepUpRate}%</Text>
              </View>
              <SliderInput value={stepUpRate} min={1} max={50} step={1} onChange={setStepUpRate} />
              <View style={{ flexDirection: 'row', gap: 6, marginTop: 12 }}>
                {STEPUP_CHIPS.map((s) => (
                  <TouchableOpacity key={s} onPress={() => setStepUpRate(s)} style={{ backgroundColor: stepUpRate === s ? 'rgba(236,72,153,0.2)' : C.input, paddingVertical: 4, paddingHorizontal: 8, borderRadius: 8, borderWidth: 1, borderColor: stepUpRate === s ? C.pink : C.inputBorder }}>
                    <Text style={{ color: stepUpRate === s ? '#fff' : C.textMuted, fontSize: 10 }}>{s}%</Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
          )}
        </View>

        {/* Compounding Milestones (Only relevant for Regular SIP if time > 15 to show all, but we'll adapt) */}
        {activeMode === "Regular" && timePeriod >= 15 && (
          <View style={{ backgroundColor: C.input, borderRadius: 16, padding: 16, marginBottom: 24, borderWidth: 1, borderColor: C.inputBorder }}>
            <Text style={{ color: '#fff', fontSize: 14, fontWeight: '700', marginBottom: 12 }}>Compounding Timeline</Text>
            <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
               <View>
                 <Text style={{ color: C.textMuted, fontSize: 11 }}>Year 5</Text>
                 <Text style={{ color: '#fff', fontSize: 13, fontWeight: '600' }}>{formatINR(calc.m5)}</Text>
               </View>
               <View>
                 <Text style={{ color: C.textMuted, fontSize: 11 }}>Year 10</Text>
                 <Text style={{ color: '#fff', fontSize: 13, fontWeight: '600' }}>{formatINR(calc.m10)}</Text>
               </View>
               <View>
                 <Text style={{ color: C.textMuted, fontSize: 11 }}>Year 15</Text>
                 <Text style={{ color: '#fff', fontSize: 13, fontWeight: '600' }}>{formatINR(calc.m15)}</Text>
               </View>
            </View>
          </View>
        )}

        {/* CTAs */}
        <View style={{ gap: 12, marginBottom: 20 }}>
          <TouchableOpacity
            onPress={() => router.push({ pathname: "/(tabs)/invest", params: { amount: activeMode === "Goal Target" ? calc.requiredSIP : monthlyInvestment } })}
            style={{
              backgroundColor: "#fff",
              borderRadius: 16,
              paddingVertical: 16,
              alignItems: 'center',
              flexDirection: 'row',
              justifyContent: 'center',
              gap: 8,
            }}
          >
            <Coins size={18} color="#000" />
            <Text style={{ color: "#000", fontSize: 15, fontWeight: "700" }}>
              Start This {activeMode === "Lumpsum" ? "Investment" : "SIP"} Now
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
             onPress={() => router.push("/(tabs)/ai-advisor")}
             style={{
               backgroundColor: "rgba(168, 85, 247, 0.15)",
               borderRadius: 16,
               paddingVertical: 16,
               alignItems: 'center',
               borderWidth: 1,
               borderColor: "rgba(168, 85, 247, 0.4)",
               flexDirection: 'row',
               justifyContent: 'center',
               gap: 8,
             }}
          >
            <Sparkles size={18} color={C.violet} />
            <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700" }}>Consult MentraFi AI</Text>
          </TouchableOpacity>
        </View>

        {/* Tax & SEBI */}
        <View style={{ backgroundColor: "rgba(255,255,255,0.03)", borderWidth: 1, borderColor: C.cardEdge, borderRadius: 16, padding: 12, marginBottom: 16 }}>
          <View style={{ flexDirection: "row", alignItems: "center", gap: 6, marginBottom: 4 }}>
            <Info size={13} color={C.textMuted} />
            <Text style={{ color: C.textMuted, fontSize: 11, fontWeight: "700" }}>Taxation Advisory (Finance Act 2024)</Text>
          </View>
          <Text style={{ color: C.textFaint, fontSize: 10, lineHeight: 15 }}>
            • Equity Funds: LTCG (over 1 yr) above ₹1.25 Lakh/yr taxed at 12.5%; STCG at 20%.{"\n"}
            • Debt & Conservative Funds: Taxed at marginal income slab rate.{"\n"}
            • Tax is paid at redemption and is not deducted from gross corpus.
          </Text>
        </View>

        <Text style={{ color: C.textFaint, fontSize: 10, textAlign: "center", fontStyle: "italic", lineHeight: 15, paddingHorizontal: 10 }}>
          Returns are illustrative estimates, not guaranteed. Actual returns depend on market performance. All mutual fund investments are subject to market risks.
        </Text>
      </ScrollView>
    </View>
  );
}