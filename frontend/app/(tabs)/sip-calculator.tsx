import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import * as Haptics from "expo-haptics";
import {
  ArrowLeft, CalendarClock, ChevronDown, ChevronUp, Coins, Info, Layers, Percent, Sparkles,
  Target, TrendingUp, Wallet,
} from "lucide-react-native";
import { useMemo, useRef, useState } from "react";
import {
  AccessibilityActionEvent, GestureResponderEvent, LayoutChangeEvent, ScrollView, Text, TextInput,
  TouchableOpacity, View,
} from "react-native";
import Svg, { Defs, G, Line, LinearGradient as SvgGrad, Rect, Stop } from "react-native-svg";
import { AuthBackground, C, depthShadow } from "../(auth)/login";
import { computeSip, INFLATION_PCT, SipMode, YearPoint } from "../../utils/sipMath";

const GAIN = "#34d399";
const INVESTED = "#8b5cf6";
const GOLD = "#fbbf24";

const MODES: { key: SipMode; label: string; Icon: typeof Wallet }[] = [
  { key: "Regular", label: "SIP", Icon: Wallet },
  { key: "Step-Up", label: "Step-Up", Icon: TrendingUp },
  { key: "Lumpsum", label: "Lump Sum", Icon: Coins },
  { key: "Goal Target", label: "Goal", Icon: Target },
];

// ─── formatting ──────────────────────────────────────────────────────────────
const inr = (v: number) => `₹${Math.round(v).toLocaleString("en-IN")}`;
function compact(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e7) return `₹${(v / 1e7).toFixed(a >= 1e9 ? 0 : 2)} Cr`;
  if (a >= 1e5) return `₹${(v / 1e5).toFixed(2)} L`;
  if (a >= 1e3) return `₹${(v / 1e3).toFixed(a >= 1e4 ? 0 : 1)}K`;
  return `₹${Math.round(v)}`;
}
const yearsLabel = (y: number) => (Number.isInteger(y) ? `${y} ${y === 1 ? "year" : "years"}` : `${Math.round(y * 12)} months`);

function tick() {
  Haptics.selectionAsync().catch(() => undefined);
}

// Touch handling for sliders and the chart: a tap or a sideways drag changes the value,
// a vertical swipe is handed to the ScrollView so scrolling never moves a control.
function useHorizontalPick(onPick: (pageX: number) => void, measure: () => void) {
  const start = useRef({ x: 0, y: 0 });
  const dragging = useRef(false);
  return {
    onStartShouldSetResponder: () => true,
    onMoveShouldSetResponder: () => dragging.current,
    onResponderGrant: (e: GestureResponderEvent) => {
      measure();
      start.current = { x: e.nativeEvent.pageX, y: e.nativeEvent.pageY };
      dragging.current = false;
    },
    onResponderMove: (e: GestureResponderEvent) => {
      const dx = e.nativeEvent.pageX - start.current.x;
      const dy = e.nativeEvent.pageY - start.current.y;
      if (!dragging.current && Math.abs(dx) > 6 && Math.abs(dx) > Math.abs(dy)) dragging.current = true;
      if (dragging.current) onPick(e.nativeEvent.pageX);
    },
    onResponderRelease: (e: GestureResponderEvent) => {
      const moved = Math.hypot(e.nativeEvent.pageX - start.current.x, e.nativeEvent.pageY - start.current.y);
      if (!dragging.current && moved < 10) onPick(e.nativeEvent.pageX);
      dragging.current = false;
    },
    onResponderTerminationRequest: () => !dragging.current,
    onResponderTerminate: () => { dragging.current = false; },
  };
}

// ─── slider ──────────────────────────────────────────────────────────────────
function Slider({ value, min, max, step, onChange, label }: {
  value: number; min: number; max: number; step: number; onChange: (v: number) => void; label: string;
}) {
  const [trackWidth, setTrackWidth] = useState(0);
  const trackRef = useRef<View>(null);
  const left = useRef(0);
  const ratio = Math.max(0, Math.min(1, (value - min) / (max - min)));

  const measure = () => trackRef.current?.measureInWindow((x) => { left.current = x; });
  const fromTouch = (pageX: number) => {
    if (!trackWidth) return;
    const r = Math.max(0, Math.min(1, (pageX - left.current) / trackWidth));
    const stepped = Number((Math.round((min + r * (max - min)) / step) * step).toFixed(4));
    const next = Math.max(min, Math.min(max, stepped));
    if (next !== value) onChange(next);
  };
  const touch = useHorizontalPick(fromTouch, measure);
  const onAction = (e: AccessibilityActionEvent) => {
    const delta = e.nativeEvent.actionName === "increment" ? step : -step;
    onChange(Math.max(min, Math.min(max, Number((value + delta).toFixed(4)))));
  };

  return (
    <View
      ref={trackRef}
      onLayout={(e: LayoutChangeEvent) => { setTrackWidth(e.nativeEvent.layout.width); measure(); }}
      style={{ height: 40, justifyContent: "center" }}
      {...touch}
      accessible
      accessibilityRole="adjustable"
      accessibilityLabel={label}
      accessibilityValue={{ min, max, now: value }}
      accessibilityActions={[{ name: "increment" }, { name: "decrement" }]}
      onAccessibilityAction={onAction}
    >
      <View pointerEvents="none" style={{ height: 6, borderRadius: 3, backgroundColor: "rgba(255,255,255,0.08)" }}>
        <LinearGradient
          colors={[INVESTED, "#06b6d4"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
          style={{ height: 6, borderRadius: 3, width: `${ratio * 100}%` }}
        />
      </View>
      <View
        pointerEvents="none"
        style={{
          position: "absolute", left: ratio * trackWidth - 12, width: 24, height: 24, borderRadius: 12,
          backgroundColor: "#fff", borderWidth: 4, borderColor: INVESTED, ...depthShadow("sm"),
        }}
      />
    </View>
  );
}

// ─── one input: label, editable value, slider, presets ───────────────────────
function InputRow({ icon: Icon, label, value, onChange, min, max, step, hardMax, prefix = "", suffix = "", presets }: {
  icon: typeof Wallet; label: string; value: number; onChange: (v: number) => void;
  min: number; max: number; step: number; hardMax: number; prefix?: string; suffix?: string;
  presets: { label: string; value: number }[];
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const display = prefix === "₹" ? Math.round(value).toLocaleString("en-IN") : String(value);
  const commit = () => {
    if (editing === null) return;
    const parsed = Number(editing.replace(/[^0-9.]/g, ""));
    if (Number.isFinite(parsed) && parsed > 0) onChange(Math.max(min, Math.min(hardMax, parsed)));
    setEditing(null);
  };

  return (
    <View style={{ gap: 6 }}>
      <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
        <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
          <Icon size={15} color={C.textMuted} />
          <Text style={{ color: C.textMuted, fontSize: 13, fontWeight: "600" }}>{label}</Text>
        </View>
        <View style={{
          flexDirection: "row", alignItems: "center", minWidth: 96, paddingHorizontal: 12, height: 38, borderRadius: 12,
          backgroundColor: "rgba(139,92,246,0.12)", borderWidth: 1,
          borderColor: editing !== null ? INVESTED : "rgba(139,92,246,0.35)", justifyContent: "flex-end",
        }}>
          {prefix ? <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700" }}>{prefix}</Text> : null}
          <TextInput
            value={editing ?? display}
            onFocus={() => setEditing(String(value))}
            onChangeText={setEditing}
            onEndEditing={commit}
            onSubmitEditing={commit}
            keyboardType="decimal-pad"
            returnKeyType="done"
            selectTextOnFocus
            accessibilityLabel={`${label} value`}
            style={{ color: "#fff", fontSize: 15, fontWeight: "700", padding: 0, textAlign: "right", minWidth: 24 }}
          />
          {suffix ? <Text style={{ color: C.textMuted, fontSize: 14, fontWeight: "600", marginLeft: 3 }}>{suffix}</Text> : null}
        </View>
      </View>
      <Slider value={Math.min(value, max)} min={min} max={max} step={step} onChange={onChange} label={label} />
      <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
        {presets.map((p) => {
          const active = Math.abs(p.value - value) < 1e-9;
          return (
            <TouchableOpacity
              key={p.label}
              onPress={() => { tick(); onChange(p.value); }}
              accessibilityRole="button"
              accessibilityState={{ selected: active }}
              style={{
                paddingVertical: 6, paddingHorizontal: 12, borderRadius: 999, borderWidth: 1,
                backgroundColor: active ? "rgba(139,92,246,0.22)" : C.input,
                borderColor: active ? INVESTED : C.inputBorder,
              }}
            >
              <Text style={{ color: active ? "#fff" : C.textMuted, fontSize: 12, fontWeight: "600" }}>{p.label}</Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
}

// ─── growth chart: one bar per year, tap to inspect ──────────────────────────
function GrowthChart({ points, selected, onSelect }: {
  points: YearPoint[]; selected: number; onSelect: (i: number) => void;
}) {
  const [width, setWidth] = useState(0);
  const ref = useRef<View>(null);
  const left = useRef(0);
  const H = 150, AXIS = 18;
  const max = Math.max(1, ...points.map((p) => p.value));
  const n = Math.max(1, points.length);
  const slot = width / n;
  const barW = Math.max(3, Math.min(22, slot * 0.62));
  const pick = (pageX: number) => {
    if (!width) return;
    const i = Math.floor((pageX - left.current) / slot);
    const next = Math.max(0, Math.min(n - 1, i));
    if (next !== selected) { tick(); onSelect(next); }
  };
  const touch = useHorizontalPick(pick, () => ref.current?.measureInWindow((x) => { left.current = x; }));
  const labels = n <= 1 ? [0] : Array.from(new Set([0, Math.floor((n - 1) / 2), n - 1]));

  return (
    <View>
      <View
        ref={ref}
        onLayout={(e) => { setWidth(e.nativeEvent.layout.width); ref.current?.measureInWindow((x) => { left.current = x; }); }}
        {...touch}
        accessibilityLabel="Year by year growth chart. Tap a bar to see that year."
        style={{ height: H + AXIS }}
      >
        {width > 0 && (
          <Svg width={width} height={H + AXIS}>
            <Defs>
              <SvgGrad id="gain" x1="0" y1="0" x2="0" y2="1">
                <Stop offset="0%" stopColor={GAIN} stopOpacity="1" />
                <Stop offset="100%" stopColor="#059669" stopOpacity="1" />
              </SvgGrad>
            </Defs>
            {[0.25, 0.5, 0.75, 1].map((g) => (
              <Line key={g} x1={0} x2={width} y1={H - g * (H - 8)} y2={H - g * (H - 8)}
                stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
            ))}
            {points.map((p, i) => {
              const x = i * slot + (slot - barW) / 2;
              const total = (p.value / max) * (H - 8);
              const inv = (Math.min(p.invested, p.value) / max) * (H - 8);
              const dim = i === selected ? 1 : 0.45;
              return (
                <G key={i}>
                  <Rect x={x} y={H - total} width={barW} height={Math.max(0, total - inv)} rx={barW / 4}
                    fill="url(#gain)" opacity={dim} />
                  <Rect x={x} y={H - inv} width={barW} height={Math.max(1, inv)} rx={barW / 4}
                    fill={INVESTED} opacity={dim} />
                </G>
              );
            })}
          </Svg>
        )}
        {width > 0 && labels.map((i) => (
          <Text key={i} style={{
            position: "absolute", top: H + 3, left: Math.max(0, Math.min(width - 40, i * slot + slot / 2 - 20)),
            width: 40, textAlign: "center", color: C.textFaint, fontSize: 10,
          }}>
            {Number.isInteger(points[i]?.year) ? `Yr ${points[i]?.year}` : `${Math.round((points[i]?.year ?? 0) * 12)}m`}
          </Text>
        ))}
      </View>
    </View>
  );
}

function Legend({ color, label, value }: { color: string; label: string; value: string }) {
  return (
    <View style={{ flex: 1, gap: 2 }}>
      <View style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
        <View style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: color }} />
        <Text style={{ color: C.textFaint, fontSize: 11, fontWeight: "600" }}>{label}</Text>
      </View>
      <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700" }}>{value}</Text>
    </View>
  );
}

function Card({ children, style }: { children: React.ReactNode; style?: object }) {
  return (
    <View style={{
      backgroundColor: C.card, borderRadius: 22, borderWidth: 1, borderColor: C.cardEdge, padding: 18,
      marginBottom: 16, ...depthShadow("md"), ...style,
    }}>
      {children}
    </View>
  );
}

// ─── screen ──────────────────────────────────────────────────────────────────
export default function SIPCalculator() {
  const router = useRouter();
  const [mode, setMode] = useState<SipMode>("Regular");
  const [monthlyInvestment, setMonthlyInvestment] = useState(5000);
  const [lumpsumInvestment, setLumpsumInvestment] = useState(100000);
  const [targetAmount, setTargetAmount] = useState(10000000);
  const [expectedReturn, setExpectedReturn] = useState(12);
  const [timePeriod, setTimePeriod] = useState(15);
  const [stepUpRate, setStepUpRate] = useState(10);
  const [selectedYear, setSelectedYear] = useState<number | null>(null);
  const [showTable, setShowTable] = useState(false);
  const [showNotes, setShowNotes] = useState(false);

  const calc = useMemo(() => computeSip({
    mode, monthlyInvestment, lumpsumInvestment, targetAmount, expectedReturn, timePeriod, stepUpRate,
  }), [mode, monthlyInvestment, lumpsumInvestment, targetAmount, expectedReturn, timePeriod, stepUpRate]);

  const points = calc.yearly;
  const sel = Math.min(selectedYear ?? points.length - 1, Math.max(0, points.length - 1));
  const selPoint = points[sel];
  const gainShare = calc.totalValue > 0 ? Math.max(0, Math.min(1, calc.estimatedReturns / calc.totalValue)) : 0;
  const multiple = calc.investedAmount > 0 ? calc.totalValue / calc.investedAmount : 0;
  const isGoal = mode === "Goal Target";
  const monthlyForCta = isGoal ? calc.requiredSIP : monthlyInvestment;

  const setModeAndReset = (m: SipMode) => { tick(); setMode(m); setSelectedYear(null); };
  const change = (setter: (v: number) => void) => (v: number) => { setter(v); setSelectedYear(null); };

  const aiPrompt = isGoal
    ? `How should I invest ${inr(calc.requiredSIP)} a month to reach ${inr(targetAmount)} in ${yearsLabel(timePeriod)}?`
    : mode === "Lumpsum"
      ? `Where should I invest a lump sum of ${inr(lumpsumInvestment)} for ${yearsLabel(timePeriod)}?`
      : `Suggest a portfolio for a ${inr(monthlyInvestment)} monthly SIP for ${yearsLabel(timePeriod)}.`;

  return (
    <View style={{ flex: 1, backgroundColor: C.bgBottom }}>
      <AuthBackground />

      {/* Header */}
      <View style={{ paddingHorizontal: 20, paddingTop: 56, paddingBottom: 14 }}>
        <View style={{ flexDirection: "row", alignItems: "center", gap: 14 }}>
          <TouchableOpacity
            onPress={() => router.back()}
            accessibilityRole="button"
            accessibilityLabel="Go back"
            style={{
              width: 40, height: 40, borderRadius: 20, backgroundColor: C.input, borderWidth: 1,
              borderColor: C.inputBorder, alignItems: "center", justifyContent: "center",
            }}
          >
            <ArrowLeft color="#fff" size={18} />
          </TouchableOpacity>
          <View style={{ flex: 1 }}>
            <Text style={{ color: "#fff", fontSize: 20, fontWeight: "800" }}>SIP Calculator</Text>
            <Text style={{ color: C.textFaint, fontSize: 12 }}>Plan your wealth, see every rupee grow</Text>
          </View>
        </View>

        {/* Mode switcher: segmented, equal width */}
        <View style={{
          flexDirection: "row", marginTop: 16, padding: 4, borderRadius: 16,
          backgroundColor: "rgba(255,255,255,0.04)", borderWidth: 1, borderColor: C.inputBorder,
        }}>
          {MODES.map(({ key, label, Icon }) => {
            const active = mode === key;
            return (
              <TouchableOpacity
                key={key}
                onPress={() => setModeAndReset(key)}
                accessibilityRole="tab"
                accessibilityState={{ selected: active }}
                style={{ flex: 1, borderRadius: 12, overflow: "hidden" }}
              >
                {active ? (
                  <LinearGradient colors={[INVESTED, "#6d28d9"]} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }}
                    style={{ paddingVertical: 9, alignItems: "center", gap: 3 }}>
                    <Icon size={15} color="#fff" />
                    <Text style={{ color: "#fff", fontSize: 12, fontWeight: "700" }}>{label}</Text>
                  </LinearGradient>
                ) : (
                  <View style={{ paddingVertical: 9, alignItems: "center", gap: 3 }}>
                    <Icon size={15} color={C.textFaint} />
                    <Text style={{ color: C.textMuted, fontSize: 12, fontWeight: "600" }}>{label}</Text>
                  </View>
                )}
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      <ScrollView
        contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 110 }}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        {/* Result */}
        <LinearGradient
          colors={["rgba(124,58,237,0.32)", "rgba(16,20,36,0.92)"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={{ borderRadius: 24, borderWidth: 1, borderColor: "rgba(139,92,246,0.35)", padding: 20, marginBottom: 16, ...depthShadow("lg") }}
        >
          <Text style={{ color: C.textMuted, fontSize: 12, fontWeight: "700", letterSpacing: 0.6 }}>
            {isGoal ? "MONTHLY SIP NEEDED" : "ESTIMATED VALUE"}
          </Text>
          <Text
            accessibilityLiveRegion="polite"
            style={{ color: "#fff", fontSize: 36, fontWeight: "800", marginTop: 4 }}
            adjustsFontSizeToFit
            numberOfLines={1}
          >
            {isGoal ? `${inr(calc.requiredSIP)}/mo` : inr(calc.totalValue)}
          </Text>
          <Text style={{ color: C.textMuted, fontSize: 13, marginTop: 2 }}>
            {isGoal
              ? `to reach ${compact(targetAmount)} in ${yearsLabel(timePeriod)}`
              : `in ${yearsLabel(timePeriod)} at ${expectedReturn}% a year`}
            {multiple > 0 && !isGoal ? `  ·  ${multiple.toFixed(2)}x your money` : ""}
          </Text>

          {/* Invested vs gains bar */}
          <View style={{ flexDirection: "row", height: 10, borderRadius: 5, overflow: "hidden", marginTop: 18, backgroundColor: "rgba(255,255,255,0.06)" }}>
            <View style={{ flex: Math.max(0.0001, 1 - gainShare), backgroundColor: INVESTED }} />
            <View style={{ flex: Math.max(0.0001, gainShare), backgroundColor: GAIN }} />
          </View>
          <View style={{ flexDirection: "row", marginTop: 14, gap: 12 }}>
            <Legend color={INVESTED} label="Invested" value={compact(calc.investedAmount)} />
            <Legend color={GAIN} label="Est. gains" value={`+${compact(calc.estimatedReturns)}`} />
            <Legend color={GOLD} label={`Today's value`} value={compact(calc.realPurchasingPower)} />
          </View>

          {/* Mode-specific insight */}
          {mode === "Step-Up" && calc.investedAmount > 0 && (
            <View style={{ marginTop: 16, padding: 12, borderRadius: 14, backgroundColor: "rgba(52,211,153,0.10)", borderWidth: 1, borderColor: "rgba(52,211,153,0.25)" }}>
              <Text style={{ color: GAIN, fontSize: 13, fontWeight: "700" }}>
                +{compact(calc.totalValue - calc.fixedSipValue)} more than a fixed {inr(monthlyInvestment)} SIP
              </Text>
              <Text style={{ color: C.textMuted, fontSize: 12, marginTop: 2 }}>
                Your last installment would be {inr(calc.lastInstallment)} a month.
              </Text>
            </View>
          )}
          {isGoal && calc.requiredSIP > 0 && (
            <View style={{ marginTop: 16, padding: 12, borderRadius: 14, backgroundColor: "rgba(139,92,246,0.12)", borderWidth: 1, borderColor: "rgba(139,92,246,0.3)" }}>
              <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }}>
                Or start at {inr(calc.goalStepUpStart)}/mo and raise it 10% every year
              </Text>
              <Text style={{ color: C.textMuted, fontSize: 12, marginTop: 2 }}>
                {compact(targetAmount)} in {yearsLabel(timePeriod)} buys what {compact(calc.realPurchasingPower)} buys today ({INFLATION_PCT}% inflation).
              </Text>
            </View>
          )}
        </LinearGradient>

        {/* Inputs */}
        <Card style={{ gap: 22 }}>
          {(mode === "Regular" || mode === "Step-Up") && (
            <InputRow icon={Wallet} label="Monthly investment" prefix="₹" value={monthlyInvestment}
              onChange={change(setMonthlyInvestment)} min={500} max={200000} step={500} hardMax={10000000}
              presets={[{ label: "₹1K", value: 1000 }, { label: "₹5K", value: 5000 }, { label: "₹10K", value: 10000 }, { label: "₹25K", value: 25000 }, { label: "₹50K", value: 50000 }]} />
          )}
          {mode === "Lumpsum" && (
            <InputRow icon={Coins} label="One-time investment" prefix="₹" value={lumpsumInvestment}
              onChange={change(setLumpsumInvestment)} min={5000} max={5000000} step={5000} hardMax={1000000000}
              presets={[{ label: "₹50K", value: 50000 }, { label: "₹1L", value: 100000 }, { label: "₹5L", value: 500000 }, { label: "₹10L", value: 1000000 }, { label: "₹25L", value: 2500000 }]} />
          )}
          {isGoal && (
            <InputRow icon={Target} label="Target amount" prefix="₹" value={targetAmount}
              onChange={change(setTargetAmount)} min={100000} max={100000000} step={100000} hardMax={10000000000}
              presets={[{ label: "₹10L", value: 1000000 }, { label: "₹25L", value: 2500000 }, { label: "₹50L", value: 5000000 }, { label: "₹1Cr", value: 10000000 }, { label: "₹5Cr", value: 50000000 }]} />
          )}
          <InputRow icon={Percent} label="Expected return (a year)" suffix="%" value={expectedReturn}
            onChange={change(setExpectedReturn)} min={1} max={30} step={0.5} hardMax={30}
            presets={[{ label: "6% Debt", value: 6 }, { label: "8% Hybrid", value: 8 }, { label: "12% Equity", value: 12 }, { label: "14% High", value: 14 }]} />
          <InputRow icon={CalendarClock} label="Time period" suffix="yrs" value={timePeriod}
            onChange={change((v) => setTimePeriod(Math.round(v)))} min={1} max={40} step={1} hardMax={40}
            presets={[{ label: "5Y", value: 5 }, { label: "10Y", value: 10 }, { label: "15Y", value: 15 }, { label: "20Y", value: 20 }, { label: "30Y", value: 30 }]} />
          {mode === "Step-Up" && (
            <InputRow icon={Layers} label="Yearly step-up" suffix="%" value={stepUpRate}
              onChange={change(setStepUpRate)} min={1} max={50} step={1} hardMax={100}
              presets={[{ label: "5%", value: 5 }, { label: "10%", value: 10 }, { label: "15%", value: 15 }, { label: "20%", value: 20 }]} />
          )}
        </Card>

        {/* Growth chart */}
        {points.length > 0 && selPoint && (
          <Card>
            <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
              <View>
                <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700" }}>Growth over time</Text>
                <Text style={{ color: C.textFaint, fontSize: 11, marginTop: 2 }}>Tap or drag across the bars</Text>
              </View>
              <View style={{ alignItems: "flex-end" }}>
                <Text style={{ color: C.textFaint, fontSize: 11 }}>
                  {Number.isInteger(selPoint.year) ? `End of year ${selPoint.year}` : `After ${Math.round(selPoint.year * 12)} months`}
                </Text>
                <Text style={{ color: "#fff", fontSize: 16, fontWeight: "800" }}>{inr(selPoint.value)}</Text>
                <Text style={{ color: C.textMuted, fontSize: 11 }}>
                  invested {compact(selPoint.invested)}  ·  gain <Text style={{ color: GAIN }}>{compact(selPoint.value - selPoint.invested)}</Text>
                </Text>
              </View>
            </View>
            <GrowthChart points={points} selected={sel} onSelect={setSelectedYear} />
            <View style={{ flexDirection: "row", gap: 16, marginTop: 10 }}>
              <View style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
                <View style={{ width: 10, height: 10, borderRadius: 3, backgroundColor: INVESTED }} />
                <Text style={{ color: C.textMuted, fontSize: 11 }}>Invested</Text>
              </View>
              <View style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
                <View style={{ width: 10, height: 10, borderRadius: 3, backgroundColor: GAIN }} />
                <Text style={{ color: C.textMuted, fontSize: 11 }}>Gains</Text>
              </View>
            </View>

            {/* Yearly table */}
            <TouchableOpacity
              onPress={() => { tick(); setShowTable((s) => !s); }}
              accessibilityRole="button"
              style={{ flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6, marginTop: 14, paddingVertical: 8 }}
            >
              <Text style={{ color: "#c4b5fd", fontSize: 13, fontWeight: "700" }}>{showTable ? "Hide" : "Show"} year-by-year table</Text>
              {showTable ? <ChevronUp size={16} color="#c4b5fd" /> : <ChevronDown size={16} color="#c4b5fd" />}
            </TouchableOpacity>
            {showTable && (
              <View style={{ marginTop: 4, borderTopWidth: 1, borderColor: C.cardEdge }}>
                <View style={{ flexDirection: "row", paddingVertical: 8 }}>
                  {["Year", "Invested", "Gains", "Value"].map((h, i) => (
                    <Text key={h} style={{ flex: i === 0 ? 0.6 : 1, textAlign: i === 0 ? "left" : "right", color: C.textFaint, fontSize: 11, fontWeight: "700" }}>{h}</Text>
                  ))}
                </View>
                {points.map((p, i) => (
                  <View key={i} style={{ flexDirection: "row", paddingVertical: 7, borderTopWidth: 1, borderColor: "rgba(255,255,255,0.04)" }}>
                    <Text style={{ flex: 0.6, color: C.textMuted, fontSize: 12 }}>{Number.isInteger(p.year) ? p.year : `${Math.round(p.year * 12)}m`}</Text>
                    <Text style={{ flex: 1, textAlign: "right", color: C.textMuted, fontSize: 12 }}>{compact(p.invested)}</Text>
                    <Text style={{ flex: 1, textAlign: "right", color: GAIN, fontSize: 12 }}>{compact(p.value - p.invested)}</Text>
                    <Text style={{ flex: 1, textAlign: "right", color: "#fff", fontSize: 12, fontWeight: "700" }}>{compact(p.value)}</Text>
                  </View>
                ))}
              </View>
            )}
          </Card>
        )}

        {/* Actions */}
        <View style={{ gap: 12, marginBottom: 16 }}>
          <TouchableOpacity
            onPress={() => router.push({ pathname: "/(tabs)/invest", params: { amount: String(mode === "Lumpsum" ? lumpsumInvestment : monthlyForCta) } })}
            accessibilityRole="button"
            style={{ borderRadius: 16, overflow: "hidden", ...depthShadow("md") }}
          >
            <LinearGradient colors={[INVESTED, "#6d28d9"]} start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}
              style={{ paddingVertical: 16, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8 }}>
              <Coins size={18} color="#fff" />
              <Text style={{ color: "#fff", fontSize: 15, fontWeight: "800" }}>
                {mode === "Lumpsum" ? `Invest ${compact(lumpsumInvestment)} now` : `Start a ${inr(monthlyForCta)}/mo SIP`}
              </Text>
            </LinearGradient>
          </TouchableOpacity>
          <TouchableOpacity
            onPress={() => router.push({ pathname: "/(tabs)/ai-advisor", params: { prompt: aiPrompt } })}
            accessibilityRole="button"
            style={{
              borderRadius: 16, paddingVertical: 15, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8,
              backgroundColor: "rgba(139,92,246,0.10)", borderWidth: 1, borderColor: "rgba(139,92,246,0.4)",
            }}
          >
            <Sparkles size={18} color="#c4b5fd" />
            <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700" }}>Ask MentraFi AI which funds fit</Text>
          </TouchableOpacity>
        </View>

        {/* Assumptions & tax */}
        <TouchableOpacity
          onPress={() => { tick(); setShowNotes((s) => !s); }}
          accessibilityRole="button"
          style={{
            backgroundColor: "rgba(255,255,255,0.03)", borderWidth: 1, borderColor: C.cardEdge, borderRadius: 16,
            padding: 14, marginBottom: 14,
          }}
        >
          <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
            <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
              <Info size={14} color={C.textMuted} />
              <Text style={{ color: C.textMuted, fontSize: 13, fontWeight: "700" }}>How this is calculated & tax</Text>
            </View>
            {showNotes ? <ChevronUp size={16} color={C.textMuted} /> : <ChevronDown size={16} color={C.textMuted} />}
          </View>
          {showNotes && (
            <Text style={{ color: C.textFaint, fontSize: 12, lineHeight: 18, marginTop: 10 }}>
              • SIP installments are invested at the start of each month; monthly rate = yearly return ÷ 12.{"\n"}
              • Step-up raises the installment once every 12 months. Lump sums compound yearly.{"\n"}
              • Today&apos;s value discounts the result by {INFLATION_PCT}% inflation a year.{"\n"}
              • Equity funds: gains after 12 months taxed at 12.5% above ₹1.25 lakh a year; within 12 months at 20%.{"\n"}
              • Debt funds bought from 1 Apr 2023: gains taxed at your slab rate.{"\n"}
              • Tax is due only when you redeem and is not deducted above (FY 2025-26 rules).
            </Text>
          )}
        </TouchableOpacity>

        <Text style={{ color: C.textFaint, fontSize: 11, textAlign: "center", lineHeight: 16, paddingHorizontal: 8 }}>
          Returns are assumptions, not guarantees. Mutual fund investments are subject to market risks; read all scheme documents carefully.
        </Text>
      </ScrollView>
    </View>
  );
}
