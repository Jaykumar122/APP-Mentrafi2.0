import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import {
  ArrowUpRight,
  Building2,
  Calendar,
  ChevronRight,
  Home,
  Info,
  PieChart,
  Search,
  ShieldCheck,
  Sparkles,
  Star,
  TrendingDown,
  TrendingUp,
  User,
  X,
} from "lucide-react-native";
import React, { memo, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  Modal,
  ScrollView,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import Svg, { Circle, Defs, Line, LinearGradient as SvgGrad, Path, Stop } from "react-native-svg";
import { API_URL } from "../../utils/api";
import BottomNav from "./BottomNav";
import { AuthBackground, C, depthShadow } from "../(auth)/login";

// ---------------------------------------------------------------------------
// Luxury Obsidian & Aurora Palette Constants
// ---------------------------------------------------------------------------
const GREEN = "#4ade80";
const PAGE_LIMIT = 20;

export type Category = "All" | "Equity" | "Debt" | "Hybrid" | "ELSS" | "Gold";
export type TenureFilter = "all" | "new" | "growth" | "established";

export type Fund = {
  id: string;
  schemeCode?: number;
  name: string;
  category: Category;
  subcategory: string;
  rating: number; // 0–5
  nav: number | null;
  navDate?: string | null;
  oneYearReturn: number | null;
  threeYearReturn: number | null;
  fiveYearReturn: number | null;
  return1m: number | null;
  return3m: number | null;
  return6m: number | null;
  launchDate: string | null;
  fundHouse: string | null;
  expenseRatio: number | null;
  aumCrores: number | null;
  sebiRiskometer: string | null;
};

const CATEGORIES: Category[] = ["All", "Equity", "Debt", "Hybrid", "ELSS", "Gold"];

const TENURE_FILTERS: { id: TenureFilter; label: string }[] = [
  { id: "all", label: "All Horizons" },
  { id: "new", label: "✨ New Launches (<1Y)" },
  { id: "growth", label: "🌱 1–3Y Track" },
  { id: "established", label: "🏆 5Y+ Track" },
];

export function getFundTenure(fund: Fund): {
  type: "new" | "growth" | "established";
  label: string;
  color: string;
  bgColor: string;
  borderColor: string;
} {
  const now = new Date();
  let monthsSinceLaunch: number | null = null;
  if (fund.launchDate) {
    const d = new Date(fund.launchDate);
    if (!isNaN(d.getTime())) {
      monthsSinceLaunch = (now.getFullYear() - d.getFullYear()) * 12 + (now.getMonth() - d.getMonth());
    }
  }

  // Under 12 months or no 1-year return yet -> New Launch
  if ((monthsSinceLaunch !== null && monthsSinceLaunch < 12) || fund.oneYearReturn == null) {
    return {
      type: "new",
      label: "✨ New (<1 Yr)",
      color: "#38bdf8",
      bgColor: "rgba(56, 189, 248, 0.12)",
      borderColor: "rgba(56, 189, 248, 0.3)",
    };
  }

  // Under 48 months or no 5Y return yet -> 1-3Y Growth Track
  if (fund.fiveYearReturn == null || (monthsSinceLaunch !== null && monthsSinceLaunch < 48)) {
    return {
      type: "growth",
      label: "🌱 1-3Y Track",
      color: "#c084fc",
      bgColor: "rgba(192, 132, 252, 0.12)",
      borderColor: "rgba(192, 132, 252, 0.3)",
    };
  }

  return {
    type: "established",
    label: "🏆 5Y+ Track",
    color: "#fbbf24",
    bgColor: "rgba(251, 191, 36, 0.12)",
    borderColor: "rgba(251, 191, 36, 0.3)",
  };
}

export function getRiskBadge(risk: string | null) {
  const r = (risk || "").toLowerCase();
  if (r.includes("low")) {
    return { label: risk || "Low Risk", color: "#4ade80", bg: "rgba(74, 222, 128, 0.12)" };
  }
  if (r.includes("moderat")) {
    return { label: risk || "Moderate Risk", color: "#fbbf24", bg: "rgba(251, 191, 36, 0.12)" };
  }
  return { label: risk || "Very High Risk", color: "#f87171", bg: "rgba(248, 113, 113, 0.12)" };
}

export function formatDate(dateStr: string | null | undefined): string {
  if (!dateStr) return "—";
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return String(dateStr);
    return d.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
  } catch {
    return String(dateStr);
  }
}

export function formatReturn(val: number | null | undefined): string {
  if (val == null) return "—";
  const num = Number(val);
  return num >= 0 ? `+${num.toFixed(1)}%` : `${num.toFixed(1)}%`;
}

export function formatNav(val: number | null | undefined): string {
  if (val == null) return "—";
  return `₹${Number(val).toFixed(2)}`;
}

function StarRow({ rating }: { rating: number }) {
  const rounded = Math.round(rating);
  return (
    <View style={{ flexDirection: "row", alignItems: "center", gap: 2 }}>
      {[1, 2, 3, 4, 5].map((i) => (
        <Star key={i} size={12} color={C.pink} fill={i <= rounded ? C.pink : "transparent"} strokeWidth={1.5} />
      ))}
    </View>
  );
}

// Memoized Fund Card — highly responsive, opens details sheet on tap,
// direct one-tap investment routing to /(tabs)/invest with schemeCode
const FundCard = memo(function FundCard({
  fund,
  onPressCard,
  onInvest,
}: {
  fund: Fund;
  onPressCard: (fund: Fund) => void;
  onInvest: (fund: Fund) => void;
}) {
  const tenure = getFundTenure(fund);

  return (
    <TouchableOpacity
      activeOpacity={0.88}
      onPress={() => onPressCard(fund)}
      style={{
        backgroundColor: C.card,
        borderWidth: 1,
        borderColor: C.cardEdge,
        borderRadius: 22,
        padding: 16,
        marginBottom: 14,
        ...depthShadow("sm"),
      }}
    >
      {/* Top Header: AMC & Tenure Badge */}
      <View
        style={{
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 10,
        }}
      >
        <View style={{ flexDirection: "row", alignItems: "center", gap: 5, flexShrink: 1 }}>
          <Building2 size={12} color={C.textMuted} />
          <Text
            numberOfLines={1}
            style={{ color: C.textMuted, fontSize: 11, fontWeight: "600", textTransform: "uppercase", letterSpacing: 0.5 }}
          >
            {fund.fundHouse || "Direct Mutual Fund"}
          </Text>
        </View>

        <View
          style={{
            paddingHorizontal: 9,
            paddingVertical: 3,
            borderRadius: 999,
            backgroundColor: tenure.bgColor,
            borderWidth: 1,
            borderColor: tenure.borderColor,
          }}
        >
          <Text style={{ color: tenure.color, fontSize: 11, fontWeight: "700" }}>
            {tenure.label}
          </Text>
        </View>
      </View>

      {/* Fund Name and Category */}
      <View style={{ marginBottom: 12 }}>
        <Text
          numberOfLines={2}
          style={{ color: "#fff", fontWeight: "700", fontSize: 15, lineHeight: 21, marginBottom: 4 }}
        >
          {fund.name}
        </Text>
        <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
          <Text style={{ color: C.textFaint, fontSize: 12 }}>
            {fund.category} · {fund.subcategory || "Growth"}
          </Text>
          <StarRow rating={fund.rating} />
        </View>
      </View>

      {/* Divider */}
      <View style={{ height: 1, backgroundColor: "rgba(255, 255, 255, 0.06)", marginBottom: 12 }} />

      {/* Adaptive Returns Matrix: Custom per tenure horizon */}
      <View
        style={{
          flexDirection: "row",
          alignItems: "flex-end",
          justifyContent: "space-between",
        }}
      >
        <View style={{ flexDirection: "row", gap: 16 }}>
          {tenure.type === "new" ? (
            <>
              <View>
                <Text style={{ color: C.textFaint, fontSize: 10, textTransform: "uppercase", marginBottom: 3 }}>
                  {fund.return6m != null ? "6M Return" : fund.return3m != null ? "3M Return" : "1M Return"}
                </Text>
                <Text style={{ color: GREEN, fontWeight: "700", fontSize: 13 }}>
                  {formatReturn(fund.return6m ?? fund.return3m ?? fund.return1m)}
                </Text>
              </View>
              <View>
                <Text style={{ color: C.textFaint, fontSize: 10, textTransform: "uppercase", marginBottom: 3 }}>
                  Launch Date
                </Text>
                <Text style={{ color: "#38bdf8", fontWeight: "600", fontSize: 12 }}>
                  {formatDate(fund.launchDate)}
                </Text>
              </View>
            </>
          ) : tenure.type === "growth" ? (
            <>
              <View>
                <Text style={{ color: C.textFaint, fontSize: 10, textTransform: "uppercase", marginBottom: 3 }}>
                  1Y Return
                </Text>
                <Text style={{ color: GREEN, fontWeight: "700", fontSize: 13 }}>
                  {formatReturn(fund.oneYearReturn)}
                </Text>
              </View>
              <View>
                <Text style={{ color: C.textFaint, fontSize: 10, textTransform: "uppercase", marginBottom: 3 }}>
                  3Y Return
                </Text>
                <Text
                  style={{
                    color: fund.threeYearReturn != null ? GREEN : C.textFaint,
                    fontWeight: "700",
                    fontSize: 13,
                  }}
                >
                  {formatReturn(fund.threeYearReturn)}
                </Text>
              </View>
            </>
          ) : (
            <>
              <View>
                <Text style={{ color: C.textFaint, fontSize: 10, textTransform: "uppercase", marginBottom: 3 }}>
                  1Y Return
                </Text>
                <Text style={{ color: GREEN, fontWeight: "700", fontSize: 13 }}>
                  {formatReturn(fund.oneYearReturn)}
                </Text>
              </View>
              <View>
                <Text style={{ color: C.textFaint, fontSize: 10, textTransform: "uppercase", marginBottom: 3 }}>
                  5Y Return
                </Text>
                <Text
                  style={{
                    color: fund.fiveYearReturn != null ? GREEN : C.textFaint,
                    fontWeight: "700",
                    fontSize: 13,
                  }}
                >
                  {formatReturn(fund.fiveYearReturn)}
                </Text>
              </View>
            </>
          )}

          <View>
            <Text style={{ color: C.textFaint, fontSize: 10, textTransform: "uppercase", marginBottom: 3 }}>
              NAV
            </Text>
            <Text style={{ color: "#fff", fontWeight: "700", fontSize: 13 }}>
              {formatNav(fund.nav)}
            </Text>
          </View>
        </View>

        {/* Invest CTA Button */}
        <TouchableOpacity
          activeOpacity={0.85}
          onPress={() => onInvest(fund)}
        >
          <LinearGradient
            colors={[C.pink, C.violet]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={{
              paddingHorizontal: 16,
              paddingVertical: 8,
              borderRadius: 999,
              flexDirection: "row",
              alignItems: "center",
              gap: 4,
            }}
          >
            <Text style={{ color: "#fff", fontWeight: "700", fontSize: 12 }}>Invest</Text>
            <ArrowUpRight size={13} color="#fff" strokeWidth={2.5} />
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </TouchableOpacity>
  );
});

// ---------------------------------------------------------------------------
// Interactive Live NAV Graph Component
// ---------------------------------------------------------------------------
type ChartPoint = { date: string; nav: number };

type ChartData = {
  schemeCode: number;
  range: string;
  periodReturn: number;
  minNav: number;
  maxNav: number;
  startNav: number;
  latestNav: number;
  startDate: string;
  latestDate: string;
  points: ChartPoint[];
};

const CHART_RANGES = ["1M", "3M", "6M", "1Y", "3Y", "5Y", "ALL"] as const;
type ChartRange = (typeof CHART_RANGES)[number];

function FundNavGraph({
  schemeCode,
  isNewFund,
  currentNav,
}: {
  schemeCode?: number;
  isNewFund: boolean;
  currentNav?: number | null;
}) {
  const [selectedRange, setSelectedRange] = useState<ChartRange>(isNewFund ? "6M" : "1Y");
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState(true);
  const [scrubPoint, setScrubPoint] = useState<ChartPoint | null>(null);
  const [chartWidth, setChartWidth] = useState(300);

  useEffect(() => {
    if (!schemeCode) return;
    let isCancelled = false;
    async function loadChart() {
      try {
        setLoading(true);
        const res = await fetch(`${API_URL}/api/funds/${schemeCode}/nav-chart?range=${selectedRange}`);
        if (res.ok) {
          const data = await res.json();
          if (!isCancelled) {
            setChartData(data);
            setScrubPoint(null);
          }
        }
      } catch (err) {
        console.error("Failed to load nav chart:", err);
      } finally {
        if (!isCancelled) setLoading(false);
      }
    }
    loadChart();
    return () => {
      isCancelled = true;
    };
  }, [schemeCode, selectedRange]);

  const height = 140;
  const paddingX = 6;
  const paddingTop = 14;
  const paddingBottom = 16;

  let pathD = "";
  let areaD = "";
  let scrubX = 0;
  let scrubY = 0;

  if (chartData && chartData.points && chartData.points.length >= 2) {
    const pts = chartData.points;
    const minVal = chartData.minNav;
    const maxVal = chartData.maxNav;
    const rangeVal = maxVal - minVal || 1;
    const plotWidth = Math.max(100, chartWidth - paddingX * 2);
    const plotHeight = height - paddingTop - paddingBottom;

    const coords = pts.map((p, i) => {
      const x = paddingX + (i / (pts.length - 1)) * plotWidth;
      const y = paddingTop + plotHeight - ((p.nav - minVal) / rangeVal) * plotHeight;
      return { x, y, point: p };
    });

    pathD = `M ${coords[0].x.toFixed(1)},${coords[0].y.toFixed(1)}`;
    for (let i = 0; i < coords.length - 1; i++) {
      const p0 = coords[Math.max(0, i - 1)];
      const p1 = coords[i];
      const p2 = coords[i + 1];
      const p3 = coords[Math.min(coords.length - 1, i + 2)];

      const cp1x = (p1.x + (p2.x - p0.x) / 6).toFixed(1);
      const cp1y = (p1.y + (p2.y - p0.y) / 6).toFixed(1);
      const cp2x = (p2.x - (p3.x - p1.x) / 6).toFixed(1);
      const cp2y = (p2.y - (p3.y - p1.y) / 6).toFixed(1);

      pathD += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${p2.x.toFixed(1)},${p2.y.toFixed(1)}`;
    }

    const last = coords[coords.length - 1];
    areaD = `${pathD} L ${last.x.toFixed(1)},${height} L ${coords[0].x.toFixed(1)},${height} Z`;

    if (scrubPoint) {
      const idx = pts.findIndex((p) => p.date === scrubPoint.date);
      if (idx >= 0) {
        scrubX = coords[idx].x;
        scrubY = coords[idx].y;
      }
    }
  }

  const isPositive = (chartData?.periodReturn ?? 0) >= 0;
  const currentDisplayNav = scrubPoint
    ? scrubPoint.nav
    : (chartData?.latestNav ?? currentNav ?? 0);
  const currentDisplayDate = scrubPoint
    ? scrubPoint.date
    : (chartData?.latestDate ?? "Latest NAV");

  const handleTouch = (event: any) => {
    if (!chartData || !chartData.points || chartData.points.length < 2) return;
    const locationX = event.nativeEvent.locationX;
    const plotWidth = Math.max(100, chartWidth - paddingX * 2);
    const clampedX = Math.max(paddingX, Math.min(paddingX + plotWidth, locationX));
    const ratio = (clampedX - paddingX) / plotWidth;
    const index = Math.round(ratio * (chartData.points.length - 1));
    const pt = chartData.points[Math.max(0, Math.min(chartData.points.length - 1, index))];
    if (pt) {
      setScrubPoint(pt);
    }
  };

  return (
    <View
      style={{
        backgroundColor: "rgba(255, 255, 255, 0.03)",
        borderWidth: 1,
        borderColor: "rgba(255, 255, 255, 0.08)",
        borderRadius: 20,
        padding: 16,
        marginBottom: 16,
      }}
    >
      {/* Header: Live NAV & Return */}
      <View
        style={{
          flexDirection: "row",
          alignItems: "flex-start",
          justifyContent: "space-between",
          marginBottom: 8,
        }}
      >
        <View>
          <View style={{ flexDirection: "row", alignItems: "center", gap: 6, marginBottom: 2 }}>
            <Text style={{ color: "#fff", fontSize: 22, fontWeight: "800" }}>
              ₹{Number(currentDisplayNav).toFixed(2)}
            </Text>
            {chartData && (
              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  gap: 3,
                  paddingHorizontal: 8,
                  paddingVertical: 2,
                  borderRadius: 999,
                  backgroundColor: isPositive ? "rgba(74, 222, 128, 0.15)" : "rgba(248, 113, 113, 0.15)",
                }}
              >
                {isPositive ? (
                  <TrendingUp size={11} color={GREEN} />
                ) : (
                  <TrendingDown size={11} color="#f87171" />
                )}
                <Text
                  style={{
                    color: isPositive ? GREEN : "#f87171",
                    fontSize: 11,
                    fontWeight: "700",
                  }}
                >
                  {isPositive ? "+" : ""}
                  {chartData.periodReturn}%
                </Text>
              </View>
            )}
          </View>
          <Text style={{ color: C.textFaint, fontSize: 11 }}>
            {scrubPoint ? `📅 ${currentDisplayDate}` : `NAV as of ${currentDisplayDate}`}
          </Text>
        </View>

        {chartData && (
          <View style={{ alignItems: "flex-end" }}>
            <Text style={{ color: C.textFaint, fontSize: 10, textTransform: "uppercase" }}>
              {selectedRange} Range
            </Text>
            <Text style={{ color: C.textMuted, fontSize: 11, fontWeight: "600" }}>
              ₹{chartData.minNav.toFixed(1)} – ₹{chartData.maxNav.toFixed(1)}
            </Text>
          </View>
        )}
      </View>

      {/* SVG Live Chart Area */}
      <View
        onLayout={(e) => {
          const w = e.nativeEvent.layout.width;
          if (w > 0) setChartWidth(w);
        }}
        onTouchStart={handleTouch}
        onTouchMove={handleTouch}
        onTouchEnd={() => {
          setTimeout(() => setScrubPoint(null), 2500);
        }}
        style={{
          height: height,
          justifyContent: "center",
          alignItems: "center",
          position: "relative",
        }}
      >
        {loading && !chartData ? (
          <ActivityIndicator color={C.pink} />
        ) : pathD ? (
          <Svg width={chartWidth} height={height}>
            <Defs>
              <SvgGrad id="chartLineGrad" x1="0" y1="0" x2="1" y2="0">
                <Stop offset="0%" stopColor={isPositive ? "#06b6d4" : "#f43f5e"} />
                <Stop offset="60%" stopColor={isPositive ? C.pink : "#fb7185"} />
                <Stop offset="100%" stopColor={isPositive ? GREEN : "#ef4444"} />
              </SvgGrad>
              <SvgGrad id="chartAreaGrad" x1="0" y1="0" x2="0" y2="1">
                <Stop offset="0%" stopColor={isPositive ? GREEN : "#f43f5e"} stopOpacity={0.28} />
                <Stop offset="80%" stopColor={isPositive ? C.pink : "#f43f5e"} stopOpacity={0.04} />
                <Stop offset="100%" stopColor={isPositive ? "#06b6d4" : "#f43f5e"} stopOpacity={0.0} />
              </SvgGrad>
            </Defs>

            {/* Min / Max Dashed Reference Lines */}
            <Line
              x1={paddingX}
              y1={paddingTop}
              x2={chartWidth - paddingX}
              y2={paddingTop}
              stroke="rgba(255, 255, 255, 0.08)"
              strokeDasharray="4, 4"
            />
            <Line
              x1={paddingX}
              y1={height - paddingBottom}
              x2={chartWidth - paddingX}
              y2={height - paddingBottom}
              stroke="rgba(255, 255, 255, 0.08)"
              strokeDasharray="4, 4"
            />

            {/* Gradient Area Fill */}
            <Path d={areaD} fill="url(#chartAreaGrad)" />

            {/* Smooth Line Path */}
            <Path
              d={pathD}
              stroke="url(#chartLineGrad)"
              strokeWidth={2.6}
              fill="none"
              strokeLinecap="round"
            />

            {/* Interactive Scrub Cursor */}
            {scrubPoint && scrubX > 0 ? (
              <>
                <Line
                  x1={scrubX}
                  y1={paddingTop}
                  x2={scrubX}
                  y2={height - paddingBottom}
                  stroke="#fff"
                  strokeWidth={1.5}
                  strokeDasharray="3, 3"
                  opacity={0.7}
                />
                <Circle cx={scrubX} cy={scrubY} r={5} fill={isPositive ? GREEN : "#f87171"} />
                <Circle cx={scrubX} cy={scrubY} r={10} fill={isPositive ? GREEN : "#f87171"} opacity={0.3} />
              </>
            ) : null}
          </Svg>
        ) : (
          <Text style={{ color: C.textFaint, fontSize: 12 }}>Live chart data loading...</Text>
        )}
      </View>

      {/* Range Selector Buttons */}
      <View
        style={{
          flexDirection: "row",
          justifyContent: "space-between",
          paddingTop: 10,
          borderTopWidth: 1,
          borderTopColor: "rgba(255, 255, 255, 0.06)",
        }}
      >
        {CHART_RANGES.map((r) => {
          const active = r === selectedRange;
          return (
            <TouchableOpacity
              key={r}
              onPress={() => setSelectedRange(r)}
              activeOpacity={0.8}
              style={{
                paddingHorizontal: 9,
                paddingVertical: 4,
                borderRadius: 8,
                backgroundColor: active ? "rgba(139, 92, 246, 0.28)" : "transparent",
                borderWidth: 1,
                borderColor: active ? "rgba(168, 85, 247, 0.5)" : "transparent",
              }}
            >
              <Text
                style={{
                  fontSize: 11,
                  fontWeight: active ? "700" : "500",
                  color: active ? "#c084fc" : C.textMuted,
                }}
              >
                {r}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Interactive Fund Details Modal Sheet
// ---------------------------------------------------------------------------
function FundDetailsModal({
  fund,
  onClose,
  onInvest,
}: {
  fund: Fund | null;
  onClose: () => void;
  onInvest: (fund: Fund) => void;
}) {
  if (!fund) return null;

  const tenure = getFundTenure(fund);
  const risk = getRiskBadge(fund.sebiRiskometer);

  return (
    <Modal
      visible={!!fund}
      animationType="slide"
      transparent={true}
      onRequestClose={onClose}
    >
      <View
        style={{
          flex: 1,
          backgroundColor: "rgba(0, 0, 0, 0.75)",
          justifyContent: "flex-end",
        }}
      >
        <TouchableOpacity
          activeOpacity={1}
          onPress={onClose}
          style={{ flex: 1 }}
        />

        <View
          style={{
            backgroundColor: "#0d0f1a",
            borderTopLeftRadius: 28,
            borderTopRightRadius: 28,
            borderWidth: 1,
            borderColor: "rgba(255, 255, 255, 0.12)",
            maxHeight: "88%",
            paddingBottom: 24,
            ...depthShadow("lg"),
          }}
        >
          {/* Grab Handle */}
          <View style={{ alignItems: "center", paddingTop: 12, paddingBottom: 8 }}>
            <View
              style={{
                width: 40,
                height: 4,
                borderRadius: 2,
                backgroundColor: "rgba(255, 255, 255, 0.2)",
              }}
            />
          </View>

          {/* Modal Header */}
          <View
            style={{
              flexDirection: "row",
              alignItems: "flex-start",
              justifyContent: "space-between",
              paddingHorizontal: 20,
              paddingBottom: 14,
              borderBottomWidth: 1,
              borderBottomColor: "rgba(255, 255, 255, 0.06)",
            }}
          >
            <View style={{ flex: 1, marginRight: 12 }}>
              <View style={{ flexDirection: "row", alignItems: "center", gap: 6, marginBottom: 4 }}>
                <Building2 size={13} color={C.textMuted} />
                <Text
                  style={{
                    color: C.textMuted,
                    fontSize: 12,
                    fontWeight: "600",
                    textTransform: "uppercase",
                  }}
                >
                  {fund.fundHouse || "Direct Mutual Fund"}
                </Text>
              </View>
              <Text
                style={{
                  color: "#fff",
                  fontSize: 18,
                  fontWeight: "700",
                  lineHeight: 24,
                  marginBottom: 6,
                }}
              >
                {fund.name}
              </Text>
              <Text style={{ color: C.textFaint, fontSize: 13 }}>
                {fund.category} · {fund.subcategory || "Growth"}
              </Text>
            </View>

            <TouchableOpacity
              onPress={onClose}
              style={{
                width: 34,
                height: 34,
                borderRadius: 17,
                backgroundColor: "rgba(255, 255, 255, 0.08)",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <X size={18} color="#fff" />
            </TouchableOpacity>
          </View>

          <ScrollView
            style={{ paddingHorizontal: 20, paddingTop: 16 }}
            showsVerticalScrollIndicator={false}
          >
            {/* Badges Row */}
            <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8, marginBottom: 18 }}>
              <View
                style={{
                  paddingHorizontal: 12,
                  paddingVertical: 5,
                  borderRadius: 999,
                  backgroundColor: tenure.bgColor,
                  borderWidth: 1,
                  borderColor: tenure.borderColor,
                }}
              >
                <Text style={{ color: tenure.color, fontSize: 12, fontWeight: "700" }}>
                  {tenure.label}
                </Text>
              </View>

              <View
                style={{
                  paddingHorizontal: 12,
                  paddingVertical: 5,
                  borderRadius: 999,
                  backgroundColor: risk.bg,
                  borderWidth: 1,
                  borderColor: risk.color + "40",
                  flexDirection: "row",
                  alignItems: "center",
                  gap: 4,
                }}
              >
                <ShieldCheck size={13} color={risk.color} />
                <Text style={{ color: risk.color, fontSize: 12, fontWeight: "700" }}>
                  {risk.label}
                </Text>
              </View>

              <View
                style={{
                  paddingHorizontal: 10,
                  paddingVertical: 5,
                  borderRadius: 999,
                  backgroundColor: "rgba(255, 255, 255, 0.05)",
                  borderWidth: 1,
                  borderColor: "rgba(255, 255, 255, 0.08)",
                  justifyContent: "center",
                }}
              >
                <StarRow rating={fund.rating} />
              </View>
            </View>

            {/* Live Interactive NAV Graph */}
            <FundNavGraph
              schemeCode={fund.schemeCode}
              isNewFund={tenure.type === "new"}
              currentNav={fund.nav}
            />

            {/* Performance Matrix */}
            <View
              style={{
                backgroundColor: "rgba(255, 255, 255, 0.03)",
                borderWidth: 1,
                borderColor: "rgba(255, 255, 255, 0.08)",
                borderRadius: 18,
                padding: 16,
                marginBottom: 16,
              }}
            >
              <View style={{ flexDirection: "row", alignItems: "center", gap: 6, marginBottom: 12 }}>
                <TrendingUp size={16} color={GREEN} />
                <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700" }}>
                  Performance Track Record
                </Text>
              </View>

              {/* 3x2 Return Grid */}
              <View style={{ flexDirection: "row", justifyContent: "space-between", marginBottom: 12 }}>
                <View style={{ flex: 1, alignItems: "center", paddingVertical: 8, borderRightWidth: 1, borderRightColor: "rgba(255, 255, 255, 0.06)" }}>
                  <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 4 }}>1 Month</Text>
                  <Text style={{ color: fund.return1m != null ? GREEN : C.textFaint, fontWeight: "700", fontSize: 14 }}>
                    {formatReturn(fund.return1m)}
                  </Text>
                </View>
                <View style={{ flex: 1, alignItems: "center", paddingVertical: 8, borderRightWidth: 1, borderRightColor: "rgba(255, 255, 255, 0.06)" }}>
                  <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 4 }}>3 Month</Text>
                  <Text style={{ color: fund.return3m != null ? GREEN : C.textFaint, fontWeight: "700", fontSize: 14 }}>
                    {formatReturn(fund.return3m)}
                  </Text>
                </View>
                <View style={{ flex: 1, alignItems: "center", paddingVertical: 8 }}>
                  <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 4 }}>6 Month</Text>
                  <Text style={{ color: fund.return6m != null ? GREEN : C.textFaint, fontWeight: "700", fontSize: 14 }}>
                    {formatReturn(fund.return6m)}
                  </Text>
                </View>
              </View>

              <View style={{ height: 1, backgroundColor: "rgba(255, 255, 255, 0.06)", marginBottom: 12 }} />

              <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                <View style={{ flex: 1, alignItems: "center", paddingVertical: 8, borderRightWidth: 1, borderRightColor: "rgba(255, 255, 255, 0.06)" }}>
                  <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 4 }}>1 Year CAGR</Text>
                  <Text style={{ color: fund.oneYearReturn != null ? GREEN : C.textFaint, fontWeight: "700", fontSize: 14 }}>
                    {formatReturn(fund.oneYearReturn)}
                  </Text>
                </View>
                <View style={{ flex: 1, alignItems: "center", paddingVertical: 8, borderRightWidth: 1, borderRightColor: "rgba(255, 255, 255, 0.06)" }}>
                  <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 4 }}>3 Year CAGR</Text>
                  <Text style={{ color: fund.threeYearReturn != null ? GREEN : C.textFaint, fontWeight: "700", fontSize: 14 }}>
                    {formatReturn(fund.threeYearReturn)}
                  </Text>
                </View>
                <View style={{ flex: 1, alignItems: "center", paddingVertical: 8 }}>
                  <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 4 }}>5 Year CAGR</Text>
                  <Text style={{ color: fund.fiveYearReturn != null ? GREEN : C.textFaint, fontWeight: "700", fontSize: 14 }}>
                    {formatReturn(fund.fiveYearReturn)}
                  </Text>
                </View>
              </View>

              {/* Informative tenure note */}
              <View
                style={{
                  marginTop: 14,
                  padding: 10,
                  borderRadius: 12,
                  backgroundColor: "rgba(255, 255, 255, 0.04)",
                  borderWidth: 1,
                  borderColor: "rgba(255, 255, 255, 0.06)",
                  flexDirection: "row",
                  alignItems: "flex-start",
                  gap: 8,
                }}
              >
                <Info size={14} color="#38bdf8" style={{ marginTop: 2 }} />
                <Text style={{ color: C.textMuted, fontSize: 11, lineHeight: 16, flex: 1 }}>
                  {tenure.type === "new"
                    ? "Newly launched scheme (<1 Yr). Short-term milestones (1M, 3M, 6M) are tracked above. Annualised 1Y/3Y/5Y returns will accumulate as the fund matures."
                    : tenure.type === "growth"
                    ? "Established 1–3 year track record. 1Y & 3Y annualised returns tracked above."
                    : "5+ year established history. Verified multi-cycle wealth compounding track record."}
                </Text>
              </View>
            </View>

            {/* Scheme Fundamentals */}
            <View
              style={{
                backgroundColor: "rgba(255, 255, 255, 0.03)",
                borderWidth: 1,
                borderColor: "rgba(255, 255, 255, 0.08)",
                borderRadius: 18,
                padding: 16,
                marginBottom: 24,
              }}
            >
              <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700", marginBottom: 12 }}>
                Scheme Fundamentals
              </Text>

              <View style={{ gap: 10 }}>
                <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
                  <Text style={{ color: C.textFaint, fontSize: 13 }}>Current NAV</Text>
                  <Text style={{ color: "#fff", fontWeight: "700", fontSize: 14 }}>
                    {formatNav(fund.nav)}
                  </Text>
                </View>

                <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
                  <Text style={{ color: C.textFaint, fontSize: 13 }}>Inception / Launch</Text>
                  <Text style={{ color: "#fff", fontWeight: "600", fontSize: 13 }}>
                    {formatDate(fund.launchDate)}
                  </Text>
                </View>

                <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
                  <Text style={{ color: C.textFaint, fontSize: 13 }}>Expense Ratio</Text>
                  <Text style={{ color: "#fff", fontWeight: "600", fontSize: 13 }}>
                    {fund.expenseRatio != null ? `${Number(fund.expenseRatio).toFixed(2)}% (Direct)` : "Direct Plan (Low TER)"}
                  </Text>
                </View>

                <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
                  <Text style={{ color: C.textFaint, fontSize: 13 }}>AUM (Assets Size)</Text>
                  <Text style={{ color: "#fff", fontWeight: "600", fontSize: 13 }}>
                    {fund.aumCrores != null ? `₹${Number(fund.aumCrores).toLocaleString("en-IN")} Cr` : "Direct AMFI Scheme"}
                  </Text>
                </View>

                <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
                  <Text style={{ color: C.textFaint, fontSize: 13 }}>Scheme Code</Text>
                  <Text style={{ color: C.textMuted, fontWeight: "600", fontSize: 13 }}>
                    {fund.schemeCode || "—"}
                  </Text>
                </View>
              </View>
            </View>
          </ScrollView>

          {/* Sticky Invest CTA */}
          <View style={{ paddingHorizontal: 20, paddingTop: 10 }}>
            <TouchableOpacity
              activeOpacity={0.88}
              onPress={() => onInvest(fund)}
            >
              <LinearGradient
                colors={[C.pink, C.violet]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={{
                  height: 52,
                  borderRadius: 16,
                  alignItems: "center",
                  justifyContent: "center",
                  flexDirection: "row",
                  gap: 8,
                }}
              >
                <Text style={{ color: "#fff", fontWeight: "700", fontSize: 15 }}>
                  Invest in this Scheme
                </Text>
                <ArrowUpRight size={18} color="#fff" strokeWidth={2.5} />
              </LinearGradient>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );
}

export default function ExploreScreen() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState("Explore");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<Category>("All");
  const [tenure, setTenure] = useState<TenureFilter>("all");
  const [funds, setFunds] = useState<Fund[]>([]);
  const [totalFunds, setTotalFunds] = useState(0);
  const [selectedFund, setSelectedFund] = useState<Fund | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);

  const requestId = useRef(0);
  const loadingMoreRef = useRef(false);

  const tabs = [
    { name: "Home", icon: Home, route: "/(tabs)/home" },
    { name: "Explore", icon: Search, route: "/(tabs)/explore" },
    { name: "Portfolio", icon: PieChart, route: "/(tabs)/portfolio" },
    { name: "AI Advisor", icon: Sparkles, route: "/(tabs)/ai-advisor" },
    { name: "Profile", icon: User, route: "/(tabs)/profile" },
  ];

  const handleInvest = (fund: Fund) => {
    setSelectedFund(null);
    router.push({
      pathname: "/(tabs)/invest",
      params: { schemeCode: String(fund.schemeCode || fund.id) },
    });
  };

  // Reset to page 1 whenever search, category, or tenure changes
  useEffect(() => {
    const controller = new AbortController();
    const timeout = setTimeout(() => {
      fetchFunds(1, controller.signal, true);
    }, 280);

    return () => {
      clearTimeout(timeout);
      controller.abort();
    };
  }, [query, category, tenure]);

  async function fetchFunds(pageToLoad: number, signal?: AbortSignal, replace = false) {
    const myRequestId = ++requestId.current;

    try {
      if (replace) {
        setLoading(true);
        setError(null);
      } else {
        setLoadingMore(true);
      }

      const params = new URLSearchParams();
      if (category !== "All") params.append("category", category);
      if (tenure !== "all") params.append("tenure", tenure);
      if (query.trim()) params.append("q", query.trim());
      params.append("page", String(pageToLoad));
      params.append("limit", String(PAGE_LIMIT));

      const res = await fetch(`${API_URL}/api/funds?${params.toString()}`, { signal });
      if (!res.ok) throw new Error("Failed to fetch funds");

      const data = await res.json();
      if (myRequestId !== requestId.current) return;

      const mapped: Fund[] = (data.funds || []).map((f: any) => ({
        id: String(f.id),
        schemeCode: f.scheme_code ?? f.schemeCode,
        name: f.name,
        category: f.category,
        subcategory: f.subcategory || "Growth",
        rating: Number(f.rating) || 4,
        nav: f.nav != null ? Number(f.nav) : null,
        navDate: f.navDate,
        oneYearReturn: f.oneYearReturn != null ? Number(f.oneYearReturn) : null,
        threeYearReturn: f.threeYearReturn != null ? Number(f.threeYearReturn) : null,
        fiveYearReturn: f.fiveYearReturn != null ? Number(f.fiveYearReturn) : null,
        return1m: f.return1m != null ? Number(f.return1m) : null,
        return3m: f.return3m != null ? Number(f.return3m) : null,
        return6m: f.return6m != null ? Number(f.return6m) : null,
        launchDate: f.launchDate,
        fundHouse: f.fundHouse,
        expenseRatio: f.expenseRatio != null ? Number(f.expenseRatio) : null,
        aumCrores: f.aumCrores != null ? Number(f.aumCrores) : null,
        sebiRiskometer: f.sebiRiskometer,
      }));

      setTotalFunds(data.pagination?.total || mapped.length);
      setFunds((prev) => {
        if (replace) return mapped;
        const existingIds = new Set(prev.map((f) => f.id));
        const newOnes = mapped.filter((f) => !existingIds.has(f.id));
        return [...prev, ...newOnes];
      });
      setPage(pageToLoad);
      setHasMore(Boolean(data.pagination?.hasMore));
    } catch (err: any) {
      if (err.name !== "AbortError") {
        console.error("Fund fetch error:", err);
        if (replace) setError("Couldn't load funds. Pull down to retry.");
      }
    } finally {
      if (myRequestId === requestId.current) {
        setLoading(false);
        setLoadingMore(false);
      }
    }
  }

  function handleLoadMore() {
    if (loadingMoreRef.current || loading || !hasMore) return;
    loadingMoreRef.current = true;
    fetchFunds(page + 1, undefined, false).finally(() => {
      loadingMoreRef.current = false;
    });
  }

  return (
    <View style={{ flex: 1, backgroundColor: C.bgBottom }}>
      <AuthBackground />
      <SafeAreaView style={{ flex: 1 }} edges={["top", "bottom"]}>
        {/* Header */}
        <View style={{ paddingHorizontal: 20, paddingTop: 14, paddingBottom: 14 }}>
          <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
            <Text style={{ color: "#fff", fontSize: 26, fontWeight: "800", letterSpacing: -0.5 }}>
              Explore Funds
            </Text>
            <View
              style={{
                flexDirection: "row",
                alignItems: "center",
                gap: 5,
                backgroundColor: "rgba(124, 58, 237, 0.16)",
                borderWidth: 1,
                borderColor: "rgba(124, 58, 237, 0.35)",
                paddingHorizontal: 11,
                paddingVertical: 5,
                borderRadius: 999,
              }}
            >
              <Sparkles size={12} color={C.pink} />
              <Text style={{ color: "#c084fc", fontSize: 11, fontWeight: "700" }}>
                {totalFunds > 0 ? `${totalFunds} Live Schemes` : "Zero Commission"}
              </Text>
            </View>
          </View>
          <Text style={{ color: C.textMuted, fontSize: 13, lineHeight: 18 }}>
            Discover top-performing schemes across all market horizons.
          </Text>
        </View>

        {/* Search — glass pill with clear button */}
        <View style={{ paddingHorizontal: 20, marginBottom: 12 }}>
          <View
            style={{
              flexDirection: "row",
              alignItems: "center",
              backgroundColor: C.input,
              borderWidth: 1,
              borderColor: C.inputBorder,
              borderRadius: 16,
              paddingHorizontal: 16,
              height: 48,
            }}
          >
            <Search size={18} color={C.textMuted} />
            <TextInput
              value={query}
              onChangeText={setQuery}
              placeholder="Search funds, AMC, category..."
              placeholderTextColor={C.textFaint}
              style={{ color: "#fff", flex: 1, fontSize: 14, marginLeft: 10 }}
            />
            {query.length > 0 ? (
              <TouchableOpacity
                onPress={() => setQuery("")}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <X size={16} color={C.textMuted} />
              </TouchableOpacity>
            ) : null}
          </View>
        </View>

        {/* Tenure Horizon Filters */}
        <View style={{ marginBottom: 10 }}>
          <FlatList
            horizontal
            data={TENURE_FILTERS}
            keyExtractor={(item) => item.id}
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={{ paddingHorizontal: 20, gap: 8 }}
            renderItem={({ item }) => {
              const active = item.id === tenure;
              return (
                <TouchableOpacity onPress={() => setTenure(item.id)} activeOpacity={0.85}>
                  {active ? (
                    <LinearGradient
                      colors={[C.pink, C.violet]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 1, y: 1 }}
                      style={{ paddingHorizontal: 16, paddingVertical: 8, borderRadius: 999 }}
                    >
                      <Text style={{ color: "#fff", fontSize: 12, fontWeight: "700" }}>{item.label}</Text>
                    </LinearGradient>
                  ) : (
                    <View
                      style={{
                        backgroundColor: C.input,
                        borderWidth: 1,
                        borderColor: C.inputBorder,
                        paddingHorizontal: 16,
                        paddingVertical: 8,
                        borderRadius: 999,
                      }}
                    >
                      <Text style={{ color: C.textMuted, fontSize: 12, fontWeight: "500" }}>{item.label}</Text>
                    </View>
                  )}
                </TouchableOpacity>
              );
            }}
          />
        </View>

        {/* Asset Category Filters */}
        <View style={{ marginBottom: 12 }}>
          <FlatList
            horizontal
            data={CATEGORIES}
            keyExtractor={(item) => item}
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={{ paddingHorizontal: 20, gap: 8 }}
            renderItem={({ item }) => {
              const active = item === category;
              return (
                <TouchableOpacity onPress={() => setCategory(item)} activeOpacity={0.85}>
                  {active ? (
                    <View
                      style={{
                        backgroundColor: "rgba(255, 255, 255, 0.16)",
                        borderWidth: 1,
                        borderColor: "rgba(255, 255, 255, 0.3)",
                        paddingHorizontal: 16,
                        paddingVertical: 7,
                        borderRadius: 999,
                      }}
                    >
                      <Text style={{ color: "#fff", fontSize: 12, fontWeight: "700" }}>{item}</Text>
                    </View>
                  ) : (
                    <View
                      style={{
                        backgroundColor: "rgba(255, 255, 255, 0.03)",
                        borderWidth: 1,
                        borderColor: "rgba(255, 255, 255, 0.06)",
                        paddingHorizontal: 16,
                        paddingVertical: 7,
                        borderRadius: 999,
                      }}
                    >
                      <Text style={{ color: C.textFaint, fontSize: 12, fontWeight: "500" }}>{item}</Text>
                    </View>
                  )}
                </TouchableOpacity>
              );
            }}
          />
        </View>

        {/* Fund list */}
        <View style={{ flex: 1 }}>
          {loading && funds.length === 0 ? (
            <View style={{ flex: 1, alignItems: "center", justifyContent: "center", paddingVertical: 64 }}>
              <ActivityIndicator color={C.pink} />
            </View>
          ) : error ? (
            <View style={{ flex: 1, alignItems: "center", justifyContent: "center", paddingVertical: 64, paddingHorizontal: 24 }}>
              <Text style={{ color: C.textMuted, fontSize: 13, textAlign: "center" }}>{error}</Text>
            </View>
          ) : (
            <FlatList
              data={funds}
              keyExtractor={(item) => item.id}
              style={{ flex: 1 }}
              contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 16 }}
              showsVerticalScrollIndicator={false}
              onRefresh={() => fetchFunds(1, undefined, true)}
              refreshing={loading}
              onEndReached={handleLoadMore}
              onEndReachedThreshold={0.4}
              removeClippedSubviews={true}
              maxToRenderPerBatch={10}
              windowSize={7}
              initialNumToRender={10}
              renderItem={({ item }) => (
                <FundCard
                  fund={item}
                  onPressCard={(f) => setSelectedFund(f)}
                  onInvest={(f) => handleInvest(f)}
                />
              )}
              ListFooterComponent={
                loadingMore ? (
                  <View style={{ paddingVertical: 16, alignItems: "center" }}>
                    <ActivityIndicator color={C.pink} size="small" />
                  </View>
                ) : null
              }
              ListEmptyComponent={
                <View style={{ alignItems: "center", justifyContent: "center", paddingVertical: 64, paddingHorizontal: 24 }}>
                  <Text style={{ color: "#fff", fontWeight: "600", fontSize: 15, marginBottom: 4 }}>No schemes found</Text>
                  <Text style={{ color: C.textMuted, fontSize: 13, textAlign: "center" }}>
                    Try selecting a different horizon or clearing your search.
                  </Text>
                </View>
              }
            />
          )}
        </View>

        {/* Interactive Fund Details Sheet */}
        <FundDetailsModal
          fund={selectedFund}
          onClose={() => setSelectedFund(null)}
          onInvest={(f) => handleInvest(f)}
        />

        {/* Bottom Navigation */}
        <BottomNav tabs={tabs} paddingBottom={10} />
      </SafeAreaView>
    </View>
  );
}