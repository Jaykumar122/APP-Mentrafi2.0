import AsyncStorage from "@react-native-async-storage/async-storage";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import {
  AlertCircle,
  ArrowDownCircle,
  ArrowLeft,
  ArrowLeftRight,
  Calendar,
  ChevronRight,
  Eye,
  EyeOff,
  History,
  Home,
  Layers,
  PieChart as PieChartIcon,
  PlusCircle,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp,
  User,
  Wallet,
  X,
} from "lucide-react-native";
import React, { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Modal,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import Svg, { Circle } from "react-native-svg";
import { API_URL } from "../../utils/api";
import { AuthBackground, C, depthShadow } from "../(auth)/login";
import BottomNav from "./BottomNav";

const GREEN = "#4ade80";
const RED = "#ff6b81";
const CYAN = "#00f2fe";
const VIOLET = "#9d4edd";
const PINK = "#ff4f81";
const AMBER = "#fbbf24";

type AssetClass = "Equity" | "Debt" | "Hybrid";

type Holding = {
  id: string;
  schemeCode: number;
  name: string;
  category: string;
  assetClass: AssetClass;
  units: number;
  avgPurchaseNav: number;
  currentNav: number;
  value: number;
  investedAmount: number;
  gainAmount: number;
  gainPercent: number;
  dayChangeAmount: number;
  dayChangePercent: number;
  change: number; // 1-year return %
  portfolioWeightPercent?: number;
};

type Allocation = {
  label: AssetClass;
  value: number;
  percent: number;
};

type PortfolioData = {
  totalValue: number;
  investedValue: number;
  gainValue: number;
  gainPercent: number;
  todayChange: number;
  todayChangePercent: number;
  xirr: number;
  totalHoldingsCount?: number;
  holdings: Holding[];
  allocations: Allocation[];
};

const ASSET_COLORS: Record<AssetClass, string> = {
  Equity: "#f43f5e",
  Hybrid: "#06b6d4",
  Debt: "#8b5cf6",
};

const FILTERS: ("All" | AssetClass)[] = ["All", "Equity", "Debt", "Hybrid"];

// Indian currency formatting: 125000 -> "1.25L", 15000000 -> "1.50Cr"
function formatCompact(value: number, decimals = 2): string {
  const abs = Math.abs(value);
  if (abs >= 1e7) return `${(value / 1e7).toFixed(decimals)}Cr`;
  if (abs >= 1e5) return `${(value / 1e5).toFixed(decimals)}L`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(decimals)}K`;
  return value.toFixed(decimals);
}

function formatCurrency(val: number): string {
  return `₹${val.toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

// Interactive SVG Donut Chart with glowing center
function AllocationDonut({
  allocations,
  totalValue,
  size = 148,
  strokeWidth = 16,
  showBalance = true,
}: {
  allocations: { label: AssetClass; percent: number; value: number }[];
  totalValue: number;
  size?: number;
  strokeWidth?: number;
  showBalance?: boolean;
}) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  let offsetAccum = 0;

  const validAllocations = allocations.filter((a) => a.percent > 0);

  return (
    <View style={{ width: size, height: size, alignItems: "center", justifyContent: "center" }}>
      <Svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <Circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={strokeWidth}
          fill="none"
        />
        {validAllocations.map((a) => {
          const segmentLength = Math.max(2, (a.percent / 100) * circumference);
          const circle = (
            <Circle
              key={a.label}
              cx={size / 2}
              cy={size / 2}
              r={radius}
              stroke={ASSET_COLORS[a.label] || PINK}
              strokeWidth={strokeWidth}
              fill="none"
              strokeDasharray={`${segmentLength} ${circumference}`}
              strokeDashoffset={-offsetAccum}
              strokeLinecap="round"
              rotation={-90}
              origin={`${size / 2}, ${size / 2}`}
            />
          );
          offsetAccum += segmentLength;
          return circle;
        })}
      </Svg>
      <View style={{ position: "absolute", alignItems: "center", justifyContent: "center" }}>
        <Text style={{ color: C.textFaint, fontSize: 9.5, fontWeight: "700", letterSpacing: 0.6 }}>
          PORTFOLIO
        </Text>
        <Text style={{ color: "#fff", fontSize: 13, fontWeight: "800", marginTop: 2 }}>
          {showBalance ? `₹${formatCompact(totalValue, 1)}` : "••••"}
        </Text>
      </View>
    </View>
  );
}

export default function PortfolioScreen() {
  const router = useRouter();

  // State
  const [portfolio, setPortfolio] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const [showBalance, setShowBalance] = useState(true);
  const [filter, setFilter] = useState<"All" | AssetClass>("All");
  const [searchQuery, setSearchQuery] = useState("");

  // Holding Detail Modal
  const [selectedHolding, setSelectedHolding] = useState<Holding | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);

  const tabs = [
    { name: "Home", icon: Home, route: "/(tabs)/home" },
    { name: "Explore", icon: Search, route: "/(tabs)/explore" },
    { name: "Portfolio", icon: PieChartIcon, route: "/(tabs)/portfolio" },
    { name: "AI Advisor", icon: Sparkles, route: "/(tabs)/ai-advisor" },
    { name: "Profile", icon: User, route: "/(tabs)/profile" },
  ];

  async function fetchPortfolio() {
    try {
      setErrorMessage("");
      const token = await AsyncStorage.getItem("userToken");
      if (!token) {
        router.replace("/(auth)/login" as any);
        return;
      }

      const res = await fetch(`${API_URL}/api/portfolio`, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data = await res.json();
      setPortfolio(data);
    } catch (err: any) {
      console.error("Portfolio fetch error:", err);
      setErrorMessage("Could not load portfolio data. Tap to retry.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    fetchPortfolio();
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    fetchPortfolio();
  };

  const handleBack = () => {
    if (typeof router.canGoBack === "function" && router.canGoBack()) {
      router.back();
    } else {
      router.replace("/(tabs)/home" as any);
    }
  };

  // Derived metrics
  const allocations = portfolio?.allocations ?? [];
  const totalValue = portfolio?.totalValue ?? 0;
  const investedValue = portfolio?.investedValue ?? 0;
  const gainValue = portfolio?.gainValue ?? 0;
  const gainPercent = portfolio?.gainPercent ?? 0;
  const todayChange = portfolio?.todayChange ?? 0;
  const todayChangePercent = portfolio?.todayChangePercent ?? 0;
  const xirr = portfolio?.xirr ?? 0;

  const isTodayPositive = todayChangePercent >= 0;
  const isGainPositive = gainValue >= 0;

  // Filtered & Searched Holdings
  const filteredHoldings = useMemo(() => {
    let list = portfolio?.holdings ?? [];
    if (filter !== "All") {
      list = list.filter((h) => h.assetClass === filter);
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      list = list.filter(
        (h) =>
          h.name.toLowerCase().includes(q) ||
          h.category.toLowerCase().includes(q) ||
          String(h.schemeCode).includes(q)
      );
    }
    return list;
  }, [portfolio, filter, searchQuery]);

  // AI Diagnostic Generator
  const aiDiagnostic = useMemo(() => {
    if (!portfolio || portfolio.holdings.length === 0) {
      return {
        title: "No Active Direct Holdings",
        desc: "Start an automated SIP in a low-cost Nifty Index fund to begin compounding.",
        prompt: "Recommend a beginner-friendly SIP portfolio of 2-3 diversified direct mutual funds for wealth creation.",
      };
    }
    const equityAlloc = allocations.find((a) => a.label === "Equity")?.percent ?? 0;
    const debtAlloc = allocations.find((a) => a.label === "Debt")?.percent ?? 0;
    const hybridAlloc = allocations.find((a) => a.label === "Hybrid")?.percent ?? 0;

    if (equityAlloc >= 85) {
      return {
        title: "High Growth Equity Matrix",
        desc: "Your capital is heavily growth-oriented. Consider adding 15-20% debt or arbitrage funds to smooth out market volatility.",
        prompt: `My portfolio is currently ${equityAlloc}% Equity and ${debtAlloc}% Debt. Please analyze and propose an optimal rebalancing strategy for steady risk-adjusted returns.`,
      };
    }
    if (debtAlloc >= 60) {
      return {
        title: "Capital Shield Conservative Mix",
        desc: "High capital safety. To beat inflation comfortably, consider shifting 25% into large-cap or balanced advantage funds.",
        prompt: `My portfolio is conservative with ${debtAlloc}% Debt. How can I prudently add Equity exposure to optimize post-tax returns?`,
      };
    }
    return {
      title: "Well-Balanced Multi-Asset Portfolio",
      desc: `Diversified across Equity (${equityAlloc}%), Debt (${debtAlloc}%), and Hybrid (${hybridAlloc}%). Aligned with standard fiduciary asset allocation.`,
      prompt: "Review my multi-asset mutual fund portfolio and check if my scheme selections have excessive sector overlap.",
    };
  }, [portfolio, allocations]);

  // Open detail modal for a holding
  const handleOpenDetail = (holding: Holding) => {
    setSelectedHolding(holding);
    setShowDetailModal(true);
  };

  return (
    <View style={{ flex: 1, backgroundColor: C.bgBottom }}>
      <AuthBackground />
      <SafeAreaView style={{ flex: 1 }} edges={["top"]}>
        {/* ─────────────────────────────────────────────────────────── */}
        {/* 1. TOP APP BAR WITH BACK ARROW & ACTIONS */}
        {/* ─────────────────────────────────────────────────────────── */}
        <View
          style={{
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "space-between",
            paddingHorizontal: 16,
            paddingVertical: 10,
            borderBottomWidth: 1,
            borderBottomColor: "rgba(255, 255, 255, 0.06)",
          }}
        >
          {/* Back Arrow */}
          <TouchableOpacity
            onPress={handleBack}
            activeOpacity={0.7}
            hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
            style={{
              width: 40,
              height: 40,
              borderRadius: 20,
              backgroundColor: "rgba(255, 255, 255, 0.08)",
              borderWidth: 1,
              borderColor: "rgba(255, 255, 255, 0.14)",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <ArrowLeft size={20} color="#fff" />
          </TouchableOpacity>

          {/* Center Title */}
          <View style={{ alignItems: "center" }}>
            <Text
              style={{
                color: "#fff",
                fontSize: 18,
                fontWeight: "700",
                letterSpacing: 0.3,
              }}
            >
              My Portfolio
            </Text>
            <View style={{ flexDirection: "row", alignItems: "center", gap: 4, marginTop: 1 }}>
              <View style={{ width: 5, height: 5, borderRadius: 2.5, backgroundColor: GREEN }} />
              <Text style={{ color: C.textFaint, fontSize: 10.5, fontWeight: "500" }}>
                SEBI Direct Verified
              </Text>
            </View>
          </View>

          {/* Right Action Buttons */}
          <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
            <TouchableOpacity
              onPress={onRefresh}
              activeOpacity={0.7}
              hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
              style={{
                width: 40,
                height: 40,
                borderRadius: 20,
                backgroundColor: "rgba(255, 255, 255, 0.08)",
                borderWidth: 1,
                borderColor: "rgba(255, 255, 255, 0.14)",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <RefreshCw size={17} color={CYAN} />
            </TouchableOpacity>

            <TouchableOpacity
              onPress={() => router.push("/(tabs)/transactions" as any)}
              activeOpacity={0.7}
              hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
              style={{
                width: 40,
                height: 40,
                borderRadius: 20,
                backgroundColor: "rgba(255, 255, 255, 0.08)",
                borderWidth: 1,
                borderColor: "rgba(255, 255, 255, 0.14)",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <History size={17} color={PINK} />
            </TouchableOpacity>

            <TouchableOpacity
              onPress={() => setShowBalance((s) => !s)}
              activeOpacity={0.7}
              hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
              style={{
                width: 40,
                height: 40,
                borderRadius: 20,
                backgroundColor: "rgba(255, 255, 255, 0.08)",
                borderWidth: 1,
                borderColor: "rgba(255, 255, 255, 0.14)",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {showBalance ? <Eye size={17} color="#fff" /> : <EyeOff size={17} color={C.textMuted} />}
            </TouchableOpacity>
          </View>
        </View>

        {/* ─────────────────────────────────────────────────────────── */}
        {/* MAIN SCROLL VIEW WITH PULL-TO-REFRESH */}
        {/* ─────────────────────────────────────────────────────────── */}
        {loading && !refreshing ? (
          <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
            <ActivityIndicator color={CYAN} size="large" />
            <Text style={{ color: C.textMuted, fontSize: 13, marginTop: 12 }}>
              Connecting to live database...
            </Text>
          </View>
        ) : (
          <ScrollView
            contentContainerStyle={{ paddingBottom: 60 }}
            refreshControl={
              <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={CYAN} />
            }
            showsVerticalScrollIndicator={false}
          >
            {/* Error banner if any */}
            {errorMessage ? (
              <TouchableOpacity
                onPress={fetchPortfolio}
                activeOpacity={0.8}
                style={{
                  marginHorizontal: 16,
                  marginTop: 12,
                  padding: 12,
                  borderRadius: 14,
                  backgroundColor: "rgba(255, 107, 129, 0.12)",
                  borderWidth: 1,
                  borderColor: "rgba(255, 107, 129, 0.3)",
                  flexDirection: "row",
                  alignItems: "center",
                  gap: 10,
                }}
              >
                <AlertCircle size={18} color={RED} />
                <Text style={{ color: "#fff", fontSize: 12.5, flex: 1 }}>{errorMessage}</Text>
                <RefreshCw size={14} color={RED} />
              </TouchableOpacity>
            ) : null}

            {/* ───────────────────────────────────────────────────────── */}
            {/* 2. HERO NET WORTH BENTO CARD */}
            {/* ───────────────────────────────────────────────────────── */}
            <View style={{ paddingHorizontal: 16, marginTop: 14 }}>
              <LinearGradient
                colors={["#121324", "#090a14"]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={{
                  borderRadius: 26,
                  borderWidth: 1,
                  borderColor: "rgba(0, 242, 254, 0.25)",
                  padding: 22,
                  ...depthShadow("lg"),
                }}
              >
                {/* Header badge */}
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "space-between",
                    marginBottom: 12,
                  }}
                >
                  <View
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      gap: 6,
                      backgroundColor: "rgba(0, 242, 254, 0.12)",
                      paddingHorizontal: 10,
                      paddingVertical: 5,
                      borderRadius: 12,
                      borderWidth: 1,
                      borderColor: "rgba(0, 242, 254, 0.3)",
                    }}
                  >
                    <Wallet size={13} color={CYAN} />
                    <Text
                      style={{
                        color: CYAN,
                        fontSize: 10.5,
                        fontWeight: "700",
                        letterSpacing: 0.8,
                      }}
                    >
                      TOTAL PORTFOLIO NET WORTH
                    </Text>
                  </View>

                  <View
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      gap: 4,
                      backgroundColor: "rgba(74, 222, 128, 0.12)",
                      paddingHorizontal: 8,
                      paddingVertical: 4,
                      borderRadius: 10,
                    }}
                  >
                    <ShieldCheck size={12} color={GREEN} />
                    <Text style={{ color: GREEN, fontSize: 10.5, fontWeight: "600" }}>Direct 0%</Text>
                  </View>
                </View>

                {/* Big Net Worth Value */}
                <Text
                  style={{
                    color: "#fff",
                    fontSize: 34,
                    fontWeight: "800",
                    letterSpacing: -0.5,
                    marginBottom: 8,
                  }}
                >
                  {showBalance ? formatCurrency(totalValue) : "₹••••••••"}
                </Text>

                {/* 1-Day Return Badge */}
                <View style={{ flexDirection: "row", alignItems: "center", gap: 8, marginBottom: 18 }}>
                  <View
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      gap: 4,
                      backgroundColor: isTodayPositive
                        ? "rgba(74, 222, 128, 0.15)"
                        : "rgba(255, 107, 129, 0.15)",
                      paddingHorizontal: 10,
                      paddingVertical: 4,
                      borderRadius: 999,
                      borderWidth: 1,
                      borderColor: isTodayPositive
                        ? "rgba(74, 222, 128, 0.3)"
                        : "rgba(255, 107, 129, 0.3)",
                    }}
                  >
                    {isTodayPositive ? (
                      <TrendingUp size={13} color={GREEN} />
                    ) : (
                      <TrendingDown size={13} color={RED} />
                    )}
                    <Text
                      style={{
                        color: isTodayPositive ? GREEN : RED,
                        fontSize: 12,
                        fontWeight: "700",
                      }}
                    >
                      {isTodayPositive ? "+" : ""}
                      {todayChangePercent.toFixed(2)}% today
                    </Text>
                  </View>
                  <Text style={{ color: C.textFaint, fontSize: 12, fontWeight: "500" }}>
                    {showBalance
                      ? `${isTodayPositive ? "+" : "-"}₹${Math.abs(todayChange).toFixed(2)}`
                      : "••••"}
                  </Text>
                </View>

                {/* Sub-grid metrics */}
                <View
                  style={{
                    flexDirection: "row",
                    justifyContent: "space-between",
                    paddingTop: 16,
                    borderTopWidth: 1,
                    borderTopColor: "rgba(255, 255, 255, 0.08)",
                  }}
                >
                  <View>
                    <Text style={{ color: C.textFaint, fontSize: 11, fontWeight: "500", marginBottom: 3 }}>
                      Total Invested
                    </Text>
                    <Text style={{ color: "#fff", fontWeight: "700", fontSize: 15 }}>
                      {showBalance ? formatCurrency(investedValue) : "••••"}
                    </Text>
                  </View>

                  <View>
                    <Text style={{ color: C.textFaint, fontSize: 11, fontWeight: "500", marginBottom: 3 }}>
                      Overall Gain
                    </Text>
                    <Text
                      style={{
                        color: isGainPositive ? GREEN : RED,
                        fontWeight: "700",
                        fontSize: 15,
                      }}
                    >
                      {showBalance
                        ? `${isGainPositive ? "+" : "-"}₹${formatCompact(Math.abs(gainValue), 1)} (${isGainPositive ? "+" : ""}${gainPercent.toFixed(1)}%)`
                        : "••••"}
                    </Text>
                  </View>

                  <View style={{ alignItems: "flex-end" }}>
                    <Text style={{ color: C.textFaint, fontSize: 11, fontWeight: "500", marginBottom: 3 }}>
                      Annualized XIRR
                    </Text>
                    <Text style={{ color: CYAN, fontWeight: "700", fontSize: 15 }}>
                      {showBalance ? `${xirr.toFixed(1)}%` : "••••"}
                    </Text>
                  </View>
                </View>
              </LinearGradient>
            </View>

            {/* ───────────────────────────────────────────────────────── */}
            {/* 3. QUICK ACTION DOCK */}
            {/* ───────────────────────────────────────────────────────── */}
            <View style={{ marginTop: 18 }}>
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={{ paddingHorizontal: 16, gap: 10 }}
              >
                {/* Invest More */}
                <TouchableOpacity
                  onPress={() => router.push("/(tabs)/invest" as any)}
                  activeOpacity={0.8}
                  style={{
                    backgroundColor: "rgba(255, 79, 129, 0.12)",
                    borderWidth: 1,
                    borderColor: "rgba(255, 79, 129, 0.35)",
                    borderRadius: 16,
                    paddingHorizontal: 16,
                    paddingVertical: 12,
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  <PlusCircle size={16} color={PINK} />
                  <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }}>Invest More</Text>
                </TouchableOpacity>

                {/* Redeem */}
                <TouchableOpacity
                  onPress={() => router.push("/(tabs)/redeem" as any)}
                  activeOpacity={0.8}
                  style={{
                    backgroundColor: C.card,
                    borderWidth: 1,
                    borderColor: C.cardEdge,
                    borderRadius: 16,
                    paddingHorizontal: 16,
                    paddingVertical: 12,
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  <ArrowDownCircle size={16} color={RED} />
                  <Text style={{ color: "#fff", fontSize: 13, fontWeight: "600" }}>Redeem</Text>
                </TouchableOpacity>

                {/* Switch */}
                <TouchableOpacity
                  onPress={() => router.push("/(tabs)/switch" as any)}
                  activeOpacity={0.8}
                  style={{
                    backgroundColor: C.card,
                    borderWidth: 1,
                    borderColor: C.cardEdge,
                    borderRadius: 16,
                    paddingHorizontal: 16,
                    paddingVertical: 12,
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  <ArrowLeftRight size={16} color={CYAN} />
                  <Text style={{ color: "#fff", fontSize: 13, fontWeight: "600" }}>Switch</Text>
                </TouchableOpacity>

                {/* SIP Dashboard */}
                <TouchableOpacity
                  onPress={() => router.push("/(tabs)/sip-dashboard" as any)}
                  activeOpacity={0.8}
                  style={{
                    backgroundColor: C.card,
                    borderWidth: 1,
                    borderColor: C.cardEdge,
                    borderRadius: 16,
                    paddingHorizontal: 16,
                    paddingVertical: 12,
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  <Calendar size={16} color={VIOLET} />
                  <Text style={{ color: "#fff", fontSize: 13, fontWeight: "600" }}>My SIPs</Text>
                </TouchableOpacity>

                {/* Calculator */}
                <TouchableOpacity
                  onPress={() => router.push("/(tabs)/sip-calculator" as any)}
                  activeOpacity={0.8}
                  style={{
                    backgroundColor: C.card,
                    borderWidth: 1,
                    borderColor: C.cardEdge,
                    borderRadius: 16,
                    paddingHorizontal: 16,
                    paddingVertical: 12,
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  <PieChartIcon size={16} color={AMBER} />
                  <Text style={{ color: "#fff", fontSize: 13, fontWeight: "600" }}>Calculator</Text>
                </TouchableOpacity>
              </ScrollView>
            </View>

            {/* ───────────────────────────────────────────────────────── */}
            {/* 4. ASSET ALLOCATION BENTO */}
            {/* ───────────────────────────────────────────────────────── */}
            <View style={{ paddingHorizontal: 16, marginTop: 22 }}>
              <View
                style={{
                  backgroundColor: C.card,
                  borderWidth: 1,
                  borderColor: C.cardEdge,
                  borderRadius: 24,
                  padding: 20,
                  ...depthShadow("md"),
                }}
              >
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "space-between",
                    marginBottom: 16,
                  }}
                >
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                    <Layers size={18} color={CYAN} />
                    <Text style={{ color: "#fff", fontWeight: "700", fontSize: 16 }}>
                      Asset Allocation
                    </Text>
                  </View>
                  <Text style={{ color: C.textFaint, fontSize: 11.5 }}>
                    {allocations.length} asset class{allocations.length !== 1 ? "es" : ""}
                  </Text>
                </View>

                {allocations.length > 0 ? (
                  <View style={{ flexDirection: "row", alignItems: "center" }}>
                    <AllocationDonut
                      allocations={allocations}
                      totalValue={totalValue}
                      showBalance={showBalance}
                    />

                    <View style={{ flex: 1, marginLeft: 16, gap: 10 }}>
                      {allocations.map((a) => {
                        const color = ASSET_COLORS[a.label] || PINK;
                        return (
                          <View key={a.label}>
                            <View
                              style={{
                                flexDirection: "row",
                                alignItems: "center",
                                justifyContent: "space-between",
                                marginBottom: 4,
                              }}
                            >
                              <View style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
                                <View
                                  style={{
                                    width: 8,
                                    height: 8,
                                    borderRadius: 4,
                                    backgroundColor: color,
                                  }}
                                />
                                <Text style={{ color: "#fff", fontSize: 13, fontWeight: "600" }}>
                                  {a.label}
                                </Text>
                              </View>
                              <View style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
                                <Text style={{ color: C.textMuted, fontSize: 12 }}>
                                  {showBalance ? `₹${formatCompact(a.value, 1)}` : "••••"}
                                </Text>
                                <Text style={{ color: "#fff", fontSize: 12.5, fontWeight: "700" }}>
                                  {a.percent}%
                                </Text>
                              </View>
                            </View>

                            {/* Progress bar */}
                            <View
                              style={{
                                height: 5,
                                borderRadius: 2.5,
                                backgroundColor: "rgba(255,255,255,0.08)",
                                overflow: "hidden",
                              }}
                            >
                              <View
                                style={{
                                  width: `${Math.min(100, Math.max(0, a.percent))}%`,
                                  height: "100%",
                                  backgroundColor: color,
                                  borderRadius: 2.5,
                                }}
                              />
                            </View>
                          </View>
                        );
                      })}
                    </View>
                  </View>
                ) : (
                  <View style={{ paddingVertical: 18, alignItems: "center" }}>
                    <Text style={{ color: C.textMuted, fontSize: 13 }}>
                      No asset allocation data yet.
                    </Text>
                  </View>
                )}
              </View>
            </View>

            {/* ───────────────────────────────────────────────────────── */}
            {/* 5. SMART AI PORTFOLIO DIAGNOSTIC BENTO */}
            {/* ───────────────────────────────────────────────────────── */}
            <View style={{ paddingHorizontal: 16, marginTop: 18 }}>
              <LinearGradient
                colors={["rgba(157, 78, 221, 0.15)", "rgba(0, 242, 254, 0.08)"]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={{
                  borderRadius: 22,
                  borderWidth: 1,
                  borderColor: "rgba(157, 78, 221, 0.35)",
                  padding: 18,
                  ...depthShadow("sm"),
                }}
              >
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "space-between",
                    marginBottom: 8,
                  }}
                >
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 7 }}>
                    <Sparkles size={16} color={CYAN} />
                    <Text style={{ color: "#fff", fontWeight: "700", fontSize: 14 }}>
                      AI Fiduciary Diagnostic
                    </Text>
                  </View>
                  <View
                    style={{
                      backgroundColor: "rgba(157, 78, 221, 0.25)",
                      paddingHorizontal: 8,
                      paddingVertical: 3,
                      borderRadius: 8,
                    }}
                  >
                    <Text style={{ color: VIOLET, fontSize: 10, fontWeight: "700" }}>
                      TRI-NETWORK
                    </Text>
                  </View>
                </View>

                <Text style={{ color: "#fff", fontSize: 13.5, fontWeight: "600", marginBottom: 4 }}>
                  {aiDiagnostic.title}
                </Text>
                <Text style={{ color: C.textMuted, fontSize: 12, lineHeight: 18, marginBottom: 14 }}>
                  {aiDiagnostic.desc}
                </Text>

                <TouchableOpacity
                  onPress={() =>
                    router.push({
                      pathname: "/(tabs)/ai-advisor",
                      params: { prompt: aiDiagnostic.prompt },
                    } as any)
                  }
                  activeOpacity={0.8}
                  style={{
                    backgroundColor: "rgba(0, 242, 254, 0.15)",
                    borderWidth: 1,
                    borderColor: "rgba(0, 242, 254, 0.35)",
                    borderRadius: 14,
                    paddingVertical: 10,
                    paddingHorizontal: 14,
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 8,
                  }}
                >
                  <Sparkles size={14} color={CYAN} />
                  <Text style={{ color: CYAN, fontSize: 12.5, fontWeight: "700" }}>
                    Ask AI Advisor to Rebalance
                  </Text>
                  <ChevronRight size={14} color={CYAN} />
                </TouchableOpacity>
              </LinearGradient>
            </View>

            {/* ───────────────────────────────────────────────────────── */}
            {/* 6. HOLDINGS SECTION & FILTERS */}
            {/* ───────────────────────────────────────────────────────── */}
            <View style={{ marginTop: 26 }}>
              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  justifyContent: "space-between",
                  paddingHorizontal: 16,
                  marginBottom: 14,
                }}
              >
                <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                  <Text style={{ color: "#fff", fontWeight: "700", fontSize: 17 }}>
                    Holdings Breakdown
                  </Text>
                  <View
                    style={{
                      backgroundColor: "rgba(255, 255, 255, 0.1)",
                      paddingHorizontal: 8,
                      paddingVertical: 2,
                      borderRadius: 10,
                    }}
                  >
                    <Text style={{ color: "#fff", fontSize: 11, fontWeight: "700" }}>
                      {filteredHoldings.length}
                    </Text>
                  </View>
                </View>

                <Text style={{ color: C.textFaint, fontSize: 11 }}>Tap holding for details</Text>
              </View>

              {/* Search bar */}
              <View style={{ paddingHorizontal: 16, marginBottom: 12 }}>
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    backgroundColor: C.card,
                    borderWidth: 1,
                    borderColor: C.cardEdge,
                    borderRadius: 14,
                    paddingHorizontal: 12,
                    height: 42,
                  }}
                >
                  <Search size={16} color={C.textMuted} style={{ marginRight: 8 }} />
                  <TextInput
                    value={searchQuery}
                    onChangeText={setSearchQuery}
                    placeholder="Search fund name or scheme code..."
                    placeholderTextColor={C.textFaint}
                    style={{ flex: 1, color: "#fff", fontSize: 13 }}
                  />
                  {searchQuery ? (
                    <TouchableOpacity onPress={() => setSearchQuery("")}>
                      <X size={16} color={C.textMuted} />
                    </TouchableOpacity>
                  ) : null}
                </View>
              </View>

              {/* Asset Class Filter Pills */}
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={{ paddingHorizontal: 16, gap: 8, marginBottom: 16 }}
              >
                {FILTERS.map((f) => {
                  const active = f === filter;
                  return (
                    <TouchableOpacity key={f} onPress={() => setFilter(f)} activeOpacity={0.85}>
                      {active ? (
                        <LinearGradient
                          colors={[PINK, VIOLET]}
                          start={{ x: 0, y: 0 }}
                          end={{ x: 1, y: 1 }}
                          style={{
                            paddingHorizontal: 18,
                            paddingVertical: 8,
                            borderRadius: 999,
                          }}
                        >
                          <Text style={{ color: "#fff", fontSize: 12.5, fontWeight: "700" }}>
                            {f}
                          </Text>
                        </LinearGradient>
                      ) : (
                        <View
                          style={{
                            backgroundColor: C.card,
                            borderWidth: 1,
                            borderColor: C.cardEdge,
                            paddingHorizontal: 18,
                            paddingVertical: 8,
                            borderRadius: 999,
                          }}
                        >
                          <Text style={{ color: C.textMuted, fontSize: 12.5, fontWeight: "500" }}>
                            {f}
                          </Text>
                        </View>
                      )}
                    </TouchableOpacity>
                  );
                })}
              </ScrollView>

              {/* Holdings List Feed */}
              <View style={{ paddingHorizontal: 16 }}>
                {filteredHoldings.length === 0 ? (
                  /* Empty State */
                  <View
                    style={{
                      backgroundColor: C.card,
                      borderWidth: 1,
                      borderColor: C.cardEdge,
                      borderRadius: 24,
                      padding: 28,
                      alignItems: "center",
                      ...depthShadow("md"),
                    }}
                  >
                    <View
                      style={{
                        width: 56,
                        height: 56,
                        borderRadius: 28,
                        backgroundColor: "rgba(0, 242, 254, 0.12)",
                        alignItems: "center",
                        justifyContent: "center",
                        marginBottom: 16,
                      }}
                    >
                      <Sparkles size={26} color={CYAN} />
                    </View>
                    <Text
                      style={{
                        color: "#fff",
                        fontSize: 16,
                        fontWeight: "700",
                        marginBottom: 6,
                        textAlign: "center",
                      }}
                    >
                      {searchQuery
                        ? "No schemes match your search"
                        : "Start Building Your Wealth"}
                    </Text>
                    <Text
                      style={{
                        color: C.textMuted,
                        fontSize: 12.5,
                        textAlign: "center",
                        lineHeight: 18,
                        marginBottom: 20,
                      }}
                    >
                      {searchQuery
                        ? "Try searching for a different fund name or switch the asset class filter."
                        : "Invest directly in India's top 37,000+ mutual funds with 0% commission and zero lock-in."}
                    </Text>

                    <TouchableOpacity
                      onPress={() => router.push("/(tabs)/explore" as any)}
                      activeOpacity={0.85}
                      style={{
                        borderRadius: 16,
                        overflow: "hidden",
                        width: "100%",
                      }}
                    >
                      <LinearGradient
                        colors={[PINK, VIOLET]}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 0 }}
                        style={{
                          paddingVertical: 13,
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <Text style={{ color: "#fff", fontSize: 13.5, fontWeight: "700" }}>
                          Explore Direct Mutual Funds
                        </Text>
                      </LinearGradient>
                    </TouchableOpacity>
                  </View>
                ) : (
                  <View style={{ gap: 10 }}>
                    {filteredHoldings.map((item) => {
                      const positiveGain = item.gainAmount >= 0;
                      const assetColor = ASSET_COLORS[item.assetClass] || PINK;

                      return (
                        <TouchableOpacity
                          key={item.id}
                          onPress={() => handleOpenDetail(item)}
                          activeOpacity={0.8}
                          style={{
                            backgroundColor: C.card,
                            borderWidth: 1,
                            borderColor: C.cardEdge,
                            borderRadius: 20,
                            padding: 16,
                            flexDirection: "row",
                            alignItems: "center",
                            ...depthShadow("sm"),
                          }}
                        >
                          {/* Accent Pill Bar */}
                          <View
                            style={{
                              width: 4,
                              height: 54,
                              borderRadius: 2,
                              backgroundColor: positiveGain ? GREEN : RED,
                              marginRight: 12,
                            }}
                          />

                          {/* Holding Meta */}
                          <View style={{ flex: 1, marginRight: 10 }}>
                            <View
                              style={{
                                flexDirection: "row",
                                alignItems: "center",
                                gap: 6,
                                marginBottom: 4,
                              }}
                            >
                              <View
                                style={{
                                  backgroundColor: "rgba(255, 255, 255, 0.08)",
                                  paddingHorizontal: 6,
                                  paddingVertical: 2,
                                  borderRadius: 6,
                                }}
                              >
                                <Text
                                  style={{
                                    color: assetColor,
                                    fontSize: 10,
                                    fontWeight: "700",
                                  }}
                                >
                                  {item.assetClass.toUpperCase()}
                                </Text>
                              </View>
                              {item.portfolioWeightPercent ? (
                                <Text style={{ color: C.textFaint, fontSize: 10.5 }}>
                                  {item.portfolioWeightPercent}% weight
                                </Text>
                              ) : null}
                            </View>

                            <Text
                              style={{
                                color: "#fff",
                                fontWeight: "700",
                                fontSize: 13.5,
                                lineHeight: 18,
                                marginBottom: 4,
                              }}
                              numberOfLines={2}
                            >
                              {item.name}
                            </Text>

                            <Text style={{ color: C.textFaint, fontSize: 11 }}>
                              {item.units.toFixed(2)} units · NAV ₹{item.currentNav.toFixed(2)}
                            </Text>
                          </View>

                          {/* Value & Gain */}
                          <View style={{ alignItems: "flex-end", marginRight: 8 }}>
                            <Text
                              style={{
                                color: "#fff",
                                fontWeight: "700",
                                fontSize: 14,
                                marginBottom: 3,
                              }}
                            >
                              {showBalance
                                ? `₹${item.value.toLocaleString("en-IN", {
                                    minimumFractionDigits: 2,
                                  })}`
                                : "••••••"}
                            </Text>

                            <View
                              style={{
                                flexDirection: "row",
                                alignItems: "center",
                                gap: 4,
                                backgroundColor: positiveGain
                                  ? "rgba(74, 222, 128, 0.15)"
                                  : "rgba(255, 107, 129, 0.15)",
                                paddingHorizontal: 7,
                                paddingVertical: 2,
                                borderRadius: 6,
                              }}
                            >
                              {positiveGain ? (
                                <TrendingUp size={10} color={GREEN} />
                              ) : (
                                <TrendingDown size={10} color={RED} />
                              )}
                              <Text
                                style={{
                                  color: positiveGain ? GREEN : RED,
                                  fontSize: 11,
                                  fontWeight: "700",
                                }}
                              >
                                {positiveGain ? "+" : ""}
                                {item.gainPercent.toFixed(1)}%
                              </Text>
                            </View>

                            {/* 1Y return tag */}
                            {item.change ? (
                              <Text
                                style={{
                                  color: C.textFaint,
                                  fontSize: 10,
                                  marginTop: 3,
                                }}
                              >
                                1Y: {item.change > 0 ? "+" : ""}
                                {item.change.toFixed(1)}%
                              </Text>
                            ) : null}
                          </View>

                          <ChevronRight size={16} color={C.textFaint} />
                        </TouchableOpacity>
                      );
                    })}
                  </View>
                )}
              </View>
            </View>
          </ScrollView>
        )}

        {/* ─────────────────────────────────────────────────────────── */}
        {/* 7. INTERACTIVE HOLDING DETAIL MODAL */}
        {/* ─────────────────────────────────────────────────────────── */}
        {selectedHolding && (
          <Modal
            visible={showDetailModal}
            transparent
            animationType="slide"
            onRequestClose={() => setShowDetailModal(false)}
          >
            <View
              style={{
                flex: 1,
                backgroundColor: "rgba(0, 0, 0, 0.75)",
                justifyContent: "flex-end",
              }}
            >
              <View
                style={{
                  backgroundColor: "#0d0e1a",
                  borderTopLeftRadius: 30,
                  borderTopRightRadius: 30,
                  borderTopWidth: 1,
                  borderColor: "rgba(0, 242, 254, 0.3)",
                  padding: 24,
                  paddingBottom: 36,
                  maxHeight: "85%",
                }}
              >
                {/* Modal Drag Handle */}
                <View
                  style={{
                    width: 40,
                    height: 4,
                    borderRadius: 2,
                    backgroundColor: "rgba(255, 255, 255, 0.2)",
                    alignSelf: "center",
                    marginBottom: 16,
                  }}
                />

                {/* Modal Header */}
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "flex-start",
                    justifyContent: "space-between",
                    marginBottom: 16,
                  }}
                >
                  <View style={{ flex: 1, marginRight: 12 }}>
                    <View
                      style={{
                        flexDirection: "row",
                        alignItems: "center",
                        gap: 6,
                        marginBottom: 6,
                      }}
                    >
                      <View
                        style={{
                          backgroundColor: "rgba(255, 255, 255, 0.08)",
                          paddingHorizontal: 8,
                          paddingVertical: 3,
                          borderRadius: 6,
                        }}
                      >
                        <Text
                          style={{
                            color: ASSET_COLORS[selectedHolding.assetClass] || PINK,
                            fontSize: 11,
                            fontWeight: "700",
                          }}
                        >
                          {selectedHolding.assetClass.toUpperCase()}
                        </Text>
                      </View>
                      <Text style={{ color: C.textFaint, fontSize: 11 }}>
                        Scheme #{selectedHolding.schemeCode}
                      </Text>
                    </View>
                    <Text
                      style={{
                        color: "#fff",
                        fontSize: 16,
                        fontWeight: "700",
                        lineHeight: 22,
                      }}
                    >
                      {selectedHolding.name}
                    </Text>
                  </View>

                  <TouchableOpacity
                    onPress={() => setShowDetailModal(false)}
                    style={{
                      width: 36,
                      height: 36,
                      borderRadius: 18,
                      backgroundColor: "rgba(255, 255, 255, 0.08)",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <X size={18} color="#fff" />
                  </TouchableOpacity>
                </View>

                {/* Financial Stats Breakdown */}
                <View
                  style={{
                    backgroundColor: C.card,
                    borderWidth: 1,
                    borderColor: C.cardEdge,
                    borderRadius: 20,
                    padding: 16,
                    marginBottom: 20,
                    gap: 12,
                  }}
                >
                  {/* Current Value & Gain Row */}
                  <View
                    style={{
                      flexDirection: "row",
                      justifyContent: "space-between",
                      borderBottomWidth: 1,
                      borderBottomColor: "rgba(255, 255, 255, 0.06)",
                      paddingBottom: 10,
                    }}
                  >
                    <View>
                      <Text style={{ color: C.textFaint, fontSize: 11.5, marginBottom: 2 }}>
                        Current Value
                      </Text>
                      <Text style={{ color: "#fff", fontSize: 18, fontWeight: "800" }}>
                        {formatCurrency(selectedHolding.value)}
                      </Text>
                    </View>
                    <View style={{ alignItems: "flex-end" }}>
                      <Text style={{ color: C.textFaint, fontSize: 11.5, marginBottom: 2 }}>
                        Total Profit / Loss
                      </Text>
                      <Text
                        style={{
                          color: selectedHolding.gainAmount >= 0 ? GREEN : RED,
                          fontSize: 16,
                          fontWeight: "700",
                        }}
                      >
                        {selectedHolding.gainAmount >= 0 ? "+" : "-"}₹
                        {Math.abs(selectedHolding.gainAmount).toFixed(2)} (
                        {selectedHolding.gainAmount >= 0 ? "+" : ""}
                        {selectedHolding.gainPercent.toFixed(2)}%)
                      </Text>
                    </View>
                  </View>

                  {/* Units & Purchase Details */}
                  <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                    <View>
                      <Text style={{ color: C.textFaint, fontSize: 11 }}>Units Held</Text>
                      <Text style={{ color: "#fff", fontSize: 13, fontWeight: "600", marginTop: 2 }}>
                        {selectedHolding.units.toFixed(4)}
                      </Text>
                    </View>
                    <View>
                      <Text style={{ color: C.textFaint, fontSize: 11 }}>Avg Purchase NAV</Text>
                      <Text style={{ color: "#fff", fontSize: 13, fontWeight: "600", marginTop: 2 }}>
                        ₹{selectedHolding.avgPurchaseNav.toFixed(4)}
                      </Text>
                    </View>
                    <View style={{ alignItems: "flex-end" }}>
                      <Text style={{ color: C.textFaint, fontSize: 11 }}>Live NAV</Text>
                      <Text style={{ color: CYAN, fontSize: 13, fontWeight: "700", marginTop: 2 }}>
                        ₹{selectedHolding.currentNav.toFixed(4)}
                      </Text>
                    </View>
                  </View>

                  {/* Invested & 1Y Return */}
                  <View
                    style={{
                      flexDirection: "row",
                      justifyContent: "space-between",
                      borderTopWidth: 1,
                      borderTopColor: "rgba(255, 255, 255, 0.06)",
                      paddingTop: 10,
                    }}
                  >
                    <View>
                      <Text style={{ color: C.textFaint, fontSize: 11 }}>Total Invested</Text>
                      <Text style={{ color: "#fff", fontSize: 13, fontWeight: "600", marginTop: 2 }}>
                        {formatCurrency(selectedHolding.investedAmount)}
                      </Text>
                    </View>
                    <View>
                      <Text style={{ color: C.textFaint, fontSize: 11 }}>Portfolio Share</Text>
                      <Text style={{ color: PINK, fontSize: 13, fontWeight: "600", marginTop: 2 }}>
                        {selectedHolding.portfolioWeightPercent ?? "—"}%
                      </Text>
                    </View>
                    <View style={{ alignItems: "flex-end" }}>
                      <Text style={{ color: C.textFaint, fontSize: 11 }}>1-Year Momentum</Text>
                      <Text
                        style={{
                          color: selectedHolding.change >= 0 ? GREEN : RED,
                          fontSize: 13,
                          fontWeight: "700",
                          marginTop: 2,
                        }}
                      >
                        {selectedHolding.change >= 0 ? "+" : ""}
                        {selectedHolding.change.toFixed(2)}%
                      </Text>
                    </View>
                  </View>
                </View>

                {/* Direct Action Buttons */}
                <View style={{ flexDirection: "row", gap: 10, marginBottom: 12 }}>
                  <TouchableOpacity
                    onPress={() => {
                      setShowDetailModal(false);
                      router.push({
                        pathname: "/(tabs)/invest",
                        params: { schemeCode: String(selectedHolding.schemeCode) },
                      } as any);
                    }}
                    activeOpacity={0.85}
                    style={{ flex: 1 }}
                  >
                    <LinearGradient
                      colors={[PINK, VIOLET]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 1, y: 0 }}
                      style={{
                        borderRadius: 16,
                        paddingVertical: 14,
                        alignItems: "center",
                        justifyContent: "center",
                        flexDirection: "row",
                        gap: 6,
                      }}
                    >
                      <PlusCircle size={16} color="#fff" />
                      <Text style={{ color: "#fff", fontSize: 13.5, fontWeight: "700" }}>
                        Invest More
                      </Text>
                    </LinearGradient>
                  </TouchableOpacity>

                  <TouchableOpacity
                    onPress={() => {
                      setShowDetailModal(false);
                      router.push("/(tabs)/redeem" as any);
                    }}
                    activeOpacity={0.85}
                    style={{
                      flex: 1,
                      backgroundColor: "rgba(255, 107, 129, 0.12)",
                      borderWidth: 1,
                      borderColor: "rgba(255, 107, 129, 0.35)",
                      borderRadius: 16,
                      paddingVertical: 14,
                      alignItems: "center",
                      justifyContent: "center",
                      flexDirection: "row",
                      gap: 6,
                    }}
                  >
                    <ArrowDownCircle size={16} color={RED} />
                    <Text style={{ color: RED, fontSize: 13.5, fontWeight: "700" }}>
                      Redeem Units
                    </Text>
                  </TouchableOpacity>
                </View>

                {/* Switch button */}
                <TouchableOpacity
                  onPress={() => {
                    setShowDetailModal(false);
                    router.push("/(tabs)/switch" as any);
                  }}
                  activeOpacity={0.8}
                  style={{
                    backgroundColor: "rgba(255, 255, 255, 0.06)",
                    borderWidth: 1,
                    borderColor: "rgba(255, 255, 255, 0.1)",
                    borderRadius: 14,
                    paddingVertical: 12,
                    alignItems: "center",
                    justifyContent: "center",
                    flexDirection: "row",
                    gap: 6,
                  }}
                >
                  <ArrowLeftRight size={15} color={CYAN} />
                  <Text style={{ color: "#fff", fontSize: 13, fontWeight: "600" }}>
                    Switch to Another Scheme
                  </Text>
                </TouchableOpacity>
              </View>
            </View>
          </Modal>
        )}

        {/* ─────────────────────────────────────────────────────────── */}
        {/* 8. BOTTOM NAVIGATION BAR */}
        {/* ─────────────────────────────────────────────────────────── */}
        <BottomNav tabs={tabs} paddingBottom={10} />
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({});