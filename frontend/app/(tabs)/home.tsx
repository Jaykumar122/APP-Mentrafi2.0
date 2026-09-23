import AsyncStorage from "@react-native-async-storage/async-storage";
import { useRouter, useFocusEffect } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import {
  ArrowDownRight,
  ArrowUpRight,
  Bell,
  Building2,
  Calculator,
  ChevronRight,
  CreditCard,
  Eye,
  EyeOff,
  History,
  Home,
  PieChart,
  Plus,
  PlusCircle,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  Star,
  TrendingDown,
  TrendingUp,
  User,
  Wallet,
  Zap,
} from "lucide-react-native";
import React, { useCallback, useEffect, useState } from "react";
import {
  ActivityIndicator,
  Image,
  RefreshControl,
  ScrollView,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import Svg, { Circle, Defs, Ellipse, LinearGradient as SvgGrad, Path, Stop } from "react-native-svg";
import { API_URL } from "../../utils/api";
import { subscribeProfileUpdates } from "../../utils/profileEvents";
import BottomNav from "./BottomNav";
import {
  AuthBackground,
  C,
  depthShadow,
} from "../(auth)/login";

// ---------------------------------------------------------------------------
// Types — mirror the shape returned by GET /api/portfolio and GET /api/funds
// ---------------------------------------------------------------------------
type Holding = {
  id: string;
  schemeCode?: number;
  name: string;
  units: number;
  value: number;
  change: number; // 1Y return %
};

type MarketFund = {
  schemeCode?: number;
  name: string;
  category: string;
  subcategory?: string;
  fundHouse?: string;
  change: number;
  rating: number;
  tenureType?: "new" | "growth" | "established";
  tenureLabel?: string;
};

type PortfolioSummary = {
  totalValue: number;
  investedValue: number;
  gainValue: number;
  gainPercent: number;
  todayChange: number;
  todayChangePercent: number;
  xirr: number;
  holdings: Holding[];
};

// Compact Indian-style number formatting: 245680.5 -> "2.46L", 12500000 -> "1.25Cr"
function formatCompact(value: number, decimals = 2): string {
  const abs = Math.abs(value);
  if (abs === 0) return "0";
  if (abs >= 1e7) return `${(value / 1e7).toFixed(decimals)}Cr`;
  if (abs >= 1e5) return `${(value / 1e5).toFixed(decimals)}L`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(decimals)}K`;
  return Number(value).toLocaleString("en-IN", { maximumFractionDigits: decimals });
}

function formatCurrency(val: number): string {
  return `₹${val.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour >= 5 && hour < 12) return "Good morning 👋";
  if (hour >= 12 && hour < 17) return "Good afternoon 👋";
  if (hour >= 17 && hour < 22) return "Good evening 👋";
  return "Good evening 🌙";
}

function getInitials(name?: string | null): string {
  const trimmed = name?.trim();
  if (!trimmed) return "M";
  const parts = trimmed.split(/\s+/).filter(Boolean);
  if (parts.length === 1) return parts[0].substring(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

const GREEN = "#4ade80";
const RED = "#ff6b81";

function StarRow({ rating }: { rating: number }) {
  const rounded = Math.round(rating);
  return (
    <View style={{ flexDirection: "row", alignItems: "center", gap: 2 }}>
      {[1, 2, 3, 4, 5].map((i) => (
        <Star key={i} size={11} color={C.pink} fill={i <= rounded ? C.pink : "transparent"} strokeWidth={1.5} />
      ))}
    </View>
  );
}

// ---------------------------------------------------------------------------
// Glass card wrapper — Obsidian Aurora material styling
// ---------------------------------------------------------------------------
function GlassPanel({ children, style }: { children: React.ReactNode; style?: any }) {
  return (
    <View
      style={{
        backgroundColor: C.card,
        borderWidth: 1,
        borderColor: C.cardEdge,
        borderRadius: 22,
        ...depthShadow("sm"),
        ...style,
      }}
    >
      {children}
    </View>
  );
}

// ---------------------------------------------------------------------------
// Sleek Curved Sparkline for Portfolio Performance
// ---------------------------------------------------------------------------
function SparklineGraph({ isPositive }: { isPositive: boolean }) {
  return (
    <Svg width="100%" height={52} viewBox="0 0 320 52">
      <Defs>
        <SvgGrad id="sparklineGrad" x1="0" y1="0" x2="1" y2="0">
          <Stop offset="0%" stopColor={isPositive ? "#06b6d4" : "#f43f5e"} />
          <Stop offset="50%" stopColor={isPositive ? C.pink : "#fb7185"} />
          <Stop offset="100%" stopColor={isPositive ? "#4ade80" : "#ef4444"} />
        </SvgGrad>
        <SvgGrad id="sparklineFill" x1="0" y1="0" x2="0" y2="1">
          <Stop offset="0%" stopColor={isPositive ? "#4ade80" : "#f43f5e"} stopOpacity={0.22} />
          <Stop offset="100%" stopColor={isPositive ? "#4ade80" : "#f43f5e"} stopOpacity={0.0} />
        </SvgGrad>
      </Defs>

      {/* Area fill */}
      <Path
        d="M0,38 Q40,42 70,30 T140,24 T210,14 T280,8 T320,4 L320,52 L0,52 Z"
        fill="url(#sparklineFill)"
      />

      {/* Primary trend line */}
      <Path
        d="M0,38 Q40,42 70,30 T140,24 T210,14 T280,8 T320,4"
        stroke="url(#sparklineGrad)"
        strokeWidth={3}
        fill="none"
        strokeLinecap="round"
      />

      {/* Pulse dot on endpoint */}
      <Circle cx="316" cy="5" r="4" fill={isPositive ? GREEN : RED} />
      <Circle cx="316" cy="5" r="8" fill={isPositive ? GREEN : RED} opacity={0.3} />
    </Svg>
  );
}

const CURRENT_TAB = "Home";

export default function HomeScreen() {
  const router = useRouter();
  const [userName, setUserName] = useState("");
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);
  const [showBalance, setShowBalance] = useState(true);
  const [portfolio, setPortfolio] = useState<PortfolioSummary | null>(null);
  const [marketPulse, setMarketPulse] = useState<MarketFund[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadUserData = useCallback(async () => {
    try {
      const [name, cachedProfile, avatar, token] = await Promise.all([
        AsyncStorage.getItem("userName"),
        AsyncStorage.getItem("cachedUserProfile"),
        AsyncStorage.getItem("userAvatarUrl"),
        AsyncStorage.getItem("userToken"),
      ]);
      if (name) setUserName(name);
      if (avatar) {
        setAvatarUrl(avatar);
      } else if (cachedProfile) {
        try {
          const parsed = JSON.parse(cachedProfile);
          if (parsed.avatarUrl) setAvatarUrl(parsed.avatarUrl);
          if (parsed.name && !name) setUserName(parsed.name);
        } catch {}
      }

      // Background profile revalidation if token is present
      if (token) {
        fetch(`${API_URL}/api/profile`, {
          headers: { Authorization: `Bearer ${token}` },
        })
          .then((res) => (res.ok ? res.json() : null))
          .then((data) => {
            if (data?.name) {
              setUserName(data.name);
              AsyncStorage.setItem("userName", data.name);
            }
            if (data?.avatarUrl) {
              setAvatarUrl(data.avatarUrl);
              AsyncStorage.setItem("userAvatarUrl", data.avatarUrl);
            }
          })
          .catch(() => {});
      }
    } catch (err) {
      console.warn("Error loading user data in home:", err);
    }
  }, []);

  const fetchPortfolio = useCallback(async () => {
    try {
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

      if (res.status === 401) {
        await AsyncStorage.removeItem("userToken");
        router.replace("/(auth)/login" as any);
        return;
      }

      if (!res.ok) throw new Error("Failed to fetch portfolio");
      const data = await res.json();
      setPortfolio(data);
    } catch (err) {
      console.error("Portfolio fetch error:", err);
    } finally {
      setLoading(false);
    }
  }, [router]);

  const fetchMarketPulse = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/funds?limit=6`);
      if (!res.ok) return;
      const data = await res.json();
      setMarketPulse(
        (data.funds || []).map((f: any) => {
          const isNew = f.oneYearReturn == null && (f.return6m != null || f.return3m != null);
          const change = f.oneYearReturn != null ? Number(f.oneYearReturn) : Number(f.return6m ?? f.return3m ?? 0);
          const tenureLabel = isNew ? "✨ New (<1Y)" : f.fiveYearReturn != null ? "🏆 5Y+ Track" : "🌱 1-3Y";
          const tenureType = isNew ? "new" : f.fiveYearReturn != null ? "established" : "growth";
          return {
            schemeCode: f.scheme_code ?? f.schemeCode,
            name: f.name,
            category: f.category,
            subcategory: f.subcategory,
            fundHouse: f.fundHouse || "Direct Mutual Fund",
            change,
            rating: Number(f.rating ?? 4),
            tenureType,
            tenureLabel,
          };
        })
      );
    } catch (err) {
      console.error("Market pulse fetch error:", err);
    }
  }, []);

  const loadAllData = useCallback(async () => {
    await Promise.all([loadUserData(), fetchPortfolio(), fetchMarketPulse()]);
  }, [loadUserData, fetchPortfolio, fetchMarketPulse]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadAllData();
    setRefreshing(false);
  }, [loadAllData]);

  // Re-fetch automatically whenever user switches to the Home tab!
  useFocusEffect(
    useCallback(() => {
      loadAllData();
    }, [loadAllData])
  );

  useEffect(() => {
    const unsubscribe = subscribeProfileUpdates(() => {
      loadUserData();
    });
    return () => unsubscribe();
  }, [loadUserData]);

  const totalValue = portfolio?.totalValue ?? 0;
  const todayChange = portfolio?.todayChange ?? 0;
  const todayChangePercent = portfolio?.todayChangePercent ?? 0;
  const investedNum = portfolio?.investedValue ?? 0;
  const gainNum = portfolio?.gainValue ?? 0;
  const gainPctNum = portfolio?.gainPercent ?? 0;

  const investedValue = `₹${formatCompact(investedNum, 2)}`;
  const isGainZero = Math.abs(gainNum) < 0.01;
  const isGainPositive = gainNum >= 0;
  const gainSign = isGainZero ? "" : isGainPositive ? "+" : "-";
  const gainValue = `${gainSign}₹${formatCompact(Math.abs(gainNum), 2)}`;
  const gainPercentValue = `${gainSign}${Math.abs(gainPctNum).toFixed(1)}%`;
  const xirrValue = `${(portfolio?.xirr ?? 0).toFixed(1)}%`;
  const holdings = (portfolio?.holdings ?? []).slice(0, 5);
  const isTodayPositive = todayChangePercent >= 0;

  const tabs = [
    { name: "Home", icon: Home, route: "/(tabs)/home" },
    { name: "Explore", icon: Search, route: "/(tabs)/explore" },
    { name: "Portfolio", icon: PieChart, route: "/(tabs)/portfolio" },
    { name: "AI Advisor", icon: Sparkles, route: "/(tabs)/ai-advisor" },
    { name: "Profile", icon: User, route: "/(tabs)/profile" },
  ];

  const quickActions = [
    {
      icon: PlusCircle,
      label: "Invest",
      sublabel: "Zero Comm.",
      route: "/(tabs)/invest",
      colors: [C.pink, C.violet],
      glowColor: "rgba(139, 92, 246, 0.25)",
    },
    {
      icon: RefreshCw,
      label: "Auto SIP",
      sublabel: "Compound",
      route: "/(tabs)/sip-dashboard",
      colors: ["#10b981", "#059669"],
      glowColor: "rgba(16, 185, 129, 0.25)",
    },
    {
      icon: Calculator,
      label: "Calculators",
      sublabel: "Step-Up & Goals",
      route: "/(tabs)/sip-calculator",
      colors: ["#06b6d4", "#0284c7"],
      glowColor: "rgba(6, 182, 212, 0.25)",
    },
    {
      icon: History,
      label: "Passbook",
      sublabel: "Transactions",
      route: "/(tabs)/transactions",
      colors: ["#f59e0b", "#d97706"],
      glowColor: "rgba(245, 158, 11, 0.25)",
    },
  ];

  const initials = getInitials(userName || "MentraFi Investor");

  return (
    <View style={{ flex: 1, backgroundColor: C.bgBottom }}>
      <AuthBackground />

      <SafeAreaView style={{ flex: 1 }} edges={["top"]}>
        <ScrollView
          showsVerticalScrollIndicator={false}
          contentContainerStyle={{ paddingBottom: 28 }}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor={C.pink}
              colors={[C.pink, C.violet]}
            />
          }
        >
          {/* Top Bar / Header */}
          <View
            style={{
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "space-between",
              paddingHorizontal: 20,
              paddingTop: 8,
              paddingBottom: 16,
            }}
          >
            {/* User Profile Avatar & Greeting */}
            <TouchableOpacity
              activeOpacity={0.85}
              onPress={() => router.push("/(tabs)/profile" as any)}
              style={{ flexDirection: "row", alignItems: "center", gap: 12 }}
            >
              <View style={{ position: "relative" }}>
                <LinearGradient
                  colors={[C.pink, C.violet]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={{
                    width: 44,
                    height: 44,
                    borderRadius: 15,
                    alignItems: "center",
                    justifyContent: "center",
                    overflow: "hidden",
                    ...depthShadow("sm"),
                  }}
                >
                  {avatarUrl ? (
                    <Image
                      source={{
                        uri: avatarUrl.startsWith("http") ? avatarUrl : `${API_URL}${avatarUrl}`,
                      }}
                      style={{ width: "100%", height: "100%" }}
                    />
                  ) : (
                    <Text style={{ color: "#fff", fontWeight: "800", fontSize: 15 }}>
                      {initials}
                    </Text>
                  )}
                </LinearGradient>
                <View
                  style={{
                    position: "absolute",
                    bottom: -1,
                    right: -1,
                    width: 12,
                    height: 12,
                    borderRadius: 6,
                    backgroundColor: GREEN,
                    borderWidth: 2,
                    borderColor: C.bgBottom,
                  }}
                />
              </View>
              <View>
                <Text style={{ color: C.textMuted, fontSize: 12 }}>
                  {getGreeting()}
                </Text>
                <Text style={{ color: "#fff", fontWeight: "700", fontSize: 16 }}>
                  {userName || "Investor"}
                </Text>
              </View>
            </TouchableOpacity>

            {/* Right actions: Live Market pill & Notification Bell */}
            <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  gap: 5,
                  paddingHorizontal: 10,
                  paddingVertical: 5,
                  borderRadius: 999,
                  backgroundColor: "rgba(74, 222, 128, 0.10)",
                  borderWidth: 1,
                  borderColor: "rgba(74, 222, 128, 0.25)",
                }}
              >
                <View style={{ width: 6, height: 6, borderRadius: 3, backgroundColor: GREEN }} />
                <Text style={{ color: GREEN, fontWeight: "700", fontSize: 11 }}>
                  Live Market
                </Text>
              </View>

              <TouchableOpacity
                activeOpacity={0.8}
                style={{
                  width: 38,
                  height: 38,
                  borderRadius: 19,
                  backgroundColor: C.input,
                  borderWidth: 1,
                  borderColor: C.inputBorder,
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Bell size={17} color="#fff" />
                <View
                  style={{
                    position: "absolute",
                    top: 8,
                    right: 9,
                    width: 7,
                    height: 7,
                    borderRadius: 3.5,
                    backgroundColor: C.pink,
                  }}
                />
              </TouchableOpacity>
            </View>
          </View>

          {/* Master Portfolio Hero Card (Bento Style) */}
          <View style={{ paddingHorizontal: 20, marginBottom: 20 }}>
            <View
              style={{
                backgroundColor: C.card,
                borderWidth: 1,
                borderColor: C.cardEdge,
                borderRadius: 28,
                padding: 20,
                overflow: "hidden",
                ...depthShadow("lg"),
              }}
            >
              {/* Aurora ambient mesh */}
              <LinearGradient
                colors={["rgba(124, 58, 237, 0.18)", "rgba(236, 72, 153, 0.08)", "transparent"]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  right: 0,
                  height: 200,
                  borderRadius: 28,
                }}
              />

              {/* Card Header */}
              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: 10,
                }}
              >
                <View style={{ flexDirection: "row", alignItems: "center", gap: 7 }}>
                  <View style={{ width: 6, height: 6, borderRadius: 3, backgroundColor: C.pink }} />
                  <Text
                    style={{
                      color: C.pink,
                      fontSize: 11,
                      fontWeight: "800",
                      letterSpacing: 1.2,
                    }}
                  >
                    PORTFOLIO VALUATION
                  </Text>
                </View>

                <TouchableOpacity
                  onPress={() => setShowBalance((s) => !s)}
                  hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                >
                  {showBalance ? (
                    <Eye size={16} color={C.textMuted} />
                  ) : (
                    <EyeOff size={16} color={C.textMuted} />
                  )}
                </TouchableOpacity>
              </View>

              {/* Valuation & Today's Change */}
              <View style={{ marginBottom: 14 }}>
                <Text
                  style={{
                    color: "#fff",
                    fontSize: 32,
                    fontWeight: "800",
                    letterSpacing: -0.5,
                    marginBottom: 6,
                  }}
                >
                  {showBalance
                    ? `₹${totalValue.toLocaleString("en-IN", { minimumFractionDigits: 2 })}`
                    : "₹••••••••"}
                </Text>

                <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                  <View
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      gap: 4,
                      backgroundColor: isTodayPositive ? "rgba(74,222,128,0.15)" : "rgba(255,107,129,0.15)",
                      paddingHorizontal: 9,
                      paddingVertical: 3,
                      borderRadius: 999,
                      borderWidth: 1,
                      borderColor: isTodayPositive ? "rgba(74,222,128,0.3)" : "rgba(255,107,129,0.3)",
                    }}
                  >
                    {isTodayPositive ? (
                      <TrendingUp size={12} color={GREEN} />
                    ) : (
                      <TrendingDown size={12} color={RED} />
                    )}
                    <Text
                      style={{
                        color: isTodayPositive ? GREEN : RED,
                        fontSize: 11,
                        fontWeight: "700",
                      }}
                    >
                      {isTodayPositive ? "+" : ""}
                      {todayChangePercent}% today
                    </Text>
                  </View>

                  <Text style={{ color: C.textFaint, fontSize: 12 }}>
                    {isTodayPositive ? "+" : "-"}₹{Math.abs(todayChange).toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                  </Text>
                </View>
              </View>

              {/* Sparkline Graph */}
              <View style={{ marginBottom: 14 }}>
                <SparklineGraph isPositive={isTodayPositive} />
              </View>

              {/* 3-Metric Summary Strip */}
              <View
                style={{
                  flexDirection: "row",
                  justifyContent: "space-between",
                  paddingTop: 14,
                  borderTopWidth: 1,
                  borderTopColor: "rgba(255, 255, 255, 0.08)",
                  marginBottom: 16,
                }}
              >
                <View style={{ flex: 1 }}>
                  <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 3 }}>Invested</Text>
                  <Text style={{ color: "#fff", fontWeight: "700", fontSize: 14 }}>
                    {showBalance ? investedValue : "₹••••"}
                  </Text>
                </View>
                <View style={{ flex: 1, alignItems: "center" }}>
                  <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 3 }}>Total Gains</Text>
                  <Text style={{ color: isGainPositive ? GREEN : RED, fontWeight: "700", fontSize: 14 }}>
                    {showBalance ? `${gainValue} (${gainPercentValue})` : "••••"}
                  </Text>
                </View>
                <View style={{ flex: 1, alignItems: "flex-end" }}>
                  <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 3 }}>Annualized XIRR</Text>
                  <Text style={{ color: C.cyan, fontWeight: "700", fontSize: 14 }}>
                    {showBalance ? xirrValue : "••••"}
                  </Text>
                </View>
              </View>

              {/* Quick Hero Shortcuts */}
              <View style={{ flexDirection: "row", gap: 10 }}>
                <TouchableOpacity
                  activeOpacity={0.88}
                  onPress={() => router.push("/(tabs)/invest" as any)}
                  style={{ flex: 1 }}
                >
                  <LinearGradient
                    colors={[C.pink, C.violet]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={{
                      height: 44,
                      borderRadius: 14,
                      flexDirection: "row",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 6,
                    }}
                  >
                    <Plus size={16} color="#fff" strokeWidth={2.5} />
                    <Text style={{ color: "#fff", fontWeight: "700", fontSize: 13 }}>Quick Invest</Text>
                  </LinearGradient>
                </TouchableOpacity>

                <TouchableOpacity
                  activeOpacity={0.85}
                  onPress={() => router.push("/(tabs)/portfolio" as any)}
                  style={{
                    height: 44,
                    paddingHorizontal: 16,
                    borderRadius: 14,
                    backgroundColor: "rgba(255, 255, 255, 0.06)",
                    borderWidth: 1,
                    borderColor: "rgba(255, 255, 255, 0.10)",
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 6,
                  }}
                >
                  <PieChart size={16} color={C.textMuted} />
                  <Text style={{ color: "#fff", fontWeight: "600", fontSize: 13 }}>Analytics</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>

          {/* Quick Actions Grid (Bento Style) */}
          <View style={{ paddingHorizontal: 20, marginBottom: 24 }}>
            <View style={{ flexDirection: "row", justifyContent: "space-between", gap: 8 }}>
              {quickActions.map((action) => (
                <TouchableOpacity
                  key={action.label}
                  activeOpacity={0.85}
                  onPress={() => action.route && router.push(action.route as any)}
                  style={{
                    flex: 1,
                    backgroundColor: C.card,
                    borderWidth: 1,
                    borderColor: C.cardEdge,
                    borderRadius: 18,
                    paddingVertical: 14,
                    paddingHorizontal: 6,
                    alignItems: "center",
                    gap: 8,
                    ...depthShadow("sm"),
                  }}
                >
                  <LinearGradient
                    colors={action.colors as any}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={{
                      width: 42,
                      height: 42,
                      borderRadius: 14,
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <action.icon size={20} color="#fff" strokeWidth={2} />
                  </LinearGradient>
                  <View style={{ alignItems: "center" }}>
                    <Text style={{ color: "#fff", fontSize: 12, fontWeight: "700" }}>
                      {action.label}
                    </Text>
                    <Text style={{ color: C.textFaint, fontSize: 10, marginTop: 1 }}>
                      {action.sublabel}
                    </Text>
                  </View>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* MentraFi AI Copilot Feature Showcase Card */}
          <View style={{ paddingHorizontal: 20, marginBottom: 24 }}>
            <TouchableOpacity
              activeOpacity={0.9}
              onPress={() => router.push("/(tabs)/ai-advisor" as any)}
            >
              <LinearGradient
                colors={["rgba(124, 58, 237, 0.22)", "rgba(219, 39, 119, 0.15)", "rgba(6, 182, 212, 0.08)"]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={{
                  borderRadius: 22,
                  borderWidth: 1,
                  borderColor: "rgba(168, 85, 247, 0.35)",
                  padding: 18,
                  ...depthShadow("md"),
                }}
              >
                <View style={{ flexDirection: "row", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 8 }}>
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
                    <LinearGradient
                      colors={[C.pink, C.violet]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 1, y: 1 }}
                      style={{
                        width: 36,
                        height: 36,
                        borderRadius: 12,
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <Sparkles size={18} color="#fff" />
                    </LinearGradient>
                    <View>
                      <Text style={{ color: "#fff", fontWeight: "800", fontSize: 15 }}>
                        MentraFi AI Copilot
                      </Text>
                      <Text style={{ color: C.cyan, fontSize: 11, fontWeight: "600" }}>
                        Fiduciary Robo-Advisory
                      </Text>
                    </View>
                  </View>

                  <View
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      gap: 4,
                      backgroundColor: "rgba(139, 92, 246, 0.25)",
                      paddingHorizontal: 8,
                      paddingVertical: 3,
                      borderRadius: 999,
                    }}
                  >
                    <Zap size={10} color="#c084fc" />
                    <Text style={{ color: "#c084fc", fontSize: 10, fontWeight: "700" }}>SEBI COMPLIANT</Text>
                  </View>
                </View>

                <Text style={{ color: C.textMuted, fontSize: 12, lineHeight: 18, marginBottom: 14 }}>
                  Ask AI about low-risk portfolios, tax harvesting, or simulate your 15-year SIP growth corpus with 1-tap.
                </Text>

                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "space-between",
                    paddingTop: 10,
                    borderTopWidth: 1,
                    borderTopColor: "rgba(255, 255, 255, 0.08)",
                  }}
                >
                  <Text style={{ color: "#c084fc", fontSize: 12, fontWeight: "600" }}>
                    "What is my ideal SIP for 15 years?"
                  </Text>
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 3 }}>
                    <Text style={{ color: "#fff", fontSize: 12, fontWeight: "700" }}>Consult AI</Text>
                    <ArrowUpRight size={14} color="#fff" strokeWidth={2.5} />
                  </View>
                </View>
              </LinearGradient>
            </TouchableOpacity>
          </View>

          {/* Market Pulse (Trending & New Schemes) */}
          <View style={{ marginBottom: 24 }}>
            <View
              style={{
                flexDirection: "row",
                alignItems: "center",
                justifyContent: "space-between",
                paddingHorizontal: 20,
                marginBottom: 12,
              }}
            >
              <View>
                <Text style={{ color: "#fff", fontWeight: "800", fontSize: 17 }}>Market Pulse</Text>
                <Text style={{ color: C.textFaint, fontSize: 12 }}>Top rated & newly launched schemes</Text>
              </View>
              <TouchableOpacity
                activeOpacity={0.8}
                onPress={() => router.push("/(tabs)/explore" as any)}
                style={{ flexDirection: "row", alignItems: "center", gap: 3 }}
              >
                <Text style={{ color: C.cyan, fontWeight: "600", fontSize: 13 }}>Explore All</Text>
                <ChevronRight size={14} color={C.cyan} />
              </TouchableOpacity>
            </View>

            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={{ paddingHorizontal: 20, gap: 12 }}
            >
              {marketPulse.map((fund, i) => (
                <TouchableOpacity
                  key={`${fund.name}-${i}`}
                  activeOpacity={0.88}
                  onPress={() => {
                    if (fund.schemeCode) {
                      router.push({
                        pathname: "/(tabs)/invest",
                        params: { schemeCode: String(fund.schemeCode) },
                      });
                    } else {
                      router.push("/(tabs)/explore" as any);
                    }
                  }}
                  style={{
                    width: 200,
                    backgroundColor: C.card,
                    borderWidth: 1,
                    borderColor: C.cardEdge,
                    borderRadius: 20,
                    padding: 14,
                    ...depthShadow("sm"),
                  }}
                >
                  {/* Card Header: Fund House & Tenure Badge */}
                  <View
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      justifyContent: "space-between",
                      marginBottom: 8,
                    }}
                  >
                    <View style={{ flexDirection: "row", alignItems: "center", gap: 4, flexShrink: 1 }}>
                      <Building2 size={11} color={C.textMuted} />
                      <Text
                        numberOfLines={1}
                        style={{ color: C.textMuted, fontSize: 10, fontWeight: "600", textTransform: "uppercase" }}
                      >
                        {fund.fundHouse || "Direct Fund"}
                      </Text>
                    </View>

                    {fund.tenureLabel ? (
                      <View
                        style={{
                          paddingHorizontal: 7,
                          paddingVertical: 2,
                          borderRadius: 999,
                          backgroundColor:
                            fund.tenureType === "new"
                              ? "rgba(56, 189, 248, 0.15)"
                              : fund.tenureType === "established"
                              ? "rgba(251, 191, 36, 0.15)"
                              : "rgba(192, 132, 252, 0.15)",
                        }}
                      >
                        <Text
                          style={{
                            fontSize: 10,
                            fontWeight: "700",
                            color:
                              fund.tenureType === "new"
                                ? "#38bdf8"
                                : fund.tenureType === "established"
                                ? "#fbbf24"
                                : "#c084fc",
                          }}
                        >
                          {fund.tenureLabel}
                        </Text>
                      </View>
                    ) : null}
                  </View>

                  {/* Fund Name */}
                  <Text
                    numberOfLines={2}
                    style={{ color: "#fff", fontWeight: "700", fontSize: 13, lineHeight: 18, marginBottom: 8, minHeight: 36 }}
                  >
                    {fund.name}
                  </Text>

                  {/* Category & Rating */}
                  <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                    <Text style={{ color: C.textFaint, fontSize: 11 }}>{fund.category}</Text>
                    <StarRow rating={fund.rating} />
                  </View>

                  {/* Returns and Invest CTA */}
                  <View
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      justifyContent: "space-between",
                      paddingTop: 8,
                      borderTopWidth: 1,
                      borderTopColor: "rgba(255, 255, 255, 0.06)",
                    }}
                  >
                    <View>
                      <Text style={{ color: C.textFaint, fontSize: 10 }}>Returns</Text>
                      <Text
                        style={{
                          color: fund.change >= 0 ? GREEN : RED,
                          fontWeight: "700",
                          fontSize: 13,
                        }}
                      >
                        {fund.change >= 0 ? "+" : ""}{fund.change.toFixed(2)}%
                      </Text>
                    </View>

                    <View
                      style={{
                        paddingHorizontal: 12,
                        paddingVertical: 6,
                        borderRadius: 999,
                        backgroundColor: "rgba(124, 58, 237, 0.2)",
                        borderWidth: 1,
                        borderColor: "rgba(124, 58, 237, 0.4)",
                        flexDirection: "row",
                        alignItems: "center",
                        gap: 2,
                      }}
                    >
                      <Text style={{ color: "#c084fc", fontWeight: "700", fontSize: 11 }}>Invest</Text>
                      <ArrowUpRight size={11} color="#c084fc" />
                    </View>
                  </View>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>

          {/* My Holdings Section */}
          <View style={{ paddingHorizontal: 20, marginBottom: 24 }}>
            <View
              style={{
                flexDirection: "row",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 12,
              }}
            >
              <Text style={{ color: "#fff", fontWeight: "800", fontSize: 17 }}>My Holdings</Text>
              <TouchableOpacity
                activeOpacity={0.8}
                onPress={() => router.push("/(tabs)/portfolio" as any)}
                style={{ flexDirection: "row", alignItems: "center", gap: 2 }}
              >
                <Text style={{ color: C.cyan, fontWeight: "600", fontSize: 13 }}>
                  View All ({portfolio?.holdings?.length || 0})
                </Text>
                <ChevronRight size={14} color={C.cyan} />
              </TouchableOpacity>
            </View>

            <GlassPanel style={{ overflow: "hidden" }}>
              {loading && !portfolio ? (
                <View style={{ paddingHorizontal: 20, paddingVertical: 28, alignItems: "center" }}>
                  <ActivityIndicator color={C.pink} size="small" />
                  <Text style={{ color: C.textFaint, fontSize: 11, marginTop: 8 }}>
                    Loading your investments...
                  </Text>
                </View>
              ) : holdings.length === 0 ? (
                <View style={{ paddingHorizontal: 20, paddingVertical: 28, alignItems: "center" }}>
                  <Text style={{ color: "#fff", fontWeight: "700", fontSize: 15, marginBottom: 4 }}>
                    No investments yet
                  </Text>
                  <Text style={{ color: C.textFaint, fontSize: 12, textAlign: "center", marginBottom: 14 }}>
                    Start compounding with direct, zero-commission mutual funds today.
                  </Text>
                  <TouchableOpacity
                    activeOpacity={0.85}
                    onPress={() => router.push("/(tabs)/explore" as any)}
                  >
                    <LinearGradient
                      colors={[C.pink, C.violet]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 1, y: 1 }}
                      style={{
                        paddingHorizontal: 18,
                        paddingVertical: 8,
                        borderRadius: 999,
                      }}
                    >
                      <Text style={{ color: "#fff", fontWeight: "700", fontSize: 12 }}>
                        Explore Funds
                      </Text>
                    </LinearGradient>
                  </TouchableOpacity>
                </View>
              ) : null}

              {holdings.map((fund, index) => {
                const positive = fund.change >= 0;
                return (
                  <TouchableOpacity
                    key={fund.id}
                    activeOpacity={0.8}
                    onPress={() => router.push("/(tabs)/portfolio" as any)}
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      paddingHorizontal: 16,
                      paddingVertical: 14,
                      borderBottomWidth: index < holdings.length - 1 ? 1 : 0,
                      borderBottomColor: "rgba(255, 255, 255, 0.06)",
                    }}
                  >
                    {/* Status Pill */}
                    <View
                      style={{
                        width: 3,
                        height: 38,
                        borderRadius: 2,
                        marginRight: 12,
                        backgroundColor: positive ? GREEN : RED,
                      }}
                    />

                    {/* Icon */}
                    <View
                      style={{
                        width: 38,
                        height: 38,
                        borderRadius: 14,
                        marginRight: 12,
                        alignItems: "center",
                        justifyContent: "center",
                        backgroundColor: positive ? "rgba(74,222,128,0.12)" : "rgba(255,107,129,0.12)",
                      }}
                    >
                      {positive ? (
                        <TrendingUp size={18} color={GREEN} />
                      ) : (
                        <TrendingDown size={18} color={RED} />
                      )}
                    </View>

                    {/* Info */}
                    <View style={{ flex: 1, marginRight: 8 }}>
                      <Text
                        style={{ color: "#fff", fontWeight: "700", fontSize: 14, marginBottom: 2 }}
                        numberOfLines={1}
                      >
                        {fund.name}
                      </Text>
                      <Text style={{ color: C.textFaint, fontSize: 11 }}>
                        {fund.units.toFixed(2)} units · Direct Plan
                      </Text>
                    </View>

                    {/* Value */}
                    <View style={{ alignItems: "flex-end", marginRight: 8 }}>
                      <Text style={{ color: "#fff", fontWeight: "700", fontSize: 14 }}>
                        {showBalance
                          ? `₹${fund.value.toLocaleString("en-IN", { minimumFractionDigits: 2 })}`
                          : "₹••••"}
                      </Text>
                      <Text style={{ color: positive ? GREEN : RED, fontSize: 11, fontWeight: "700" }}>
                        {positive ? "+" : ""}
                        {Number(fund.change || 0).toFixed(1)}%
                      </Text>
                    </View>

                    <ChevronRight size={16} color={C.textFaint} />
                  </TouchableOpacity>
                );
              })}
            </GlassPanel>
          </View>

          {/* SIP Wealth Goal Banner */}
          <View style={{ paddingHorizontal: 20 }}>
            <TouchableOpacity
              activeOpacity={0.88}
              onPress={() => router.push("/(tabs)/sip-calculator" as any)}
              style={{ ...depthShadow("md") }}
            >
              <LinearGradient
                colors={["rgba(6, 182, 212, 0.18)", "rgba(124, 58, 237, 0.18)"]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  borderRadius: 22,
                  padding: 16,
                  borderWidth: 1,
                  borderColor: "rgba(6, 182, 212, 0.3)",
                }}
              >
                <View
                  style={{
                    width: 44,
                    height: 44,
                    borderRadius: 14,
                    backgroundColor: "rgba(6, 182, 212, 0.2)",
                    alignItems: "center",
                    justifyContent: "center",
                    marginRight: 12,
                  }}
                >
                  <Calculator size={22} color={C.cyan} />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={{ color: "#fff", fontWeight: "700", fontSize: 14 }}>
                    Compound Wealth Calculator
                  </Text>
                  <Text style={{ color: C.textMuted, fontSize: 11, marginTop: 2 }}>
                    See how ₹5,000/mo SIP compounds to ₹24.98L in 15 years
                  </Text>
                </View>
                <ChevronRight size={18} color="#fff" />
              </LinearGradient>
            </TouchableOpacity>
          </View>
        </ScrollView>

        {/* Bottom Navigation */}
        <BottomNav tabs={tabs} paddingBottom={10} />
      </SafeAreaView>
    </View>
  );
}