import React, { useState, useRef, useEffect, useMemo, useCallback } from "react";
import {
  View,
  Text,
  Image,
  TextInput,
  TouchableOpacity,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Dimensions,
  ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter, useFocusEffect } from "expo-router";
import {
  Sparkles,
  Send,
  Home,
  Search,
  PieChart,
  User,
  RotateCcw,
  TrendingUp,
  ShieldCheck,
  Zap,
  ArrowLeft,
  ArrowUpRight,
  Briefcase,
  Wallet,
  Target as TargetIcon,
  ChevronRight,
  CheckCircle2,
  Copy,
  Check,
} from "lucide-react-native";
import * as Clipboard from "expo-clipboard";
import * as Haptics from "expo-haptics";
import Svg, {
  Defs,
  RadialGradient,
  Ellipse,
  Stop,
  Path,
  Circle as SvgCircle,
  LinearGradient as SvgGrad,
} from "react-native-svg";
import { C, depthShadow, GlowBackdrop } from "../(auth)/login";
import BottomNav from "./BottomNav";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { API_URL } from "../../utils/api";
import { subscribeProfileUpdates } from "../../utils/profileEvents";

const { width: SCREEN_WIDTH } = Dimensions.get("window");
const GREEN = "#4ade80";

type FundCardData = {
  id: string;
  name: string;
  category: string;
  subcategory: string;
  rating: number;
  nav: number;
  oneYearReturn: number;
  fiveYearReturn: number;
  planType: string;
  monthlyAmount?: number;
  allocationPercent?: number;
};

type ChatMessage = {
  id: string;
  role: "user" | "ai";
  text: string;
  timestamp: string;
  funds?: FundCardData[];
  keyHighlights?: string[];
};

const PRESET_RESPONSES: Record<string, { text: string; funds?: FundCardData[]; highlights?: string[] }> = {
  sip_definition: {
    text: `A **SIP (Systematic Investment Plan)** is a disciplined method of investing a fixed sum of money in mutual funds at regular intervals (typically monthly or quarterly).\n\n### 🔑 Key Advantages:\n• **Rupee-Cost Averaging:** When market prices drop, your fixed installment automatically buys more units; when prices rise, you buy fewer units. You never have to time the market.\n• **Power of Compounding:** Small, consistent contributions compound exponentially over 5, 10, or 15+ years into substantial wealth.\n• **Low Minimum Entry:** Start investing with as little as ₹500/month via automated bank mandate (UPI/eNACH).\n• **Complete Flexibility:** You can pause, increase (Step-up), or stop your SIP anytime with zero penalties (for open-ended non-ELSS funds).\n• **Financial Discipline:** Automates wealth accumulation on your monthly salary date before discretionary expenses.`,
    highlights: [
      "Disciplined automated investing on your monthly salary cycle.",
      "Eliminates market timing risk via rupee-cost averaging.",
      "Start with just ₹500/month in Direct-Growth mutual funds.",
    ],
  },
  nav: {
    text: `**NAV (Net Asset Value)** represents the per-unit market price of a mutual fund scheme, calculated at the end of every business day (around 9:00 PM IST).\n\n### 📊 How It Is Calculated:\nNAV = (Total Scheme Assets − Total Liabilities & Operating Expenses) ÷ Total Outstanding Units\n\n### 💡 Important Myth Buster:\nA higher NAV (e.g. ₹200) does **NOT** mean a fund is 'expensive' compared to one with an NAV of ₹20. Mutual fund returns depend strictly on the percentage growth of the underlying portfolio, not the absolute NAV. A 15% gain on a ₹20 NAV fund and a 15% gain on a ₹200 NAV fund produce the exact same rupee profit.`,
    highlights: [
      "Calculated daily after market closing at 9:00 PM IST.",
      "A lower NAV does NOT mean a fund is cheaper or will give higher returns.",
      "Returns depend strictly on scheme percentage growth.",
    ],
  },
  sip: {
    text: `Here is a personalized Direct-Growth asset allocation crafted for your profile:\n\n• **Flexi Cap (50% — ₹5,000/mo):** Consistent multi-cap wealth builder with downside protection.\n• **Large Cap (30% — ₹3,000/mo):** Core stability backed by India's top 100 bluechip companies.\n• **Mid Cap (20% — ₹2,000/mo):** High alpha engine for long-term outperformance.\n\n💰 **Deterministic SIP Compounding:**\n• Total Invested over 15 Years: **₹18.00 Lakh**\n• Estimated Maturity Value (@ ~14% CAGR): **₹1.11 Crore**`,
    highlights: [
      "100% Direct-Growth schemes to save 0.8% - 1.5% yearly commission.",
      "Strict zero-math-drift allocation constraint applied.",
      "LTCG above ₹1.25 Lakh taxed at 12.5% under current 2026 tax norms.",
    ],
    funds: [
      {
        id: "118955",
        name: "HDFC Flexi Cap Fund - Direct Plan",
        category: "Equity",
        subcategory: "Flexi Cap",
        rating: 5,
        nav: 2241.68,
        oneYearReturn: 28.4,
        fiveYearReturn: 19.8,
        planType: "Direct Growth",
      },
      {
        id: "120586",
        name: "Quant Active Fund - Direct Plan",
        category: "Equity",
        subcategory: "Multi Cap",
        rating: 5,
        nav: 116.34,
        oneYearReturn: 34.2,
        fiveYearReturn: 24.1,
        planType: "Direct Growth",
      },
      {
        id: "125497",
        name: "SBI Small Cap Fund - Direct Plan",
        category: "Equity",
        subcategory: "Small Cap",
        rating: 4,
        nav: 215.67,
        oneYearReturn: 31.8,
        fiveYearReturn: 21.5,
        planType: "Direct Growth",
      },
    ],
  },
  elss: {
    text: `An **ELSS (Equity Linked Savings Scheme)** is an equity mutual fund offering tax deduction benefits under Section 80C of the Income Tax Act:\n\n• **Tax Deduction:** Up to ₹1.50 Lakh under the Old Tax Regime.\n• **Lock-in Period:** 3 years — the shortest lock-in among all Section 80C options (PPF is 15 years, Tax-Saving FDs are 5 years).\n• **Current 2026 LTCG Tax:** Capital gains up to ₹1.25 Lakh per financial year are tax-free; gains beyond ₹1.25 Lakh are taxed at **12.5%** without indexation.`,
    highlights: [
      "Shortest lock-in period (36 months) across all 80C instruments.",
      "Compounding equity growth rather than fixed-income interest.",
      "SIP installments lock in independently for 3 years each.",
    ],
    funds: [
      {
        id: "120510",
        name: "Parag Parikh ELSS Tax Saver Fund - Direct",
        category: "Equity",
        subcategory: "ELSS",
        rating: 5,
        nav: 36.3,
        oneYearReturn: 26.5,
        fiveYearReturn: 20.4,
        planType: "Direct Growth",
      },
      {
        id: "119551",
        name: "Mirae Asset ELSS Tax Saver Fund - Direct",
        category: "Equity",
        subcategory: "ELSS",
        rating: 4,
        nav: 48.9,
        oneYearReturn: 23.8,
        fiveYearReturn: 18.2,
        planType: "Direct Growth",
      },
    ],
  },
  direct: {
    text: `The core distinction between **Direct** and **Regular** mutual funds lies in distributor commissions:\n\n• **Direct Plans:** You invest directly with the AMC. There are **zero broker commissions**, resulting in a significantly lower Expense Ratio (often 0.5% – 1.5% lower per year).\n• **Regular Plans:** You invest via an intermediary/broker who receives ongoing trail commissions deducted directly from your fund's NAV every day.\n\nOver a 15–20 year investment horizon, that 1% difference in expense ratio can save you **15% to 20% of your total wealth** due to compounding! MentraFi strictly recommends Direct-Growth schemes.`,
    highlights: [
      "Zero distributor trail commission deducted from your daily NAV.",
      "Identical portfolio holdings, but Direct consistently generates higher returns.",
      "Over ₹10 Lakh extra accumulated on a ₹10,000 SIP over 20 years.",
    ],
  },
  greeting: {
    text: `Hello! 👋 Welcome to MentraFi AI, your intelligent Indian mutual fund wealth advisor.\n\nI can help you build personalized goal-based SIP portfolios, compare 37,000+ Direct-Growth schemes, calculate compounding projections, or answer any mutual fund queries.\n\nHow can I help you invest today?`,
    highlights: [
      "Ask for a customized SIP portfolio (e.g. 'Invest ₹5,000/month for 10 years').",
      "Ask any concept question (e.g. 'What is SIP?', 'What is NAV?', 'Direct vs Regular').",
      "Explore ELSS tax-saving funds under Section 80C.",
    ],
  },
  default: {
    text: `I have analyzed your query through our SEBI-aligned wealth advisory framework. \n\nWhen investing in Indian mutual funds, prioritizing **Direct-Growth schemes** with a minimum 5-year horizon ensures maximum compound growth and tax efficiency under the current 2026 capital gains norms.\n\nWould you like a tailored SIP allocation, an ELSS tax comparison, or fund performance metrics?`,
    highlights: [
      "Grounded across 37,768 SEBI registered mutual fund schemes.",
      "Current 2026 taxation engine applied (12.5% LTCG above ₹1.25 Lakh).",
      "Instant portfolio risk profiling and compounding projections.",
    ],
  },
};

function AIAvatarHero() {
  return (
    <View
      style={{
        borderRadius: 24,
        backgroundColor: C.card,
        borderWidth: 1,
        borderColor: C.cardEdge,
        padding: 18,
        alignItems: "center",
        ...depthShadow("md"),
        overflow: "hidden",
      }}
    >
      <LinearGradient
        colors={["rgba(139, 92, 246, 0.18)", "rgba(6, 182, 212, 0.12)", "transparent"]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0 }}
      />
      <View
        style={{
          width: 58,
          height: 58,
          borderRadius: 29,
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "rgba(13, 27, 42, 0.9)",
          borderWidth: 1.5,
          borderColor: "rgba(6, 182, 212, 0.4)",
          marginBottom: 10,
        }}
      >
        <Image
          source={require("../../assets/images/mentrafi-emblem.png")}
          style={{ width: 42, height: 34 }}
          resizeMode="contain"
        />
      </View>
      <Text style={{ color: "#fff", fontSize: 16, fontWeight: "700", letterSpacing: 0.2 }}>
        MentraFi Neuro-Symbolic AI
      </Text>
      <Text style={{ color: C.textMuted, fontSize: 12, textAlign: "center", marginTop: 4, lineHeight: 18 }}>
        Tri-network intelligence grounded across 37,882 SEBI mutual fund schemes with zero mathematical drift.
      </Text>

      {/* 3 feature badges */}
      <View style={{ flexDirection: "row", alignItems: "center", gap: 6, marginTop: 12, flexWrap: "wrap", justifyContent: "center" }}>
        <View style={{ backgroundColor: "rgba(16, 185, 129, 0.15)", paddingHorizontal: 9, paddingVertical: 4, borderRadius: 12, borderWidth: 1, borderColor: "rgba(16, 185, 129, 0.3)" }}>
          <Text style={{ color: "#34d399", fontSize: 10, fontWeight: "700" }}>🛡️ SEBI Grounded</Text>
        </View>
        <View style={{ backgroundColor: "rgba(6, 182, 212, 0.15)", paddingHorizontal: 9, paddingVertical: 4, borderRadius: 12, borderWidth: 1, borderColor: "rgba(6, 182, 212, 0.3)" }}>
          <Text style={{ color: "#38bdf8", fontSize: 10, fontWeight: "700" }}>⚡ Zero Math Drift</Text>
        </View>
        <View style={{ backgroundColor: "rgba(139, 92, 246, 0.15)", paddingHorizontal: 9, paddingVertical: 4, borderRadius: 12, borderWidth: 1, borderColor: "rgba(139, 92, 246, 0.3)" }}>
          <Text style={{ color: "#a78bfa", fontSize: 10, fontWeight: "700" }}>💎 100% Direct-Growth</Text>
        </View>
      </View>
    </View>
  );
}

function FundCard({ fund }: { fund: FundCardData }) {
  const router = useRouter();
  return (
    <View
      style={{
        backgroundColor: "rgba(13, 27, 42, 0.8)",
        borderWidth: 1,
        borderColor: "rgba(6, 182, 212, 0.25)",
        borderRadius: 20,
        padding: 15,
        marginTop: 10,
        ...depthShadow("sm"),
      }}
    >
      <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
        <View style={{ flex: 1, marginRight: 10 }}>
          <Text style={{ color: "#fff", fontWeight: "700", fontSize: 14, lineHeight: 19 }}>
            {fund.name}
          </Text>
          <View style={{ flexDirection: "row", alignItems: "center", gap: 6, marginTop: 4, flexWrap: "wrap" }}>
            <Text style={{ color: C.textMuted, fontSize: 11 }}>
              {fund.category} • {fund.subcategory}
            </Text>
            {fund.monthlyAmount ? (
              <View style={{ backgroundColor: "rgba(16, 185, 129, 0.2)", paddingHorizontal: 6, paddingVertical: 2, borderRadius: 6 }}>
                <Text style={{ color: "#34d399", fontSize: 10, fontWeight: "700" }}>
                  ₹{fund.monthlyAmount.toLocaleString("en-IN")}/mo ({fund.allocationPercent}%)
                </Text>
              </View>
            ) : null}
          </View>
        </View>
        <View
          style={{
            flexDirection: "row",
            alignItems: "center",
            gap: 3,
            backgroundColor: "rgba(245, 158, 11, 0.15)",
            paddingHorizontal: 8,
            paddingVertical: 3,
            borderRadius: 12,
            borderWidth: 1,
            borderColor: "rgba(245, 158, 11, 0.3)",
          }}
        >
          <Sparkles size={11} color="#fbbf24" />
          <Text style={{ color: "#fbbf24", fontSize: 11, fontWeight: "700" }}>{fund.rating}.0</Text>
        </View>
      </View>

      <View
        style={{
          flexDirection: "row",
          justifyContent: "space-between",
          alignItems: "center",
          backgroundColor: "rgba(255, 255, 255, 0.04)",
          borderRadius: 14,
          paddingHorizontal: 12,
          paddingVertical: 9,
          marginTop: 6,
          borderWidth: 1,
          borderColor: "rgba(255, 255, 255, 0.08)",
        }}
      >
        <View>
          <Text style={{ color: C.textFaint, fontSize: 10, textTransform: "uppercase", letterSpacing: 0.5 }}>1Y Return</Text>
          <Text style={{ color: GREEN, fontWeight: "700", fontSize: 13, marginTop: 1 }}>
            +{fund.oneYearReturn}%
          </Text>
        </View>

        <View>
          <Text style={{ color: C.textFaint, fontSize: 10, textTransform: "uppercase", letterSpacing: 0.5 }}>5Y Return</Text>
          <Text style={{ color: GREEN, fontWeight: "700", fontSize: 13, marginTop: 1 }}>
            +{fund.fiveYearReturn}%
          </Text>
        </View>

        <View>
          <Text style={{ color: C.textFaint, fontSize: 10, textTransform: "uppercase", letterSpacing: 0.5 }}>Live NAV</Text>
          <Text style={{ color: "#fff", fontWeight: "700", fontSize: 13, marginTop: 1 }}>
            ₹{fund.nav ? fund.nav.toFixed(2) : "N/A"}
          </Text>
        </View>

        <TouchableOpacity
          onPress={() => router.push("/(tabs)/invest" as any)}
          activeOpacity={0.8}
          style={{
            flexDirection: "row",
            alignItems: "center",
            gap: 3,
            backgroundColor: "rgba(6, 182, 212, 0.15)",
            paddingHorizontal: 10,
            paddingVertical: 5,
            borderRadius: 10,
            borderWidth: 1,
            borderColor: "rgba(6, 182, 212, 0.3)",
          }}
        >
          <Text style={{ color: "#38bdf8", fontSize: 11, fontWeight: "700" }}>Invest</Text>
          <ArrowUpRight size={12} color="#38bdf8" />
        </TouchableOpacity>
      </View>
    </View>
  );
}

function FormattedMessageText({ text }: { text: string }) {
  if (!text) return null;
  const lines = text.split("\n");

  return (
    <View style={{ gap: 4 }}>
      {lines.map((line, lineIdx) => {
        const trimmed = line.trim();
        if (!trimmed) {
          return <View key={lineIdx} style={{ height: 6 }} />;
        }

        const isHeader = trimmed.startsWith("### ") || trimmed.startsWith("## ");
        const content = isHeader ? trimmed.replace(/^#{2,3}\s*/, "") : line;
        const parts = content.split(/(\*\*.*?\*\*)/g);

        return (
          <Text
            key={lineIdx}
            selectable={true}
            selectionColor="#FF4F81"
            style={{
              color: isHeader ? "#F472B6" : "#FFFFFF",
              fontSize: isHeader ? 14 : 13.5,
              lineHeight: isHeader ? 22 : 21,
              fontWeight: isHeader ? "700" : "400",
              marginTop: isHeader && lineIdx > 0 ? 6 : 0,
            }}
          >
            {parts.map((part, partIdx) => {
              if (part.startsWith("**") && part.endsWith("**") && part.length >= 4) {
                return (
                  <Text
                    key={partIdx}
                    selectable={true}
                    selectionColor="#FF4F81"
                    style={{
                      fontWeight: "700",
                      color: isHeader ? "#F472B6" : "#FFFFFF",
                    }}
                  >
                    {part.slice(2, -2)}
                  </Text>
                );
              }
              return part;
            })}
          </Text>
        );
      })}
    </View>
  );
}

export default function AIAdvisorScreen() {
  const router = useRouter();
  const [inputText, setInputText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [userProfile, setUserProfile] = useState<{
    name?: string;
    age?: number;
    budget?: number;
    risk?: string;
    occupation?: string;
    goal?: string;
  } | null>(null);

  const scrollViewRef = useRef<ScrollView>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleBack = () => {
    if (typeof router.canGoBack === "function" && router.canGoBack()) {
      router.back();
    } else {
      router.replace("/(tabs)/home" as any);
    }
  };

  const handleCopy = async (id: string, textToCopy: string) => {
    try {
      await Clipboard.setStringAsync(textToCopy);
      try {
        await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      } catch {}
      setCopiedId(id);
      setTimeout(() => {
        setCopiedId((prev) => (prev === id ? null : prev));
      }, 2000);
    } catch (err) {
      console.error("Failed to copy text:", err);
    }
  };

  const loadUserProfile = useCallback(async () => {
    try {
      // 1. Fast local cache check for instant 0ms rendering
      const cached = await AsyncStorage.getItem("cachedUserProfile");
      if (cached) {
        try {
          const parsed = JSON.parse(cached);
          const pInfo = parsed.personalInfo || {};
          const rawRisk = (pInfo.riskAppetite || "").trim();
          const normalizedRisk = rawRisk
            ? rawRisk.charAt(0).toUpperCase() + rawRisk.slice(1).toLowerCase()
            : undefined;
          setUserProfile({
            name: parsed.name || pInfo.fullName,
            age: pInfo.age ? Number(pInfo.age) : undefined,
            budget: pInfo.monthlySipBudget ? Number(pInfo.monthlySipBudget) : undefined,
            risk: normalizedRisk,
            occupation: pInfo.occupation || undefined,
            goal: pInfo.investmentGoal || undefined,
          });
        } catch {}
      }

      // 2. Fetch fresh verified data from backend database with cache-busting
      const token = await AsyncStorage.getItem("userToken");
      if (!token) return;
      const res = await fetch(`${API_URL}/api/profile?_t=${Date.now()}`, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
          "Cache-Control": "no-cache, no-store, must-revalidate",
          Pragma: "no-cache",
        },
      });
      if (res.ok) {
        const data = await res.json();
        const pInfo = data.personalInfo || {};
        const rawRisk = (pInfo.riskAppetite || "").trim();
        const normalizedRisk = rawRisk
          ? rawRisk.charAt(0).toUpperCase() + rawRisk.slice(1).toLowerCase()
          : undefined;
        setUserProfile({
          name: data.name || pInfo.fullName,
          age: pInfo.age ? Number(pInfo.age) : undefined,
          budget: pInfo.monthlySipBudget ? Number(pInfo.monthlySipBudget) : undefined,
          risk: normalizedRisk,
          occupation: pInfo.occupation || undefined,
          goal: pInfo.investmentGoal || undefined,
        });
        await AsyncStorage.setItem("cachedUserProfile", JSON.stringify(data));
      }
    } catch (err) {
      console.error("Failed to load user profile in AI Advisor:", err);
    }
  }, []);

  // Run on mount and listen to real-time profile update events
  useEffect(() => {
    loadUserProfile();
    const unsubscribe = subscribeProfileUpdates(() => {
      loadUserProfile();
    });
    return () => unsubscribe();
  }, [loadUserProfile]);

  // Re-fetch automatically whenever user switches to the AI Advisor tab!
  useFocusEffect(
    useCallback(() => {
      loadUserProfile();
    }, [loadUserProfile])
  );

  const userOccupation = userProfile?.occupation || "Doing Job / Salaried";
  const userGoal = userProfile?.goal || "Retirement Planning";
  const userAge = userProfile?.age ?? 20;
  const userBudget = userProfile?.budget ?? 5000;
  const rawRisk = (userProfile?.risk || "Moderate").trim();
  const userRisk = rawRisk ? rawRisk.charAt(0).toUpperCase() + rawRisk.slice(1).toLowerCase() : "Moderate";

  const suggestions = useMemo(() => {
    const budgetFmt = userBudget >= 1000 ? `${Math.round(userBudget / 1000)}k` : `${userBudget}`;
    const isHighRisk = userRisk.toLowerCase() === "high";
    const isLowRisk = userRisk.toLowerCase() === "low";

    return [
      {
        title: `${budgetFmt} Monthly SIP Plan`,
        desc: `${userRisk} plan for age ${userAge} • ${userGoal}`,
        prompt: `I am ${userAge}, want to invest ${userBudget} monthly SIP for 15 years with ${userRisk.toLowerCase()} risk for ${userGoal.toLowerCase()}.`,
        icon: isHighRisk ? Zap : isLowRisk ? ShieldCheck : TrendingUp,
      },
      {
        title: "ELSS Tax Saving",
        desc: "Section 80C deductions & rules",
        prompt: `What is an ELSS mutual fund, how does Section 80C tax deduction work, and is it suitable for a ${userRisk.toLowerCase()} risk investor?`,
        icon: ShieldCheck,
      },
      {
        title: `${userRisk} Allocation Strategy`,
        desc: `${userOccupation} portfolio split`,
        prompt: `As a ${userAge}-year-old (${userOccupation}) with ${userRisk} risk appetite, how should I allocate my ₹${userBudget}/month across Equity, Debt, and Hybrid funds?`,
        icon: isHighRisk ? Zap : TrendingUp,
      },
      {
        title: "10% Step-Up Compounding",
        desc: `Project returns on ₹${budgetFmt}/month`,
        prompt: `Calculate how a 10% annual Step-Up SIP on ₹${userBudget}/month increases my maturity corpus over 15 years with ${userRisk.toLowerCase()} risk.`,
        icon: ArrowUpRight,
      },
    ];
  }, [userOccupation, userGoal, userBudget, userAge, userRisk]);

  const tabs = [
    { name: "Home", icon: Home, route: "/(tabs)/home" },
    { name: "Explore", icon: Search, route: "/(tabs)/explore" },
    { name: "Portfolio", icon: PieChart, route: "/(tabs)/portfolio" },
    { name: "AI Advisor", icon: Sparkles, route: "/(tabs)/ai-advisor" },
    { name: "Profile", icon: User, route: "/(tabs)/profile" },
  ];

  useEffect(() => {
    scrollViewRef.current?.scrollToEnd({ animated: true });
  }, [messages, isTyping]);

  const handleSend = async (userText: string) => {
    const trimmed = userText.trim();
    if (!trimmed || isTyping) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      text: trimmed,
      timestamp: "Just now",
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputText("");
    setIsTyping(true);

    const token = await AsyncStorage.getItem("userToken");
    const aiMsgId = `ai-${Date.now()}`;

    // Helper for fallback preset response
    const runFallback = () => {
      const lower = trimmed.toLowerCase();
      let responseKey = "default";
      if (
        lower === "hi" ||
        lower === "hello" ||
        lower === "hey" ||
        lower.startsWith("hi ") ||
        lower.startsWith("hello ") ||
        lower.startsWith("hey ") ||
        lower.includes("good morning") ||
        lower.includes("good evening") ||
        lower.includes("namaste")
      ) {
        responseKey = "greeting";
      } else if (lower.includes("what is sip") || lower.includes("explain sip") || lower === "sip" || lower.includes("sip kya") || lower.includes("about sip")) {
        responseKey = "sip_definition";
      } else if (lower.includes("nav") || lower.includes("net asset value")) {
        responseKey = "nav";
      } else if (lower.includes("sip") || lower.includes("10000") || lower.includes("portfolio") || lower.includes("invest")) {
        responseKey = "sip";
      } else if (lower.includes("elss") || lower.includes("tax") || lower.includes("80c")) {
        responseKey = "elss";
      } else if (lower.includes("direct") || lower.includes("regular") || lower.includes("difference")) {
        responseKey = "direct";
      }

      const preset = PRESET_RESPONSES[responseKey];
      setMessages((prev) => [
        ...prev,
        {
          id: aiMsgId,
          role: "ai",
          text: preset.text,
          timestamp: "Just now",
          funds: preset.funds,
          keyHighlights: preset.highlights,
        },
      ]);
      setIsTyping(false);
    };

    if (!token) {
      setTimeout(runFallback, 500);
      return;
    }

    try {
      let hasStartedStreaming = false;
      let lastProcessedLength = 0;
      let sseBuffer = "";

      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_URL}/api/advisor/chat/stream`);
      xhr.setRequestHeader("Content-Type", "application/json");
      xhr.setRequestHeader("Authorization", `Bearer ${token}`);
      xhr.timeout = 30000;

      xhr.onprogress = () => {
        const newText = xhr.responseText.slice(lastProcessedLength);
        lastProcessedLength = xhr.responseText.length;
        sseBuffer += newText;

        const lines = sseBuffer.split("\n");
        sseBuffer = lines.pop() || "";

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine.startsWith("data:")) continue;
          const jsonStr = trimmedLine.slice(5).trim();
          if (jsonStr === "[DONE]") {
            setIsTyping(false);
            continue;
          }

          try {
            const parsed = JSON.parse(jsonStr);

            if (parsed.recommendedFunds && Array.isArray(parsed.recommendedFunds)) {
              if (!hasStartedStreaming) {
                hasStartedStreaming = true;
                setMessages((prev) => [
                  ...prev,
                  {
                    id: aiMsgId,
                    role: "ai",
                    text: "",
                    timestamp: "Just now",
                    funds: parsed.recommendedFunds,
                  },
                ]);
              } else {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === aiMsgId ? { ...m, funds: parsed.recommendedFunds } : m
                  )
                );
              }
            }

            if (typeof parsed.chunk === "string") {
              if (!hasStartedStreaming) {
                hasStartedStreaming = true;
                setMessages((prev) => [
                  ...prev,
                  {
                    id: aiMsgId,
                    role: "ai",
                    text: parsed.chunk,
                    timestamp: "Just now",
                  },
                ]);
              } else {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === aiMsgId ? { ...m, text: m.text + parsed.chunk } : m
                  )
                );
              }
            }
          } catch {
            // Partial JSON chunk
          }
        }
      };

      xhr.onload = () => {
        setIsTyping(false);
        if (!hasStartedStreaming) {
          runFallback();
        }
      };

      xhr.onerror = () => {
        if (!hasStartedStreaming) {
          runFallback();
        } else {
          setIsTyping(false);
        }
      };

      xhr.ontimeout = () => {
        if (!hasStartedStreaming) {
          runFallback();
        } else {
          setIsTyping(false);
        }
      };

      xhr.send(
        JSON.stringify({
          message: trimmed,
          age: userProfile?.age ?? userAge,
          name: userProfile?.name,
          occupation: userProfile?.occupation ?? userOccupation,
          monthly_sip_budget: userProfile?.budget ?? userBudget,
          risk_appetite: userProfile?.risk ?? userRisk,
          investment_goal: userProfile?.goal ?? userGoal,
          history: messages.slice(-10).map((m) => ({
            role: m.role === "ai" ? "assistant" : "user",
            content: m.text,
          })),
        })
      );
    } catch {
      runFallback();
    }
  };

  const handleResetChat = () => {
    setMessages([]);
  };

  return (
    <View style={{ flex: 1, backgroundColor: C.bgBottom }}>
      {/* Background Gradients & Ambient Glows */}
      <LinearGradient
        colors={[C.bgTop, C.bgMid, C.bgBottom]}
        start={{ x: 0.5, y: 0 }}
        end={{ x: 0.5, y: 1 }}
        style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0 }}
      />
      <View
        pointerEvents="none"
        style={{
          position: "absolute",
          top: -100,
          right: -80,
          width: 320,
          height: 320,
          borderRadius: 160,
          backgroundColor: "rgba(255,79,129,0.12)",
        }}
      />
      <View
        pointerEvents="none"
        style={{
          position: "absolute",
          top: 250,
          left: -100,
          width: 300,
          height: 300,
          borderRadius: 150,
          backgroundColor: "rgba(123,63,242,0.12)",
        }}
      />

      <SafeAreaView style={{ flex: 1 }} edges={["top"]}>
        {/* Top App Bar with Back Navigation */}
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
          <TouchableOpacity
            onPress={handleBack}
            activeOpacity={0.7}
            hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
            style={{
              width: 38,
              height: 38,
              borderRadius: 19,
              backgroundColor: "rgba(255, 255, 255, 0.08)",
              borderWidth: 1,
              borderColor: "rgba(255, 255, 255, 0.12)",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <ArrowLeft size={18} color="#fff" />
          </TouchableOpacity>

          <View style={{ alignItems: "center" }}>
            <Text style={{ color: "#fff", fontWeight: "700", fontSize: 16, letterSpacing: 0.3 }}>
              MentraFi AI Advisor
            </Text>
            <View style={{ flexDirection: "row", alignItems: "center", gap: 5, marginTop: 2 }}>
              <View style={{ width: 6, height: 6, borderRadius: 3, backgroundColor: GREEN }} />
              <Text style={{ color: "#34d399", fontSize: 10.5, fontWeight: "600" }}>
                Neural Engine Online (SEBI Reg.)
              </Text>
            </View>
          </View>

          <TouchableOpacity
            onPress={handleResetChat}
            activeOpacity={0.7}
            hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
            style={{
              width: 38,
              height: 38,
              borderRadius: 19,
              backgroundColor: "rgba(255, 255, 255, 0.08)",
              borderWidth: 1,
              borderColor: "rgba(255, 255, 255, 0.12)",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <RotateCcw size={16} color={C.textMuted} />
          </TouchableOpacity>
        </View>

        {/* Investor Profile Capsule Bento Bar */}
        <TouchableOpacity
          onPress={() => router.push("/(tabs)/profile" as any)}
          activeOpacity={0.85}
          style={{
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "space-between",
            backgroundColor: "rgba(13, 27, 42, 0.75)",
            marginHorizontal: 16,
            marginTop: 8,
            marginBottom: 4,
            paddingHorizontal: 12,
            paddingVertical: 9,
            borderRadius: 18,
            borderWidth: 1,
            borderColor: "rgba(6, 182, 212, 0.25)",
            ...depthShadow("sm"),
          }}
        >
          <View style={{ flexDirection: "row", alignItems: "center", gap: 6, flexWrap: "wrap", flex: 1 }}>
            <View style={{ flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: "rgba(139, 92, 246, 0.15)", paddingHorizontal: 8, paddingVertical: 3.5, borderRadius: 10, borderWidth: 1, borderColor: "rgba(139, 92, 246, 0.3)" }}>
              <Briefcase size={10} color="#a78bfa" />
              <Text style={{ color: "#c4b5fd", fontSize: 10.5, fontWeight: "700" }}>{userOccupation}</Text>
            </View>

            <View style={{ flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: "rgba(16, 185, 129, 0.15)", paddingHorizontal: 8, paddingVertical: 3.5, borderRadius: 10, borderWidth: 1, borderColor: "rgba(16, 185, 129, 0.3)" }}>
              <Wallet size={10} color="#34d399" />
              <Text style={{ color: "#6ee7b7", fontSize: 10.5, fontWeight: "700" }}>₹{userBudget.toLocaleString("en-IN")}/mo</Text>
            </View>

            <View
              style={{
                flexDirection: "row",
                alignItems: "center",
                gap: 4,
                backgroundColor:
                  userRisk === "High"
                    ? "rgba(244, 63, 94, 0.15)"
                    : userRisk === "Low"
                    ? "rgba(16, 185, 129, 0.15)"
                    : "rgba(6, 182, 212, 0.15)",
                paddingHorizontal: 8,
                paddingVertical: 3.5,
                borderRadius: 10,
                borderWidth: 1,
                borderColor:
                  userRisk === "High"
                    ? "rgba(244, 63, 94, 0.35)"
                    : userRisk === "Low"
                    ? "rgba(16, 185, 129, 0.35)"
                    : "rgba(6, 182, 212, 0.35)",
              }}
            >
              <ShieldCheck
                size={10}
                color={userRisk === "High" ? "#fb7185" : userRisk === "Low" ? "#34d399" : "#38bdf8"}
              />
              <Text
                style={{
                  color: userRisk === "High" ? "#fca5a5" : userRisk === "Low" ? "#6ee7b7" : "#7dd3fc",
                  fontSize: 10.5,
                  fontWeight: "700",
                }}
              >
                {userRisk} Risk
              </Text>
            </View>

            <View style={{ flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: "rgba(6, 182, 212, 0.15)", paddingHorizontal: 8, paddingVertical: 3.5, borderRadius: 10, borderWidth: 1, borderColor: "rgba(6, 182, 212, 0.3)" }}>
              <TargetIcon size={10} color="#38bdf8" />
              <Text style={{ color: "#7dd3fc", fontSize: 10.5, fontWeight: "700" }}>{userGoal}</Text>
            </View>
          </View>

          <View style={{ flexDirection: "row", alignItems: "center", gap: 2, paddingLeft: 6 }}>
            <Text style={{ color: C.textFaint, fontSize: 10, fontWeight: "600" }}>Edit</Text>
            <ChevronRight size={12} color={C.textFaint} />
          </View>
        </TouchableOpacity>

        {/* Scrollable Conversation */}
        <KeyboardAvoidingView
          style={{ flex: 1 }}
          behavior={Platform.OS === "ios" ? "padding" : undefined}
          keyboardVerticalOffset={Platform.OS === "ios" ? 10 : 0}
        >
          <ScrollView
            ref={scrollViewRef}
            contentContainerStyle={{ paddingHorizontal: 16, paddingTop: 16, paddingBottom: 24 }}
            showsVerticalScrollIndicator={false}
          >
            {/* Quick Hero Banner */}
            <View style={{ marginBottom: 20 }}>
              <AIAvatarHero />
            </View>

            {/* Quick Suggestion Chips (Hero Carousel) */}
            <View style={{ marginBottom: 20 }}>
              <Text style={{ color: C.textFaint, fontSize: 11, fontWeight: "700", letterSpacing: 1, marginBottom: 10, textTransform: "uppercase" }}>
                Suggested Inquiries
              </Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 10 }}>
                {suggestions.map((item, index) => {
                  const Icon = item.icon;
                  return (
                    <TouchableOpacity
                      key={index}
                      onPress={() => handleSend(item.prompt)}
                      style={{
                        backgroundColor: C.card,
                        borderRadius: 18,
                        borderWidth: 1,
                        borderColor: C.cardEdge,
                        padding: 14,
                        width: 200,
                        ...depthShadow("sm"),
                      }}
                    >
                      <View
                        style={{
                          width: 32,
                          height: 32,
                          borderRadius: 10,
                          backgroundColor: "rgba(255,79,129,0.12)",
                          alignItems: "center",
                          justifyContent: "center",
                          marginBottom: 8,
                        }}
                      >
                        <Icon size={16} color={C.pink} />
                      </View>
                      <Text style={{ color: "#fff", fontWeight: "700", fontSize: 13, marginBottom: 2 }}>
                        {item.title}
                      </Text>
                      <Text style={{ color: C.textFaint, fontSize: 11 }} numberOfLines={2}>
                        {item.desc}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </ScrollView>
            </View>

            {/* Chat Messages */}
            {messages.map((msg) => {
              const isUser = msg.role === "user";
              return (
                <View
                  key={msg.id}
                  style={{
                    alignSelf: isUser ? "flex-end" : "flex-start",
                    maxWidth: SCREEN_WIDTH * 0.88,
                    marginBottom: 16,
                  }}
                >
                  <View
                    style={{
                      backgroundColor: isUser ? "rgba(255,79,129,0.15)" : C.card,
                      borderWidth: 1,
                      borderColor: isUser ? "rgba(255,79,129,0.35)" : C.cardEdge,
                      borderRadius: 22,
                      borderBottomRightRadius: isUser ? 4 : 22,
                      borderBottomLeftRadius: isUser ? 22 : 4,
                      padding: 16,
                      ...depthShadow("sm"),
                    }}
                  >
                    {!isUser && (
                      <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                        <View style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
                          <View style={{ width: 6, height: 6, borderRadius: 3, backgroundColor: C.pink }} />
                          <Text style={{ color: C.pink, fontSize: 11, fontWeight: "700", letterSpacing: 0.5 }}>
                            MENTRAFI AI ADVISORY
                          </Text>
                        </View>
                        <TouchableOpacity
                          onPress={() => handleCopy(msg.id, msg.text)}
                          activeOpacity={0.7}
                          style={{
                            flexDirection: "row",
                            alignItems: "center",
                            gap: 4,
                            paddingHorizontal: 8,
                            paddingVertical: 3,
                            borderRadius: 8,
                            backgroundColor: copiedId === msg.id ? "rgba(74,222,128,0.15)" : "rgba(255,255,255,0.06)",
                            borderWidth: 1,
                            borderColor: copiedId === msg.id ? "rgba(74,222,128,0.3)" : "rgba(255,255,255,0.1)",
                          }}
                        >
                          {copiedId === msg.id ? (
                            <>
                              <Check size={11} color={GREEN} />
                              <Text style={{ color: GREEN, fontSize: 10, fontWeight: "600" }}>Copied</Text>
                            </>
                          ) : (
                            <>
                              <Copy size={11} color={C.textMuted} />
                              <Text style={{ color: C.textMuted, fontSize: 10, fontWeight: "500" }}>Copy</Text>
                            </>
                          )}
                        </TouchableOpacity>
                      </View>
                    )}

                    {isUser ? (
                      <View>
                        <Text
                          selectable={true}
                          selectionColor="#FF4F81"
                          style={{
                            color: "#fff",
                            fontSize: 14,
                            lineHeight: 22,
                            fontWeight: "400",
                          }}
                        >
                          {msg.text}
                        </Text>
                        <TouchableOpacity
                          onPress={() => handleCopy(msg.id, msg.text)}
                          activeOpacity={0.7}
                          style={{
                            alignSelf: "flex-end",
                            flexDirection: "row",
                            alignItems: "center",
                            gap: 4,
                            marginTop: 6,
                            paddingHorizontal: 7,
                            paddingVertical: 2.5,
                            borderRadius: 8,
                            backgroundColor: copiedId === msg.id ? "rgba(74,222,128,0.15)" : "rgba(255,255,255,0.06)",
                          }}
                        >
                          {copiedId === msg.id ? (
                            <>
                              <Check size={10} color={GREEN} />
                              <Text style={{ color: GREEN, fontSize: 9, fontWeight: "600" }}>Copied</Text>
                            </>
                          ) : (
                            <>
                              <Copy size={10} color={C.textMuted} />
                              <Text style={{ color: C.textMuted, fontSize: 9, fontWeight: "500" }}>Copy</Text>
                            </>
                          )}
                        </TouchableOpacity>
                      </View>
                    ) : (
                      <FormattedMessageText text={msg.text} />
                    )}

                    {/* Key Highlights Bullet points */}
                    {msg.keyHighlights && msg.keyHighlights.length > 0 && (
                      <View style={{ marginTop: 12, borderTopWidth: 1, borderTopColor: C.inputBorder, paddingTop: 10, gap: 6 }}>
                        {msg.keyHighlights.map((hl, i) => (
                          <View key={i} style={{ flexDirection: "row", alignItems: "flex-start", gap: 6 }}>
                            <CheckCircle2 size={13} color={C.cyan} style={{ marginTop: 3 }} />
                            <Text style={{ color: C.textMuted, fontSize: 12, flex: 1, lineHeight: 18 }}>
                              {hl}
                            </Text>
                          </View>
                        ))}
                      </View>
                    )}

                    {/* Recommended Funds Cards */}
                    {msg.funds && msg.funds.length > 0 && (
                      <View style={{ marginTop: 8 }}>
                        <Text
                          style={{
                            color: C.cyan,
                            fontSize: 11,
                            fontWeight: "700",
                            letterSpacing: 0.8,
                            marginTop: 8,
                            marginBottom: 4,
                            textTransform: "uppercase",
                          }}
                        >
                          Top Recommended Direct Funds
                        </Text>
                        {msg.funds.map((fund) => (
                          <FundCard key={fund.id} fund={fund} />
                        ))}
                      </View>
                    )}
                  </View>
                  <Text
                    style={{
                      color: C.textFaint,
                      fontSize: 10,
                      marginTop: 4,
                      alignSelf: isUser ? "flex-end" : "flex-start",
                      paddingHorizontal: 4,
                    }}
                  >
                    {msg.timestamp}
                  </Text>
                </View>
              );
            })}

            {/* Typing Indicator */}
            {isTyping && (
              <View
                style={{
                  alignSelf: "flex-start",
                  backgroundColor: C.card,
                  borderWidth: 1,
                  borderColor: C.cardEdge,
                  borderRadius: 18,
                  paddingHorizontal: 16,
                  paddingVertical: 12,
                  marginBottom: 16,
                  flexDirection: "row",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <ActivityIndicator size="small" color={C.pink} />
                <Text style={{ color: C.textMuted, fontSize: 12, fontStyle: "italic" }}>
                  Analyzing schemes & compounding math...
                </Text>
              </View>
            )}
          </ScrollView>

          {/* Input Bar */}
          <View
            style={{
              paddingHorizontal: 16,
              paddingTop: 10,
              paddingBottom: 12,
              backgroundColor: "rgba(10,10,20,0.95)",
              borderTopWidth: 1,
              borderTopColor: C.cardEdge,
            }}
          >
            <View
              style={{
                flexDirection: "row",
                alignItems: "center",
                backgroundColor: C.input,
                borderRadius: 25,
                borderWidth: 1,
                borderColor: C.inputBorder,
                paddingHorizontal: 14,
                paddingVertical: Platform.OS === "ios" ? 10 : 4,
              }}
            >
              <TextInput
                style={{
                  flex: 1,
                  color: "#fff",
                  fontSize: 14,
                  paddingHorizontal: 6,
                  maxHeight: 90,
                }}
                placeholder={`Ask MentraFi about ${userOccupation} SIP, tax, or funds...`}
                placeholderTextColor={C.textFaint}
                value={inputText}
                onChangeText={setInputText}
                onSubmitEditing={() => handleSend(inputText)}
                returnKeyType="send"
              />
              <TouchableOpacity
                onPress={() => handleSend(inputText)}
                disabled={!inputText.trim() || isTyping}
                style={{
                  opacity: inputText.trim() && !isTyping ? 1 : 0.4,
                }}
              >
                <LinearGradient
                  colors={[C.cyan, C.violet]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: 18,
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Send size={15} color="#fff" />
                </LinearGradient>
              </TouchableOpacity>
            </View>

            <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 5, marginTop: 7 }}>
              <ShieldCheck size={11} color={C.textFaint} />
              <Text style={{ color: C.textFaint, fontSize: 10, textAlign: "center" }}>
                SEBI Grounded • Zero Math Drift • 100% Direct-Growth
              </Text>
            </View>
          </View>
        </KeyboardAvoidingView>

        {/* Global Bottom Navigation */}
        <BottomNav tabs={tabs} paddingBottom={Platform.OS === "ios" ? 18 : 10} />
      </SafeAreaView>
    </View>
  );
}
