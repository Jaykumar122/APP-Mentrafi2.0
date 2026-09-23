import AsyncStorage from "@react-native-async-storage/async-storage";
import { useLocalSearchParams, useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import {
  ArrowLeft,
  CheckCircle2,
  ChevronRight,
  CreditCard,
  History,
  PieChart,
  Search,
  ShieldCheck,
  Smartphone,
  TrendingUp,
  X,
  Zap,
} from "lucide-react-native";
import React, { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Modal,
  ScrollView,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { AuthBackground, C, depthShadow } from "../(auth)/login";
import { API_URL } from "../../utils/api";

const GREEN = "#4ade80";

type Fund = {
  id: number;
  scheme_code: number;
  name: string;
  category: string;
  subcategory?: string;
  nav: number | string;
  oneYearReturn?: number | string;
  rating?: number | string;
};

const AMOUNT_PRESETS = [1000, 2500, 5000, 10000, 25000, 50000];

const PAYMENT_METHODS = [
  { id: "UPI_GPAY", name: "Google Pay", type: "UPI", icon: Smartphone, tag: "Instant Allotment" },
  { id: "UPI_PHONEPE", name: "PhonePe", type: "UPI", icon: Smartphone, tag: "Fast UPI" },
  { id: "UPI_PAYTM", name: "Paytm UPI", type: "UPI", icon: Smartphone, tag: "Instant" },
  { id: "UPI_CUSTOM", name: "Other UPI ID / VPA", type: "UPI_MANUAL", icon: Zap, tag: "Any UPI App" },
  { id: "NET_BANKING_HDFC", name: "HDFC NetBanking", type: "NET_BANKING", icon: CreditCard, tag: "Preferred" },
  { id: "NET_BANKING_SBI", name: "SBI NetBanking", type: "NET_BANKING", icon: CreditCard, tag: "Supported" },
  { id: "NET_BANKING_ICICI", name: "ICICI NetBanking", type: "NET_BANKING", icon: CreditCard, tag: "Supported" },
  { id: "DEBIT_CARD", name: "Debit Card / RuPay", type: "CARD", icon: CreditCard, tag: "Visa/MC/RuPay" },
];

export default function InvestScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ schemeCode?: string }>();

  const [selectedFund, setSelectedFund] = useState<Fund | null>(null);
  const [loadingInitialFund, setLoadingInitialFund] = useState(false);

  // Search funds modal
  const [showFundSearch, setShowFundSearch] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Fund[]>([]);
  const [searching, setSearching] = useState(false);

  // Amount
  const [amountStr, setAmountStr] = useState("5000");

  // Payment method
  const [selectedMethod, setSelectedMethod] = useState("UPI_GPAY");
  const [customUpiId, setCustomUpiId] = useState("");

  // Payment process modal
  const [processing, setProcessing] = useState(false);
  const [processStep, setProcessStep] = useState<"connecting" | "authorizing" | "confirming">("connecting");
  const [successData, setSuccessData] = useState<any | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Load fund if schemeCode was passed as query param, or fetch starter fund
  useEffect(() => {
    async function loadFund() {
      try {
        setLoadingInitialFund(true);
        const query = params.schemeCode ? `schemeCode=${params.schemeCode}` : "limit=1";
        const res = await fetch(`${API_URL}/api/funds?${query}`);
        if (res.ok) {
          const data = await res.json();
          if (data.funds && data.funds.length > 0) {
            setSelectedFund(data.funds[0]);
          }
        }
      } catch (e) {
        console.error("Error loading fund:", e);
      } finally {
        setLoadingInitialFund(false);
      }
    }
    loadFund();
  }, [params.schemeCode]);

  // Search funds
  useEffect(() => {
    if (!searchQuery.trim() || searchQuery.length < 2) {
      setSearchResults([]);
      return;
    }
    const timer = setTimeout(async () => {
      try {
        setSearching(true);
        const res = await fetch(`${API_URL}/api/funds?q=${encodeURIComponent(searchQuery)}&limit=15`);
        if (res.ok) {
          const data = await res.json();
          setSearchResults(data.funds || []);
        }
      } catch (e) {
        console.error("Search error:", e);
      } finally {
        setSearching(false);
      }
    }, 250);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const amountNum = parseFloat(amountStr) || 0;
  const navNum = selectedFund ? parseFloat(String(selectedFund.nav)) || 10 : 10;
  const estimatedUnits = navNum > 0 ? (amountNum / navNum).toFixed(4) : "0.0000";

  async function handleProceedPayment() {
    if (!selectedFund) {
      setErrorMessage("Please select a mutual fund to invest in");
      return;
    }
    if (amountNum < 100) {
      setErrorMessage("Minimum investment amount is ₹100");
      return;
    }
    if (selectedMethod === "UPI_CUSTOM" && (!customUpiId.includes("@") || customUpiId.length < 4)) {
      setErrorMessage("Please enter a valid UPI ID (e.g. mobile@upi)");
      return;
    }

    setErrorMessage(null);
    setProcessing(true);
    setProcessStep("connecting");

    try {
      const token = await AsyncStorage.getItem("userToken");
      if (!token) {
        setProcessing(false);
        router.replace("/(auth)/login");
        return;
      }

      // Step 1: Simulated gateway handshake
      await new Promise((r) => setTimeout(r, 650));
      setProcessStep("authorizing");

      // Step 2: Simulated bank authorization
      await new Promise((r) => setTimeout(r, 750));
      setProcessStep("confirming");

      // Step 3: Real atomic execution on backend
      const res = await fetch(`${API_URL}/api/portfolio/invest`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          schemeCode: selectedFund.scheme_code,
          amount: amountNum,
          paymentMethod: selectedMethod,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Investment request failed");
      }

      await new Promise((r) => setTimeout(r, 400));
      setSuccessData({
        ...data,
        amount: amountNum,
        units: estimatedUnits,
        fundName: selectedFund.name,
        nav: navNum,
      });
    } catch (err: any) {
      console.error("Investment error:", err);
      setErrorMessage(err.message || "Failed to complete transaction");
      setProcessing(false);
    }
  }

  return (
    <View style={{ flex: 1, backgroundColor: C.bgBottom }}>
      <AuthBackground />
      <SafeAreaView style={{ flex: 1 }} edges={["top", "bottom"]}>
        {/* Header */}
        <View
          style={{
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "space-between",
            paddingHorizontal: 20,
            paddingVertical: 14,
            borderBottomWidth: 1,
            borderBottomColor: C.cardEdge,
          }}
        >
          <TouchableOpacity
            onPress={() => router.back()}
            style={{
              width: 40,
              height: 40,
              borderRadius: 20,
              backgroundColor: C.input,
              alignItems: "center",
              justifyContent: "center",
              borderWidth: 1,
              borderColor: C.inputBorder,
            }}
          >
            <ArrowLeft size={20} color="#fff" />
          </TouchableOpacity>
          <Text style={{ color: "#fff", fontSize: 18, fontWeight: "700" }}>Invest (Lump Sum)</Text>
          <TouchableOpacity
            onPress={() => router.push("/transactions" as any)}
            style={{
              width: 40,
              height: 40,
              borderRadius: 20,
              backgroundColor: C.input,
              alignItems: "center",
              justifyContent: "center",
              borderWidth: 1,
              borderColor: C.inputBorder,
            }}
          >
            <History size={18} color={C.cyan} />
          </TouchableOpacity>
        </View>

        <ScrollView
          showsVerticalScrollIndicator={false}
          contentContainerStyle={{ paddingHorizontal: 20, paddingVertical: 20, paddingBottom: 40 }}
        >
          {errorMessage && (
            <View
              style={{
                backgroundColor: "rgba(255,107,129,0.15)",
                borderWidth: 1,
                borderColor: "rgba(255,107,129,0.4)",
                padding: 14,
                borderRadius: 16,
                marginBottom: 16,
              }}
            >
              <Text style={{ color: "#ff6b81", fontSize: 13, fontWeight: "600" }}>{errorMessage}</Text>
            </View>
          )}

          {/* Fund Selection Section */}
          <Text style={{ color: C.pink, fontSize: 11, fontWeight: "700", letterSpacing: 1.2, marginBottom: 8 }}>
            SELECT SCHEME
          </Text>

          {loadingInitialFund ? (
            <View
              style={{
                backgroundColor: C.card,
                padding: 24,
                borderRadius: 20,
                alignItems: "center",
                borderWidth: 1,
                borderColor: C.cardEdge,
              }}
            >
              <ActivityIndicator size="small" color={C.pink} />
              <Text style={{ color: C.textMuted, fontSize: 12, marginTop: 8 }}>Loading scheme details...</Text>
            </View>
          ) : selectedFund ? (
            <View
              style={{
                backgroundColor: C.card,
                borderRadius: 22,
                borderWidth: 1,
                borderColor: C.cardEdge,
                padding: 16,
                marginBottom: 20,
                ...depthShadow("sm"),
              }}
            >
              <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                <View style={{ flex: 1, marginRight: 12 }}>
                  <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700", marginBottom: 4 }} numberOfLines={2}>
                    {selectedFund.name}
                  </Text>
                  <Text style={{ color: C.textFaint, fontSize: 12 }}>{selectedFund.category}</Text>
                </View>
                <TouchableOpacity
                  onPress={() => setShowFundSearch(true)}
                  style={{
                    backgroundColor: "rgba(255,79,129,0.12)",
                    paddingHorizontal: 12,
                    paddingVertical: 6,
                    borderRadius: 999,
                    borderWidth: 1,
                    borderColor: "rgba(255,79,129,0.3)",
                  }}
                >
                  <Text style={{ color: C.pink, fontSize: 11, fontWeight: "700" }}>Change</Text>
                </TouchableOpacity>
              </View>

              <View
                style={{
                  flexDirection: "row",
                  justifyContent: "space-between",
                  paddingTop: 12,
                  borderTopWidth: 1,
                  borderTopColor: C.inputBorder,
                }}
              >
                <View>
                  <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 2 }}>Current NAV</Text>
                  <Text style={{ color: "#fff", fontWeight: "700", fontSize: 14 }}>
                    ₹{parseFloat(String(selectedFund.nav || 0)).toFixed(4)}
                  </Text>
                </View>
                {selectedFund.oneYearReturn != null && (
                  <View style={{ alignItems: "flex-end" }}>
                    <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 2 }}>1Y Return</Text>
                    <Text style={{ color: GREEN, fontWeight: "700", fontSize: 14 }}>
                      +{selectedFund.oneYearReturn}%
                    </Text>
                  </View>
                )}
              </View>
            </View>
          ) : (
            <TouchableOpacity
              onPress={() => setShowFundSearch(true)}
              style={{
                backgroundColor: C.card,
                padding: 18,
                borderRadius: 20,
                borderWidth: 1,
                borderColor: C.cardEdge,
                flexDirection: "row",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 20,
              }}
            >
              <View style={{ flexDirection: "row", alignItems: "center", gap: 12 }}>
                <Search size={20} color={C.pink} />
                <Text style={{ color: C.textMuted, fontSize: 14 }}>Search and select a mutual fund...</Text>
              </View>
              <ChevronRight size={18} color={C.textMuted} />
            </TouchableOpacity>
          )}

          {/* Investment Amount Input */}
          <Text style={{ color: C.pink, fontSize: 11, fontWeight: "700", letterSpacing: 1.2, marginBottom: 8 }}>
            INVESTMENT AMOUNT
          </Text>

          <View
            style={{
              backgroundColor: C.card,
              borderRadius: 22,
              borderWidth: 1,
              borderColor: C.cardEdge,
              padding: 20,
              marginBottom: 20,
              ...depthShadow("sm"),
            }}
          >
            <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 14 }}>
              <Text style={{ color: "#fff", fontSize: 32, fontWeight: "700", marginRight: 8 }}>₹</Text>
              <TextInput
                keyboardType="numeric"
                value={amountStr}
                onChangeText={setAmountStr}
                placeholder="5000"
                placeholderTextColor={C.textFaint}
                style={{
                  flex: 1,
                  color: "#fff",
                  fontSize: 32,
                  fontWeight: "700",
                  padding: 0,
                }}
              />
            </View>

            {/* Amount Presets */}
            <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
              {AMOUNT_PRESETS.map((p) => {
                const isSelected = amountNum === p;
                return (
                  <TouchableOpacity
                    key={p}
                    onPress={() => setAmountStr(String(p))}
                    style={{
                      paddingHorizontal: 12,
                      paddingVertical: 7,
                      borderRadius: 999,
                      backgroundColor: isSelected ? "rgba(255,79,129,0.2)" : C.input,
                      borderWidth: 1,
                      borderColor: isSelected ? C.pink : C.inputBorder,
                    }}
                  >
                    <Text
                      style={{
                        color: isSelected ? "#fff" : C.textMuted,
                        fontSize: 12,
                        fontWeight: isSelected ? "700" : "500",
                      }}
                    >
                      +₹{p.toLocaleString("en-IN")}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
          </View>

          {/* Estimated Units Card */}
          <LinearGradient
            colors={["rgba(255,79,129,0.08)", "rgba(121,82,252,0.08)"]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={{
              borderRadius: 20,
              borderWidth: 1,
              borderColor: "rgba(255,255,255,0.08)",
              padding: 16,
              marginBottom: 24,
            }}
          >
            <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <Text style={{ color: C.textMuted, fontSize: 13 }}>Estimated Units Allotted</Text>
              <Text style={{ color: C.cyan, fontSize: 16, fontWeight: "700" }}>{estimatedUnits} units</Text>
            </View>
            <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
              <Text style={{ color: C.textFaint, fontSize: 11 }}>Applicable NAV</Text>
              <Text style={{ color: "#fff", fontSize: 12, fontWeight: "600" }}>₹{navNum.toFixed(4)} (Today's NAV)</Text>
            </View>
            <View style={{ marginTop: 10, paddingTop: 8, borderTopWidth: 1, borderTopColor: "rgba(255,255,255,0.05)" }}>
              <Text style={{ color: C.textFaint, fontSize: 10 }}>
                Orders placed before 2:00 PM IST qualify for today's NAV execution. Zero transaction fee.
              </Text>
            </View>
          </LinearGradient>

          {/* Payment Method Selector */}
          <Text style={{ color: C.pink, fontSize: 11, fontWeight: "700", letterSpacing: 1.2, marginBottom: 12 }}>
            PAYMENT METHOD
          </Text>

          <View style={{ gap: 10, marginBottom: 28 }}>
            {PAYMENT_METHODS.map((method) => {
              const active = selectedMethod === method.id;
              const Icon = method.icon;
              return (
                <TouchableOpacity
                  key={method.id}
                  onPress={() => setSelectedMethod(method.id)}
                  style={{
                    backgroundColor: active ? "rgba(255,79,129,0.12)" : C.card,
                    borderWidth: 1,
                    borderColor: active ? C.pink : C.cardEdge,
                    borderRadius: 18,
                    padding: 14,
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "space-between",
                  }}
                >
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 14 }}>
                    <View
                      style={{
                        width: 38,
                        height: 38,
                        borderRadius: 12,
                        backgroundColor: active ? C.pink : C.input,
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <Icon size={18} color="#fff" />
                    </View>
                    <View>
                      <Text style={{ color: "#fff", fontWeight: "600", fontSize: 14 }}>{method.name}</Text>
                      <Text style={{ color: C.textFaint, fontSize: 11 }}>{method.tag}</Text>
                    </View>
                  </View>

                  <View
                    style={{
                      width: 20,
                      height: 20,
                      borderRadius: 10,
                      borderWidth: 2,
                      borderColor: active ? C.pink : C.textFaint,
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    {active && (
                      <View style={{ width: 10, height: 10, borderRadius: 5, backgroundColor: C.pink }} />
                    )}
                  </View>
                </TouchableOpacity>
              );
            })}

            {selectedMethod === "UPI_CUSTOM" && (
              <View
                style={{
                  backgroundColor: C.card,
                  borderRadius: 18,
                  borderWidth: 1,
                  borderColor: C.cardEdge,
                  padding: 14,
                  marginTop: 4,
                }}
              >
                <Text style={{ color: C.textMuted, fontSize: 12, marginBottom: 8 }}>Enter your VPA / UPI ID:</Text>
                <TextInput
                  value={customUpiId}
                  onChangeText={setCustomUpiId}
                  placeholder="e.g. yourname@okaxis"
                  placeholderTextColor={C.textFaint}
                  autoCapitalize="none"
                  style={{
                    backgroundColor: C.input,
                    borderWidth: 1,
                    borderColor: C.inputBorder,
                    borderRadius: 12,
                    paddingHorizontal: 14,
                    paddingVertical: 10,
                    color: "#fff",
                    fontSize: 14,
                  }}
                />
              </View>
            )}
          </View>

          {/* Invest Now CTA */}
          <TouchableOpacity
            onPress={handleProceedPayment}
            activeOpacity={0.85}
            style={{ borderRadius: 20, overflow: "hidden", ...depthShadow("md") }}
          >
            <LinearGradient
              colors={[C.pink, C.violet]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={{
                paddingVertical: 18,
                alignItems: "center",
                justifyContent: "center",
                flexDirection: "row",
                gap: 8,
              }}
            >
              <ShieldCheck size={20} color="#fff" />
              <Text style={{ color: "#fff", fontSize: 16, fontWeight: "700" }}>
                Pay ₹{amountNum.toLocaleString("en-IN")} & Invest
              </Text>
            </LinearGradient>
          </TouchableOpacity>
        </ScrollView>

        {/* Modal 1: Fund Search Modal */}
        <Modal visible={showFundSearch} animationType="slide" transparent>
          <View style={{ flex: 1, backgroundColor: "rgba(0,0,0,0.85)", justifyContent: "flex-end" }}>
            <View
              style={{
                backgroundColor: "#0d0b17",
                borderTopLeftRadius: 30,
                borderTopRightRadius: 30,
                borderTopWidth: 1,
                borderColor: C.cardEdge,
                paddingTop: 20,
                paddingHorizontal: 20,
                height: "80%",
              }}
            >
              <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                <Text style={{ color: "#fff", fontSize: 18, fontWeight: "700" }}>Select Fund</Text>
                <TouchableOpacity
                  onPress={() => setShowFundSearch(false)}
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: 18,
                    backgroundColor: C.input,
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <X size={18} color="#fff" />
                </TouchableOpacity>
              </View>

              {/* Search input */}
              <View
                style={{
                  backgroundColor: C.input,
                  borderWidth: 1,
                  borderColor: C.inputBorder,
                  borderRadius: 16,
                  flexDirection: "row",
                  alignItems: "center",
                  paddingHorizontal: 14,
                  paddingVertical: 10,
                  marginBottom: 16,
                }}
              >
                <Search size={18} color={C.textMuted} style={{ marginRight: 10 }} />
                <TextInput
                  value={searchQuery}
                  onChangeText={setSearchQuery}
                  placeholder="Search funds by name or AMC..."
                  placeholderTextColor={C.textFaint}
                  autoFocus
                  style={{ flex: 1, color: "#fff", fontSize: 14, padding: 0 }}
                />
                {searching && <ActivityIndicator size="small" color={C.pink} />}
              </View>

              <ScrollView showsVerticalScrollIndicator={false}>
                {searchResults.length > 0 ? (
                  searchResults.map((fund) => (
                    <TouchableOpacity
                      key={fund.scheme_code}
                      onPress={() => {
                        setSelectedFund(fund);
                        setShowFundSearch(false);
                      }}
                      style={{
                        paddingVertical: 14,
                        borderBottomWidth: 1,
                        borderBottomColor: C.cardEdge,
                      }}
                    >
                      <Text style={{ color: "#fff", fontSize: 14, fontWeight: "600", marginBottom: 4 }}>
                        {fund.name}
                      </Text>
                      <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                        <Text style={{ color: C.pink, fontSize: 11, fontWeight: "600" }}>{fund.category}</Text>
                        <Text style={{ color: C.textFaint, fontSize: 11 }}>•</Text>
                        <Text style={{ color: C.textMuted, fontSize: 11 }}>NAV: ₹{parseFloat(String(fund.nav || 0)).toFixed(2)}</Text>
                      </View>
                    </TouchableOpacity>
                  ))
                ) : (
                  <View style={{ alignItems: "center", paddingTop: 40 }}>
                    <Text style={{ color: C.textMuted, fontSize: 13 }}>
                      {searchQuery.length < 2 ? "Type to search over 1,000+ mutual funds" : "No matching funds found"}
                    </Text>
                  </View>
                )}
              </ScrollView>
            </View>
          </View>
        </Modal>

        {/* Modal 2: Payment Gateway Processing & Success */}
        <Modal visible={processing || !!successData} transparent animationType="fade">
          <View
            style={{
              flex: 1,
              backgroundColor: "rgba(0,0,0,0.92)",
              alignItems: "center",
              justifyContent: "center",
              paddingHorizontal: 24,
            }}
          >
            {processing && !successData && (
              <View
                style={{
                  width: "100%",
                  backgroundColor: C.card,
                  borderRadius: 28,
                  borderWidth: 1,
                  borderColor: C.cardEdge,
                  padding: 28,
                  alignItems: "center",
                  ...depthShadow("lg"),
                }}
              >
                <ActivityIndicator size="large" color={C.pink} style={{ marginBottom: 20 }} />
                <Text style={{ color: "#fff", fontSize: 18, fontWeight: "700", marginBottom: 8 }}>
                  {processStep === "connecting"
                    ? "Connecting to Payment Gateway..."
                    : processStep === "authorizing"
                    ? "Authorizing with Bank..."
                    : "Confirming Unit Allotment..."}
                </Text>
                <Text style={{ color: C.textMuted, fontSize: 12, textAlign: "center", marginBottom: 20 }}>
                  Securing 256-bit encrypted connection with NPCI & Exchange
                </Text>
                <View
                  style={{
                    width: "100%",
                    height: 4,
                    backgroundColor: C.input,
                    borderRadius: 2,
                    overflow: "hidden",
                  }}
                >
                  <LinearGradient
                    colors={[C.pink, C.violet]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                    style={{
                      width: processStep === "connecting" ? "35%" : processStep === "authorizing" ? "70%" : "95%",
                      height: "100%",
                    }}
                  />
                </View>
              </View>
            )}

            {successData && (
              <View
                style={{
                  width: "100%",
                  backgroundColor: C.card,
                  borderRadius: 28,
                  borderWidth: 1,
                  borderColor: C.cardEdge,
                  padding: 28,
                  alignItems: "center",
                  ...depthShadow("lg"),
                }}
              >
                <View
                  style={{
                    width: 64,
                    height: 64,
                    borderRadius: 32,
                    backgroundColor: "rgba(74,222,128,0.15)",
                    alignItems: "center",
                    justifyContent: "center",
                    marginBottom: 16,
                  }}
                >
                  <CheckCircle2 size={36} color={GREEN} />
                </View>

                <Text style={{ color: "#fff", fontSize: 20, fontWeight: "700", marginBottom: 4 }}>
                  Investment Successful!
                </Text>
                <Text style={{ color: C.textMuted, fontSize: 13, textAlign: "center", marginBottom: 20 }}>
                  ₹{successData.amount.toLocaleString("en-IN")} allocated to {successData.fundName}
                </Text>

                {/* Receipt Card */}
                <View
                  style={{
                    width: "100%",
                    backgroundColor: C.input,
                    borderRadius: 18,
                    borderWidth: 1,
                    borderColor: C.inputBorder,
                    padding: 16,
                    gap: 10,
                    marginBottom: 24,
                  }}
                >
                  <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                    <Text style={{ color: C.textFaint, fontSize: 12 }}>Units Allotted</Text>
                    <Text style={{ color: C.cyan, fontWeight: "700", fontSize: 13 }}>{successData.units} Units</Text>
                  </View>
                  <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                    <Text style={{ color: C.textFaint, fontSize: 12 }}>Allotment NAV</Text>
                    <Text style={{ color: "#fff", fontWeight: "600", fontSize: 13 }}>₹{successData.nav.toFixed(4)}</Text>
                  </View>
                  <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                    <Text style={{ color: C.textFaint, fontSize: 12 }}>Transaction Ref</Text>
                    <Text style={{ color: C.textMuted, fontSize: 12 }}>TXN-{successData.transaction?.id || Date.now()}</Text>
                  </View>
                  <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                    <Text style={{ color: C.textFaint, fontSize: 12 }}>Payment Method</Text>
                    <Text style={{ color: GREEN, fontWeight: "600", fontSize: 12 }}>{selectedMethod}</Text>
                  </View>
                </View>

                <View style={{ width: "100%", gap: 10 }}>
                  <TouchableOpacity
                    onPress={() => {
                      setSuccessData(null);
                      setProcessing(false);
                      router.replace("/portfolio" as any);
                    }}
                    style={{
                      borderRadius: 16,
                      overflow: "hidden",
                    }}
                  >
                    <LinearGradient
                      colors={[C.pink, C.violet]}
                      style={{ paddingVertical: 14, alignItems: "center" }}
                    >
                      <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700" }}>Go to My Portfolio</Text>
                    </LinearGradient>
                  </TouchableOpacity>

                  <TouchableOpacity
                    onPress={() => {
                      setSuccessData(null);
                      setProcessing(false);
                      router.replace("/transactions" as any);
                    }}
                    style={{
                      paddingVertical: 14,
                      alignItems: "center",
                      backgroundColor: C.input,
                      borderRadius: 16,
                      borderWidth: 1,
                      borderColor: C.inputBorder,
                    }}
                  >
                    <Text style={{ color: C.textMuted, fontSize: 14, fontWeight: "600" }}>View Transaction History</Text>
                  </TouchableOpacity>
                </View>
              </View>
            )}
          </View>
        </Modal>
      </SafeAreaView>
    </View>
  );
}
