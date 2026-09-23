import AsyncStorage from "@react-native-async-storage/async-storage";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import {
  AlertCircle,
  ArrowDownCircle,
  ArrowLeft,
  Banknote,
  Building2,
  CheckCircle2,
  History,
  Info,
  Layers,
  PieChart,
  ShieldCheck,
  TrendingDown,
} from "lucide-react-native";
import React, { useEffect, useState } from "react";
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
const RED = "#ff6b81";

type Holding = {
  id: string;
  schemeCode: number;
  name: string;
  assetClass: string;
  units: number;
  avgPurchaseNav: number;
  currentNav: number;
  value: number;
  investedAmount: number;
};

const PERCENT_PRESETS = [25, 50, 75, 100];

export default function RedeemScreen() {
  const router = useRouter();

  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedHolding, setSelectedHolding] = useState<Holding | null>(null);

  const [unitsStr, setUnitsStr] = useState("");
  const [activePercent, setActivePercent] = useState<number | null>(null);

  const [bankAccount, setBankAccount] = useState("HDFC Bank •••• 4821 (Primary)");
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [redeeming, setRedeeming] = useState(false);
  const [successData, setSuccessData] = useState<any | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function fetchHoldings() {
    try {
      setLoading(true);
      const token = await AsyncStorage.getItem("userToken");
      if (!token) {
        router.replace("/(auth)/login");
        return;
      }
      const res = await fetch(`${API_URL}/api/portfolio`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        const list: Holding[] = data.holdings || [];
        setHoldings(list);
        if (list.length > 0) {
          setSelectedHolding(list[0]);
          const defaultUnits = (list[0].units * 0.5).toFixed(4);
          setUnitsStr(defaultUnits);
          setActivePercent(50);
        }
      }
    } catch (e) {
      console.error("Error fetching holdings:", e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchHoldings();
  }, []);

  function handleSelectHolding(h: Holding) {
    setSelectedHolding(h);
    setActivePercent(100);
    setUnitsStr(h.units.toFixed(4));
    setErrorMessage(null);
  }

  function handlePercentSelect(pct: number) {
    if (!selectedHolding) return;
    setActivePercent(pct);
    const calculated = (selectedHolding.units * (pct / 100)).toFixed(4);
    setUnitsStr(calculated);
    setErrorMessage(null);
  }

  const unitsNum = parseFloat(unitsStr) || 0;
  const currentNav = selectedHolding ? selectedHolding.currentNav || 10 : 10;
  const estimatedPayout = Number((unitsNum * currentNav).toFixed(2));
  const maxUnits = selectedHolding ? selectedHolding.units : 0;

  async function handleExecuteRedeem() {
    if (!selectedHolding) return;
    if (unitsNum <= 0) {
      setErrorMessage("Please enter a positive units amount to redeem");
      return;
    }
    if (unitsNum > maxUnits + 0.0001) {
      setErrorMessage(`Cannot redeem more than your available ${maxUnits.toFixed(4)} units`);
      return;
    }

    setShowConfirmModal(false);
    setRedeeming(true);
    setErrorMessage(null);

    try {
      const token = await AsyncStorage.getItem("userToken");
      if (!token) {
        setRedeeming(false);
        router.replace("/(auth)/login");
        return;
      }

      const res = await fetch(`${API_URL}/api/portfolio/redeem`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          schemeCode: selectedHolding.schemeCode,
          units: unitsNum,
          payoutMethod: "BANK_TRANSFER",
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Redemption request failed");
      }

      setSuccessData({
        ...data,
        unitsRedeemed: unitsNum,
        payoutAmount: estimatedPayout,
        fundName: selectedHolding.name,
        bankAccount,
      });
    } catch (err: any) {
      console.error("Redeem error:", err);
      setErrorMessage(err.message || "Failed to execute redemption");
    } finally {
      setRedeeming(false);
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
          <Text style={{ color: "#fff", fontSize: 18, fontWeight: "700" }}>Redeem / Sell Units</Text>
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
                flexDirection: "row",
                alignItems: "center",
                gap: 8,
              }}
            >
              <AlertCircle size={18} color={RED} />
              <Text style={{ color: RED, fontSize: 13, fontWeight: "600", flex: 1 }}>{errorMessage}</Text>
            </View>
          )}

          {loading ? (
            <View style={{ alignItems: "center", paddingVertical: 50 }}>
              <ActivityIndicator size="large" color={C.pink} />
              <Text style={{ color: C.textMuted, fontSize: 13, marginTop: 12 }}>Loading portfolio holdings...</Text>
            </View>
          ) : holdings.length === 0 ? (
            <View
              style={{
                backgroundColor: C.card,
                padding: 30,
                borderRadius: 24,
                alignItems: "center",
                borderWidth: 1,
                borderColor: C.cardEdge,
                marginTop: 20,
              }}
            >
              <Layers size={44} color={C.textMuted} style={{ marginBottom: 14 }} />
              <Text style={{ color: "#fff", fontSize: 18, fontWeight: "700", marginBottom: 6 }}>
                No Active Holdings
              </Text>
              <Text style={{ color: C.textMuted, fontSize: 13, textAlign: "center", marginBottom: 20 }}>
                You do not have any mutual fund units in your portfolio to redeem yet.
              </Text>
              <TouchableOpacity
                onPress={() => router.replace("/invest" as any)}
                style={{ borderRadius: 16, overflow: "hidden" }}
              >
                <LinearGradient colors={[C.pink, C.violet]} style={{ paddingHorizontal: 24, paddingVertical: 12 }}>
                  <Text style={{ color: "#fff", fontWeight: "700", fontSize: 14 }}>Invest Now</Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          ) : (
            <>
              {/* Select Holding Section */}
              <Text style={{ color: C.pink, fontSize: 11, fontWeight: "700", letterSpacing: 1.2, marginBottom: 10 }}>
                SELECT HOLDING TO REDEEM
              </Text>

              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={{ gap: 12, marginBottom: 20 }}
              >
                {holdings.map((h) => {
                  const isSelected = selectedHolding?.schemeCode === h.schemeCode;
                  return (
                    <TouchableOpacity
                      key={h.id}
                      onPress={() => handleSelectHolding(h)}
                      style={{
                        backgroundColor: isSelected ? "rgba(255,79,129,0.12)" : C.card,
                        borderWidth: 1,
                        borderColor: isSelected ? C.pink : C.cardEdge,
                        borderRadius: 20,
                        padding: 16,
                        width: 220,
                      }}
                    >
                      <Text
                        style={{ color: "#fff", fontSize: 13, fontWeight: "700", marginBottom: 6 }}
                        numberOfLines={1}
                      >
                        {h.name}
                      </Text>
                      <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 8 }}>
                        {h.units.toFixed(2)} units available
                      </Text>
                      <Text style={{ color: C.cyan, fontSize: 16, fontWeight: "700" }}>
                        ₹{h.value.toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </ScrollView>

              {/* Units Input Section */}
              <Text style={{ color: C.pink, fontSize: 11, fontWeight: "700", letterSpacing: 1.2, marginBottom: 8 }}>
                REDEMPTION QUANTITY
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
                <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                  <Text style={{ color: C.textFaint, fontSize: 12 }}>Units to Sell</Text>
                  <Text style={{ color: C.textMuted, fontSize: 12 }}>
                    Max: <Text style={{ color: "#fff", fontWeight: "700" }}>{maxUnits.toFixed(4)}</Text>
                  </Text>
                </View>

                <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 16 }}>
                  <TextInput
                    keyboardType="numeric"
                    value={unitsStr}
                    onChangeText={(val) => {
                      setUnitsStr(val);
                      setActivePercent(null);
                      setErrorMessage(null);
                    }}
                    placeholder="0.0000"
                    placeholderTextColor={C.textFaint}
                    style={{
                      flex: 1,
                      color: "#fff",
                      fontSize: 28,
                      fontWeight: "700",
                      padding: 0,
                    }}
                  />
                  <Text style={{ color: C.textMuted, fontSize: 15, fontWeight: "600" }}>Units</Text>
                </View>

                {/* Percentage Chips */}
                <View style={{ flexDirection: "row", gap: 8 }}>
                  {PERCENT_PRESETS.map((pct) => {
                    const isSelected = activePercent === pct;
                    return (
                      <TouchableOpacity
                        key={pct}
                        onPress={() => handlePercentSelect(pct)}
                        style={{
                          flex: 1,
                          paddingVertical: 8,
                          borderRadius: 12,
                          alignItems: "center",
                          backgroundColor: isSelected ? "rgba(255,79,129,0.2)" : C.input,
                          borderWidth: 1,
                          borderColor: isSelected ? C.pink : C.inputBorder,
                        }}
                      >
                        <Text
                          style={{
                            color: isSelected ? "#fff" : C.textMuted,
                            fontSize: 12,
                            fontWeight: isSelected ? "700" : "600",
                          }}
                        >
                          {pct === 100 ? "All" : `${pct}%`}
                        </Text>
                      </TouchableOpacity>
                    );
                  })}
                </View>
              </View>

              {/* Payout Calculation Card */}
              <View
                style={{
                  backgroundColor: C.card,
                  borderRadius: 22,
                  borderWidth: 1,
                  borderColor: C.cardEdge,
                  padding: 18,
                  gap: 12,
                  marginBottom: 20,
                  ...depthShadow("sm"),
                }}
              >
                <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                  <Text style={{ color: C.textMuted, fontSize: 13 }}>Current NAV</Text>
                  <Text style={{ color: "#fff", fontWeight: "600", fontSize: 13 }}>₹{currentNav.toFixed(4)}</Text>
                </View>
                <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                  <Text style={{ color: C.textMuted, fontSize: 13 }}>Exit Load</Text>
                  <Text style={{ color: GREEN, fontWeight: "600", fontSize: 13 }}>₹0.00 (Nil)</Text>
                </View>
                <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                  <Text style={{ color: C.textMuted, fontSize: 13 }}>Settlement Timeline</Text>
                  <Text style={{ color: C.cyan, fontWeight: "600", fontSize: 13 }}>T+2 Business Days</Text>
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
                  <Text style={{ color: "#fff", fontWeight: "700", fontSize: 15 }}>Estimated Payout</Text>
                  <Text style={{ color: GREEN, fontWeight: "700", fontSize: 18 }}>
                    ₹{estimatedPayout.toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                  </Text>
                </View>
              </View>

              {/* Destination Bank Account */}
              <Text style={{ color: C.pink, fontSize: 11, fontWeight: "700", letterSpacing: 1.2, marginBottom: 8 }}>
                PAYOUT DESTINATION
              </Text>

              <View
                style={{
                  backgroundColor: C.card,
                  borderRadius: 20,
                  borderWidth: 1,
                  borderColor: C.cardEdge,
                  padding: 16,
                  flexDirection: "row",
                  alignItems: "center",
                  gap: 14,
                  marginBottom: 28,
                }}
              >
                <View
                  style={{
                    width: 42,
                    height: 42,
                    borderRadius: 14,
                    backgroundColor: "rgba(0,242,254,0.12)",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Building2 size={22} color={C.cyan} />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={{ color: "#fff", fontWeight: "600", fontSize: 14 }}>{bankAccount}</Text>
                  <Text style={{ color: C.textFaint, fontSize: 11 }}>Direct NEFT/RTGS credit to linked account</Text>
                </View>
              </View>

              {/* Redeem CTA */}
              <TouchableOpacity
                onPress={() => {
                  if (unitsNum <= 0) {
                    setErrorMessage("Please enter units to redeem");
                    return;
                  }
                  if (unitsNum > maxUnits + 0.0001) {
                    setErrorMessage(`Cannot exceed available ${maxUnits.toFixed(4)} units`);
                    return;
                  }
                  setShowConfirmModal(true);
                }}
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
                  <ArrowDownCircle size={20} color="#fff" />
                  <Text style={{ color: "#fff", fontSize: 16, fontWeight: "700" }}>
                    Redeem ₹{estimatedPayout.toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                  </Text>
                </LinearGradient>
              </TouchableOpacity>
            </>
          )}
        </ScrollView>

        {/* Confirmation Modal */}
        <Modal visible={showConfirmModal} transparent animationType="fade">
          <View
            style={{
              flex: 1,
              backgroundColor: "rgba(0,0,0,0.85)",
              alignItems: "center",
              justifyContent: "center",
              paddingHorizontal: 24,
            }}
          >
            <View
              style={{
                width: "100%",
                backgroundColor: C.card,
                borderRadius: 24,
                borderWidth: 1,
                borderColor: C.cardEdge,
                padding: 24,
                ...depthShadow("lg"),
              }}
            >
              <Text style={{ color: "#fff", fontSize: 18, fontWeight: "700", marginBottom: 12 }}>
                Confirm Redemption?
              </Text>
              <Text style={{ color: C.textMuted, fontSize: 13, lineHeight: 20, marginBottom: 20 }}>
                You are redeeming <Text style={{ color: "#fff", fontWeight: "700" }}>{unitsNum.toFixed(4)} units</Text> from{" "}
                <Text style={{ color: "#fff", fontWeight: "700" }}>{selectedHolding?.name}</Text>. Estimated net payout of{" "}
                <Text style={{ color: GREEN, fontWeight: "700" }}>
                  ₹{estimatedPayout.toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                </Text>{" "}
                will be credited to your linked bank account.
              </Text>

              <View style={{ flexDirection: "row", gap: 12 }}>
                <TouchableOpacity
                  onPress={() => setShowConfirmModal(false)}
                  style={{
                    flex: 1,
                    paddingVertical: 14,
                    borderRadius: 16,
                    backgroundColor: C.input,
                    alignItems: "center",
                    borderWidth: 1,
                    borderColor: C.inputBorder,
                  }}
                >
                  <Text style={{ color: C.textMuted, fontSize: 14, fontWeight: "600" }}>Cancel</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  onPress={handleExecuteRedeem}
                  style={{ flex: 1, borderRadius: 16, overflow: "hidden" }}
                >
                  <LinearGradient
                    colors={[C.pink, C.violet]}
                    style={{ paddingVertical: 14, alignItems: "center" }}
                  >
                    <Text style={{ color: "#fff", fontSize: 14, fontWeight: "700" }}>Confirm Sell</Text>
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>

        {/* Success Modal */}
        <Modal visible={redeeming || !!successData} transparent animationType="fade">
          <View
            style={{
              flex: 1,
              backgroundColor: "rgba(0,0,0,0.92)",
              alignItems: "center",
              justifyContent: "center",
              paddingHorizontal: 24,
            }}
          >
            {redeeming && !successData && (
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
                  Processing Redemption...
                </Text>
                <Text style={{ color: C.textMuted, fontSize: 12, textAlign: "center" }}>
                  Submitting redemption order to AMC and registrar (CAMS/KFintech)
                </Text>
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
                  Redemption Placed!
                </Text>
                <Text style={{ color: C.textMuted, fontSize: 13, textAlign: "center", marginBottom: 20 }}>
                  ₹{successData.payoutAmount.toLocaleString("en-IN", { minimumFractionDigits: 2 })} will be credited to your bank.
                </Text>

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
                    <Text style={{ color: C.textFaint, fontSize: 12 }}>Units Redeemed</Text>
                    <Text style={{ color: RED, fontWeight: "700", fontSize: 13 }}>-{successData.unitsRedeemed} Units</Text>
                  </View>
                  <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                    <Text style={{ color: C.textFaint, fontSize: 12 }}>Remaining Units</Text>
                    <Text style={{ color: "#fff", fontWeight: "600", fontSize: 13 }}>{successData.remainingUnits} Units</Text>
                  </View>
                  <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                    <Text style={{ color: C.textFaint, fontSize: 12 }}>Destination</Text>
                    <Text style={{ color: C.cyan, fontWeight: "600", fontSize: 12 }}>{successData.bankAccount}</Text>
                  </View>
                  <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                    <Text style={{ color: C.textFaint, fontSize: 12 }}>Status</Text>
                    <Text style={{ color: GREEN, fontWeight: "700", fontSize: 12 }}>COMPLETED</Text>
                  </View>
                </View>

                <View style={{ width: "100%", gap: 10 }}>
                  <TouchableOpacity
                    onPress={() => {
                      setSuccessData(null);
                      router.replace("/portfolio" as any);
                    }}
                    style={{ borderRadius: 16, overflow: "hidden" }}
                  >
                    <LinearGradient
                      colors={[C.pink, C.violet]}
                      style={{ paddingVertical: 14, alignItems: "center" }}
                    >
                      <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700" }}>Back to Portfolio</Text>
                    </LinearGradient>
                  </TouchableOpacity>

                  <TouchableOpacity
                    onPress={() => {
                      setSuccessData(null);
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
