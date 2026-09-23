import AsyncStorage from "@react-native-async-storage/async-storage";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import {
  AlertCircle,
  ArrowLeft,
  ArrowLeftRight,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  History,
  Layers,
  Search,
  Sparkles,
  X,
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
};

type Fund = {
  id: number;
  scheme_code: number;
  name: string;
  category: string;
  nav: number | string;
  oneYearReturn?: number | string;
};

const PERCENT_PRESETS = [25, 50, 75, 100];

export default function SwitchScreen() {
  const router = useRouter();

  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [loadingHoldings, setLoadingHoldings] = useState(true);

  const [sourceHolding, setSourceHolding] = useState<Holding | null>(null);
  const [targetFund, setTargetFund] = useState<Fund | null>(null);

  // Units
  const [unitsStr, setUnitsStr] = useState("");
  const [activePercent, setActivePercent] = useState<number | null>(null);

  // Fund Search Modal for Target
  const [showTargetSearch, setShowTargetSearch] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Fund[]>([]);
  const [searching, setSearching] = useState(false);

  // Processing state
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [successData, setSuccessData] = useState<any | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function fetchHoldings() {
    try {
      setLoadingHoldings(true);
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
          setSourceHolding(list[0]);
          setUnitsStr((list[0].units * 0.5).toFixed(4));
          setActivePercent(50);
        }
      }
    } catch (e) {
      console.error("Error fetching holdings:", e);
    } finally {
      setLoadingHoldings(false);
    }
  }

  useEffect(() => {
    fetchHoldings();
  }, []);

  // Search target funds
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
          const filtered = (data.funds || []).filter(
            (f: Fund) => f.scheme_code !== sourceHolding?.schemeCode
          );
          setSearchResults(filtered);
        }
      } catch (e) {
        console.error("Search error:", e);
      } finally {
        setSearching(false);
      }
    }, 250);
    return () => clearTimeout(timer);
  }, [searchQuery, sourceHolding]);

  function handleSelectSource(h: Holding) {
    setSourceHolding(h);
    setActivePercent(50);
    setUnitsStr((h.units * 0.5).toFixed(4));
    setErrorMessage(null);
    if (targetFund?.scheme_code === h.schemeCode) {
      setTargetFund(null);
    }
  }

  function handlePercentSelect(pct: number) {
    if (!sourceHolding) return;
    setActivePercent(pct);
    const calculated = (sourceHolding.units * (pct / 100)).toFixed(4);
    setUnitsStr(calculated);
    setErrorMessage(null);
  }

  const unitsNum = parseFloat(unitsStr) || 0;
  const sourceNav = sourceHolding ? sourceHolding.currentNav || 10 : 10;
  const targetNav = targetFund ? parseFloat(String(targetFund.nav)) || 10 : 10;
  const switchAmount = Number((unitsNum * sourceNav).toFixed(2));
  const estimatedTargetUnits = targetNav > 0 ? (switchAmount / targetNav).toFixed(4) : "0.0000";
  const maxSourceUnits = sourceHolding ? sourceHolding.units : 0;

  async function handleExecuteSwitch() {
    if (!sourceHolding) return;
    if (!targetFund) {
      setErrorMessage("Please select a target fund to switch into");
      return;
    }
    if (unitsNum <= 0) {
      setErrorMessage("Please enter positive units to switch");
      return;
    }
    if (unitsNum > maxSourceUnits + 0.0001) {
      setErrorMessage(`Cannot switch more than your available ${maxSourceUnits.toFixed(4)} units`);
      return;
    }

    setShowConfirmModal(false);
    setSwitching(true);
    setErrorMessage(null);

    try {
      const token = await AsyncStorage.getItem("userToken");
      if (!token) {
        setSwitching(false);
        router.replace("/(auth)/login");
        return;
      }

      const res = await fetch(`${API_URL}/api/portfolio/switch`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          sourceSchemeCode: sourceHolding.schemeCode,
          targetSchemeCode: targetFund.scheme_code,
          units: unitsNum,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Fund switch failed");
      }

      setSuccessData({
        ...data,
        sourceName: sourceHolding.name,
        targetName: targetFund.name,
        switchAmount,
        sourceUnits: unitsNum,
        targetUnits: estimatedTargetUnits,
      });
    } catch (err: any) {
      console.error("Switch error:", err);
      setErrorMessage(err.message || "Failed to execute switch");
    } finally {
      setSwitching(false);
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
          <Text style={{ color: "#fff", fontSize: 18, fontWeight: "700" }}>Switch Funds</Text>
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

          {loadingHoldings ? (
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
                No Funds to Switch
              </Text>
              <Text style={{ color: C.textMuted, fontSize: 13, textAlign: "center", marginBottom: 20 }}>
                You must have existing fund holdings in your portfolio before switching.
              </Text>
              <TouchableOpacity
                onPress={() => router.replace("/invest" as any)}
                style={{ borderRadius: 16, overflow: "hidden" }}
              >
                <LinearGradient colors={[C.pink, C.violet]} style={{ paddingHorizontal: 24, paddingVertical: 12 }}>
                  <Text style={{ color: "#fff", fontWeight: "700", fontSize: 14 }}>Invest in a Fund</Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          ) : (
            <>
              {/* Step 1: Source Holding */}
              <Text style={{ color: C.pink, fontSize: 11, fontWeight: "700", letterSpacing: 1.2, marginBottom: 10 }}>
                FROM (SOURCE FUND)
              </Text>

              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={{ gap: 12, marginBottom: 20 }}
              >
                {holdings.map((h) => {
                  const isSelected = sourceHolding?.schemeCode === h.schemeCode;
                  return (
                    <TouchableOpacity
                      key={h.id}
                      onPress={() => handleSelectSource(h)}
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

              {/* Step 2: Destination Fund */}
              <Text style={{ color: C.pink, fontSize: 11, fontWeight: "700", letterSpacing: 1.2, marginBottom: 10 }}>
                TO (TARGET FUND)
              </Text>

              {targetFund ? (
                <View
                  style={{
                    backgroundColor: C.card,
                    borderRadius: 20,
                    borderWidth: 1,
                    borderColor: C.cardEdge,
                    padding: 16,
                    marginBottom: 20,
                    ...depthShadow("sm"),
                  }}
                >
                  <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                    <View style={{ flex: 1, marginRight: 12 }}>
                      <Text style={{ color: "#fff", fontSize: 14, fontWeight: "700", marginBottom: 4 }} numberOfLines={2}>
                        {targetFund.name}
                      </Text>
                      <Text style={{ color: C.textFaint, fontSize: 11 }}>{targetFund.category}</Text>
                    </View>
                    <TouchableOpacity
                      onPress={() => setShowTargetSearch(true)}
                      style={{
                        backgroundColor: "rgba(0,242,254,0.12)",
                        paddingHorizontal: 12,
                        paddingVertical: 6,
                        borderRadius: 999,
                        borderWidth: 1,
                        borderColor: "rgba(0,242,254,0.3)",
                      }}
                    >
                      <Text style={{ color: C.cyan, fontSize: 11, fontWeight: "700" }}>Change</Text>
                    </TouchableOpacity>
                  </View>
                  <View style={{ flexDirection: "row", justifyContent: "space-between", paddingTop: 10, borderTopWidth: 1, borderTopColor: C.inputBorder }}>
                    <Text style={{ color: C.textMuted, fontSize: 12 }}>Target NAV</Text>
                    <Text style={{ color: "#fff", fontWeight: "700", fontSize: 12 }}>
                      ₹{parseFloat(String(targetFund.nav || 0)).toFixed(4)}
                    </Text>
                  </View>
                </View>
              ) : (
                <TouchableOpacity
                  onPress={() => setShowTargetSearch(true)}
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
                    <Search size={20} color={C.cyan} />
                    <Text style={{ color: C.textMuted, fontSize: 14 }}>Search target fund to transfer into...</Text>
                  </View>
                  <ChevronRight size={18} color={C.textMuted} />
                </TouchableOpacity>
              )}

              {/* Step 3: Units to Switch */}
              <Text style={{ color: C.pink, fontSize: 11, fontWeight: "700", letterSpacing: 1.2, marginBottom: 8 }}>
                UNITS TO SWITCH
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
                  <Text style={{ color: C.textFaint, fontSize: 12 }}>Transfer Quantity</Text>
                  <Text style={{ color: C.textMuted, fontSize: 12 }}>
                    Max: <Text style={{ color: "#fff", fontWeight: "700" }}>{maxSourceUnits.toFixed(4)}</Text>
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

              {/* Conversion Preview */}
              <LinearGradient
                colors={["rgba(255,79,129,0.08)", "rgba(0,242,254,0.08)"]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={{
                  borderRadius: 22,
                  borderWidth: 1,
                  borderColor: "rgba(255,255,255,0.08)",
                  padding: 18,
                  gap: 12,
                  marginBottom: 28,
                }}
              >
                <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
                  <Text style={{ color: C.textMuted, fontSize: 13 }}>Switch Value</Text>
                  <Text style={{ color: "#fff", fontWeight: "700", fontSize: 15 }}>
                    ₹{switchAmount.toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                  </Text>
                </View>

                <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
                  <Text style={{ color: C.textMuted, fontSize: 13 }}>Estimated Target Units</Text>
                  <Text style={{ color: C.cyan, fontWeight: "700", fontSize: 15 }}>
                    {targetFund ? `${estimatedTargetUnits} units` : "Select Target"}
                  </Text>
                </View>

                <View style={{ paddingTop: 10, borderTopWidth: 1, borderTopColor: "rgba(255,255,255,0.05)" }}>
                  <Text style={{ color: C.textFaint, fontSize: 11 }}>
                    Direct portfolio reallocation. No bank withdrawal or manual re-deposit required.
                  </Text>
                </View>
              </LinearGradient>

              {/* Switch CTA */}
              <TouchableOpacity
                onPress={() => {
                  if (!targetFund) {
                    setErrorMessage("Please choose a target fund first");
                    return;
                  }
                  if (unitsNum <= 0) {
                    setErrorMessage("Please enter units to switch");
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
                  <ArrowLeftRight size={20} color="#fff" />
                  <Text style={{ color: "#fff", fontSize: 16, fontWeight: "700" }}>
                    Switch ₹{switchAmount.toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                  </Text>
                </LinearGradient>
              </TouchableOpacity>
            </>
          )}
        </ScrollView>

        {/* Modal 1: Target Fund Search Modal */}
        <Modal visible={showTargetSearch} animationType="slide" transparent>
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
                <Text style={{ color: "#fff", fontSize: 18, fontWeight: "700" }}>Select Destination Fund</Text>
                <TouchableOpacity
                  onPress={() => setShowTargetSearch(false)}
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
                  placeholder="Search funds by AMC or category..."
                  placeholderTextColor={C.textFaint}
                  autoFocus
                  style={{ flex: 1, color: "#fff", fontSize: 14, padding: 0 }}
                />
                {searching && <ActivityIndicator size="small" color={C.cyan} />}
              </View>

              <ScrollView showsVerticalScrollIndicator={false}>
                {searchResults.length > 0 ? (
                  searchResults.map((fund) => (
                    <TouchableOpacity
                      key={fund.scheme_code}
                      onPress={() => {
                        setTargetFund(fund);
                        setShowTargetSearch(false);
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
                        <Text style={{ color: C.cyan, fontSize: 11, fontWeight: "600" }}>{fund.category}</Text>
                        <Text style={{ color: C.textFaint, fontSize: 11 }}>•</Text>
                        <Text style={{ color: C.textMuted, fontSize: 11 }}>NAV: ₹{parseFloat(String(fund.nav || 0)).toFixed(2)}</Text>
                      </View>
                    </TouchableOpacity>
                  ))
                ) : (
                  <View style={{ alignItems: "center", paddingTop: 40 }}>
                    <Text style={{ color: C.textMuted, fontSize: 13 }}>
                      {searchQuery.length < 2 ? "Type to search destination funds" : "No matching funds found"}
                    </Text>
                  </View>
                )}
              </ScrollView>
            </View>
          </View>
        </Modal>

        {/* Modal 2: Confirm Modal */}
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
                Confirm Fund Switch
              </Text>
              <Text style={{ color: C.textMuted, fontSize: 13, lineHeight: 20, marginBottom: 16 }}>
                You are transferring <Text style={{ color: "#fff", fontWeight: "700" }}>{unitsNum.toFixed(4)} units</Text> from{" "}
                <Text style={{ color: C.pink, fontWeight: "700" }}>{sourceHolding?.name}</Text> into{" "}
                <Text style={{ color: C.cyan, fontWeight: "700" }}>{targetFund?.name}</Text>.
              </Text>

              <View
                style={{
                  backgroundColor: C.input,
                  borderRadius: 16,
                  padding: 14,
                  gap: 8,
                  marginBottom: 20,
                }}
              >
                <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                  <Text style={{ color: C.textFaint, fontSize: 12 }}>Switch Value</Text>
                  <Text style={{ color: "#fff", fontWeight: "700", fontSize: 13 }}>
                    ₹{switchAmount.toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                  </Text>
                </View>
                <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                  <Text style={{ color: C.textFaint, fontSize: 12 }}>Target Units Allotted</Text>
                  <Text style={{ color: C.cyan, fontWeight: "700", fontSize: 13 }}>
                    ~{estimatedTargetUnits} Units
                  </Text>
                </View>
              </View>

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
                  onPress={handleExecuteSwitch}
                  style={{ flex: 1, borderRadius: 16, overflow: "hidden" }}
                >
                  <LinearGradient
                    colors={[C.pink, C.violet]}
                    style={{ paddingVertical: 14, alignItems: "center" }}
                  >
                    <Text style={{ color: "#fff", fontSize: 14, fontWeight: "700" }}>Execute Switch</Text>
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>

        {/* Modal 3: Success Modal */}
        <Modal visible={switching || !!successData} transparent animationType="fade">
          <View
            style={{
              flex: 1,
              backgroundColor: "rgba(0,0,0,0.92)",
              alignItems: "center",
              justifyContent: "center",
              paddingHorizontal: 24,
            }}
          >
            {switching && !successData && (
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
                  Executing Switch Order...
                </Text>
                <Text style={{ color: C.textMuted, fontSize: 12, textAlign: "center" }}>
                  Adjusting unit allocation across portfolio schemes
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
                    backgroundColor: "rgba(0,242,254,0.15)",
                    alignItems: "center",
                    justifyContent: "center",
                    marginBottom: 16,
                  }}
                >
                  <CheckCircle2 size={36} color={C.cyan} />
                </View>

                <Text style={{ color: "#fff", fontSize: 20, fontWeight: "700", marginBottom: 4 }}>
                  Switch Successful!
                </Text>
                <Text style={{ color: C.textMuted, fontSize: 13, textAlign: "center", marginBottom: 20 }}>
                  ₹{successData.switchAmount.toLocaleString("en-IN", { minimumFractionDigits: 2 })} transferred into {successData.targetName}
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
                    <Text style={{ color: C.textFaint, fontSize: 12 }}>Source Units Deducted</Text>
                    <Text style={{ color: RED, fontWeight: "700", fontSize: 13 }}>-{successData.sourceUnits} Units</Text>
                  </View>
                  <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                    <Text style={{ color: C.textFaint, fontSize: 12 }}>Target Units Credited</Text>
                    <Text style={{ color: GREEN, fontWeight: "700", fontSize: 13 }}>+{successData.targetUnits} Units</Text>
                  </View>
                  <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                    <Text style={{ color: C.textFaint, fontSize: 12 }}>Transaction Ref</Text>
                    <Text style={{ color: C.textMuted, fontSize: 12 }}>SW-{successData.transaction?.id || Date.now()}</Text>
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
                      <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700" }}>View in Portfolio</Text>
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
