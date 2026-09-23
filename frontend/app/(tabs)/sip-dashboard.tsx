import AsyncStorage from "@react-native-async-storage/async-storage";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import {
  ArrowLeft,
  Calculator,
  Calendar,
  Pause,
  Pencil,
  Play,
  Plus,
  Search,
  TrendingUp,
  X,
} from "lucide-react-native";
import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Modal,
  RefreshControl,
  ScrollView,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { AuthBackground, C, depthShadow } from "../(auth)/login";
import { API_URL } from "../../utils/api";

const GREEN = "#4ade80";

type SIPStatus = "active" | "paused" | "completed" | "cancelled";

type SIP = {
  id: string;
  schemeCode: number;
  name: string;
  type: "Equity" | "Debt";
  status: SIPStatus;
  completed: number;
  total: number;
  monthlyAmount: number;
  installmentDay: number;
  nextDate: string; // ISO date, or "Paused"
  totalInvested: number;
  currentValue: number;
  returns: number;
};

type FundSearchResult = {
  id: string;
  schemeCode: number;
  name: string;
  category: string;
};

function formatINR(val: number) {
  if (val >= 100000) return `₹${(val / 100000).toFixed(2)}L`;
  return `₹${val.toLocaleString("en-IN")}`;
}

function formatNextDate(sip: SIP) {
  if (sip.status === "paused" || sip.nextDate === "Paused") return "Paused";
  const d = new Date(sip.nextDate);
  if (isNaN(d.getTime())) return sip.nextDate;
  return d.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" });
}

export default function SIPDashboard() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<"active" | "paused">("active");

  const [activeSIPs, setActiveSIPs] = useState<SIP[]>([]);
  const [pausedSIPs, setPausedSIPs] = useState<SIP[]>([]);
  const [monthlySIPTotal, setMonthlySIPTotal] = useState(0);
  const [totalInvested, setTotalInvested] = useState(0);

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // --- Add/Edit SIP modal state ------------------------------------------
  const [showSipModal, setShowSipModal] = useState(false);
  const [modalMode, setModalMode] = useState<"add" | "edit">("add");
  const [editingSip, setEditingSip] = useState<SIP | null>(null);

  const [fundQuery, setFundQuery] = useState("");
  const [fundResults, setFundResults] = useState<FundSearchResult[]>([]);
  const [searchingFunds, setSearchingFunds] = useState(false);
  const [selectedFund, setSelectedFund] = useState<{ schemeCode: number; name: string } | null>(null);

  const [amountInput, setAmountInput] = useState("5000");
  const [dayInput, setDayInput] = useState("1");
  const [durationInput, setDurationInput] = useState("60");
  const [formError, setFormError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const sips = activeTab === "active" ? activeSIPs : pausedSIPs;

  async function fetchSIPs() {
    try {
      setError(null);
      const token = await AsyncStorage.getItem("userToken");
      if (!token) {
        setError("Please log in to view your SIPs.");
        return;
      }

      const res = await fetch(`${API_URL}/api/sip`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!res.ok) {
        let serverMessage = "";
        try {
          const body = await res.json();
          serverMessage = body?.error ?? "";
        } catch {
          // response wasn't JSON — ignore, we'll fall back to status code
        }
        console.error(`SIP fetch failed (${res.status}): ${serverMessage || "no error body"}`);

        if (res.status === 401) {
          setError("Your session expired. Please log in again.");
        } else {
          setError(
            serverMessage ||
              `Couldn't load your SIPs (server error ${res.status}). Make sure the backend migration has been run.`
          );
        }
        return;
      }

      const data = await res.json();
      setActiveSIPs(data.activeSIPs ?? []);
      setPausedSIPs(data.pausedSIPs ?? []);
      setMonthlySIPTotal(data.monthlySIPTotal ?? 0);
      setTotalInvested(data.totalInvested ?? 0);
    } catch (err) {
      console.error("SIP fetch error:", err);
      setError("Couldn't reach the server. Check your connection and try again.");
    }
  }

  useEffect(() => {
    (async () => {
      setLoading(true);
      await fetchSIPs();
      setLoading(false);
    })();
  }, []);

  async function onRefresh() {
    setRefreshing(true);
    await fetchSIPs();
    setRefreshing(false);
  }

  async function togglePause(sip: SIP) {
    try {
      const token = await AsyncStorage.getItem("userToken");
      if (!token) return;
      const action = sip.status === "paused" ? "resume" : "pause";
      const res = await fetch(`${API_URL}/api/sip/${sip.id}/${action}`, {
        method: "PATCH",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error(`Failed to ${action} SIP`);
      await fetchSIPs();
    } catch (err) {
      console.error("SIP pause/resume error:", err);
    }
  }

  // --- Fund search (debounced) for the "add SIP" flow ---------------------
  useEffect(() => {
    if (modalMode !== "add" || !showSipModal) return;

    const controller = new AbortController();
    const timeout = setTimeout(async () => {
      if (!fundQuery.trim()) {
        setFundResults([]);
        return;
      }
      try {
        setSearchingFunds(true);
        const res = await fetch(
          `${API_URL}/api/funds?q=${encodeURIComponent(fundQuery.trim())}&limit=10`,
          { signal: controller.signal }
        );
        if (!res.ok) throw new Error("Fund search failed");
        const data = await res.json();
        setFundResults(
          (data.funds ?? []).map((f: any) => ({
            id: String(f.id),
            schemeCode: f.scheme_code,
            name: f.name,
            category: f.category,
          }))
        );
      } catch (err: any) {
        if (err.name !== "AbortError") console.error("Fund search error:", err);
      } finally {
        setSearchingFunds(false);
      }
    }, 300);

    return () => {
      clearTimeout(timeout);
      controller.abort();
    };
  }, [fundQuery, modalMode, showSipModal]);

  function openAddModal() {
    setModalMode("add");
    setEditingSip(null);
    setSelectedFund(null);
    setFundQuery("");
    setFundResults([]);
    setAmountInput("5000");
    setDayInput("1");
    setDurationInput("60");
    setFormError(null);
    setShowSipModal(true);
  }

  function openEditModal(sip: SIP) {
    setModalMode("edit");
    setEditingSip(sip);
    setSelectedFund({ schemeCode: sip.schemeCode, name: sip.name });
    setAmountInput(String(sip.monthlyAmount));
    setDayInput(String(sip.installmentDay));
    setDurationInput(String(sip.total));
    setFormError(null);
    setShowSipModal(true);
  }

  async function submitSipForm() {
    const amount = parseFloat(amountInput);
    const day = parseInt(dayInput, 10);
    const duration = parseInt(durationInput, 10);

    if (!amount || amount <= 0) {
      setFormError("Enter a valid monthly amount.");
      return;
    }
    if (!day || day < 1 || day > 28) {
      setFormError("Installment day must be between 1 and 28.");
      return;
    }
    if (!duration || duration < 1) {
      setFormError("Enter a valid SIP duration (in months).");
      return;
    }
    if (modalMode === "add" && !selectedFund) {
      setFormError("Search and select a fund first.");
      return;
    }

    setFormError(null);
    setSubmitting(true);
    try {
      const token = await AsyncStorage.getItem("userToken");
      if (!token) throw new Error("Not logged in");

      if (modalMode === "add") {
        const res = await fetch(`${API_URL}/api/sip`, {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
          body: JSON.stringify({
            schemeCode: selectedFund!.schemeCode,
            monthlyAmount: amount,
            installmentDay: day,
            totalInstallments: duration,
          }),
        });
        if (!res.ok) throw new Error("Failed to create SIP");
      } else if (editingSip) {
        const res = await fetch(`${API_URL}/api/sip/${editingSip.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
          body: JSON.stringify({
            monthlyAmount: amount,
            installmentDay: day,
            totalInstallments: duration,
          }),
        });
        if (!res.ok) throw new Error("Failed to update SIP");
      }

      setShowSipModal(false);
      await fetchSIPs();
    } catch (err) {
      console.error("SIP save error:", err);
      setFormError(modalMode === "add" ? "Couldn't start the SIP. Try again." : "Couldn't update the SIP. Try again.");
    } finally {
      setSubmitting(false);
    }
  }

  async function cancelSip() {
    if (!editingSip) return;
    setSubmitting(true);
    try {
      const token = await AsyncStorage.getItem("userToken");
      if (!token) return;
      const res = await fetch(`${API_URL}/api/sip/${editingSip.id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error("Failed to cancel SIP");
      setShowSipModal(false);
      await fetchSIPs();
    } catch (err) {
      console.error("SIP cancel error:", err);
      setFormError("Couldn't cancel the SIP. Try again.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <View style={{ flex: 1, backgroundColor: C.bgBottom }}>
      <AuthBackground />

      {/* Header — same glass-card gradient as the rest of the app */}
      <LinearGradient
        colors={[C.card, "#050508"]}
        start={{ x: 0.2, y: 0 }}
        end={{ x: 0.8, y: 1 }}
        style={{
          paddingHorizontal: 20,
          paddingTop: 56,
          paddingBottom: 24,
          borderBottomLeftRadius: 32,
          borderBottomRightRadius: 32,
          borderBottomWidth: 1,
          borderColor: C.cardEdge,
        }}
      >
        <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
          <View style={{ flexDirection: "row", alignItems: "center", gap: 12 }}>
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
              <ArrowLeft size={18} color="white" />
            </TouchableOpacity>
            <Text style={{ color: "#fff", fontWeight: "700", fontSize: 20 }}>My SIPs</Text>
          </View>

          <View style={{ flexDirection: "row", gap: 8 }}>
            <TouchableOpacity
              onPress={() => router.push("/sip-calculator" as any)}
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
              <Calculator size={16} color={C.pink} />
            </TouchableOpacity>
            <TouchableOpacity onPress={openAddModal} activeOpacity={0.85}>
              <LinearGradient
                colors={[C.pink, C.violet]}
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
                <Plus size={16} color="#fff" />
              </LinearGradient>
            </TouchableOpacity>
          </View>
        </View>

        {/* Stats row */}
        <View style={{ flexDirection: "row", gap: 12, marginBottom: 20 }}>
          <View
            style={{
              flex: 1,
              backgroundColor: C.input,
              borderWidth: 1,
              borderColor: C.inputBorder,
              borderRadius: 16,
              padding: 16,
            }}
          >
            <Text style={{ color: C.textFaint, fontSize: 12, marginBottom: 4 }}>Monthly SIP</Text>
            <Text style={{ color: "#fff", fontSize: 20, fontWeight: "700" }}>{formatINR(monthlySIPTotal)}</Text>
          </View>
          <View
            style={{
              flex: 1,
              backgroundColor: C.input,
              borderWidth: 1,
              borderColor: C.inputBorder,
              borderRadius: 16,
              padding: 16,
            }}
          >
            <Text style={{ color: C.textFaint, fontSize: 12, marginBottom: 4 }}>Total Invested</Text>
            <Text style={{ color: C.pink, fontSize: 20, fontWeight: "700" }}>{formatINR(totalInvested)}</Text>
          </View>
        </View>

        {/* Segmented tabs */}
        <View
          style={{
            flexDirection: "row",
            backgroundColor: C.input,
            borderWidth: 1,
            borderColor: C.inputBorder,
            borderRadius: 999,
            padding: 4,
          }}
        >
          <TouchableOpacity
            onPress={() => setActiveTab("active")}
            style={{ flex: 1, borderRadius: 999, overflow: "hidden" }}
          >
            {activeTab === "active" ? (
              <LinearGradient
                colors={[C.pink, C.violet]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={{ paddingVertical: 10, alignItems: "center", borderRadius: 999 }}
              >
                <Text style={{ color: "#fff", fontSize: 12, fontWeight: "700" }}>
                  Active ({activeSIPs.length})
                </Text>
              </LinearGradient>
            ) : (
              <View style={{ paddingVertical: 10, alignItems: "center" }}>
                <Text style={{ color: C.textMuted, fontSize: 12, fontWeight: "700" }}>
                  Active ({activeSIPs.length})
                </Text>
              </View>
            )}
          </TouchableOpacity>
          <TouchableOpacity
            onPress={() => setActiveTab("paused")}
            style={{ flex: 1, borderRadius: 999, overflow: "hidden" }}
          >
            {activeTab === "paused" ? (
              <LinearGradient
                colors={[C.pink, C.violet]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={{ paddingVertical: 10, alignItems: "center", borderRadius: 999 }}
              >
                <Text style={{ color: "#fff", fontSize: 12, fontWeight: "700" }}>
                  Paused ({pausedSIPs.length})
                </Text>
              </LinearGradient>
            ) : (
              <View style={{ paddingVertical: 10, alignItems: "center" }}>
                <Text style={{ color: C.textMuted, fontSize: 12, fontWeight: "700" }}>
                  Paused ({pausedSIPs.length})
                </Text>
              </View>
            )}
          </TouchableOpacity>
        </View>
      </LinearGradient>

      {/* SIP Cards */}
      {loading ? (
        <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
          <ActivityIndicator color={C.pink} />
        </View>
      ) : error ? (
        <View style={{ flex: 1, alignItems: "center", justifyContent: "center", paddingHorizontal: 24 }}>
          <Text style={{ color: C.textMuted, fontSize: 13, textAlign: "center", marginBottom: 12 }}>{error}</Text>
          <TouchableOpacity onPress={onRefresh} activeOpacity={0.85}>
            <LinearGradient
              colors={[C.pink, C.violet]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={{ paddingHorizontal: 20, paddingVertical: 10, borderRadius: 999 }}
            >
              <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }}>Retry</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>
      ) : (
        <ScrollView
          style={{ flex: 1 }}
          contentContainerStyle={{ padding: 20, gap: 12, paddingBottom: 40 }}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={C.pink} />
          }
        >
          {sips.length === 0 ? (
            <View style={{ alignItems: "center", paddingVertical: 48 }}>
              <Text style={{ color: C.textMuted, fontSize: 13, textAlign: "center" }}>
                {activeTab === "active"
                  ? "No active SIPs yet. Tap + to start one."
                  : "No paused SIPs."}
              </Text>
            </View>
          ) : (
            sips.map((sip) => {
              const progress = Math.min(100, (sip.completed / sip.total) * 100);
              return (
                <View
                  key={sip.id}
                  style={{
                    backgroundColor: C.input,
                    borderWidth: 1,
                    borderColor: C.inputBorder,
                    borderRadius: 20,
                    padding: 16,
                  }}
                >
                  {/* Top row */}
                  <View style={{ flexDirection: "row", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 12 }}>
                    <View style={{ flex: 1, marginRight: 8 }}>
                      <Text style={{ color: "#fff", fontWeight: "700", fontSize: 14, marginBottom: 6 }}>
                        {sip.name}
                      </Text>
                      <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                        <View
                          style={{
                            backgroundColor: "rgba(255,79,129,0.14)",
                            paddingHorizontal: 10,
                            paddingVertical: 4,
                            borderRadius: 999,
                          }}
                        >
                          <Text style={{ color: C.pink, fontSize: 10, fontWeight: "700" }}>{sip.type}</Text>
                        </View>
                        <Text style={{ color: C.textFaint, fontSize: 11 }}>
                          {sip.completed}/{sip.total} completed
                        </Text>
                      </View>
                    </View>

                    <View style={{ flexDirection: "row", gap: 8 }}>
                      <TouchableOpacity
                        onPress={() => openEditModal(sip)}
                        style={{
                          width: 32,
                          height: 32,
                          borderRadius: 10,
                          backgroundColor: C.card,
                          borderWidth: 1,
                          borderColor: C.inputBorder,
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <Pencil size={14} color={C.textMuted} />
                      </TouchableOpacity>
                      <TouchableOpacity
                        onPress={() => togglePause(sip)}
                        style={{
                          width: 32,
                          height: 32,
                          borderRadius: 10,
                          backgroundColor: C.card,
                          borderWidth: 1,
                          borderColor: C.inputBorder,
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        {sip.status === "paused" ? (
                          <Play size={14} color={GREEN} />
                        ) : (
                          <Pause size={14} color={C.textMuted} />
                        )}
                      </TouchableOpacity>
                    </View>
                  </View>

                  {/* Monthly + next date */}
                  <View style={{ flexDirection: "row", justifyContent: "space-between", marginBottom: 12 }}>
                    <View>
                      <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 2 }}>Monthly Amount</Text>
                      <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700" }}>
                        {formatINR(sip.monthlyAmount)}
                      </Text>
                    </View>
                    <View style={{ alignItems: "flex-end" }}>
                      <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 2 }}>Next Installment</Text>
                      <View style={{ flexDirection: "row", alignItems: "center", gap: 4 }}>
                        <Calendar size={12} color={C.textMuted} />
                        <Text style={{ color: "#fff", fontSize: 12, fontWeight: "600" }}>
                          {formatNextDate(sip)}
                        </Text>
                      </View>
                    </View>
                  </View>

                  {/* Progress bar */}
                  <View
                    style={{
                      backgroundColor: C.inputBorder,
                      height: 6,
                      borderRadius: 999,
                      marginBottom: 12,
                      overflow: "hidden",
                    }}
                  >
                    <LinearGradient
                      colors={[C.pink, C.violet]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 1, y: 0 }}
                      style={{ height: "100%", width: `${progress}%`, borderRadius: 999 }}
                    />
                  </View>

                  {/* Bottom row */}
                  <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
                    <View>
                      <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 2 }}>Total Invested</Text>
                      <Text style={{ color: "#fff", fontSize: 12, fontWeight: "700" }}>
                        {formatINR(sip.totalInvested)}
                      </Text>
                    </View>
                    <View>
                      <Text style={{ color: C.textFaint, fontSize: 11, marginBottom: 2 }}>Current Value</Text>
                      <Text style={{ color: "#fff", fontSize: 12, fontWeight: "700" }}>
                        {formatINR(sip.currentValue)}
                      </Text>
                    </View>
                    <View
                      style={{
                        flexDirection: "row",
                        alignItems: "center",
                        gap: 4,
                        backgroundColor: sip.returns >= 0 ? "rgba(74,222,128,0.14)" : "rgba(255,79,129,0.14)",
                        paddingHorizontal: 10,
                        paddingVertical: 6,
                        borderRadius: 999,
                      }}
                    >
                      <TrendingUp size={12} color={sip.returns >= 0 ? GREEN : C.pink} />
                      <Text style={{ color: sip.returns >= 0 ? GREEN : C.pink, fontSize: 12, fontWeight: "700" }}>
                        {sip.returns >= 0 ? "+" : ""}
                        {sip.returns}%
                      </Text>
                    </View>
                  </View>
                </View>
              );
            })
          )}
        </ScrollView>
      )}

      {/* Add / Edit SIP modal */}
      <Modal visible={showSipModal} transparent animationType="slide" onRequestClose={() => setShowSipModal(false)}>
        <View style={{ flex: 1, backgroundColor: "rgba(0,0,0,0.6)", justifyContent: "flex-end" }}>
          <View
            style={{
              backgroundColor: C.card,
              borderTopLeftRadius: 28,
              borderTopRightRadius: 28,
              borderWidth: 1,
              borderColor: C.cardEdge,
              padding: 20,
              paddingBottom: 32,
              maxHeight: "88%",
            }}
          >
            <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
              <Text style={{ color: "#fff", fontSize: 17, fontWeight: "700" }}>
                {modalMode === "add" ? "Start a New SIP" : "Edit SIP"}
              </Text>
              <TouchableOpacity
                onPress={() => setShowSipModal(false)}
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: 16,
                  backgroundColor: C.input,
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <X size={16} color="#fff" />
              </TouchableOpacity>
            </View>

            <ScrollView showsVerticalScrollIndicator={false}>
              {modalMode === "add" ? (
                <View style={{ marginBottom: 16 }}>
                  <Text style={{ color: C.textFaint, fontSize: 12, marginBottom: 8 }}>Fund</Text>
                  {selectedFund ? (
                    <View
                      style={{
                        flexDirection: "row",
                        alignItems: "center",
                        justifyContent: "space-between",
                        backgroundColor: C.input,
                        borderWidth: 1,
                        borderColor: C.inputBorder,
                        borderRadius: 14,
                        paddingHorizontal: 14,
                        paddingVertical: 12,
                      }}
                    >
                      <Text style={{ color: "#fff", fontSize: 13, fontWeight: "600", flex: 1, marginRight: 8 }}>
                        {selectedFund.name}
                      </Text>
                      <TouchableOpacity onPress={() => setSelectedFund(null)}>
                        <Text style={{ color: C.pink, fontSize: 12, fontWeight: "700" }}>Change</Text>
                      </TouchableOpacity>
                    </View>
                  ) : (
                    <>
                      <View
                        style={{
                          flexDirection: "row",
                          alignItems: "center",
                          backgroundColor: C.input,
                          borderWidth: 1,
                          borderColor: C.inputBorder,
                          borderRadius: 14,
                          paddingHorizontal: 14,
                          height: 46,
                          marginBottom: 8,
                        }}
                      >
                        <Search size={16} color={C.textMuted} />
                        <TextInput
                          value={fundQuery}
                          onChangeText={setFundQuery}
                          placeholder="Search a fund to invest in..."
                          placeholderTextColor={C.textFaint}
                          style={{ color: "#fff", flex: 1, marginLeft: 10, fontSize: 13 }}
                        />
                      </View>
                      {searchingFunds && <ActivityIndicator color={C.pink} style={{ marginVertical: 8 }} />}
                      {fundResults.map((f) => (
                        <TouchableOpacity
                          key={f.id}
                          onPress={() => {
                            setSelectedFund({ schemeCode: f.schemeCode, name: f.name });
                            setFundResults([]);
                          }}
                          style={{
                            paddingVertical: 10,
                            paddingHorizontal: 12,
                            borderBottomWidth: 1,
                            borderColor: C.inputBorder,
                          }}
                        >
                          <Text style={{ color: "#fff", fontSize: 13, fontWeight: "600" }}>{f.name}</Text>
                          <Text style={{ color: C.textFaint, fontSize: 11, marginTop: 2 }}>{f.category}</Text>
                        </TouchableOpacity>
                      ))}
                    </>
                  )}
                </View>
              ) : (
                <View style={{ marginBottom: 16 }}>
                  <Text style={{ color: C.textFaint, fontSize: 12, marginBottom: 8 }}>Fund</Text>
                  <Text style={{ color: "#fff", fontSize: 14, fontWeight: "700" }}>{selectedFund?.name}</Text>
                </View>
              )}

              <View style={{ marginBottom: 16 }}>
                <Text style={{ color: C.textFaint, fontSize: 12, marginBottom: 8 }}>Monthly Amount (₹)</Text>
                <TextInput
                  value={amountInput}
                  onChangeText={setAmountInput}
                  keyboardType="decimal-pad"
                  placeholder="e.g. 5000"
                  placeholderTextColor={C.textFaint}
                  style={{
                    backgroundColor: C.input,
                    borderWidth: 1,
                    borderColor: C.inputBorder,
                    borderRadius: 14,
                    paddingHorizontal: 14,
                    height: 46,
                    color: "#fff",
                    fontSize: 14,
                  }}
                />
              </View>

              <View style={{ flexDirection: "row", gap: 12, marginBottom: 16 }}>
                <View style={{ flex: 1 }}>
                  <Text style={{ color: C.textFaint, fontSize: 12, marginBottom: 8 }}>Installment Day</Text>
                  <TextInput
                    value={dayInput}
                    onChangeText={setDayInput}
                    keyboardType="number-pad"
                    placeholder="1-28"
                    placeholderTextColor={C.textFaint}
                    style={{
                      backgroundColor: C.input,
                      borderWidth: 1,
                      borderColor: C.inputBorder,
                      borderRadius: 14,
                      paddingHorizontal: 14,
                      height: 46,
                      color: "#fff",
                      fontSize: 14,
                    }}
                  />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={{ color: C.textFaint, fontSize: 12, marginBottom: 8 }}>Duration (months)</Text>
                  <TextInput
                    value={durationInput}
                    onChangeText={setDurationInput}
                    keyboardType="number-pad"
                    placeholder="e.g. 60"
                    placeholderTextColor={C.textFaint}
                    style={{
                      backgroundColor: C.input,
                      borderWidth: 1,
                      borderColor: C.inputBorder,
                      borderRadius: 14,
                      paddingHorizontal: 14,
                      height: 46,
                      color: "#fff",
                      fontSize: 14,
                    }}
                  />
                </View>
              </View>

              {formError && (
                <Text style={{ color: C.pink, fontSize: 12, marginBottom: 12, textAlign: "center" }}>
                  {formError}
                </Text>
              )}

              <TouchableOpacity
                onPress={submitSipForm}
                disabled={submitting}
                activeOpacity={0.88}
                style={{ opacity: submitting ? 0.6 : 1, ...depthShadow("sm") }}
              >
                <LinearGradient
                  colors={[C.pink, C.violet]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={{ borderRadius: 16, paddingVertical: 14, alignItems: "center" }}
                >
                  {submitting ? (
                    <ActivityIndicator color="#fff" />
                  ) : (
                    <Text style={{ color: "#fff", fontWeight: "700", fontSize: 13 }}>
                      {modalMode === "add" ? "Start SIP" : "Save Changes"}
                    </Text>
                  )}
                </LinearGradient>
              </TouchableOpacity>

              {modalMode === "edit" && (
                <TouchableOpacity onPress={cancelSip} disabled={submitting} style={{ marginTop: 14, alignItems: "center" }}>
                  <Text style={{ color: C.textMuted, fontSize: 12, fontWeight: "600" }}>Cancel this SIP</Text>
                </TouchableOpacity>
              )}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </View>
  );
}
