import AsyncStorage from "@react-native-async-storage/async-storage";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import {
  ArrowDownCircle,
  ArrowLeft,
  ArrowLeftRight,
  ArrowUpCircle,
  Calendar,
  CreditCard,
  History,
  Layers,
  PlusCircle,
  RefreshCw,
  Search,
} from "lucide-react-native";
import React, { useEffect, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  RefreshControl,
  ScrollView,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { AuthBackground, C, depthShadow } from "../(auth)/login";
import { API_URL } from "../../utils/api";

const GREEN = "#4ade80";
const RED = "#ff6b81";

type Transaction = {
  id: string;
  schemeCode: number;
  fundName: string;
  fundCategory: string;
  type: "BUY" | "REDEEM" | "SWITCH";
  amount: number;
  units: number;
  nav: number;
  status: string;
  paymentMethod: string;
  targetSchemeCode?: number | null;
  targetFundName?: string | null;
  targetFundCategory?: string | null;
  targetUnits?: number | null;
  targetNav?: number | null;
  createdAt: string;
};

const FILTERS: { label: string; value: "ALL" | "BUY" | "REDEEM" | "SWITCH" }[] = [
  { label: "All", value: "ALL" },
  { label: "Investments", value: "BUY" },
  { label: "Redemptions", value: "REDEEM" },
  { label: "Switches", value: "SWITCH" },
];

function formatDate(isoString: string) {
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleDateString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return isoString;
  }
}

export default function TransactionsScreen() {
  const router = useRouter();

  const [filter, setFilter] = useState<"ALL" | "BUY" | "REDEEM" | "SWITCH">("ALL");
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  async function fetchTransactions(typeFilter = filter) {
    try {
      const token = await AsyncStorage.getItem("userToken");
      if (!token) {
        router.replace("/(auth)/login");
        return;
      }
      const query = typeFilter !== "ALL" ? `?type=${typeFilter}` : "";
      const res = await fetch(`${API_URL}/api/portfolio/transactions${query}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setTransactions(data.transactions || []);
      }
    } catch (e) {
      console.error("Error fetching transactions:", e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    setLoading(true);
    fetchTransactions(filter);
  }, [filter]);

  function onRefresh() {
    setRefreshing(true);
    fetchTransactions(filter);
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
          <Text style={{ color: "#fff", fontSize: 18, fontWeight: "700" }}>Transaction History</Text>
          <TouchableOpacity
            onPress={onRefresh}
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
            <RefreshCw size={18} color={C.pink} />
          </TouchableOpacity>
        </View>

        {/* Filter Chips */}
        <View style={{ paddingVertical: 14 }}>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={{ paddingHorizontal: 20, gap: 8 }}
          >
            {FILTERS.map((f) => {
              const active = filter === f.value;
              return (
                <TouchableOpacity key={f.value} onPress={() => setFilter(f.value)}>
                  {active ? (
                    <LinearGradient
                      colors={[C.pink, C.violet]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 1, y: 1 }}
                      style={{ paddingHorizontal: 18, paddingVertical: 8, borderRadius: 999 }}
                    >
                      <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }}>{f.label}</Text>
                    </LinearGradient>
                  ) : (
                    <View
                      style={{
                        backgroundColor: C.input,
                        borderWidth: 1,
                        borderColor: C.inputBorder,
                        paddingHorizontal: 18,
                        paddingVertical: 8,
                        borderRadius: 999,
                      }}
                    >
                      <Text style={{ color: C.textMuted, fontSize: 13, fontWeight: "500" }}>{f.label}</Text>
                    </View>
                  )}
                </TouchableOpacity>
              );
            })}
          </ScrollView>
        </View>

        {/* Transactions List */}
        {loading ? (
          <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
            <ActivityIndicator size="large" color={C.pink} />
            <Text style={{ color: C.textMuted, fontSize: 13, marginTop: 12 }}>Loading transactions...</Text>
          </View>
        ) : (
          <FlatList
            data={transactions}
            keyExtractor={(item) => item.id}
            contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 30 }}
            refreshControl={
              <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={C.pink} />
            }
            renderItem={({ item }) => {
              const isBuy = item.type === "BUY";
              const isRedeem = item.type === "REDEEM";
              const isSwitch = item.type === "SWITCH";

              const badgeColor = isBuy ? GREEN : isRedeem ? RED : C.cyan;
              const Icon = isBuy ? ArrowUpCircle : isRedeem ? ArrowDownCircle : ArrowLeftRight;

              return (
                <View
                  style={{
                    backgroundColor: C.card,
                    borderRadius: 20,
                    borderWidth: 1,
                    borderColor: C.cardEdge,
                    padding: 16,
                    marginBottom: 12,
                    ...depthShadow("sm"),
                  }}
                >
                  <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                    <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                      <View
                        style={{
                          width: 32,
                          height: 32,
                          borderRadius: 16,
                          backgroundColor: `${badgeColor}20`,
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <Icon size={18} color={badgeColor} />
                      </View>
                      <View>
                        <Text style={{ color: badgeColor, fontSize: 11, fontWeight: "700", letterSpacing: 0.8 }}>
                          {item.type}
                        </Text>
                        <Text style={{ color: C.textFaint, fontSize: 11 }}>{formatDate(item.createdAt)}</Text>
                      </View>
                    </View>

                    <View style={{ alignItems: "flex-end" }}>
                      <Text style={{ color: "#fff", fontSize: 16, fontWeight: "700" }}>
                        ₹{item.amount.toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                      </Text>
                      <View
                        style={{
                          backgroundColor: "rgba(74,222,128,0.12)",
                          paddingHorizontal: 8,
                          paddingVertical: 2,
                          borderRadius: 999,
                          marginTop: 2,
                        }}
                      >
                        <Text style={{ color: GREEN, fontSize: 10, fontWeight: "700" }}>
                          {item.status || "COMPLETED"}
                        </Text>
                      </View>
                    </View>
                  </View>

                  {/* Fund details */}
                  <View style={{ marginTop: 6, paddingTop: 10, borderTopWidth: 1, borderTopColor: C.inputBorder }}>
                    <Text style={{ color: "#fff", fontSize: 14, fontWeight: "600", marginBottom: 4 }}>
                      {item.fundName}
                    </Text>

                    {isSwitch && item.targetFundName && (
                      <View style={{ flexDirection: "row", alignItems: "center", gap: 6, marginBottom: 6 }}>
                        <Text style={{ color: C.textFaint, fontSize: 12 }}>Transferred to:</Text>
                        <Text style={{ color: C.cyan, fontSize: 12, fontWeight: "600", flex: 1 }} numberOfLines={1}>
                          {item.targetFundName}
                        </Text>
                      </View>
                    )}

                    <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginTop: 4 }}>
                      <Text style={{ color: C.textMuted, fontSize: 12 }}>
                        {item.units} Units @ NAV ₹{item.nav}
                      </Text>
                      <Text style={{ color: C.textFaint, fontSize: 11 }}>
                        via {item.paymentMethod || "Direct"}
                      </Text>
                    </View>
                  </View>
                </View>
              );
            }}
            ListEmptyComponent={
              <View
                style={{
                  alignItems: "center",
                  justifyContent: "center",
                  paddingVertical: 60,
                }}
              >
                <Layers size={48} color={C.textMuted} style={{ marginBottom: 14 }} />
                <Text style={{ color: "#fff", fontSize: 16, fontWeight: "700", marginBottom: 6 }}>
                  No Transactions Found
                </Text>
                <Text style={{ color: C.textMuted, fontSize: 13, textAlign: "center", marginBottom: 20 }}>
                  {filter === "ALL"
                    ? "You haven't made any investments, redemptions, or switches yet."
                    : `No ${filter.toLowerCase()} transactions recorded.`}
                </Text>
                <TouchableOpacity
                  onPress={() => router.push("/invest" as any)}
                  style={{ borderRadius: 16, overflow: "hidden" }}
                >
                  <LinearGradient colors={[C.pink, C.violet]} style={{ paddingHorizontal: 22, paddingVertical: 12 }}>
                    <Text style={{ color: "#fff", fontWeight: "700", fontSize: 14 }}>Make an Investment</Text>
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            }
          />
        )}
      </SafeAreaView>
    </View>
  );
}
