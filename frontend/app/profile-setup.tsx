import AsyncStorage from "@react-native-async-storage/async-storage";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import {
  AlertCircle,
  ArrowRight,
  Briefcase,
  Building2,
  Calendar,
  Check,
  CheckCircle2,
  ChevronRight,
  GraduationCap,
  Home,
  Laptop,
  Palmtree,
  Phone,
  Shield,
  Sparkles,
  Target as TargetIcon,
  User,
  Wallet,
} from "lucide-react-native";
import React, { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Dimensions,
  Image,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StatusBar,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import Animated, { FadeIn, FadeInDown, LinearTransition } from "react-native-reanimated";
import { API_URL } from "../utils/api";
import { notifyProfileUpdated } from "../utils/profileEvents";
import {
  AuthBackground,
  C,
  depthShadow,
  ErrorBanner,
} from "./(auth)/login";

let Haptics: any = null;
try {
  Haptics = require("expo-haptics");
} catch {
  // Graceful fallback
}

const { width: SCREEN_WIDTH } = Dimensions.get("window");
const CARD_WIDTH = Math.min(SCREEN_WIDTH - 32, 430);

// ─────────────────────────────────────────────
// OCCUPATION / JOB CHOICES
// ─────────────────────────────────────────────
export const OCCUPATIONS = [
  {
    id: "job",
    label: "Doing Job / Salaried",
    desc: "Private or Govt. employee",
    icon: Briefcase,
    color: "#38bdf8",
  },
  {
    id: "student",
    label: "Student",
    desc: "College or University student",
    icon: GraduationCap,
    color: "#a78bfa",
  },
  {
    id: "business",
    label: "Business / Self-Employed",
    desc: "Entrepreneur, founder, trader",
    icon: Building2,
    color: "#34d399",
  },
  {
    id: "retired",
    label: "Retired Person",
    desc: "Senior citizen or Pensioner",
    icon: Palmtree,
    color: "#fbbf24",
  },
  {
    id: "professional",
    label: "Freelancer / Professional",
    desc: "CA, Doctor, Tech, Creator",
    icon: Laptop,
    color: "#f472b6",
  },
  {
    id: "homemaker",
    label: "Homemaker",
    desc: "Managing family & household",
    icon: Home,
    color: "#818cf8",
  },
] as const;

// ─────────────────────────────────────────────
// RISK APPETITE & SIP PRESETS
// ─────────────────────────────────────────────
const RISK_OPTIONS = [
  { id: "Low", label: "Low", desc: "Capital Safety • Debt & Liquid", color: "#10b981" },
  { id: "Moderate", label: "Moderate", desc: "Balanced • Index & Large Cap", color: "#06b6d4" },
  { id: "High", label: "High", desc: "Aggressive • Mid & Small Cap", color: "#8b5cf6" },
] as const;

const SIP_PRESETS = [2500, 5000, 10000, 25000];

const GOAL_PRESETS = [
  "🚀 Wealth Creation",
  "🏖️ Retirement Planning",
  "🏠 Buy a House",
  "🛡️ Emergency Fund",
  "⚖️ Tax Saving (ELSS)",
];

// ─────────────────────────────────────────────
// DOB HELPER FUNCTIONS
// ─────────────────────────────────────────────
function formatDobInput(raw: string): string {
  const digits = raw.replace(/\D/g, "").slice(0, 8);
  if (digits.length <= 2) return digits;
  if (digits.length <= 4) return `${digits.slice(0, 2)} / ${digits.slice(2)}`;
  return `${digits.slice(0, 2)} / ${digits.slice(2, 4)} / ${digits.slice(4)}`;
}

function computeAge(dobFormatted: string): {
  age: number | null;
  isValid: boolean;
  error: string | null;
  isoDate: string | null;
} {
  const digits = dobFormatted.replace(/\D/g, "");
  if (digits.length < 8) {
    return { age: null, isValid: false, error: null, isoDate: null };
  }
  const day = parseInt(digits.slice(0, 2), 10);
  const month = parseInt(digits.slice(2, 4), 10);
  const year = parseInt(digits.slice(4, 8), 10);

  if (month < 1 || month > 12) {
    return { age: null, isValid: false, error: "Month must be between 01 and 12", isoDate: null };
  }
  if (day < 1 || day > 31) {
    return { age: null, isValid: false, error: "Day must be between 01 and 31", isoDate: null };
  }
  const currentYear = new Date().getFullYear();
  if (year < 1920 || year > currentYear) {
    return { age: null, isValid: false, error: `Birth year must be between 1920 and ${currentYear}`, isoDate: null };
  }

  const birthDate = new Date(year, month - 1, day);
  if (birthDate.getFullYear() !== year || birthDate.getMonth() !== month - 1 || birthDate.getDate() !== day) {
    return { age: null, isValid: false, error: "Invalid calendar date", isoDate: null };
  }

  const today = new Date();
  let calculatedAge = today.getFullYear() - year;
  const m = today.getMonth() - (month - 1);
  if (m < 0 || (m === 0 && today.getDate() < day)) {
    calculatedAge--;
  }

  const isoDate = `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;

  if (calculatedAge < 18) {
    return {
      age: calculatedAge,
      isValid: false,
      error: "SEBI requires investors to be at least 18 years old",
      isoDate,
    };
  }

  if (calculatedAge > 100) {
    return { age: calculatedAge, isValid: false, error: "Age must be under 100", isoDate };
  }

  return { age: calculatedAge, isValid: true, error: null, isoDate };
}

// ─────────────────────────────────────────────
// PROFILE SETUP SCREEN
// ─────────────────────────────────────────────
export default function ProfileSetup() {
  const router = useRouter();

  // Form State
  const [phone, setPhone] = useState("");
  const [dob, setDob] = useState("");
  const [occupation, setOccupation] = useState<string>("job");
  const [monthlySipBudget, setMonthlySipBudget] = useState("5000");
  const [riskAppetite, setRiskAppetite] = useState<string>("Moderate");
  const [investmentGoal, setInvestmentGoal] = useState("🚀 Wealth Creation");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Derived Values
  const dobInfo = computeAge(dob);
  const cleanPhone = phone.replace(/\D/g, "");
  const isPhoneValid = cleanPhone.length === 10 && /^[6-9]\d{9}$/.test(cleanPhone);

  const isFormValid =
    isPhoneValid &&
    dobInfo.isValid &&
    dobInfo.age !== null &&
    occupation.trim() !== "" &&
    monthlySipBudget.trim() !== "" &&
    parseFloat(monthlySipBudget) > 0 &&
    riskAppetite.trim() !== "" &&
    investmentGoal.trim() !== "";

  const handlePhoneChange = (text: string) => {
    const digits = text.replace(/\D/g, "").slice(0, 10);
    setPhone(digits);
    if (error) setError("");
  };

  const handleDobChange = (text: string) => {
    const formatted = formatDobInput(text);
    setDob(formatted);
    if (error) setError("");
  };

  const handleSelectOccupation = (occLabel: string) => {
    if (Haptics?.impactAsync) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    }
    setOccupation(occLabel);
  };

  const handleSelectRisk = (risk: string) => {
    if (Haptics?.impactAsync) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    }
    setRiskAppetite(risk);
  };

  const handleSelectGoal = (goal: string) => {
    if (Haptics?.impactAsync) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    }
    setInvestmentGoal(goal);
  };

  const handleQuickBudget = (amount: number) => {
    if (Haptics?.impactAsync) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    }
    setMonthlySipBudget(String(amount));
  };

  async function handleSubmit() {
    setError("");

    if (!cleanPhone) {
      setError("Please enter your 10-digit mobile number.");
      return;
    }
    if (!isPhoneValid) {
      setError("Please enter a valid 10-digit Indian mobile number (starts with 6-9).");
      return;
    }

    if (!dob) {
      setError("Please enter your Date of Birth (DD / MM / YYYY).");
      return;
    }
    if (dobInfo.error) {
      setError(dobInfo.error);
      return;
    }
    if (!dobInfo.isValid || dobInfo.age === null) {
      setError("Please enter a complete and valid Date of Birth.");
      return;
    }

    if (!occupation) {
      setError("Please select your occupation.");
      return;
    }

    const budgetNum = parseFloat(monthlySipBudget.trim());
    if (isNaN(budgetNum) || budgetNum <= 0) {
      setError("Please enter a valid monthly SIP budget (₹).");
      return;
    }

    if (Haptics?.impactAsync) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
    }

    setLoading(true);
    try {
      const token = await AsyncStorage.getItem("userToken");
      if (!token) {
        router.replace("/(auth)/login");
        return;
      }

      const res = await fetch(`${API_URL}/api/profile`, {
        method: "PATCH",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          phone: `+91 ${cleanPhone}`,
          dateOfBirth: dobInfo.isoDate,
          age: dobInfo.age,
          occupation,
          monthlySipBudget: budgetNum,
          riskAppetite: riskAppetite.trim(),
          investmentGoal: investmentGoal.trim(),
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || "Failed to save profile");
      }

      const updatedData = await res.json().catch(() => null);
      if (updatedData) {
        await AsyncStorage.setItem("cachedUserProfile", JSON.stringify(updatedData));
      }
      notifyProfileUpdated();

      // Clear pending setup flag
      await AsyncStorage.removeItem("pendingProfileSetup");
      await AsyncStorage.setItem("profileCompleted", "true");

      // Seamlessly navigate to main app
      router.replace("/(tabs)/home");
    } catch (err: any) {
      console.error("Profile setup save error:", err);
      setError(err?.message || "Could not save profile. Check connection.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <View style={{ flex: 1, backgroundColor: C.bgBottom }}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />
      <AuthBackground />

      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        style={{ flex: 1 }}
      >
        <ScrollView
          contentContainerStyle={{
            flexGrow: 1,
            alignItems: "center",
            justifyContent: "center",
            paddingVertical: 32,
            paddingHorizontal: 16,
          }}
          keyboardShouldPersistTaps="handled"
        >
          {/* Main Glassmorphic Bento Card */}
          <View
            style={{
              width: CARD_WIDTH,
              borderRadius: 32,
              backgroundColor: C.card,
              borderWidth: 1,
              borderColor: C.cardEdge,
              overflow: "hidden",
              ...depthShadow("lg"),
            }}
          >
            {/* Top 3D Showcase Banner */}
            <View
              style={{
                height: 145,
                width: "100%",
                overflow: "hidden",
                position: "relative",
                backgroundColor: "#000",
              }}
            >
              <Image
                source={require("../assets/images/onboarding_3d_hero.jpg")}
                style={{
                  width: "100%",
                  height: 310,
                  position: "absolute",
                  top: -24,
                }}
                resizeMode="cover"
              />
              <LinearGradient
                colors={["transparent", "rgba(16, 20, 36, 0.65)", C.card]}
                style={{
                  position: "absolute",
                  bottom: 0,
                  left: 0,
                  right: 0,
                  height: 80,
                }}
              />

              {/* Mentrafi Brand Emblem */}
              <View
                style={{
                  position: "absolute",
                  top: 14,
                  left: 16,
                  flexDirection: "row",
                  alignItems: "center",
                  gap: 8,
                  paddingHorizontal: 12,
                  paddingVertical: 6,
                  borderRadius: 20,
                  backgroundColor: "rgba(0, 0, 0, 0.6)",
                  borderWidth: 1,
                  borderColor: "rgba(255, 255, 255, 0.15)",
                }}
              >
                <Image
                  source={require("../assets/images/mentrafi-emblem.png")}
                  style={{ width: 18, height: 16 }}
                  resizeMode="contain"
                />
                <Text style={{ color: "#fff", fontSize: 13, fontWeight: "800", letterSpacing: 0.4 }}>
                  Mentrafi
                </Text>
              </View>

              {/* SEBI KYC Pill */}
              <View
                style={{
                  position: "absolute",
                  top: 14,
                  right: 16,
                  paddingHorizontal: 10,
                  paddingVertical: 5,
                  borderRadius: 14,
                  backgroundColor: "rgba(16, 185, 129, 0.25)",
                  borderWidth: 1,
                  borderColor: "rgba(16, 185, 129, 0.5)",
                }}
              >
                <Text style={{ color: "#34d399", fontSize: 10.5, fontWeight: "700", letterSpacing: 0.3 }}>
                  🇮🇳 SEBI KYC VERIFIED
                </Text>
              </View>
            </View>

            {/* Header Content */}
            <View style={{ paddingHorizontal: 22, paddingTop: 16, paddingBottom: 26 }}>
              <View style={{ alignItems: "center", marginBottom: 16 }}>
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 6,
                    paddingHorizontal: 12,
                    paddingVertical: 4,
                    borderRadius: 14,
                    backgroundColor: "rgba(255, 255, 255, 0.06)",
                    borderWidth: 1,
                    borderColor: "rgba(255, 255, 255, 0.1)",
                    marginBottom: 8,
                  }}
                >
                  <Sparkles size={12} color={C.cyan} />
                  <Text style={{ color: C.cyan, fontSize: 10.5, fontWeight: "700", letterSpacing: 0.8 }}>
                    STEP 2 OF 2 • INVESTOR SETUP
                  </Text>
                </View>

                <Text
                  style={{
                    color: "#fff",
                    fontSize: 23,
                    lineHeight: 28,
                    fontWeight: "800",
                    textAlign: "center",
                    letterSpacing: -0.3,
                    marginBottom: 4,
                  }}
                >
                  Set up your profile
                </Text>
                <Text
                  style={{
                    color: C.textMuted,
                    fontSize: 12.5,
                    lineHeight: 18,
                    textAlign: "center",
                  }}
                >
                  Personalize your AI advisory & verified mutual fund account.
                </Text>
              </View>

              <ErrorBanner message={error} />

              {/* ─────────────────────────────────────────────────────────── */}
              {/* 1. PHONE NUMBER */}
              {/* ─────────────────────────────────────────────────────────── */}
              <View style={{ marginBottom: 18 }}>
                <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 7, paddingLeft: 2 }}>
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
                    <Phone size={14} color={C.cyan} />
                    <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }}>
                      Phone Number
                    </Text>
                  </View>
                  {isPhoneValid && (
                    <View style={{ flexDirection: "row", alignItems: "center", gap: 4 }}>
                      <CheckCircle2 size={13} color="#34d399" />
                      <Text style={{ color: "#34d399", fontSize: 11, fontWeight: "600" }}>
                        Valid Mobile
                      </Text>
                    </View>
                  )}
                </View>

                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    backgroundColor: C.input,
                    borderWidth: 1,
                    borderColor: isPhoneValid ? "rgba(52, 211, 153, 0.4)" : C.inputBorder,
                    borderRadius: 18,
                    height: 52,
                    paddingHorizontal: 12,
                  }}
                >
                  {/* Indian Country Code Pill */}
                  <View
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      gap: 4,
                      backgroundColor: "rgba(255, 255, 255, 0.08)",
                      paddingHorizontal: 10,
                      paddingVertical: 6,
                      borderRadius: 12,
                      borderWidth: 1,
                      borderColor: "rgba(255, 255, 255, 0.09)",
                      marginRight: 10,
                    }}
                  >
                    <Text style={{ fontSize: 13 }}>🇮🇳</Text>
                    <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }}>+91</Text>
                  </View>

                  <TextInput
                    value={phone}
                    onChangeText={handlePhoneChange}
                    placeholder="10-digit mobile number"
                    placeholderTextColor={C.textFaint}
                    keyboardType="phone-pad"
                    maxLength={10}
                    style={{
                      flex: 1,
                      color: "#fff",
                      fontSize: 15,
                      fontWeight: "600",
                      letterSpacing: 1.2,
                    }}
                  />
                </View>
              </View>

              {/* ─────────────────────────────────────────────────────────── */}
              {/* 2. DATE OF BIRTH & AUTO-CALCULATED AGE */}
              {/* ─────────────────────────────────────────────────────────── */}
              <View style={{ marginBottom: 18 }}>
                <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 7, paddingLeft: 2 }}>
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
                    <Calendar size={14} color={C.emerald} />
                    <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }}>
                      Date of Birth
                    </Text>
                  </View>
                  <Text style={{ color: C.textFaint, fontSize: 11, fontWeight: "500" }}>
                    DD / MM / YYYY
                  </Text>
                </View>

                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    backgroundColor: C.input,
                    borderWidth: 1,
                    borderColor: dobInfo.isValid ? "rgba(52, 211, 153, 0.4)" : C.inputBorder,
                    borderRadius: 18,
                    height: 52,
                    paddingHorizontal: 16,
                  }}
                >
                  <Calendar size={18} color={dobInfo.isValid ? "#34d399" : C.emerald} />
                  <TextInput
                    value={dob}
                    onChangeText={handleDobChange}
                    placeholder="DD / MM / YYYY (e.g. 15 / 08 / 1998)"
                    placeholderTextColor={C.textFaint}
                    keyboardType="number-pad"
                    maxLength={14}
                    style={{
                      flex: 1,
                      marginLeft: 12,
                      color: "#fff",
                      fontSize: 14.5,
                      fontWeight: "600",
                      letterSpacing: 0.8,
                    }}
                  />
                </View>

                {/* Real-Time Auto-Calculated Age Banner */}
                {dobInfo.age !== null && dobInfo.isValid && (
                  <Animated.View
                    entering={FadeInDown.duration(250)}
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      gap: 6,
                      backgroundColor: "rgba(16, 185, 129, 0.12)",
                      borderWidth: 1,
                      borderColor: "rgba(16, 185, 129, 0.35)",
                      borderRadius: 12,
                      paddingHorizontal: 12,
                      paddingVertical: 7,
                      marginTop: 8,
                    }}
                  >
                    <CheckCircle2 size={14} color="#34d399" />
                    <Text style={{ color: "#34d399", fontSize: 12, fontWeight: "600" }}>
                      🎂 Auto-calculated:{" "}
                      <Text style={{ fontWeight: "800", color: "#fff" }}>
                        {dobInfo.age} years old
                      </Text>{" "}
                      • SEBI KYC Eligible (18+)
                    </Text>
                  </Animated.View>
                )}

                {dobInfo.error && (
                  <Animated.View
                    entering={FadeInDown.duration(200)}
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      gap: 6,
                      backgroundColor: "rgba(239, 68, 68, 0.12)",
                      borderWidth: 1,
                      borderColor: "rgba(239, 68, 68, 0.35)",
                      borderRadius: 12,
                      paddingHorizontal: 12,
                      paddingVertical: 7,
                      marginTop: 8,
                    }}
                  >
                    <AlertCircle size={14} color="#f87171" />
                    <Text style={{ color: "#f87171", fontSize: 11.5, fontWeight: "600" }}>
                      {dobInfo.error}
                    </Text>
                  </Animated.View>
                )}
              </View>

              {/* ─────────────────────────────────────────────────────────── */}
              {/* 3. OCCUPATION / JOB */}
              {/* ─────────────────────────────────────────────────────────── */}
              <View style={{ marginBottom: 18 }}>
                <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 8, paddingLeft: 2 }}>
                  <Briefcase size={14} color={C.violet} />
                  <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700", marginLeft: 6 }}>
                    What do you do? (Occupation)
                  </Text>
                </View>

                <View style={{ gap: 8 }}>
                  {OCCUPATIONS.map((occ) => {
                    const isSelected = occupation === occ.label;
                    const IconComp = occ.icon;

                    return (
                      <TouchableOpacity
                        key={occ.id}
                        activeOpacity={0.82}
                        onPress={() => handleSelectOccupation(occ.label)}
                        style={{
                          borderRadius: 16,
                          overflow: "hidden",
                          borderWidth: 1,
                          borderColor: isSelected ? occ.color : "rgba(255, 255, 255, 0.08)",
                          backgroundColor: isSelected
                            ? "rgba(255, 255, 255, 0.08)"
                            : "rgba(255, 255, 255, 0.03)",
                        }}
                      >
                        <LinearGradient
                          colors={
                            isSelected
                              ? ["rgba(139, 92, 246, 0.25)", "rgba(6, 182, 212, 0.2)"]
                              : ["transparent", "transparent"]
                          }
                          start={{ x: 0, y: 0 }}
                          end={{ x: 1, y: 0 }}
                          style={{
                            flexDirection: "row",
                            alignItems: "center",
                            paddingHorizontal: 14,
                            paddingVertical: 10,
                          }}
                        >
                          <View
                            style={{
                              width: 36,
                              height: 36,
                              borderRadius: 12,
                              backgroundColor: isSelected
                                ? "rgba(255, 255, 255, 0.16)"
                                : "rgba(255, 255, 255, 0.05)",
                              alignItems: "center",
                              justifyContent: "center",
                              marginRight: 12,
                            }}
                          >
                            <IconComp size={18} color={isSelected ? occ.color : C.textMuted} />
                          </View>

                          <View style={{ flex: 1 }}>
                            <Text
                              style={{
                                color: isSelected ? "#fff" : "rgba(255, 255, 255, 0.85)",
                                fontSize: 13.5,
                                fontWeight: isSelected ? "700" : "600",
                              }}
                            >
                              {occ.label}
                            </Text>
                            <Text style={{ color: C.textFaint, fontSize: 11, marginTop: 1 }}>
                              {occ.desc}
                            </Text>
                          </View>

                          {isSelected && (
                            <View
                              style={{
                                width: 22,
                                height: 22,
                                borderRadius: 11,
                                backgroundColor: occ.color,
                                alignItems: "center",
                                justifyContent: "center",
                              }}
                            >
                              <Check size={13} color="#000" strokeWidth={3} />
                            </View>
                          )}
                        </LinearGradient>
                      </TouchableOpacity>
                    );
                  })}
                </View>
              </View>

              {/* ─────────────────────────────────────────────────────────── */}
              {/* 4. MONTHLY SIP BUDGET */}
              {/* ─────────────────────────────────────────────────────────── */}
              <View style={{ marginBottom: 18 }}>
                <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 7, paddingLeft: 2 }}>
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
                    <Wallet size={14} color={C.cyan} />
                    <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }}>
                      Monthly SIP Budget (₹)
                    </Text>
                  </View>
                </View>

                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    backgroundColor: C.input,
                    borderWidth: 1,
                    borderColor: C.inputBorder,
                    borderRadius: 18,
                    height: 52,
                    paddingHorizontal: 16,
                    marginBottom: 10,
                  }}
                >
                  <Text style={{ color: C.cyan, fontSize: 18, fontWeight: "800", marginRight: 8 }}>
                    ₹
                  </Text>
                  <TextInput
                    value={monthlySipBudget}
                    onChangeText={setMonthlySipBudget}
                    placeholder="e.g. 5000"
                    placeholderTextColor={C.textFaint}
                    keyboardType="numeric"
                    style={{
                      flex: 1,
                      color: "#fff",
                      fontSize: 16,
                      fontWeight: "700",
                    }}
                  />
                </View>

                {/* Quick Budget Preset Chips */}
                <View style={{ flexDirection: "row", gap: 8 }}>
                  {SIP_PRESETS.map((amt) => {
                    const isSelected = monthlySipBudget === String(amt);
                    return (
                      <TouchableOpacity
                        key={amt}
                        onPress={() => handleQuickBudget(amt)}
                        activeOpacity={0.8}
                        style={{
                          flex: 1,
                          paddingVertical: 7,
                          borderRadius: 12,
                          alignItems: "center",
                          backgroundColor: isSelected
                            ? "rgba(6, 182, 212, 0.25)"
                            : "rgba(255, 255, 255, 0.04)",
                          borderWidth: 1,
                          borderColor: isSelected ? C.cyan : "rgba(255, 255, 255, 0.08)",
                        }}
                      >
                        <Text
                          style={{
                            color: isSelected ? "#fff" : C.textMuted,
                            fontSize: 11.5,
                            fontWeight: isSelected ? "700" : "500",
                          }}
                        >
                          ₹{(amt / 1000).toFixed(amt % 1000 === 0 ? 0 : 1)}k
                        </Text>
                      </TouchableOpacity>
                    );
                  })}
                </View>
              </View>

              {/* ─────────────────────────────────────────────────────────── */}
              {/* 5. RISK APPETITE */}
              {/* ─────────────────────────────────────────────────────────── */}
              <View style={{ marginBottom: 18 }}>
                <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 8, paddingLeft: 2 }}>
                  <Shield size={14} color={C.emerald} />
                  <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700", marginLeft: 6 }}>
                    Risk Appetite
                  </Text>
                </View>

                <View style={{ flexDirection: "row", gap: 8 }}>
                  {RISK_OPTIONS.map((opt) => {
                    const selected = riskAppetite === opt.id;
                    return (
                      <TouchableOpacity
                        key={opt.id}
                        onPress={() => handleSelectRisk(opt.id)}
                        activeOpacity={0.85}
                        style={{ flex: 1 }}
                      >
                        <View
                          style={{
                            borderRadius: 16,
                            paddingVertical: 10,
                            paddingHorizontal: 6,
                            alignItems: "center",
                            backgroundColor: selected
                              ? "rgba(255, 255, 255, 0.09)"
                              : C.input,
                            borderWidth: 1,
                            borderColor: selected ? opt.color : C.inputBorder,
                          }}
                        >
                          <Text
                            style={{
                              color: selected ? opt.color : C.textMuted,
                              fontSize: 13,
                              fontWeight: selected ? "800" : "600",
                              marginBottom: 2,
                            }}
                          >
                            {opt.label}
                          </Text>
                          <Text
                            style={{
                              color: C.textFaint,
                              fontSize: 9.5,
                              textAlign: "center",
                              lineHeight: 12,
                            }}
                            numberOfLines={2}
                          >
                            {opt.desc}
                          </Text>
                        </View>
                      </TouchableOpacity>
                    );
                  })}
                </View>
              </View>

              {/* ─────────────────────────────────────────────────────────── */}
              {/* 6. INVESTMENT GOAL */}
              {/* ─────────────────────────────────────────────────────────── */}
              <View style={{ marginBottom: 22 }}>
                <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 8, paddingLeft: 2 }}>
                  <TargetIcon size={14} color={C.pink} />
                  <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700", marginLeft: 6 }}>
                    Primary Investment Goal
                  </Text>
                </View>

                {/* Quick Goal Chips */}
                <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 7, marginBottom: 10 }}>
                  {GOAL_PRESETS.map((preset) => {
                    const isSelected = investmentGoal === preset;
                    return (
                      <TouchableOpacity
                        key={preset}
                        activeOpacity={0.8}
                        onPress={() => handleSelectGoal(preset)}
                        style={{
                          paddingHorizontal: 11,
                          paddingVertical: 6,
                          borderRadius: 12,
                          backgroundColor: isSelected
                            ? "rgba(139, 92, 246, 0.25)"
                            : "rgba(255, 255, 255, 0.04)",
                          borderWidth: 1,
                          borderColor: isSelected ? C.pink : "rgba(255, 255, 255, 0.08)",
                        }}
                      >
                        <Text
                          style={{
                            color: isSelected ? "#fff" : C.textMuted,
                            fontSize: 11.5,
                            fontWeight: isSelected ? "700" : "500",
                          }}
                        >
                          {preset}
                        </Text>
                      </TouchableOpacity>
                    );
                  })}
                </View>

                <View
                  style={{
                    backgroundColor: C.input,
                    borderWidth: 1,
                    borderColor: C.inputBorder,
                    borderRadius: 16,
                    paddingHorizontal: 14,
                    paddingVertical: 10,
                  }}
                >
                  <TextInput
                    value={investmentGoal}
                    onChangeText={setInvestmentGoal}
                    placeholder="Custom goal — e.g. Buy electric car in 3 years"
                    placeholderTextColor={C.textFaint}
                    style={{
                      color: "#fff",
                      fontSize: 13.5,
                      fontWeight: "500",
                    }}
                  />
                </View>
              </View>

              {/* ─────────────────────────────────────────────────────────── */}
              {/* SUBMIT BUTTON */}
              {/* ─────────────────────────────────────────────────────────── */}
              <TouchableOpacity
                onPress={handleSubmit}
                activeOpacity={0.88}
                disabled={loading || !isFormValid}
                style={{
                  height: 54,
                  borderRadius: 27,
                  overflow: "hidden",
                  opacity: loading || !isFormValid ? 0.6 : 1,
                  shadowColor: C.emerald,
                  shadowOffset: { width: 0, height: 6 },
                  shadowOpacity: 0.35,
                  shadowRadius: 16,
                  elevation: 8,
                }}
              >
                <LinearGradient
                  colors={
                    isFormValid
                      ? ["#8b5cf6", "#06b6d4", "#10b981"]
                      : ["#334155", "#1e293b"]
                  }
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={{
                    flex: 1,
                    alignItems: "center",
                    justifyContent: "center",
                    flexDirection: "row",
                    gap: 8,
                  }}
                >
                  {loading ? (
                    <ActivityIndicator color="#fff" />
                  ) : (
                    <>
                      <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700", letterSpacing: 0.3 }}>
                        Complete Setup & Launch
                      </Text>
                      <ArrowRight size={18} color="#fff" />
                    </>
                  )}
                </LinearGradient>
              </TouchableOpacity>

              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  justifyContent: "center",
                  marginTop: 16,
                  gap: 5,
                }}
              >
                <Shield size={12} color={C.emerald} />
                <Text style={{ color: C.textFaint, fontSize: 10.5, fontWeight: "500" }}>
                  SEBI Registered • 256-Bit Bank Level Encryption
                </Text>
              </View>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}
