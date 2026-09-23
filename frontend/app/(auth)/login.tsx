import AsyncStorage from "@react-native-async-storage/async-storage";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import { AlertCircle, ArrowRight, Eye, EyeOff, Lock, Mail, ShieldCheck, User } from "lucide-react-native";
import Svg, {
  Circle,
  Defs,
  Ellipse,
  LinearGradient as SvgGrad,
  Path,
  RadialGradient,
  Stop,
} from "react-native-svg";
import React, { useState } from "react";
import {
  ActivityIndicator,
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
import Animated, {
  FadeIn,
  FadeInDown,
  FadeOutUp,
  LinearTransition,
} from "react-native-reanimated";
import { API_URL } from "../../utils/api";
import { AnimatedTabToggle, AuthTab } from "../../components/AnimatedTabToggle";

let Haptics: any = null;
try {
  Haptics = require("expo-haptics");
} catch {
  // Graceful fallback
}

const { width: SCREEN_WIDTH } = Dimensions.get("window");
const CARD_WIDTH = Math.min(SCREEN_WIDTH - 32, 420);

// ─────────────────────────────────────────────
// DESIGN SYSTEM: LUXURY OBSIDIAN & AURORA PALETTE
// ─────────────────────────────────────────────
export const C = {
  bgTop: "#0a0f1d",
  bgMid: "#06070c",
  bgBottom: "#000000",
  card: "rgba(16, 20, 36, 0.78)",
  cardEdge: "rgba(255, 255, 255, 0.09)",
  input: "rgba(255, 255, 255, 0.05)",
  inputBorder: "rgba(255, 255, 255, 0.10)",
  pink: "#8b5cf6",
  magenta: "#06b6d4",
  violet: "#7c3aed",
  cyan: "#06b6d4",
  emerald: "#10b981",
  textPrimary: "#ffffff",
  textMuted: "rgba(255, 255, 255, 0.65)",
  textFaint: "rgba(255, 255, 255, 0.38)",
};

export const depthShadow = (level: "sm" | "md" | "lg" = "md") => {
  const map = {
    sm: { h: 4, r: 8, op: 0.25, elev: 4 },
    md: { h: 10, r: 18, op: 0.45, elev: 8 },
    lg: { h: 18, r: 30, op: 0.6, elev: 16 },
  }[level];
  return {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: map.h },
    shadowRadius: map.r,
    shadowOpacity: map.op,
    elevation: map.elev,
  };
};

export const cardTilt = {};

// ─────────────────────────────────────────────
// GLOW BACKDROP
// ─────────────────────────────────────────────
export function GlowBackdrop({ from = "#8b5cf6", to = "#06b6d4" }: { from?: string; to?: string }) {
  return (
    <Svg width={CARD_WIDTH} height={190} viewBox="0 0 300 190">
      <Defs>
        <RadialGradient id="glowA" cx="50%" cy="65%" r="60%">
          <Stop offset="0%" stopColor={from} stopOpacity="0.8" />
          <Stop offset="55%" stopColor={to} stopOpacity="0.3" />
          <Stop offset="100%" stopColor="#000" stopOpacity="0" />
        </RadialGradient>
        <RadialGradient id="glowB" cx="50%" cy="75%" r="35%">
          <Stop offset="0%" stopColor="#fff" stopOpacity="0.4" />
          <Stop offset="100%" stopColor="#fff" stopOpacity="0" />
        </RadialGradient>
      </Defs>
      <Ellipse cx="150" cy="130" rx="140" ry="75" fill="url(#glowA)" />
      <Ellipse cx="150" cy="140" rx="60" ry="24" fill="url(#glowB)" />
      <Ellipse cx="150" cy="162" rx="40" ry="8" fill="#000" opacity="0.4" />
    </Svg>
  );
}

// ─────────────────────────────────────────────
// LOCK / SECURITY HERO ICON
// ─────────────────────────────────────────────
export function LockIcon() {
  return (
    <Svg width="120" height="120" viewBox="0 0 140 140">
      <Defs>
        <SvgGrad id="lkGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <Stop offset="0%" stopColor="#38bdf8" />
          <Stop offset="50%" stopColor="#818cf8" />
          <Stop offset="100%" stopColor="#10b981" />
        </SvgGrad>
        <RadialGradient id="lkShackle" cx="40%" cy="30%" r="70%">
          <Stop offset="0%" stopColor="#ffffff" />
          <Stop offset="50%" stopColor="#38bdf8" />
          <Stop offset="100%" stopColor="#1e1b4b" />
        </RadialGradient>
      </Defs>
      <Path
        d="M 46 62 V 42 a 24 24 0 0 1 48 0 V 62"
        stroke="url(#lkShackle)"
        strokeWidth="12"
        fill="none"
        strokeLinecap="round"
      />
      <Path
        d="M 32 62 h 76 a 12 12 0 0 1 12 12 v 44 a 12 12 0 0 1 -12 12 h -76 a 12 12 0 0 1 -12 -12 v -44 a 12 12 0 0 1 12 -12 z"
        fill="rgba(16, 22, 40, 0.9)"
        stroke="url(#lkGrad)"
        strokeWidth="2"
      />
      <Circle cx="70" cy="88" r="6" fill="#38bdf8" />
      <Path d="M 67 92 L 73 92 L 75 106 L 65 106 Z" fill="#38bdf8" />
    </Svg>
  );
}

export function FloatingSphere({
  size,
  top,
  left,
  right,
  color,
  opacity = 0.4,
}: {
  size: number;
  top: number;
  left?: number;
  right?: number;
  color: string;
  opacity?: number;
}) {
  return (
    <View pointerEvents="none" style={{ position: "absolute", top, left, right, opacity }}>
      <Svg width={size} height={size} viewBox="0 0 100 100">
        <Defs>
          <RadialGradient id={`sph-${size}-${top}`} cx="35%" cy="30%" r="75%">
            <Stop offset="0%" stopColor="#fff" stopOpacity="0.8" />
            <Stop offset="40%" stopColor={color} stopOpacity="0.7" />
            <Stop offset="100%" stopColor={color} stopOpacity="0.05" />
          </RadialGradient>
        </Defs>
        <Circle cx="50" cy="50" r="46" fill={`url(#sph-${size}-${top})`} />
      </Svg>
    </View>
  );
}

// ─────────────────────────────────────────────
// AUTH BACKGROUND
// ─────────────────────────────────────────────
export function AuthBackground() {
  return (
    <>
      <LinearGradient
        colors={[C.bgTop, C.bgMid, C.bgBottom]}
        start={{ x: 0.5, y: 0 }}
        end={{ x: 0.5, y: 1 }}
        style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0 }}
      />
      {/* Upper Violet Aurora Glow */}
      <View
        pointerEvents="none"
        style={{
          position: "absolute",
          top: -100,
          left: -80,
          width: 320,
          height: 320,
          borderRadius: 160,
          backgroundColor: "rgba(124, 58, 237, 0.14)",
        }}
      />
      {/* Lower Emerald Aurora Glow */}
      <View
        pointerEvents="none"
        style={{
          position: "absolute",
          bottom: -100,
          right: -80,
          width: 340,
          height: 340,
          borderRadius: 170,
          backgroundColor: "rgba(16, 185, 129, 0.12)",
        }}
      />
    </>
  );
}

// ─────────────────────────────────────────────
// INPUT FIELD
// ─────────────────────────────────────────────
export function FieldInput({
  icon,
  placeholder,
  value,
  onChangeText,
  secure,
  toggleSecure,
  keyboardType,
  autoCapitalize,
}: {
  icon: React.ReactNode;
  placeholder: string;
  value: string;
  onChangeText: (v: string) => void;
  secure?: boolean;
  toggleSecure?: () => void;
  keyboardType?: "default" | "email-address";
  autoCapitalize?: "none" | "words" | "sentences" | "characters";
}) {
  const [isFocused, setIsFocused] = useState(false);

  return (
    <View
      style={{
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: isFocused ? "rgba(255, 255, 255, 0.07)" : C.input,
        borderWidth: 1,
        borderColor: isFocused ? "#38bdf8" : C.inputBorder,
        borderRadius: 18,
        paddingHorizontal: 16,
        height: 54,
        marginBottom: 14,
      }}
    >
      <View style={{ opacity: isFocused ? 1 : 0.7 }}>{icon}</View>
      <TextInput
        value={value}
        onChangeText={onChangeText}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        placeholder={placeholder}
        placeholderTextColor={C.textFaint}
        secureTextEntry={secure}
        autoCapitalize={autoCapitalize ?? "none"}
        keyboardType={keyboardType || "default"}
        style={{
          flex: 1,
          marginLeft: 12,
          color: "#fff",
          fontSize: 14.5,
          fontWeight: "500",
        }}
      />
      {toggleSecure && (
        <TouchableOpacity onPress={toggleSecure} activeOpacity={0.7} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
          {secure ? <Eye size={18} color={C.textMuted} /> : <EyeOff size={18} color={C.textMuted} />}
        </TouchableOpacity>
      )}
    </View>
  );
}

// ─────────────────────────────────────────────
// ERROR BANNER
// ─────────────────────────────────────────────
export function ErrorBanner({ message }: { message: string }) {
  if (!message) return null;
  return (
    <View
      style={{
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: "rgba(239, 68, 68, 0.12)",
        borderWidth: 1,
        borderColor: "rgba(239, 68, 68, 0.35)",
        borderRadius: 14,
        paddingHorizontal: 14,
        paddingVertical: 10,
        marginBottom: 14,
        gap: 8,
      }}
    >
      <AlertCircle size={16} color="#f87171" />
      <Text style={{ color: "#f87171", fontSize: 12.5, fontWeight: "500", flex: 1 }}>{message}</Text>
    </View>
  );
}

// ─────────────────────────────────────────────
// UNIFIED AUTH SCREEN COMPONENT WITH SPRING ANIMATION
// ─────────────────────────────────────────────
export function AuthScreen({ initialMode = "login" }: { initialMode?: AuthTab }) {
  const router = useRouter();
  const [mode, setMode] = useState<AuthTab>(initialMode);
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [hidePassword, setHidePassword] = useState(true);
  const [hideConfirmPassword, setHideConfirmPassword] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  const validateLogin = () => {
    const cleanEmail = email.trim();
    if (!cleanEmail) {
      setError("Please enter your email address.");
      return false;
    }
    if (!emailRegex.test(cleanEmail)) {
      setError("Please enter a valid email address (e.g. name@example.com).");
      return false;
    }
    if (!password) {
      setError("Please enter your password.");
      return false;
    }
    if (password.length < 8) {
      setError("Password must be at least 8 characters long.");
      return false;
    }
    return true;
  };

  const validateSignup = () => {
    if (!fullName.trim()) {
      setError("Please enter your full name.");
      return false;
    }
    const cleanEmail = email.trim();
    if (!cleanEmail) {
      setError("Please enter your email address.");
      return false;
    }
    if (!emailRegex.test(cleanEmail)) {
      setError("Please enter a valid email address (e.g. name@example.com).");
      return false;
    }
    if (!password) {
      setError("Please enter a password.");
      return false;
    }
    if (password.length < 8) {
      setError("Password must be at least 8 characters long.");
      return false;
    }
    if (confirmPassword !== password) {
      setError("Passwords do not match.");
      return false;
    }
    return true;
  };

  const handleTabChange = (newTab: AuthTab) => {
    setError("");
    setMode(newTab);
  };

  const toggleMode = () => {
    if (Haptics?.impactAsync) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    }
    setError("");
    setMode((prev) => (prev === "login" ? "signup" : "login"));
  };

  async function handleSubmit() {
    setError("");
    if (mode === "login") {
      if (!validateLogin()) return;

      if (Haptics?.impactAsync) {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
      }

      setLoading(true);
      try {
        const res = await fetch(`${API_URL}/api/auth/signin`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email: email.trim().toLowerCase(),
            password,
          }),
        });

        const data = await res.json();

        if (!res.ok) {
          setError(data.error || "Couldn't log in. Please check your credentials.");
          setLoading(false);
          return;
        }

        await AsyncStorage.setItem("userToken", data.token);
        if (data.user?.name) {
          await AsyncStorage.setItem("userName", data.user.name);
        }
        await AsyncStorage.removeItem("cachedUserProfile");
        router.replace("/(tabs)/home");
      } catch {
        setError("Network error. Check your connection.");
      } finally {
        setLoading(false);
      }
    } else {
      if (!validateSignup()) return;

      if (Haptics?.impactAsync) {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
      }

      setLoading(true);
      try {
        const cleanEmail = email.trim().toLowerCase();
        const cleanName = fullName.trim();

        const signupRes = await fetch(`${API_URL}/api/auth/signup`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: cleanName,
            email: cleanEmail,
            password,
          }),
        });

        const signupData = await signupRes.json();

        if (!signupRes.ok) {
          setError(signupData.error || "Signup failed. Please try again.");
          setLoading(false);
          return;
        }

        const signinRes = await fetch(`${API_URL}/api/auth/signin`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email: cleanEmail,
            password,
          }),
        });

        const signinData = await signinRes.json();

        if (!signinRes.ok) {
          setError(signinData.error || "Sign in failed after registration.");
          setLoading(false);
          return;
        }

        await AsyncStorage.setItem("userToken", signinData.token);
        if (signinData.user?.name) {
          await AsyncStorage.setItem("userName", signinData.user.name);
        }
        await AsyncStorage.removeItem("cachedUserProfile");
        await AsyncStorage.setItem("pendingProfileSetup", "true");
        router.replace("/profile-setup" as any);
      } catch {
        setError("Network error. Check your connection.");
      } finally {
        setLoading(false);
      }
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
            paddingVertical: 36,
            paddingHorizontal: 16,
          }}
          keyboardShouldPersistTaps="handled"
        >
          {/* Main Glassmorphism Bento Card */}
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
            {/* Top Onboarding-Related Hero Banner */}
            <View
              style={{
                height: 155,
                width: "100%",
                overflow: "hidden",
                position: "relative",
                backgroundColor: "#000000",
              }}
            >
              <Image
                source={require("../../assets/images/onboarding_3d_hero.jpg")}
                style={{
                  width: "100%",
                  height: 310,
                  position: "absolute",
                  top: -24,
                }}
                resizeMode="cover"
              />
              <LinearGradient
                colors={["transparent", "rgba(16, 20, 36, 0.6)", C.card]}
                style={{
                  position: "absolute",
                  bottom: 0,
                  left: 0,
                  right: 0,
                  height: 85,
                }}
              />
              {/* Mentrafi Brand Pill */}
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
                  backgroundColor: "rgba(0, 0, 0, 0.55)",
                  borderWidth: 1,
                  borderColor: "rgba(255, 255, 255, 0.14)",
                }}
              >
                <Image
                  source={require("../../assets/images/mentrafi-emblem.png")}
                  style={{ width: 18, height: 16 }}
                  resizeMode="contain"
                />
                <Text style={{ color: "#fff", fontSize: 13, fontWeight: "800", letterSpacing: 0.4 }}>
                  Mentrafi
                </Text>
              </View>

              {/* Fiduciary / Direct Scheme Tag */}
              <Animated.View
                key={mode}
                entering={FadeIn.duration(280)}
                style={{
                  position: "absolute",
                  top: 14,
                  right: 16,
                  paddingHorizontal: 10,
                  paddingVertical: 5,
                  borderRadius: 14,
                  backgroundColor: "rgba(16, 185, 129, 0.2)",
                  borderWidth: 1,
                  borderColor: "rgba(16, 185, 129, 0.4)",
                }}
              >
                <Text style={{ color: "#34d399", fontSize: 10.5, fontWeight: "700", letterSpacing: 0.3 }}>
                  {mode === "login" ? "🇮🇳 DIRECT FUNDS" : "🇮🇳 ZERO COMMISSION"}
                </Text>
              </Animated.View>
            </View>

            {/* Content Body */}
            <View style={{ paddingHorizontal: 24, paddingBottom: 28, paddingTop: 16 }}>
              {/* Animated Tab Switcher */}
              <AnimatedTabToggle activeTab={mode} onTabChange={handleTabChange} />

              {/* Category Pill & Titles */}
              <Animated.View
                key={`header-${mode}`}
                entering={FadeInDown.duration(280)}
                style={{ alignItems: "center", marginBottom: 20 }}
              >
                <View
                  style={{
                    paddingHorizontal: 12,
                    paddingVertical: 4,
                    borderRadius: 14,
                    backgroundColor: "rgba(255, 255, 255, 0.06)",
                    borderWidth: 1,
                    borderColor: "rgba(255, 255, 255, 0.09)",
                    marginBottom: 10,
                  }}
                >
                  <Text
                    style={{
                      color: mode === "login" ? C.cyan : C.emerald,
                      fontSize: 11,
                      fontWeight: "700",
                      letterSpacing: 0.6,
                    }}
                  >
                    {mode === "login" ? "WELCOME BACK" : "GET STARTED"}
                  </Text>
                </View>

                <Text
                  style={{
                    color: "#fff",
                    fontSize: 24,
                    lineHeight: 30,
                    fontWeight: "800",
                    letterSpacing: -0.4,
                    marginBottom: 6,
                    textAlign: "center",
                  }}
                >
                  {mode === "login" ? "Log in to Mentrafi" : "Create your account"}
                </Text>
                <Text
                  style={{
                    color: C.textMuted,
                    fontSize: 13,
                    lineHeight: 18,
                    textAlign: "center",
                  }}
                >
                  {mode === "login"
                    ? "Direct mutual funds, portfolio analytics & AI advisory."
                    : "Start direct mutual fund investing in under 3 minutes."}
                </Text>
              </Animated.View>

              {/* Animated Form Fields & Buttons */}
              <Animated.View layout={LinearTransition.springify().damping(18).stiffness(180)}>
                <ErrorBanner message={error} />

                {/* Full Name Input (Signup Only) */}
                {mode === "signup" && (
                  <Animated.View
                    key="field-name"
                    entering={FadeInDown.duration(260)}
                    exiting={FadeOutUp.duration(180)}
                    layout={LinearTransition.springify().damping(18).stiffness(180)}
                  >
                    <FieldInput
                      icon={<User size={18} color={C.cyan} />}
                      placeholder="Full name"
                      value={fullName}
                      onChangeText={setFullName}
                      autoCapitalize="words"
                    />
                  </Animated.View>
                )}

                {/* Email Address Input */}
                <Animated.View layout={LinearTransition.springify().damping(18).stiffness(180)}>
                  <FieldInput
                    icon={<Mail size={18} color={mode === "login" ? C.cyan : C.emerald} />}
                    placeholder="Email address"
                    value={email}
                    onChangeText={setEmail}
                    keyboardType="email-address"
                    autoCapitalize="none"
                  />
                </Animated.View>

                {/* Password Input */}
                <Animated.View layout={LinearTransition.springify().damping(18).stiffness(180)}>
                  <FieldInput
                    icon={<Lock size={18} color={mode === "login" ? C.emerald : C.violet} />}
                    placeholder="Password (min. 8 characters)"
                    value={password}
                    onChangeText={setPassword}
                    secure={hidePassword}
                    toggleSecure={() => setHidePassword((v) => !v)}
                  />
                </Animated.View>

                {/* Confirm Password Input (Signup Only) */}
                {mode === "signup" && (
                  <Animated.View
                    key="field-confirm"
                    entering={FadeInDown.duration(260)}
                    exiting={FadeOutUp.duration(180)}
                    layout={LinearTransition.springify().damping(18).stiffness(180)}
                  >
                    <FieldInput
                      icon={<Lock size={18} color={C.cyan} />}
                      placeholder="Confirm password (min. 8 characters)"
                      value={confirmPassword}
                      onChangeText={setConfirmPassword}
                      secure={hideConfirmPassword}
                      toggleSecure={() => setHideConfirmPassword((v) => !v)}
                    />
                  </Animated.View>
                )}

                {/* Forgot Password Link (Login Only) */}
                {mode === "login" && (
                  <Animated.View
                    key="forgot-btn"
                    entering={FadeInDown.duration(200)}
                    exiting={FadeOutUp.duration(150)}
                    layout={LinearTransition.springify().damping(18).stiffness(180)}
                    style={{ alignSelf: "flex-end", marginBottom: 18 }}
                  >
                    <TouchableOpacity
                      activeOpacity={0.7}
                      onPress={() => router.push("/(auth)/forgot-password")}
                    >
                      <Text style={{ color: C.cyan, fontSize: 12.5, fontWeight: "600" }}>
                        Forgot password?
                      </Text>
                    </TouchableOpacity>
                  </Animated.View>
                )}

                {/* Luminous Gradient Action Button */}
                <Animated.View layout={LinearTransition.springify().damping(18).stiffness(180)}>
                  <TouchableOpacity
                    onPress={handleSubmit}
                    activeOpacity={0.88}
                    disabled={loading}
                    style={{
                      height: 54,
                      borderRadius: 27,
                      overflow: "hidden",
                      shadowColor: mode === "login" ? C.cyan : C.emerald,
                      shadowOffset: { width: 0, height: 6 },
                      shadowOpacity: 0.35,
                      shadowRadius: 16,
                      elevation: 8,
                      opacity: loading ? 0.75 : 1,
                      marginTop: mode === "signup" ? 4 : 0,
                    }}
                  >
                    <LinearGradient
                      colors={
                        mode === "login"
                          ? ["#8b5cf6", "#06b6d4", "#10b981"]
                          : ["#06b6d4", "#8b5cf6", "#10b981"]
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
                          <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700", letterSpacing: 0.2 }}>
                            {mode === "login" ? "Log In" : "Create Account"}
                          </Text>
                          <ArrowRight size={18} color="#fff" />
                        </>
                      )}
                    </LinearGradient>
                  </TouchableOpacity>
                </Animated.View>

                {/* Divider */}
                <Animated.View
                  layout={LinearTransition.springify().damping(18).stiffness(180)}
                  style={{ flexDirection: "row", alignItems: "center", marginVertical: 20 }}
                >
                  <View style={{ flex: 1, height: 1, backgroundColor: "rgba(255, 255, 255, 0.08)" }} />
                  <Text style={{ color: C.textFaint, fontSize: 10.5, marginHorizontal: 12, fontWeight: "600" }}>
                    OR CONTINUE WITH
                  </Text>
                  <View style={{ flex: 1, height: 1, backgroundColor: "rgba(255, 255, 255, 0.08)" }} />
                </Animated.View>

                {/* Social Login Buttons */}
                <Animated.View
                  layout={LinearTransition.springify().damping(18).stiffness(180)}
                  style={{ flexDirection: "row", gap: 12, marginBottom: 20 }}
                >
                  {["Google", "Apple"].map((label) => (
                    <TouchableOpacity
                      key={label}
                      activeOpacity={0.8}
                      style={{
                        flex: 1,
                        height: 46,
                        borderRadius: 16,
                        borderWidth: 1,
                        borderColor: C.inputBorder,
                        backgroundColor: "rgba(255, 255, 255, 0.04)",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <Text style={{ color: "#fff", fontSize: 13, fontWeight: "600" }}>
                        {label}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </Animated.View>

                {/* Mode Switch Link */}
                <Animated.View
                  layout={LinearTransition.springify().damping(18).stiffness(180)}
                  style={{ flexDirection: "row", justifyContent: "center" }}
                >
                  <Text style={{ color: C.textMuted, fontSize: 13 }}>
                    {mode === "login" ? "Don't have an account? " : "Already have an account? "}
                  </Text>
                  <TouchableOpacity activeOpacity={0.7} onPress={toggleMode}>
                    <Text style={{ color: C.cyan, fontSize: 13, fontWeight: "700" }}>
                      {mode === "login" ? "Sign up" : "Log in"}
                    </Text>
                  </TouchableOpacity>
                </Animated.View>

                {/* Fiduciary / SEBI Disclaimer */}
                <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "center", marginTop: 20, gap: 5 }}>
                  <ShieldCheck size={13} color={C.emerald} />
                  <Text style={{ color: C.textFaint, fontSize: 10.5, fontWeight: "500" }}>
                    SEBI Reg. Direct Schemes • 256-Bit Bank Encryption
                  </Text>
                </View>
              </Animated.View>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}

// ─────────────────────────────────────────────
// DEFAULT ROUTE EXPORT
// ─────────────────────────────────────────────
export default function LoginScreen() {
  return <AuthScreen initialMode="login" />;
}