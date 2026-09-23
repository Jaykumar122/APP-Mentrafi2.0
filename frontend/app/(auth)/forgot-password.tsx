import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  Eye,
  EyeOff,
  KeyRound,
  Lock,
  Mail,
} from "lucide-react-native";
import { useState } from "react";
import {
  ActivityIndicator,
  Dimensions,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import Animated, { FadeIn, FadeInDown, LinearTransition } from "react-native-reanimated";
import Svg, { Circle, Defs, Ellipse, LinearGradient as SvgGrad, Path, RadialGradient, Stop } from "react-native-svg";
import {
  AuthBackground,
  C,
  cardTilt,
  depthShadow,
  ErrorBanner,
  FieldInput,
  GlowBackdrop,
} from "./login";

const { width: SCREEN_WIDTH } = Dimensions.get("window");
const CARD_WIDTH = SCREEN_WIDTH - 44;

type Step = "email" | "otp" | "password" | "success";

// ─────────────────────────────────────────────
// KEY ICON — isometric extruded key, built the same way as LockIcon
// (front/top/side faces + gradient sheen) so the hero matches login
// ─────────────────────────────────────────────
function KeyIcon() {
  return (
    <Svg width="130" height="130" viewBox="0 0 150 150">
      <Defs>
        <SvgGrad id="keyHead" x1="0" y1="0" x2="1" y2="1">
          <Stop offset="0%" stopColor="#ff9dbc" />
          <Stop offset="100%" stopColor={C.pink} />
        </SvgGrad>
        <SvgGrad id="keyHeadSide" x1="0" y1="0" x2="0" y2="1">
          <Stop offset="0%" stopColor="#b23360" />
          <Stop offset="100%" stopColor="#7a1f42" />
        </SvgGrad>
        <SvgGrad id="keyShaft" x1="0" y1="0" x2="1" y2="1">
          <Stop offset="0%" stopColor="#c9b3ff" />
          <Stop offset="100%" stopColor={C.violet} />
        </SvgGrad>
      </Defs>

      {/* shaft */}
      <Path d="M78,66 L120,108 L108,120 L100,112 L92,120 L82,110 L94,98 L78,82 Z" fill="url(#keyShaft)" />
      <Path d="M120,108 L124,104 L108,88 L104,92 Z" fill="#4c2794" opacity="0.6" />

      {/* teeth */}
      <Path d="M100,112 L92,120 L86,114 L94,106 Z" fill="url(#keyShaft)" />
      <Path d="M108,120 L98,130 L92,124 L102,114 Z" fill="url(#keyShaft)" opacity="0.9" />

      {/* bow (head) — isometric ring */}
      <Ellipse cx="54" cy="58" rx="30" ry="30" fill="url(#keyHeadSide)" />
      <Ellipse cx="54" cy="52" rx="30" ry="30" fill="url(#keyHead)" />
      <Circle cx="54" cy="52" r="13" fill="#050310" opacity="0.85" />
      <Circle cx="54" cy="52" r="13" stroke="#fff" strokeOpacity="0.25" strokeWidth="1.5" fill="none" />

      {/* highlight sheen */}
      <Ellipse cx="44" cy="44" rx="10" ry="5" fill="#fff" opacity="0.35" />
    </Svg>
  );
}

// ─────────────────────────────────────────────
// FORGOT PASSWORD SCREEN — same dark glass-card material, gradient
// background, and hero construction as the login screen
// ─────────────────────────────────────────────
export default function ForgotPasswordScreen() {
  const router = useRouter();
  const [step, setStep] = useState<Step>("email");
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  const handleSendOTP = async () => {
    setError("");
    const cleanEmail = email.trim();
    if (!cleanEmail) {
      setError("Please enter your registered email address.");
      return;
    }
    if (!emailRegex.test(cleanEmail)) {
      setError("Please enter a valid email address (e.g. name@example.com).");
      return;
    }

    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setStep("otp");
    }, 1500);
  };

  const handleVerifyOTP = async () => {
    setError("");
    if (!otp.trim()) {
      setError("OTP is required.");
      return;
    }
    if (otp.length !== 6) {
      setError("OTP must be 6 digits.");
      return;
    }

    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setStep("password");
    }, 1500);
  };

  const handleResetPassword = async () => {
    setError("");
    if (!newPassword.trim()) {
      setError("Password is required.");
      return;
    }
    if (newPassword.length < 8) {
      setError("Password must be at least 8 characters long.");
      return;
    }
    if (!confirmPassword.trim()) {
      setError("Please confirm your password.");
      return;
    }
    if (confirmPassword !== newPassword) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setStep("success");
      setTimeout(() => {
        router.replace("/(auth)");
      }, 2000);
    }, 1500);
  };

  const handleBack = () => {
    setError("");
    if (step === "email") {
      router.back();
    } else if (step === "otp") {
      setStep("email");
      setOtp("");
    } else if (step === "password") {
      setStep("otp");
      setNewPassword("");
      setConfirmPassword("");
    }
  };

  const getEyebrow = () => {
    switch (step) {
      case "email":
        return "RESET PASSWORD";
      case "otp":
        return "VERIFY IT'S YOU";
      case "password":
        return "NEW PASSWORD";
      case "success":
        return "ALL SET";
    }
  };

  const getHeadline = () => {
    switch (step) {
      case "email":
        return "Forgot your password?";
      case "otp":
        return "Verify your identity";
      case "password":
        return "Create a new password";
      case "success":
        return "Password reset!";
    }
  };

  const getSubtext = () => {
    switch (step) {
      case "email":
        return "Enter your registered email address to receive a verification code.";
      case "otp":
        return `We sent a 6-digit code to ${email}. Enter it below.`;
      case "password":
        return "Choose a strong password (at least 8 characters) to secure your account.";
      case "success":
        return "Your password has been reset successfully. Redirecting to login...";
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: C.bgBottom }}>
      <AuthBackground />

      <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined} style={{ flex: 1 }}>
        <ScrollView
          contentContainerStyle={{
            flexGrow: 1,
            alignItems: "center",
            justifyContent: "center",
            paddingVertical: 40,
          }}
          keyboardShouldPersistTaps="handled"
        >
          <View
            style={{
              width: CARD_WIDTH,
              ...depthShadow("lg"),
              ...cardTilt,
            }}
          >
            <LinearGradient
              colors={[C.card, "#050508"]}
              start={{ x: 0.2, y: 0 }}
              end={{ x: 0.8, y: 1 }}
              style={{
                borderRadius: 36,
                borderWidth: 1,
                borderColor: C.cardEdge,
                overflow: "hidden",
                paddingBottom: 30,
              }}
            >
              <View
                pointerEvents="none"
                style={{
                  position: "absolute",
                  top: -40,
                  left: -60,
                  width: 140,
                  height: 700,
                  backgroundColor: "rgba(255,255,255,0.05)",
                  transform: [{ rotate: "18deg" }],
                }}
              />

              {/* Back button, floating above the hero */}
              {step !== "success" && (
                <TouchableOpacity
                  onPress={handleBack}
                  activeOpacity={0.7}
                  style={{
                    position: "absolute",
                    top: 18,
                    left: 18,
                    width: 36,
                    height: 36,
                    borderRadius: 18,
                    backgroundColor: C.input,
                    borderWidth: 1,
                    borderColor: C.inputBorder,
                    alignItems: "center",
                    justifyContent: "center",
                    zIndex: 5,
                  }}
                >
                  <ArrowLeft size={16} color="#fff" />
                </TouchableOpacity>
              )}

              <View style={{ height: 190, alignItems: "center", justifyContent: "flex-end" }}>
                <View style={{ position: "absolute", bottom: 0 }}>
                  <GlowBackdrop from={C.pink} to={C.violet} />
                </View>
                <View style={{ marginBottom: 26, ...depthShadow("md") }}>
                  {step === "success" ? (
                    <View
                      style={{
                        width: 96,
                        height: 96,
                        borderRadius: 48,
                        backgroundColor: "rgba(74,222,128,0.15)",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <CheckCircle size={48} color="#4ade80" />
                    </View>
                  ) : (
                    <KeyIcon />
                  )}
                </View>
              </View>

              <View style={{ paddingHorizontal: 26 }}>
                <Animated.View key={step} entering={FadeIn.duration(250)} layout={LinearTransition.duration(250)}>
                  <Text
                    style={{
                      color: C.pink,
                      fontSize: 10,
                      fontWeight: "700",
                      letterSpacing: 1.2,
                      marginBottom: 10,
                      textAlign: "center",
                    }}
                  >
                    {getEyebrow()}
                  </Text>
                  <Text
                    style={{
                      color: "#fff",
                      fontSize: 23,
                      lineHeight: 29,
                      fontWeight: "700",
                      marginBottom: 6,
                      textAlign: "center",
                    }}
                  >
                    {getHeadline()}
                  </Text>
                  <Text
                    style={{
                      color: C.textMuted,
                      fontSize: 12.5,
                      lineHeight: 19,
                      textAlign: "center",
                      marginBottom: 26,
                    }}
                  >
                    {getSubtext()}
                  </Text>
                </Animated.View>

                {step !== "success" && <ErrorBanner message={error} />}

                {/* STEP 1: Email / Mobile */}
                {step === "email" && (
                  <Animated.View
                    entering={FadeInDown.duration(300).springify()}
                    layout={LinearTransition.duration(300)}
                  >
                    <FieldInput
                      icon={<Mail size={18} color={C.cyan} />}
                      placeholder="Email address"
                      value={email}
                      onChangeText={setEmail}
                      keyboardType="email-address"
                      autoCapitalize="none"
                    />

                    <TouchableOpacity
                      onPress={handleSendOTP}
                      activeOpacity={0.88}
                      disabled={loading}
                      style={{ ...depthShadow("md"), opacity: loading ? 0.75 : 1, marginTop: 8 }}
                    >
                      <LinearGradient
                        colors={["#8b5cf6", "#06b6d4", "#10b981"]}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 1 }}
                        style={{
                          height: 54,
                          borderRadius: 27,
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
                            <Text style={{ color: "#fff", fontSize: 14, fontWeight: "700" }}>
                              Send verification code
                            </Text>
                            <ArrowRight size={17} color="#fff" />
                          </>
                        )}
                      </LinearGradient>
                    </TouchableOpacity>
                  </Animated.View>
                )}

                {/* STEP 2: OTP */}
                {step === "otp" && (
                  <Animated.View
                    entering={FadeInDown.duration(300).springify()}
                    layout={LinearTransition.duration(300)}
                  >
                    <FieldInput
                      icon={<KeyRound size={18} color={C.textMuted} />}
                      placeholder="000000"
                      value={otp}
                      onChangeText={(text) => setOtp(text.replace(/[^0-9]/g, ""))}
                      keyboardType="default"
                    />

                    <TouchableOpacity activeOpacity={0.7} style={{ marginBottom: 8 }}>
                      <Text style={{ color: C.cyan, fontSize: 12.5, fontWeight: "600", textAlign: "center" }}>
                        Did not receive? Resend code
                      </Text>
                    </TouchableOpacity>

                    <TouchableOpacity
                      onPress={handleVerifyOTP}
                      activeOpacity={0.88}
                      disabled={loading}
                      style={{ ...depthShadow("md"), opacity: loading ? 0.75 : 1, marginTop: 8 }}
                    >
                      <LinearGradient
                        colors={["#8b5cf6", "#06b6d4", "#10b981"]}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 1 }}
                        style={{
                          height: 54,
                          borderRadius: 27,
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
                            <Text style={{ color: "#fff", fontSize: 14, fontWeight: "700" }}>Verify code</Text>
                            <ArrowRight size={17} color="#fff" />
                          </>
                        )}
                      </LinearGradient>
                    </TouchableOpacity>
                  </Animated.View>
                )}

                {/* STEP 3: New Password */}
                {step === "password" && (
                  <Animated.View
                    entering={FadeInDown.duration(300).springify()}
                    layout={LinearTransition.duration(300)}
                  >
                    <FieldInput
                      icon={<Lock size={18} color={C.emerald} />}
                      placeholder="New password (min. 8 characters)"
                      value={newPassword}
                      onChangeText={setNewPassword}
                      secure={!showPassword}
                      toggleSecure={() => setShowPassword((v) => !v)}
                    />
                    <FieldInput
                      icon={<Lock size={18} color={C.cyan} />}
                      placeholder="Confirm password (min. 8 characters)"
                      value={confirmPassword}
                      onChangeText={setConfirmPassword}
                      secure={!showConfirmPassword}
                      toggleSecure={() => setShowConfirmPassword((v) => !v)}
                    />

                    <TouchableOpacity
                      onPress={handleResetPassword}
                      activeOpacity={0.88}
                      disabled={loading}
                      style={{ ...depthShadow("md"), opacity: loading ? 0.75 : 1, marginTop: 8 }}
                    >
                      <LinearGradient
                        colors={["#8b5cf6", "#06b6d4", "#10b981"]}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 1 }}
                        style={{
                          height: 54,
                          borderRadius: 27,
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
                            <Text style={{ color: "#fff", fontSize: 14, fontWeight: "700" }}>Reset password</Text>
                            <ArrowRight size={17} color="#fff" />
                          </>
                        )}
                      </LinearGradient>
                    </TouchableOpacity>
                  </Animated.View>
                )}

                {/* STEP 4: Success — just the headline/subtext above, no extra content */}

                <Text
                  style={{
                    marginTop: 22,
                    textAlign: "center",
                    color: C.textFaint,
                    fontSize: 10,
                    lineHeight: 15,
                  }}
                >
                  SEBI Reg. No. MF/021/05 · Investments subject to market risks. Please read all
                  scheme documents carefully.
                </Text>
              </View>
            </LinearGradient>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}