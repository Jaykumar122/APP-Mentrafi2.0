import AsyncStorage from "@react-native-async-storage/async-storage";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import { ArrowRight, ArrowUpRight, ShieldCheck, Sparkles, TrendingUp } from "lucide-react-native";
import React, { useRef, useState } from "react";
import {
  Dimensions,
  FlatList,
  Image,
  Platform,
  StatusBar,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

let Haptics: any = null;
try {
  Haptics = require("expo-haptics");
} catch {
  // Graceful fallback if haptics unavailable
}

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get("window");

const THEME = {
  bg: "#000000",
  textPrimary: "#ffffff",
  textSecondary: "rgba(255, 255, 255, 0.72)",
  textMuted: "rgba(255, 255, 255, 0.45)",
  violet: "#8b5cf6",
  cyan: "#06b6d4",
  emerald: "#10b981",
  glassTagBg: "rgba(255, 255, 255, 0.07)",
  glassTagBorder: "rgba(255, 255, 255, 0.12)",
};

const SLIDES = [
  {
    id: "1",
    tag: "🇮🇳 Indian Direct Mutual Funds",
    headline: "Take Control of\nYour Wealth",
    sub: "Direct mutual funds, intelligent compounding, and automated execution in one place.",
    pills: ["SEBI Registered", "Direct-Growth", "Zero Commission"],
  },
  {
    id: "2",
    tag: "⚡ 100% Capital Conservation",
    headline: "Zero Intermediaries,\nMax Compounding",
    sub: "Bypass trailing distributor fees. Save up to 1.5% every year compounded straight into your portfolio.",
    pills: ["14,360+ Schemes", "₹0 Brokerage", "+26.2% 1Y Return"],
  },
  {
    id: "3",
    tag: "🧠 Neuro-Symbolic AI Core",
    headline: "Invest with\nMathematical Certainty",
    sub: "Sub-second portfolio recommendations with 99.5% closed-form compounding algebra fidelity.",
    pills: ["99.5% Accuracy", "<0.95s Latency", "Fiduciary Guard"],
  },
];

export default function OnboardingScreen() {
  const router = useRouter();
  const [currentIndex, setCurrentIndex] = useState(0);
  const flatListRef = useRef<FlatList>(null);

  const isLast = currentIndex === SLIDES.length - 1;

  async function handleFinish() {
    if (Haptics?.notificationAsync) {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
    }
    try {
      await AsyncStorage.setItem("hasSeenOnboarding", "true");
      router.replace("/(auth)/login");
    } catch (error) {
      console.error("Failed to save onboarding status:", error);
      router.replace("/(auth)/login");
    }
  }

  function goNext() {
    if (Haptics?.impactAsync) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    }
    if (currentIndex < SLIDES.length - 1) {
      const next = currentIndex + 1;
      flatListRef.current?.scrollToIndex({ index: next, animated: true });
      setCurrentIndex(next);
    } else {
      handleFinish();
    }
  }

  return (
    <View style={{ flex: 1, backgroundColor: THEME.bg }}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />

      {/* Top Header: Brand & Skip */}
      <View
        style={{
          position: "absolute",
          top: Platform.OS === "android" ? 44 : 54,
          left: 20,
          right: 20,
          zIndex: 30,
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
          <Image
            source={require("../assets/images/mentrafi-emblem.png")}
            style={{ width: 28, height: 24 }}
            resizeMode="contain"
          />
          <Text
            style={{
              color: THEME.textPrimary,
              fontSize: 18,
              fontWeight: "800",
              letterSpacing: 0.4,
            }}
          >
            Mentrafi
          </Text>
        </View>

        {!isLast ? (
          <TouchableOpacity
            onPress={handleFinish}
            activeOpacity={0.7}
            hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
            style={{
              paddingHorizontal: 12,
              paddingVertical: 6,
              borderRadius: 14,
              backgroundColor: "rgba(0, 0, 0, 0.4)",
            }}
          >
            <Text
              style={{
                color: THEME.textMuted,
                fontSize: 13,
                fontWeight: "600",
              }}
            >
              Skip
            </Text>
          </TouchableOpacity>
        ) : (
          <View style={{ width: 40 }} />
        )}
      </View>

      {/* 3D Showcase: Top Half Hero Section with Indian Flag & Indian Equities */}
      <View
        style={{
          width: SCREEN_WIDTH,
          height: SCREEN_HEIGHT * 0.52,
          overflow: "hidden",
          backgroundColor: "#000000",
        }}
      >
        <Image
          source={require("../assets/images/onboarding_3d_hero.jpg")}
          style={{
            width: SCREEN_WIDTH,
            height: SCREEN_HEIGHT * 0.96,
            position: "absolute",
            top: -SCREEN_HEIGHT * 0.015,
          }}
          resizeMode="cover"
        />

        {/* Ambient bottom vignette / gradient to fade seamlessly into text area */}
        <LinearGradient
          colors={["transparent", "rgba(0, 0, 0, 0.65)", "#000000"]}
          style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            height: 90,
          }}
        />
      </View>

      {/* Interactive Content Slider (Bottom Half) */}
      <View style={{ flex: 1, justifyContent: "space-between", paddingBottom: Platform.OS === "android" ? 32 : 44 }}>
        <FlatList
          ref={flatListRef}
          data={SLIDES}
          horizontal
          pagingEnabled
          scrollEnabled={true}
          showsHorizontalScrollIndicator={false}
          keyExtractor={(item) => item.id}
          onMomentumScrollEnd={(e) => {
            const idx = Math.round(e.nativeEvent.contentOffset.x / SCREEN_WIDTH);
            setCurrentIndex(idx);
          }}
          renderItem={({ item }) => (
            <View
              style={{
                width: SCREEN_WIDTH,
                paddingHorizontal: 24,
                alignItems: "flex-start",
                justifyContent: "center",
              }}
            >
              {/* Category Pill Tag */}
              <View
                style={{
                  paddingHorizontal: 12,
                  paddingVertical: 5,
                  borderRadius: 16,
                  backgroundColor: THEME.glassTagBg,
                  borderWidth: 1,
                  borderColor: THEME.glassTagBorder,
                  marginBottom: 12,
                }}
              >
                <Text
                  style={{
                    color: THEME.textSecondary,
                    fontSize: 12,
                    fontWeight: "600",
                    letterSpacing: 0.3,
                  }}
                >
                  {item.tag}
                </Text>
              </View>

              {/* Bold Headline */}
              <Text
                style={{
                  color: THEME.textPrimary,
                  fontSize: 32,
                  lineHeight: 38,
                  fontWeight: "800",
                  letterSpacing: -0.6,
                  marginBottom: 10,
                }}
              >
                {item.headline}
              </Text>

              {/* Clean Subtitle */}
              <Text
                style={{
                  color: THEME.textMuted,
                  fontSize: 14,
                  lineHeight: 20,
                  marginBottom: 16,
                }}
              >
                {item.sub}
              </Text>

              {/* Feature Micro-Pills */}
              <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
                {item.pills.map((pill: string, idx: number) => (
                  <View
                    key={idx}
                    style={{
                      paddingHorizontal: 10,
                      paddingVertical: 4,
                      borderRadius: 12,
                      backgroundColor: "rgba(255, 255, 255, 0.05)",
                      borderWidth: 1,
                      borderColor: "rgba(255, 255, 255, 0.08)",
                    }}
                  >
                    <Text style={{ color: THEME.textSecondary, fontSize: 11, fontWeight: "600" }}>
                      {pill}
                    </Text>
                  </View>
                ))}
              </View>
            </View>
          )}
        />

        {/* Bottom Bar: Action Button & Modern Dot Indicator */}
        <View
          style={{
            paddingHorizontal: 24,
            paddingTop: 12,
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          {/* Main Action Button */}
          <TouchableOpacity
            onPress={goNext}
            activeOpacity={0.85}
            style={{
              flex: 1,
              marginRight: 16,
              height: 54,
              borderRadius: 27,
              overflow: "hidden",
              shadowColor: THEME.emerald,
              shadowOffset: { width: 0, height: 6 },
              shadowOpacity: 0.35,
              shadowRadius: 14,
              elevation: 8,
            }}
          >
            <LinearGradient
              colors={["#8b5cf6", "#06b6d4", "#10b981"]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={{
                flex: 1,
                flexDirection: "row",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
                paddingHorizontal: 20,
              }}
            >
              <Text
                style={{
                  color: THEME.textPrimary,
                  fontSize: 16,
                  fontWeight: "700",
                  letterSpacing: 0.2,
                }}
              >
                {isLast ? "Get Started" : "Continue"}
              </Text>
              {isLast ? (
                <ArrowUpRight size={20} color={THEME.textPrimary} />
              ) : (
                <ArrowRight size={20} color={THEME.textPrimary} />
              )}
            </LinearGradient>
          </TouchableOpacity>

          {/* Minimalist Pagination Dots */}
          <View style={{ flexDirection: "row", gap: 7, alignItems: "center" }}>
            {SLIDES.map((_, i) => (
              <View
                key={i}
                style={{
                  width: i === currentIndex ? 18 : 6,
                  height: 6,
                  borderRadius: 3,
                  backgroundColor:
                    i === currentIndex
                      ? "#ffffff"
                      : "rgba(255, 255, 255, 0.22)",
                }}
              />
            ))}
          </View>
        </View>
      </View>
    </View>
  );
}