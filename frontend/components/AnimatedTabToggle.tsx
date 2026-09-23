import { LinearGradient } from "expo-linear-gradient";
import React, { useEffect, useState } from "react";
import { LayoutChangeEvent, Text, TouchableOpacity, View } from "react-native";
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from "react-native-reanimated";

let Haptics: any = null;
try {
  Haptics = require("expo-haptics");
} catch {
  // Graceful fallback
}

export type AuthTab = "login" | "signup";

type Props = {
  activeTab: AuthTab;
  onTabChange: (tab: AuthTab) => void;
};

export function AnimatedTabToggle({ activeTab, onTabChange }: Props) {
  const [tabWidth, setTabWidth] = useState(0);
  const translateX = useSharedValue(activeTab === "login" ? 0 : 1);

  useEffect(() => {
    translateX.value = withSpring(activeTab === "login" ? 0 : 1, {
      damping: 18,
      stiffness: 200,
      mass: 0.6,
    });
  }, [activeTab, translateX]);

  const handleLayout = (event: LayoutChangeEvent) => {
    const { width } = event.nativeEvent.layout;
    const calculatedTabWidth = (width - 8) / 2; // subtract 4px padding each side
    setTabWidth(calculatedTabWidth);
  };

  const animatedIndicatorStyle = useAnimatedStyle(() => {
    return {
      transform: [{ translateX: translateX.value * tabWidth }],
      width: tabWidth,
    };
  });

  const handlePress = (tab: AuthTab) => {
    if (activeTab === tab) return;
    if (Haptics?.impactAsync) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    }
    onTabChange(tab);
  };

  return (
    <View
      onLayout={handleLayout}
      style={{
        flexDirection: "row",
        backgroundColor: "rgba(255, 255, 255, 0.05)",
        borderWidth: 1,
        borderColor: "rgba(255, 255, 255, 0.08)",
        borderRadius: 18,
        padding: 4,
        marginBottom: 20,
        position: "relative",
        height: 48,
        alignItems: "center",
      }}
    >
      {/* Animated Sliding Pill Indicator */}
      {tabWidth > 0 && (
        <Animated.View
          style={[
            {
              position: "absolute",
              top: 4,
              bottom: 4,
              left: 4,
              borderRadius: 14,
              overflow: "hidden",
              shadowColor: "#06b6d4",
              shadowOffset: { width: 0, height: 2 },
              shadowOpacity: 0.3,
              shadowRadius: 8,
              elevation: 4,
            },
            animatedIndicatorStyle,
          ]}
        >
          <LinearGradient
            colors={["rgba(139, 92, 246, 0.45)", "rgba(6, 182, 212, 0.4)"]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={{
              flex: 1,
              borderWidth: 1,
              borderColor: "rgba(56, 189, 248, 0.45)",
              borderRadius: 14,
            }}
          />
        </Animated.View>
      )}

      {/* Log In Tab */}
      <TouchableOpacity
        activeOpacity={0.8}
        onPress={() => handlePress("login")}
        style={{
          flex: 1,
          height: "100%",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 2,
        }}
      >
        <Text
          style={{
            color: activeTab === "login" ? "#ffffff" : "rgba(255, 255, 255, 0.55)",
            fontWeight: activeTab === "login" ? "700" : "600",
            fontSize: 13.5,
            letterSpacing: 0.2,
          }}
        >
          Log In
        </Text>
      </TouchableOpacity>

      {/* Sign Up Tab */}
      <TouchableOpacity
        activeOpacity={0.8}
        onPress={() => handlePress("signup")}
        style={{
          flex: 1,
          height: "100%",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 2,
        }}
      >
        <Text
          style={{
            color: activeTab === "signup" ? "#ffffff" : "rgba(255, 255, 255, 0.55)",
            fontWeight: activeTab === "signup" ? "700" : "600",
            fontSize: 13.5,
            letterSpacing: 0.2,
          }}
        >
          Create Account
        </Text>
      </TouchableOpacity>
    </View>
  );
}
