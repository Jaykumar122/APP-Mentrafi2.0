import React from "react";
import {
  Image,
  StyleProp,
  StyleSheet,
  Text,
  View,
  ViewStyle,
} from "react-native";
import Svg, { Defs, Ellipse, RadialGradient, Stop } from "react-native-svg";

export interface Rotating3DLogoProps {
  /** Width of the emblem in pixels (default: 110) */
  size?: number;
  /** Custom container style */
  style?: StyleProp<ViewStyle>;
  /** Optional badge label */
  badgeLabel?: string;
}

/**
 * Onboarding-aligned Mentrafi Brand Emblem Header:
 * Clean, flat glassmorphic badge with subtle ambient aurora glow,
 * matching the modern onboarding design language.
 */
export default function Rotating3DLogo({
  size = 110,
  style,
  badgeLabel = "SEBI Registered Direct Schemes",
}: Rotating3DLogoProps) {
  const height = Math.round(size * 0.85);

  return (
    <View style={[styles.root, { width: size + 30, height: height + 30 }, style]}>
      {/* Ambient Aurora Glow Backdrop */}
      <View pointerEvents="none" style={styles.glowBackdrop}>
        <Svg width={size + 60} height={height + 50} viewBox="0 0 200 160">
          <Defs>
            <RadialGradient id="auroraAura" cx="50%" cy="50%" r="50%">
              <Stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.4" />
              <Stop offset="50%" stopColor="#06b6d4" stopOpacity="0.2" />
              <Stop offset="100%" stopColor="#06070c" stopOpacity="0" />
            </RadialGradient>
          </Defs>
          <Ellipse cx="100" cy="80" rx="90" ry="60" fill="url(#auroraAura)" />
        </Svg>
      </View>

      {/* Modern Glassmorphic Emblem Container */}
      <View style={styles.emblemBadge}>
        <Image
          source={require("../assets/images/mentrafi-emblem.png")}
          style={{ width: size * 0.55, height: size * 0.46 }}
          resizeMode="contain"
        />
        <Text style={styles.brandTitle}>Mentrafi</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  },
  glowBackdrop: {
    position: "absolute",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 0,
  },
  emblemBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 24,
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.12)",
    shadowColor: "#06b6d4",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.25,
    shadowRadius: 14,
    elevation: 6,
    zIndex: 2,
  },
  brandTitle: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "800",
    letterSpacing: 0.5,
  },
});
