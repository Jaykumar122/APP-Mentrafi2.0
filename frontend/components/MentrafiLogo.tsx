import React from "react";
import { Image, View, Text, StyleSheet, ViewStyle, StyleProp } from "react-native";

export interface MentrafiLogoProps {
  variant?: "emblem" | "card" | "vertical" | "horizontal";
  size?: number;
  showText?: boolean;
  style?: StyleProp<ViewStyle>;
}

export default function MentrafiLogo({
  variant = "emblem",
  size = 48,
  showText = false,
  style,
}: MentrafiLogoProps) {
  if (variant === "card") {
    return (
      <View style={[styles.container, style]}>
        <Image
          source={require("../assets/images/mentrafi-card.png")}
          style={{
            width: size * 1.36,
            height: size,
            borderRadius: Math.min(18, size * 0.1),
          }}
          resizeMode="contain"
        />
      </View>
    );
  }

  if (variant === "horizontal" || (showText && variant !== "vertical")) {
    return (
      <View style={[styles.horizontalContainer, style]}>
        <Image
          source={require("../assets/images/mentrafi-emblem.png")}
          style={{ width: size, height: size * 0.8 }}
          resizeMode="contain"
        />
        <View style={styles.textContainer}>
          <Text style={[styles.brandText, { fontSize: Math.max(14, size * 0.36) }]}>
            Mentrafi
          </Text>
          <Text style={[styles.taglineText, { fontSize: Math.max(9, size * 0.18) }]}>
            Mutual Fund Recommendations
          </Text>
        </View>
      </View>
    );
  }

  if (variant === "vertical") {
    return (
      <View style={[styles.container, style]}>
        <Image
          source={require("../assets/images/mentrafi-emblem.png")}
          style={{ width: size, height: size * 0.8 }}
          resizeMode="contain"
        />
        <View style={styles.verticalTextBlock}>
          <Text style={[styles.brandText, { fontSize: Math.max(16, size * 0.28) }]}>
            Mentrafi
          </Text>
          <Text style={[styles.taglineText, { fontSize: Math.max(9, size * 0.14) }]}>
            Mutual Fund Recommendations
          </Text>
        </View>
      </View>
    );
  }

  // Default: "emblem"
  return (
    <View style={[styles.container, style]}>
      <Image
        source={require("../assets/images/mentrafi-emblem.png")}
        style={{ width: size, height: size * 0.8 }}
        resizeMode="contain"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    justifyContent: "center",
  },
  horizontalContainer: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  textContainer: {
    justifyContent: "center",
  },
  verticalTextBlock: {
    alignItems: "center",
    marginTop: 6,
  },
  brandText: {
    color: "#FFFFFF",
    fontWeight: "800",
    letterSpacing: 0.5,
  },
  taglineText: {
    color: "rgba(255,255,255,0.6)",
    fontWeight: "500",
    letterSpacing: 0.3,
    marginTop: 1,
  },
});
