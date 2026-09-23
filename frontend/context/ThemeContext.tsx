import AsyncStorage from "@react-native-async-storage/async-storage";
import React, { createContext, useContext, useEffect, useState } from "react";
import { useColorScheme } from "react-native";

export type ThemeMode = "light" | "dark" | "system";

interface ThemeContextType {
  themeMode: ThemeMode;
  setThemeMode: (mode: ThemeMode) => void;
  isDark: boolean;
  colors: {
    // Background colors
    bg: string;
    cardBg: string;
    inputBg: string;
    errorBg: string;
    successBg: string;
    warningBg: string;

    // Text colors
    text: string;
    textSecondary: string;
    textTertiary: string;
    textInverse: string;

    // Border colors
    border: string;
    inputBorder: string;
    divider: string;

    // Status colors
    error: string;
    errorText: string;
    success: string;
    successText: string;
    warning: string;
    warningText: string;

    // Component colors
    placeholder: string;
    icon: string;
    iconSecondary: string;

    // Gradient colors
    gradientStart: string;
    gradientEnd: string;
  };
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// In-memory fallback storage for when AsyncStorage is unavailable
let inMemoryStorage: Record<string, string> = {};

const safeAsyncStorage = {
  getItem: async (key: string) => {
    try {
      return await AsyncStorage.getItem(key);
    } catch (error) {
      console.warn("AsyncStorage getItem failed, using fallback:", error);
      return inMemoryStorage[key] || null;
    }
  },
  setItem: async (key: string, value: string) => {
    try {
      await AsyncStorage.setItem(key, value);
    } catch (error) {
      console.warn("AsyncStorage setItem failed, using fallback:", error);
      inMemoryStorage[key] = value;
    }
  },
};

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const systemColorScheme = useColorScheme();
  const [themeMode, setThemeModeState] = useState<ThemeMode>("system");

  // Load theme preference from AsyncStorage on mount
  useEffect(() => {
    const loadTheme = async () => {
      try {
        const savedTheme = await safeAsyncStorage.getItem("themeMode");
        if (savedTheme) {
          setThemeModeState(savedTheme as ThemeMode);
        }
      } catch (error) {
        console.warn("Failed to load theme preference:", error);
      }
    };

    loadTheme();
  }, []);

  const setThemeMode = (mode: ThemeMode) => {
    setThemeModeState(mode);
    // Save to storage in background, don't wait for it
    safeAsyncStorage.setItem("themeMode", mode).catch((error) => {
      console.warn("Failed to save theme:", error);
    });
  };

  // Determine actual theme
  const actualTheme =
    themeMode === "system" ? systemColorScheme || "light" : themeMode;
  const isDark = actualTheme === "dark";

  // Define colors
  const colors = {
    // Light mode colors
    light: {
      bg: "#faf5ff",
      cardBg: "#ffffff",
      inputBg: "#f3e8ff",
      errorBg: "#fee2e2",
      successBg: "#dcfce7",
      warningBg: "#fef3c7",

      text: "#0f172a",
      textSecondary: "#64748b",
      textTertiary: "#94a3b8",
      textInverse: "#ffffff",

      border: "#e2e8f0",
      inputBorder: "#e9d5ff",
      divider: "#e2e8f0",

      error: "#dc2626",
      errorText: "#dc2626",
      success: "#16a34a",
      successText: "#16a34a",
      warning: "#ea580c",
      warningText: "#ea580c",

      placeholder: "#94a3b8",
      icon: "#64748b",
      iconSecondary: "#cbd5e1",

      gradientStart: "#7c3aed",
      gradientEnd: "#fb923c",
    },
    // Dark mode colors
    dark: {
      bg: "#0f172a",
      cardBg: "#1e293b",
      inputBg: "#334155",
      errorBg: "#7f1d1d",
      successBg: "#166534",
      warningBg: "#78350f",

      text: "#f1f5f9",
      textSecondary: "#cbd5e1",
      textTertiary: "#94a3b8",
      textInverse: "#0f172a",

      border: "#334155",
      inputBorder: "#475569",
      divider: "#334155",

      error: "#fca5a5",
      errorText: "#fca5a5",
      success: "#86efac",
      successText: "#86efac",
      warning: "#fdba74",
      warningText: "#fdba74",

      placeholder: "#64748b",
      icon: "#cbd5e1",
      iconSecondary: "#475569",

      gradientStart: "#7c3aed",
      gradientEnd: "#fb923c",
    },
  };

  const themeColors = isDark ? colors.dark : colors.light;

  return (
    <ThemeContext.Provider
      value={{
        themeMode,
        setThemeMode,
        isDark,
        colors: themeColors,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
};
