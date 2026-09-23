import AsyncStorage from "@react-native-async-storage/async-storage";
import { Stack, useRouter } from "expo-router";
import * as SplashScreen from "expo-splash-screen";
import { useEffect, useState } from "react";
import { ThemeProvider } from "../context/ThemeContext";
import "../global.css";
import { GestureHandlerRootView } from "react-native-gesture-handler";

// Prevent splash screen from auto-hiding until initial auth & routing checks resolve
SplashScreen.preventAutoHideAsync().catch(() => {});

export default function RootLayout() {
  const [isLoggedIn, setIsLoggedIn] = useState<boolean | null>(null);
  const [hasSeenOnboarding, setHasSeenOnboarding] = useState<boolean | null>(null);
  const [pendingProfileSetup, setPendingProfileSetup] = useState<boolean | null>(null);
  const router = useRouter();

  useEffect(() => {
    // Check auth, onboarding and pending profile setup flags
    Promise.all([
      AsyncStorage.getItem("userToken"),
      AsyncStorage.getItem("hasSeenOnboarding"),
      AsyncStorage.getItem("pendingProfileSetup"),
    ]).then(([token, onboarding, pending]) => {
      setIsLoggedIn(!!token);
      setHasSeenOnboarding(!!onboarding);
      setPendingProfileSetup(!!pending);
    });
  }, []);

  useEffect(() => {
    // Wait for checks to complete
    if (isLoggedIn === null || hasSeenOnboarding === null || pendingProfileSetup === null) return;

    // Smoothly hide native splash once route is ready
    SplashScreen.hideAsync().catch(() => {});

    // If user has a pending profile-setup flow, send them there first
    if (pendingProfileSetup && isLoggedIn) {
      router.replace("/profile-setup");
      return;
    }

    if (isLoggedIn) {
      // User is logged in → go to home
      router.replace("/(tabs)/home");
    } else {
      // User is not logged in → go to auth
      router.replace("/(auth)" as any);
    }
  }, [isLoggedIn, hasSeenOnboarding, pendingProfileSetup, router]);

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <ThemeProvider>
        <Stack screenOptions={{ headerShown: false }} />
      </ThemeProvider>
    </GestureHandlerRootView>
  );
}