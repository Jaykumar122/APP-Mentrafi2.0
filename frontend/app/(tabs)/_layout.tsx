import { Tabs } from "expo-router";

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: { display: "none" }, // custom bottom nav is rendered per-screen
      }}
    >
      <Tabs.Screen name="home" />
      <Tabs.Screen name="explore" />
      <Tabs.Screen name="portfolio" />
      <Tabs.Screen name="ai-advisor" />
      <Tabs.Screen name="profile" />

      {/* Reachable via router.push but not shown in the tab bar */}
      <Tabs.Screen name="sip-dashboard" options={{ href: null }} />
      <Tabs.Screen name="sip-calculator" options={{ href: null }} />
      <Tabs.Screen name="invest" options={{ href: null }} />
      <Tabs.Screen name="redeem" options={{ href: null }} />
      <Tabs.Screen name="switch" options={{ href: null }} />
      <Tabs.Screen name="transactions" options={{ href: null }} />
    </Tabs>
  );
}