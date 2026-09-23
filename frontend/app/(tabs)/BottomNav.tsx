import React from "react";
import { View, TouchableOpacity, Text } from "react-native";
import { useRouter, usePathname } from "expo-router";
import { C } from "../(auth)/login";

export function getSafeTabRoute(route?: string): string {
  if (!route) return "/(tabs)/home";
  if (route.startsWith("/(tabs)/")) return route;
  const clean = route.replace(/^\//, "").replace(/^\(tabs\)\//, "");
  return `/(tabs)/${clean}`;
}

export default function BottomNav({
  tabs,
  paddingBottom = 10,
}: {
  tabs: { name: string; icon: any; route?: string }[];
  paddingBottom?: number;
}) {
  const router = useRouter();
  const pathname = usePathname() || "";

  function isTabActive(tabRoute?: string) {
    if (!tabRoute) return false;
    const clean = tabRoute.replace(/^\//, "").replace(/^\(tabs\)\//, "");
    return pathname.includes(clean);
  }

  return (
    <View
      style={{
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-around",
        paddingHorizontal: 8,
        paddingTop: 10,
        paddingBottom: paddingBottom,
        backgroundColor: "rgba(10,10,20,0.92)",
        borderTopWidth: 1,
        borderTopColor: C.inputBorder,
      }}
    >
      {tabs.map((tab) => {
        const isActive = isTabActive(tab.route);
        return (
          <TouchableOpacity
            key={tab.name}
            onPress={() => {
              if (tab.route && !isActive) {
                const target = getSafeTabRoute(tab.route);
                router.replace(target as any);
              }
            }}
            style={{ alignItems: "center", gap: 4, flex: 1 }}
          >
            <tab.icon size={22} color={isActive ? C.pink : C.textFaint} />
            <Text
              style={{
                fontSize: 10,
                color: isActive ? C.pink : C.textFaint,
                fontWeight: isActive ? "700" : "400",
              }}
            >
              {tab.name}
            </Text>
            {isActive && (
              <View style={{ width: 4, height: 4, borderRadius: 2, backgroundColor: C.pink, marginTop: 1 }} />
            )}
          </TouchableOpacity>
        );
      })}
    </View>
  );
}
