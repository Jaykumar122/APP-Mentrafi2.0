import { Platform } from "react-native";
import Constants from "expo-constants";

// Dynamically detect the host IP address when running via Expo on physical device or emulator
const debuggerHost = Constants.expoConfig?.hostUri;
const lanIp = debuggerHost ? debuggerHost.split(":")[0] : "10.213.203.188";

const DEV_API_HOST = `http://${lanIp}:3001`;
const ANDROID_EMULATOR_HOST = "http://10.0.2.2:3001";
const WEB_HOST = "http://localhost:3001";

export const API_URL =
  Platform.OS === "web"
    ? WEB_HOST
    : Platform.OS === "android" && !debuggerHost
    ? ANDROID_EMULATOR_HOST
    : DEV_API_HOST;
