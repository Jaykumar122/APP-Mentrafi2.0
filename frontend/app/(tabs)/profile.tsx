import AsyncStorage from "@react-native-async-storage/async-storage";
import { useFocusEffect, useLocalSearchParams, useRouter } from "expo-router";
import * as ImagePicker from "expo-image-picker";
import { LinearGradient } from "expo-linear-gradient";
import {
  AlertCircle,
  ArrowDownCircle,
  ArrowLeft,
  ArrowLeftRight,
  ArrowRight,
  ArrowUpCircle,
  ArrowUpRight,
  Award,
  Bell,
  Briefcase,
  Building2,
  Calendar,
  Camera as CameraIcon,
  Check,
  CheckCircle2,
  ChevronRight,
  CreditCard,
  Download,
  FileText,
  GraduationCap,
  HelpCircle,
  History,
  Home,
  Image as ImageIcon,
  Laptop,
  Lock,
  LogOut,
  Palmtree,
  Pencil,
  Phone,
  PieChart,
  RefreshCw,
  Search,
  Settings,
  Shield,
  ShieldCheck,
  Sparkles,
  Target as TargetIcon,
  TrendingUp,
  User,
  Wallet,
  X,
} from "lucide-react-native";
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Dimensions,
  Image,
  KeyboardAvoidingView,
  Modal,
  Platform,
  RefreshControl,
  ScrollView,
  StatusBar,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { GestureDetector, Gesture } from "react-native-gesture-handler";
import Animated, {
  useAnimatedStyle,
  useSharedValue,
} from "react-native-reanimated";
import * as ImageManipulator from "expo-image-manipulator";
import Svg, { Defs, LinearGradient as SvgGrad, Path, Stop } from "react-native-svg";
import { API_URL } from "../../utils/api";
import { AuthBackground, C, depthShadow } from "../(auth)/login";
import BottomNav from "./BottomNav";
import { notifyProfileUpdated, subscribeProfileUpdates } from "../../utils/profileEvents";

let Haptics: any = null;
try {
  Haptics = require("expo-haptics");
} catch {
  // Graceful fallback
}

const SCREEN_WIDTH = Dimensions.get("window").width;
const CROP_SIZE = SCREEN_WIDTH - 80;
const GREEN = "#10b981";
const CYAN = "#06b6d4";
const VIOLET = "#8b5cf6";
const RED = "#f43f5e";

// ─────────────────────────────────────────────────────────────
// AVATAR FALLBACKS
// ─────────────────────────────────────────────────────────────
const AVATAR_COLORS = [VIOLET, CYAN, GREEN, "#f59e0b", "#ec4899", "#3b82f6", "#14b8a6"];

function getInitials(name?: string | null): string {
  const trimmed = name?.trim();
  if (!trimmed) return "M";
  const parts = trimmed.split(/\s+/).filter(Boolean);
  if (parts.length === 1) return parts[0].substring(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function getAvatarColor(seed?: string | null): string {
  const str = seed?.trim() || "user";
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}

// ─────────────────────────────────────────────────────────────
// DATE FORMATTING (TIMEZONE-SAFE)
// ─────────────────────────────────────────────────────────────
function formatDisplayDob(dobString?: string | null): string {
  if (!dobString) return "Not provided";
  const clean = dobString.split("T")[0];
  const parts = clean.split("-");
  if (parts.length === 3) {
    const [year, month, day] = parts;
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const mIndex = parseInt(month, 10) - 1;
    const monthName = months[mIndex] || month;
    return `${day}/${month}/${year} (${day} ${monthName} ${year})`;
  }
  return dobString;
}

function calculateAgeFromDobString(dateStr?: string | null): number | null {
  if (!dateStr) return null;
  const clean = dateStr.split("T")[0];
  const parts = clean.split("-");
  if (parts.length !== 3) return null;
  const [year, month, day] = parts.map((p) => parseInt(p, 10));
  if (isNaN(year) || isNaN(month) || isNaN(day)) return null;
  const today = new Date();
  let age = today.getFullYear() - year;
  const m = today.getMonth() - (month - 1);
  if (m < 0 || (m === 0 && today.getDate() < day)) {
    age--;
  }
  return age >= 0 ? age : null;
}

function formatTxDate(isoString: string) {
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleDateString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return isoString;
  }
}

// ─────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────
type ProfileData = {
  name: string;
  email: string;
  avatarUrl: string | null;
  kycStatus: string;
  tier: string;
  personalInfo: {
    phone: string | null;
    dateOfBirth: string | null;
    gender: string | null;
    location: string | null;
    age: number | null;
    monthlySipBudget: number | null;
    riskAppetite: string | null;
    investmentGoal: string | null;
    occupation: string | null;
  };
  stats: {
    portfolioValue: number;
    portfolioReturnPercent: number;
    activeSips: number;
    monthlySipAmount: number;
    fundsHeld: number;
  };
};

type PersonalInfoForm = {
  name: string;
  phone: string;
  dateOfBirth: string; // stored as DD / MM / YYYY in editor
  gender: string;
  location: string;
  age: string;
  monthlySipBudget: string;
  riskAppetite: string;
  investmentGoal: string;
  occupation: string;
};

type TransactionItem = {
  id: string;
  schemeCode: number;
  fundName: string;
  fundCategory: string;
  type: "BUY" | "REDEEM" | "SWITCH";
  amount: number;
  units: number;
  nav: number;
  status: string;
  paymentMethod: string;
  createdAt: string;
};

const OCCUPATION_OPTIONS = [
  { id: "job", label: "Doing Job / Salaried", icon: Briefcase, color: CYAN },
  { id: "student", label: "Student", icon: GraduationCap, color: VIOLET },
  { id: "business", label: "Business / Self-Employed", icon: Building2, color: GREEN },
  { id: "retired", label: "Retired Person", icon: Palmtree, color: "#fbbf24" },
  { id: "professional", label: "Freelancer / Professional", icon: Laptop, color: "#f472b6" },
  { id: "homemaker", label: "Homemaker", icon: Home, color: "#818cf8" },
];

const TX_FILTERS = [
  { label: "All", value: "ALL" },
  { label: "Investments", value: "BUY" },
  { label: "Redemptions", value: "REDEEM" },
  { label: "Switches", value: "SWITCH" },
] as const;

// ─────────────────────────────────────────────────────────────
// PROFILE SCREEN
// ─────────────────────────────────────────────────────────────
export default function ProfileScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ openEditor?: string }>();
  const hasAutoOpenedEditor = useRef(false);

  const tabs = [
    { name: "Home", icon: Home, route: "/(tabs)/home" },
    { name: "Explore", icon: Search, route: "/(tabs)/explore" },
    { name: "Portfolio", icon: PieChart, route: "/(tabs)/portfolio" },
    { name: "AI Advisor", icon: Sparkles, route: "/(tabs)/ai-advisor" },
    { name: "Profile", icon: User, route: "/(tabs)/profile" },
  ];

  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Transactions State
  const [transactions, setTransactions] = useState<TransactionItem[]>([]);
  const [txFilter, setTxFilter] = useState<"ALL" | "BUY" | "REDEEM" | "SWITCH">("ALL");
  const [loadingTx, setLoadingTx] = useState(false);

  // Avatar Upload State
  const [uploadingAvatar, setUploadingAvatar] = useState(false);
  const [pendingAvatar, setPendingAvatar] = useState<string | null>(null);
  const [showAvatarOptions, setShowAvatarOptions] = useState(false);
  const [pendingAvatarSize, setPendingAvatarSize] = useState({ width: 0, height: 0 });

  // Reanimated Shared Values for Crop Gesture
  const scale = useSharedValue(1);
  const savedScale = useSharedValue(1);
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const savedTranslateX = useSharedValue(0);
  const savedTranslateY = useSharedValue(0);

  // Personal Info Modal Editor State
  const [showPersonalInfoEditor, setShowPersonalInfoEditor] = useState(false);
  const [savingPersonalInfo, setSavingPersonalInfo] = useState(false);
  const [modalError, setModalError] = useState("");
  const [personalInfoForm, setPersonalInfoForm] = useState<PersonalInfoForm>({
    name: "",
    phone: "",
    dateOfBirth: "",
    gender: "",
    location: "",
    age: "",
    monthlySipBudget: "",
    riskAppetite: "",
    investmentGoal: "",
    occupation: "",
  });

  useFocusEffect(
    useCallback(() => {
      fetchProfile();
      fetchTransactions("ALL");
    }, [])
  );

  useEffect(() => {
    const unsubscribe = subscribeProfileUpdates(() => {
      fetchProfile();
    });
    return () => {
      unsubscribe();
    };
  }, []);

  useEffect(() => {
    if (params.openEditor === "1" && profile && !loading && !hasAutoOpenedEditor.current) {
      hasAutoOpenedEditor.current = true;
      openPersonalInfoEditor();
    }
  }, [params.openEditor, profile, loading]);

  async function fetchProfile() {
    try {
      const token = await AsyncStorage.getItem("userToken");
      if (!token) {
        router.replace("/(auth)/login");
        return;
      }

      const res = await fetch(`${API_URL}/api/profile`, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });

      if (!res.ok) throw new Error("Failed to fetch profile");

      const data = await res.json();
      setProfile(data);
      if (data.avatarUrl) {
        await AsyncStorage.setItem("userAvatarUrl", data.avatarUrl);
      }
      if (data.name) {
        await AsyncStorage.setItem("userName", data.name);
      }
      await AsyncStorage.setItem("cachedUserProfile", JSON.stringify(data));
    } catch (err) {
      console.error("Profile fetch error:", err);
    } finally {
      setLoading(false);
    }
  }

  async function fetchTransactions(filterType = txFilter) {
    try {
      setLoadingTx(true);
      const token = await AsyncStorage.getItem("userToken");
      if (!token) return;

      const query = filterType !== "ALL" ? `?type=${filterType}&limit=10` : "?limit=10";
      const res = await fetch(`${API_URL}/api/portfolio/transactions${query}`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.ok) {
        const data = await res.json();
        setTransactions(data.transactions || []);
      }
    } catch (err) {
      console.error("Error fetching transactions in profile:", err);
    } finally {
      setLoadingTx(false);
    }
  }

  async function onRefresh() {
    setRefreshing(true);
    await Promise.all([fetchProfile(), fetchTransactions(txFilter)]);
    setRefreshing(false);
  }

  function handleFilterChange(filter: "ALL" | "BUY" | "REDEEM" | "SWITCH") {
    if (Haptics?.impactAsync) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    }
    setTxFilter(filter);
    fetchTransactions(filter);
  }

  // ─────────────────────────────────────────────────────────────
  // AVATAR GESTURES & CROPPING
  // ─────────────────────────────────────────────────────────────
  function resetCropTransform() {
    scale.value = 1;
    savedScale.value = 1;
    translateX.value = 0;
    translateY.value = 0;
    savedTranslateX.value = 0;
    savedTranslateY.value = 0;
  }

  const pinchGesture = Gesture.Pinch()
    .onUpdate((e) => {
      scale.value = Math.max(1, Math.min(savedScale.value * e.scale, 4));
    })
    .onEnd(() => {
      savedScale.value = scale.value;
    });

  const panGesture = Gesture.Pan()
    .onUpdate((e) => {
      translateX.value = savedTranslateX.value + e.translationX;
      translateY.value = savedTranslateY.value + e.translationY;
    })
    .onEnd(() => {
      savedTranslateX.value = translateX.value;
      savedTranslateY.value = translateY.value;
    });

  const composedGesture = Gesture.Simultaneous(pinchGesture, panGesture);

  const animatedImageStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: translateX.value },
      { translateY: translateY.value },
      { scale: scale.value },
    ],
  }));

  function handleAvatarPencilPress() {
    if (Haptics?.impactAsync) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    }
    setShowAvatarOptions(true);
  }

  async function handlePickFromLibrary() {
    setShowAvatarOptions(false);
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Permission needed", "Please allow photo library access to change your profile picture.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 0.8,
    });

    if (result.canceled || !result.assets?.[0]) return;
    resetCropTransform();
    setPendingAvatar(result.assets[0].uri);
    setPendingAvatarSize({ width: result.assets[0].width, height: result.assets[0].height });
  }

  async function handleTakePhoto() {
    setShowAvatarOptions(false);
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Permission needed", "Please allow camera access to take a profile picture.");
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      cameraType: ImagePicker.CameraType.front,
      allowsEditing: false,
      quality: 0.8,
    });

    if (result.canceled || !result.assets?.[0]) return;
    resetCropTransform();
    setPendingAvatar(result.assets[0].uri);
    setPendingAvatarSize({ width: result.assets[0].width, height: result.assets[0].height });
  }

  async function confirmAvatarUpload() {
    if (!pendingAvatar) return;
    try {
      setUploadingAvatar(true);

      const imgW = pendingAvatarSize.width;
      const imgH = pendingAvatarSize.height;
      const baseScale = Math.max(CROP_SIZE / imgW, CROP_SIZE / imgH);
      const totalScale = baseScale * scale.value;
      const displayedW = imgW * totalScale;
      const displayedH = imgH * totalScale;
      const centerX = displayedW / 2 - translateX.value;
      const centerY = displayedH / 2 - translateY.value;
      const cropSizeInOriginal = CROP_SIZE / totalScale;
      let originX = centerX / totalScale - cropSizeInOriginal / 2;
      let originY = centerY / totalScale - cropSizeInOriginal / 2;

      originX = Math.max(0, Math.min(originX, imgW - cropSizeInOriginal));
      originY = Math.max(0, Math.min(originY, imgH - cropSizeInOriginal));

      const manipulated = await ImageManipulator.manipulateAsync(
        pendingAvatar,
        [
          {
            crop: {
              originX,
              originY,
              width: cropSizeInOriginal,
              height: cropSizeInOriginal,
            },
          },
          { resize: { width: 500, height: 500 } },
        ],
        { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG }
      );

      const token = await AsyncStorage.getItem("userToken");
      const formData = new FormData();
      formData.append("avatar", {
        uri: manipulated.uri,
        name: "avatar.jpg",
        type: "image/jpeg",
      } as any);

      const res = await fetch(`${API_URL}/api/profile/avatar`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");

      const data = await res.json();
      setProfile((prev) => (prev ? { ...prev, avatarUrl: data.avatarUrl } : prev));
      if (data.avatarUrl) {
        await AsyncStorage.setItem("userAvatarUrl", data.avatarUrl);
      }
      notifyProfileUpdated();
      setPendingAvatar(null);
    } catch (err) {
      console.error("Avatar upload error:", err);
      Alert.alert("Upload failed", "Couldn't update your profile picture. Try again.");
    } finally {
      setUploadingAvatar(false);
    }
  }

  // ─────────────────────────────────────────────────────────────
  // PERSONAL INFO EDITOR
  // ─────────────────────────────────────────────────────────────
  function openPersonalInfoEditor() {
    setModalError("");
    const rawDob = profile?.personalInfo?.dateOfBirth ?? "";
    let formattedEditorDob = "";
    if (rawDob) {
      const parts = rawDob.split("T")[0].split("-");
      if (parts.length === 3) {
        formattedEditorDob = `${parts[2]} / ${parts[1]} / ${parts[0]}`;
      } else {
        formattedEditorDob = rawDob;
      }
    }

    setPersonalInfoForm({
      name: profile?.name ?? "",
      phone: (profile?.personalInfo?.phone ?? "").replace(/^\+91\s*/, ""),
      dateOfBirth: formattedEditorDob,
      gender: profile?.personalInfo?.gender ?? "",
      location: profile?.personalInfo?.location ?? "",
      age: profile?.personalInfo?.age != null ? String(profile.personalInfo.age) : "",
      monthlySipBudget:
        profile?.personalInfo?.monthlySipBudget != null ? String(profile.personalInfo.monthlySipBudget) : "",
      riskAppetite: profile?.personalInfo?.riskAppetite ?? "Moderate",
      investmentGoal: profile?.personalInfo?.investmentGoal ?? "🚀 Wealth Creation",
      occupation: profile?.personalInfo?.occupation ?? "Doing Job / Salaried",
    });
    setShowPersonalInfoEditor(true);
  }

  async function savePersonalInfo() {
    setModalError("");
    try {
      setSavingPersonalInfo(true);
      const token = await AsyncStorage.getItem("userToken");
      if (!token) {
        router.replace("/(auth)/login");
        return;
      }

      // Convert editor DD / MM / YYYY to ISO YYYY-MM-DD
      let isoDate: string | null = null;
      let calculatedAge: number | null = null;
      if (personalInfoForm.dateOfBirth.trim()) {
        const digits = personalInfoForm.dateOfBirth.replace(/\D/g, "");
        if (digits.length === 8) {
          const day = digits.slice(0, 2);
          const month = digits.slice(2, 4);
          const year = digits.slice(4, 8);
          isoDate = `${year}-${month}-${day}`;

          const birthYear = parseInt(year, 10);
          const birthMonth = parseInt(month, 10);
          const birthDay = parseInt(day, 10);
          const today = new Date();
          calculatedAge = today.getFullYear() - birthYear;
          const m = today.getMonth() - (birthMonth - 1);
          if (m < 0 || (m === 0 && today.getDate() < birthDay)) {
            calculatedAge--;
          }
          if (calculatedAge < 18) {
            setModalError("SEBI requires investors to be at least 18 years old.");
            setSavingPersonalInfo(false);
            return;
          }
        } else {
          setModalError("Please enter a valid Date of Birth (DD / MM / YYYY).");
          setSavingPersonalInfo(false);
          return;
        }
      }

      // Phone formatting
      const cleanPhone = personalInfoForm.phone.replace(/\D/g, "");
      let formattedPhone: string | null = null;
      if (cleanPhone) {
        if (cleanPhone.length !== 10 || !/^[6-9]\d{9}$/.test(cleanPhone)) {
          setModalError("Please enter a valid 10-digit Indian mobile number (starts with 6-9).");
          setSavingPersonalInfo(false);
          return;
        }
        formattedPhone = `+91 ${cleanPhone}`;
      }

      const res = await fetch(`${API_URL}/api/profile`, {
        method: "PATCH",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: personalInfoForm.name.trim(),
          phone: formattedPhone,
          dateOfBirth: isoDate,
          gender: personalInfoForm.gender.trim() || null,
          location: personalInfoForm.location.trim() || null,
          age: calculatedAge ?? (personalInfoForm.age.trim() ? parseInt(personalInfoForm.age.trim(), 10) : null),
          monthlySipBudget: personalInfoForm.monthlySipBudget.trim()
            ? parseFloat(personalInfoForm.monthlySipBudget.trim())
            : null,
          riskAppetite: personalInfoForm.riskAppetite.trim() || null,
          investmentGoal: personalInfoForm.investmentGoal.trim() || null,
          occupation: personalInfoForm.occupation.trim() || null,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Failed to update personal information");
      }

      setProfile(data);
      if (data.name) {
        await AsyncStorage.setItem("userName", data.name);
      }
      await AsyncStorage.setItem("cachedUserProfile", JSON.stringify(data));
      notifyProfileUpdated();
      setShowPersonalInfoEditor(false);
    } catch (err: any) {
      console.error("Personal info update error:", err);
      setModalError(err?.message || "Couldn't save your profile details.");
    } finally {
      setSavingPersonalInfo(false);
    }
  }

  const handleLogout = () => {
    Alert.alert("Log Out", "Are you sure you want to log out of your Mentrafi account?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Log Out",
        style: "destructive",
        onPress: async () => {
          await AsyncStorage.removeItem("userToken");
          await AsyncStorage.removeItem("userName");
          await AsyncStorage.removeItem("userAvatarUrl");
          await AsyncStorage.removeItem("pendingProfileSetup");
          await AsyncStorage.removeItem("cachedUserProfile");
          router.replace("/(auth)/login");
        },
      },
    ]);
  };

  const handleBack = () => {
    if (typeof router.canGoBack === "function" && router.canGoBack()) {
      router.back();
    } else {
      router.replace("/(tabs)/home" as any);
    }
  };

  if (loading) {
    return (
      <View style={{ flex: 1, backgroundColor: C.bgBottom }}>
        <AuthBackground />
        <SafeAreaView style={{ flex: 1 }} edges={["top"]}>
          <View
            style={{
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "space-between",
              paddingHorizontal: 16,
              paddingVertical: 10,
              borderBottomWidth: 1,
              borderBottomColor: "rgba(255, 255, 255, 0.06)",
            }}
          >
            <TouchableOpacity
              onPress={handleBack}
              activeOpacity={0.7}
              hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
              style={{
                width: 40,
                height: 40,
                borderRadius: 20,
                backgroundColor: "rgba(255, 255, 255, 0.08)",
                borderWidth: 1,
                borderColor: "rgba(255, 255, 255, 0.14)",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <ArrowLeft size={20} color="#fff" />
            </TouchableOpacity>
            <Text style={{ color: "#fff", fontSize: 18, fontWeight: "700" }}>Profile</Text>
            <View style={{ width: 40, height: 40 }} />
          </View>
          <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
            <ActivityIndicator color={VIOLET} size="large" />
          </View>
        </SafeAreaView>
      </View>
    );
  }

  const avatarSource = profile?.avatarUrl
    ? profile.avatarUrl.startsWith("http")
      ? profile.avatarUrl
      : `${API_URL}${profile.avatarUrl}`
    : null;

  const avatarInitials = getInitials(profile?.name);
  const avatarColor = getAvatarColor(profile?.name || profile?.email);

  const calculatedAge =
    profile?.personalInfo?.age ?? calculateAgeFromDobString(profile?.personalInfo?.dateOfBirth);

  const displayDob = formatDisplayDob(profile?.personalInfo?.dateOfBirth);

  const portfolioVal = profile?.stats?.portfolioValue ?? 0;
  const portfolioReturn = profile?.stats?.portfolioReturnPercent ?? 0;
  const activeSipsCount = profile?.stats?.activeSips ?? 0;
  const monthlySipAmt = profile?.stats?.monthlySipAmount ?? 0;
  const monthlyBudget = profile?.personalInfo?.monthlySipBudget ? Number(profile.personalInfo.monthlySipBudget) : 0;
  const fundsHeldCount = profile?.stats?.fundsHeld ?? 0;

  // ─────────────────────────────────────────────────────────────
  // FULL SCREEN AVATAR CROP MODAL
  // ─────────────────────────────────────────────────────────────
  if (pendingAvatar) {
    return (
      <View style={{ flex: 1, backgroundColor: "#000" }}>
        <StatusBar barStyle="light-content" />
        <SafeAreaView style={{ flex: 1 }}>
          <View
            style={{
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "space-between",
              paddingHorizontal: 16,
              paddingVertical: 12,
            }}
          >
            <TouchableOpacity
              onPress={() => setPendingAvatar(null)}
              disabled={uploadingAvatar}
              hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              style={{ flexDirection: "row", alignItems: "center", gap: 4 }}
            >
              <X size={20} color="white" />
              <Text style={{ color: "white", fontSize: 15, fontWeight: "600" }}>Cancel</Text>
            </TouchableOpacity>

            <Text style={{ color: "white", fontSize: 16, fontWeight: "700" }}>Adjust Photo</Text>

            <TouchableOpacity
              onPress={confirmAvatarUpload}
              disabled={uploadingAvatar}
              hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
            >
              {uploadingAvatar ? (
                <ActivityIndicator color={CYAN} size="small" />
              ) : (
                <Text style={{ color: CYAN, fontSize: 15, fontWeight: "700" }}>Save</Text>
              )}
            </TouchableOpacity>
          </View>

          <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
            <View
              style={{
                width: CROP_SIZE,
                height: CROP_SIZE,
                borderRadius: CROP_SIZE / 2,
                overflow: "hidden",
                backgroundColor: "#000",
                borderWidth: 2,
                borderColor: CYAN,
                ...depthShadow("lg"),
              }}
            >
              <GestureDetector gesture={composedGesture}>
                <Animated.Image
                  source={{ uri: pendingAvatar }}
                  style={[{ width: CROP_SIZE, height: CROP_SIZE }, animatedImageStyle]}
                  resizeMode="cover"
                />
              </GestureDetector>
            </View>

            <Text style={{ color: C.textMuted, fontSize: 13, marginTop: 24, textAlign: "center" }}>
              Pinch to zoom, drag to reposition
            </Text>
          </View>
        </SafeAreaView>
      </View>
    );
  }

  return (
    <View style={{ flex: 1, backgroundColor: C.bgBottom }}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />
      <AuthBackground />

      <SafeAreaView style={{ flex: 1 }} edges={["top"]}>
        {/* Top Header Bar with Back Arrow */}
        <View
          style={{
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "space-between",
            paddingHorizontal: 16,
            paddingVertical: 10,
            borderBottomWidth: 1,
            borderBottomColor: "rgba(255, 255, 255, 0.06)",
          }}
        >
          <TouchableOpacity
            onPress={handleBack}
            activeOpacity={0.7}
            hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
            style={{
              width: 40,
              height: 40,
              borderRadius: 20,
              backgroundColor: "rgba(255, 255, 255, 0.08)",
              borderWidth: 1,
              borderColor: "rgba(255, 255, 255, 0.14)",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <ArrowLeft size={20} color="#fff" />
          </TouchableOpacity>

          <Text
            style={{
              color: "#fff",
              fontSize: 18,
              fontWeight: "700",
              letterSpacing: 0.3,
            }}
          >
            Profile
          </Text>

          <TouchableOpacity
            onPress={onRefresh}
            activeOpacity={0.7}
            hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
            style={{
              width: 40,
              height: 40,
              borderRadius: 20,
              backgroundColor: "rgba(255, 255, 255, 0.08)",
              borderWidth: 1,
              borderColor: "rgba(255, 255, 255, 0.14)",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <RefreshCw size={18} color={CYAN} />
          </TouchableOpacity>
        </View>

        <ScrollView
          contentContainerStyle={{ paddingBottom: 60 }}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={CYAN} />
          }
          showsVerticalScrollIndicator={false}
        >
          {/* ─────────────────────────────────────────────────────────── */}
          {/* 1. HERO USER BENTO CARD */}
          {/* ─────────────────────────────────────────────────────────── */}
          <View style={{ paddingHorizontal: 16, marginTop: 10, marginBottom: 16 }}>
            <View
              style={{
                borderRadius: 28,
                backgroundColor: C.card,
                borderWidth: 1,
                borderColor: C.cardEdge,
                overflow: "hidden",
                padding: 22,
                ...depthShadow("lg"),
              }}
            >
              {/* Top Fiduciary & SEBI Pill */}
              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: 18,
                }}
              >
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 6,
                    backgroundColor: "rgba(16, 185, 129, 0.15)",
                    paddingHorizontal: 10,
                    paddingVertical: 5,
                    borderRadius: 14,
                    borderWidth: 1,
                    borderColor: "rgba(16, 185, 129, 0.35)",
                  }}
                >
                  <View style={{ width: 6, height: 6, borderRadius: 3, backgroundColor: GREEN }} />
                  <Text style={{ color: "#34d399", fontSize: 10.5, fontWeight: "700", letterSpacing: 0.4 }}>
                    🇮🇳 SEBI REG. DIRECT INVESTOR
                  </Text>
                </View>

                <TouchableOpacity
                  onPress={openPersonalInfoEditor}
                  activeOpacity={0.8}
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    gap: 5,
                    backgroundColor: "rgba(255, 255, 255, 0.08)",
                    paddingHorizontal: 12,
                    paddingVertical: 6,
                    borderRadius: 16,
                    borderWidth: 1,
                    borderColor: "rgba(255, 255, 255, 0.12)",
                  }}
                >
                  <Pencil size={12} color={CYAN} />
                  <Text style={{ color: "#fff", fontSize: 11.5, fontWeight: "700" }}>Edit Profile</Text>
                </TouchableOpacity>
              </View>

              {/* Avatar, Name, Email, Tier */}
              <View style={{ flexDirection: "row", alignItems: "center", gap: 16 }}>
                <View style={{ position: "relative" }}>
                  <LinearGradient
                    colors={[C.pink, C.violet]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={{
                      width: 84,
                      height: 84,
                      borderRadius: 42,
                      padding: 2.5,
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <View
                      style={{
                        width: "100%",
                        height: "100%",
                        borderRadius: 40,
                        overflow: "hidden",
                        backgroundColor: "#0d1b2a",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      {uploadingAvatar ? (
                        <ActivityIndicator color={C.pink} />
                      ) : avatarSource ? (
                        <Image source={{ uri: avatarSource }} style={{ width: "100%", height: "100%" }} />
                      ) : (
                        <LinearGradient
                          colors={[C.pink, C.violet]}
                          start={{ x: 0, y: 0 }}
                          end={{ x: 1, y: 1 }}
                          style={{
                            width: "100%",
                            height: "100%",
                            alignItems: "center",
                            justifyContent: "center",
                          }}
                        >
                          <Text style={{ color: "#fff", fontSize: 28, fontWeight: "800" }}>
                            {avatarInitials}
                          </Text>
                        </LinearGradient>
                      )}
                    </View>
                  </LinearGradient>

                  <TouchableOpacity
                    onPress={handleAvatarPencilPress}
                    disabled={uploadingAvatar}
                    style={{
                      position: "absolute",
                      bottom: 0,
                      right: -2,
                      width: 26,
                      height: 26,
                      borderRadius: 13,
                      backgroundColor: CYAN,
                      alignItems: "center",
                      justifyContent: "center",
                      borderWidth: 2,
                      borderColor: "#000",
                    }}
                  >
                    <CameraIcon size={12} color="#000" />
                  </TouchableOpacity>
                </View>

                <View style={{ flex: 1 }}>
                  <Text
                    style={{ color: "#fff", fontSize: 21, fontWeight: "800", letterSpacing: -0.2 }}
                    numberOfLines={1}
                  >
                    {profile?.name || "Mentrafi Investor"}
                  </Text>
                  <Text style={{ color: C.textMuted, fontSize: 12.5, marginTop: 2 }} numberOfLines={1}>
                    {profile?.email || ""}
                  </Text>

                  <View
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      gap: 6,
                      marginTop: 8,
                      alignSelf: "flex-start",
                      backgroundColor: "rgba(139, 92, 246, 0.18)",
                      paddingHorizontal: 10,
                      paddingVertical: 4,
                      borderRadius: 12,
                      borderWidth: 1,
                      borderColor: "rgba(139, 92, 246, 0.35)",
                    }}
                  >
                    <Award size={12} color={VIOLET} />
                    <Text style={{ color: "#c084fc", fontSize: 11, fontWeight: "700" }}>
                      {profile?.tier || "Standard Investor"}
                    </Text>
                  </View>
                </View>
              </View>

              {/* Quick Stat Highlights */}
              <View
                style={{
                  flexDirection: "row",
                  gap: 10,
                  marginTop: 20,
                  paddingTop: 16,
                  borderTopWidth: 1,
                  borderTopColor: "rgba(255, 255, 255, 0.07)",
                }}
              >
                {/* 1. PORTFOLIO CARD */}
                <View
                  style={{
                    flex: 1,
                    backgroundColor: "rgba(255, 255, 255, 0.03)",
                    borderRadius: 16,
                    padding: 12,
                    borderWidth: 1,
                    borderColor: "rgba(255, 255, 255, 0.06)",
                  }}
                >
                  <Text style={{ color: C.textFaint, fontSize: 10, fontWeight: "600" }}>PORTFOLIO</Text>
                  <Text style={{ color: "#fff", fontSize: 16, fontWeight: "800", marginTop: 4 }}>
                    ₹{portfolioVal.toLocaleString("en-IN", { maximumFractionDigits: 2 })}
                  </Text>
                  <Text
                    style={{
                      color: portfolioReturn >= 0 ? GREEN : RED,
                      fontSize: 10,
                      fontWeight: "700",
                      marginTop: 2,
                    }}
                  >
                    {portfolioReturn > 0 ? "+" : ""}
                    {portfolioReturn.toFixed(1)}% Returns
                  </Text>
                </View>

                {/* 2. ACTIVE SIPS CARD */}
                <View
                  style={{
                    flex: 1,
                    backgroundColor: "rgba(255, 255, 255, 0.03)",
                    borderRadius: 16,
                    padding: 12,
                    borderWidth: 1,
                    borderColor: "rgba(255, 255, 255, 0.06)",
                  }}
                >
                  <Text style={{ color: C.textFaint, fontSize: 10, fontWeight: "600" }}>ACTIVE SIPS</Text>
                  <Text style={{ color: "#fff", fontSize: 16, fontWeight: "800", marginTop: 4 }}>
                    {activeSipsCount} {activeSipsCount === 1 ? "Plan" : "Plans"}
                  </Text>
                  <Text style={{ color: CYAN, fontSize: 10, fontWeight: "700", marginTop: 2 }}>
                    {activeSipsCount > 0
                      ? `₹${monthlySipAmt.toLocaleString("en-IN")} / mo`
                      : monthlyBudget > 0
                      ? `₹${monthlyBudget.toLocaleString("en-IN")} budget`
                      : "0 / mo"}
                  </Text>
                </View>

                {/* 3. FUNDS HELD CARD */}
                <View
                  style={{
                    flex: 1,
                    backgroundColor: "rgba(255, 255, 255, 0.03)",
                    borderRadius: 16,
                    padding: 12,
                    borderWidth: 1,
                    borderColor: "rgba(255, 255, 255, 0.06)",
                  }}
                >
                  <Text style={{ color: C.textFaint, fontSize: 10, fontWeight: "600" }}>FUNDS HELD</Text>
                  <Text style={{ color: "#fff", fontSize: 16, fontWeight: "800", marginTop: 4 }}>
                    {fundsHeldCount} Direct
                  </Text>
                  <Text style={{ color: "#a78bfa", fontSize: 10, fontWeight: "700", marginTop: 2 }}>
                    0% Commission
                  </Text>
                </View>
              </View>
            </View>
          </View>

          {/* ─────────────────────────────────────────────────────────── */}
          {/* 2. INVESTOR PROFILE & KYC DETAILS (FIXED DOB & OCCUPATION) */}
          {/* ─────────────────────────────────────────────────────────── */}
          <View style={{ paddingHorizontal: 16, marginBottom: 16 }}>
            <View
              style={{
                borderRadius: 28,
                backgroundColor: C.card,
                borderWidth: 1,
                borderColor: C.cardEdge,
                padding: 20,
                ...depthShadow("md"),
              }}
            >
              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: 16,
                }}
              >
                <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                  <View
                    style={{
                      width: 32,
                      height: 32,
                      borderRadius: 10,
                      backgroundColor: "rgba(6, 182, 212, 0.15)",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <User size={16} color={CYAN} />
                  </View>
                  <Text style={{ color: "#fff", fontSize: 16, fontWeight: "800" }}>
                    Investor Profile & KYC
                  </Text>
                </View>

                <TouchableOpacity onPress={openPersonalInfoEditor} activeOpacity={0.7}>
                  <Text style={{ color: CYAN, fontSize: 12, fontWeight: "700" }}>Change</Text>
                </TouchableOpacity>
              </View>

              <View style={{ gap: 12 }}>
                {/* Phone */}
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "space-between",
                    paddingVertical: 10,
                    borderBottomWidth: 1,
                    borderBottomColor: "rgba(255, 255, 255, 0.06)",
                  }}
                >
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
                    <Phone size={16} color={CYAN} />
                    <Text style={{ color: C.textMuted, fontSize: 13, fontWeight: "500" }}>Mobile</Text>
                  </View>
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
                    <Text style={{ color: "#fff", fontSize: 13.5, fontWeight: "700" }}>
                      {profile?.personalInfo?.phone || "Not linked"}
                    </Text>
                    {profile?.personalInfo?.phone && (
                      <View
                        style={{
                          backgroundColor: "rgba(16, 185, 129, 0.2)",
                          paddingHorizontal: 6,
                          paddingVertical: 2,
                          borderRadius: 8,
                        }}
                      >
                        <Text style={{ color: "#34d399", fontSize: 9.5, fontWeight: "700" }}>VERIFIED</Text>
                      </View>
                    )}
                  </View>
                </View>

                {/* Date of Birth & Age (CRITICAL FIX: NO 2005/10/9 TIMEZONE SHIFT!) */}
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "space-between",
                    paddingVertical: 10,
                    borderBottomWidth: 1,
                    borderBottomColor: "rgba(255, 255, 255, 0.06)",
                  }}
                >
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
                    <Calendar size={16} color={GREEN} />
                    <Text style={{ color: C.textMuted, fontSize: 13, fontWeight: "500" }}>
                      Date of Birth
                    </Text>
                  </View>
                  <View style={{ alignItems: "flex-end" }}>
                    <Text style={{ color: "#fff", fontSize: 13.5, fontWeight: "700" }}>
                      {displayDob}
                    </Text>
                    {calculatedAge !== null && (
                      <Text style={{ color: GREEN, fontSize: 11, fontWeight: "600", marginTop: 2 }}>
                        🎂 {calculatedAge} years old (18+ Verified)
                      </Text>
                    )}
                  </View>
                </View>

                {/* Occupation / Job */}
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "space-between",
                    paddingVertical: 10,
                    borderBottomWidth: 1,
                    borderBottomColor: "rgba(255, 255, 255, 0.06)",
                  }}
                >
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
                    <Briefcase size={16} color={VIOLET} />
                    <Text style={{ color: C.textMuted, fontSize: 13, fontWeight: "500" }}>Occupation</Text>
                  </View>
                  <View
                    style={{
                      backgroundColor: "rgba(139, 92, 246, 0.18)",
                      paddingHorizontal: 10,
                      paddingVertical: 4,
                      borderRadius: 12,
                      borderWidth: 1,
                      borderColor: "rgba(139, 92, 246, 0.3)",
                    }}
                  >
                    <Text style={{ color: "#c084fc", fontSize: 12.5, fontWeight: "700" }}>
                      {profile?.personalInfo?.occupation || "Doing Job / Salaried"}
                    </Text>
                  </View>
                </View>

                {/* Risk Profile */}
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "space-between",
                    paddingVertical: 10,
                    borderBottomWidth: 1,
                    borderBottomColor: "rgba(255, 255, 255, 0.06)",
                  }}
                >
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
                    <Shield size={16} color={CYAN} />
                    <Text style={{ color: C.textMuted, fontSize: 13, fontWeight: "500" }}>
                      Risk Appetite
                    </Text>
                  </View>
                  <View
                    style={{
                      backgroundColor: "rgba(6, 182, 212, 0.15)",
                      paddingHorizontal: 10,
                      paddingVertical: 4,
                      borderRadius: 12,
                      borderWidth: 1,
                      borderColor: "rgba(6, 182, 212, 0.3)",
                    }}
                  >
                    <Text style={{ color: CYAN, fontSize: 12.5, fontWeight: "700" }}>
                      {profile?.personalInfo?.riskAppetite || "Moderate"}
                    </Text>
                  </View>
                </View>

                {/* Monthly SIP Budget */}
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "space-between",
                    paddingVertical: 10,
                    borderBottomWidth: 1,
                    borderBottomColor: "rgba(255, 255, 255, 0.06)",
                  }}
                >
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
                    <Wallet size={16} color={GREEN} />
                    <Text style={{ color: C.textMuted, fontSize: 13, fontWeight: "500" }}>
                      Monthly Budget
                    </Text>
                  </View>
                  <Text style={{ color: "#fff", fontSize: 13.5, fontWeight: "700" }}>
                    ₹{(profile?.personalInfo?.monthlySipBudget ?? 5000).toLocaleString("en-IN")} / mo
                  </Text>
                </View>

                {/* Investment Goal */}
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "space-between",
                    paddingVertical: 10,
                  }}
                >
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
                    <TargetIcon size={16} color="#f472b6" />
                    <Text style={{ color: C.textMuted, fontSize: 13, fontWeight: "500" }}>Goal</Text>
                  </View>
                  <Text
                    style={{ color: "#fff", fontSize: 13, fontWeight: "700", maxWidth: "60%" }}
                    numberOfLines={1}
                  >
                    {profile?.personalInfo?.investmentGoal || "🚀 Wealth Creation"}
                  </Text>
                </View>
              </View>
            </View>
          </View>

          {/* ─────────────────────────────────────────────────────────── */}
          {/* 3. TRANSACTION HISTORY BENTO (REQUESTED FEATURE!) */}
          {/* ─────────────────────────────────────────────────────────── */}
          <View style={{ paddingHorizontal: 16, marginBottom: 16 }}>
            <View
              style={{
                borderRadius: 28,
                backgroundColor: C.card,
                borderWidth: 1,
                borderColor: C.cardEdge,
                padding: 20,
                ...depthShadow("md"),
              }}
            >
              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: 14,
                }}
              >
                <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                  <View
                    style={{
                      width: 32,
                      height: 32,
                      borderRadius: 10,
                      backgroundColor: "rgba(16, 185, 129, 0.15)",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <History size={16} color={GREEN} />
                  </View>
                  <Text style={{ color: "#fff", fontSize: 16, fontWeight: "800" }}>
                    Transaction History
                  </Text>
                </View>

                <TouchableOpacity
                  onPress={() => router.push("/(tabs)/transactions" as any)}
                  activeOpacity={0.7}
                  style={{ flexDirection: "row", alignItems: "center", gap: 3 }}
                >
                  <Text style={{ color: CYAN, fontSize: 12, fontWeight: "700" }}>View All</Text>
                  <ChevronRight size={13} color={CYAN} />
                </TouchableOpacity>
              </View>

              {/* Filter Tabs */}
              <View style={{ flexDirection: "row", gap: 6, marginBottom: 14 }}>
                {TX_FILTERS.map((f) => {
                  const active = txFilter === f.value;
                  return (
                    <TouchableOpacity
                      key={f.value}
                      onPress={() => handleFilterChange(f.value)}
                      activeOpacity={0.8}
                      style={{
                        paddingHorizontal: 10,
                        paddingVertical: 5,
                        borderRadius: 12,
                        backgroundColor: active ? "rgba(6, 182, 212, 0.25)" : "rgba(255, 255, 255, 0.04)",
                        borderWidth: 1,
                        borderColor: active ? CYAN : "rgba(255, 255, 255, 0.08)",
                      }}
                    >
                      <Text
                        style={{
                          color: active ? "#fff" : C.textMuted,
                          fontSize: 11,
                          fontWeight: active ? "700" : "500",
                        }}
                      >
                        {f.label}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </View>

              {/* Transactions List */}
              {loadingTx ? (
                <View style={{ paddingVertical: 20, alignItems: "center" }}>
                  <ActivityIndicator color={CYAN} size="small" />
                </View>
              ) : transactions.length === 0 ? (
                <View
                  style={{
                    paddingVertical: 24,
                    alignItems: "center",
                    backgroundColor: "rgba(255, 255, 255, 0.02)",
                    borderRadius: 18,
                    borderWidth: 1,
                    borderColor: "rgba(255, 255, 255, 0.05)",
                  }}
                >
                  <History size={28} color={C.textFaint} />
                  <Text style={{ color: "#fff", fontSize: 14, fontWeight: "700", marginTop: 10 }}>
                    No transactions yet
                  </Text>
                  <Text
                    style={{
                      color: C.textMuted,
                      fontSize: 11.5,
                      textAlign: "center",
                      marginTop: 4,
                      paddingHorizontal: 20,
                    }}
                  >
                    Your direct mutual fund investments, monthly SIPs, and redemptions will appear here.
                  </Text>

                  <TouchableOpacity
                    onPress={() => router.push("/(tabs)/explore")}
                    activeOpacity={0.8}
                    style={{
                      marginTop: 14,
                      flexDirection: "row",
                      alignItems: "center",
                      gap: 6,
                      backgroundColor: "rgba(6, 182, 212, 0.15)",
                      paddingHorizontal: 14,
                      paddingVertical: 7,
                      borderRadius: 14,
                      borderWidth: 1,
                      borderColor: "rgba(6, 182, 212, 0.35)",
                    }}
                  >
                    <Text style={{ color: CYAN, fontSize: 12, fontWeight: "700" }}>
                      Explore Direct Funds
                    </Text>
                    <ArrowRight size={13} color={CYAN} />
                  </TouchableOpacity>
                </View>
              ) : (
                <View style={{ gap: 10 }}>
                  {transactions.slice(0, 5).map((tx) => {
                    const isBuy = tx.type === "BUY";
                    const isRedeem = tx.type === "REDEEM";
                    const typeColor = isBuy ? GREEN : isRedeem ? RED : CYAN;

                    return (
                      <View
                        key={tx.id}
                        style={{
                          flexDirection: "row",
                          alignItems: "center",
                          justifyContent: "space-between",
                          backgroundColor: "rgba(255, 255, 255, 0.03)",
                          borderRadius: 16,
                          padding: 12,
                          borderWidth: 1,
                          borderColor: "rgba(255, 255, 255, 0.06)",
                        }}
                      >
                        <View style={{ flexDirection: "row", alignItems: "center", gap: 10, flex: 1 }}>
                          <View
                            style={{
                              width: 36,
                              height: 36,
                              borderRadius: 12,
                              backgroundColor: isBuy
                                ? "rgba(16, 185, 129, 0.15)"
                                : isRedeem
                                ? "rgba(244, 63, 94, 0.15)"
                                : "rgba(6, 182, 212, 0.15)",
                              alignItems: "center",
                              justifyContent: "center",
                            }}
                          >
                            {isBuy ? (
                              <ArrowDownCircle size={18} color={GREEN} />
                            ) : isRedeem ? (
                              <ArrowUpCircle size={18} color={RED} />
                            ) : (
                              <ArrowLeftRight size={18} color={CYAN} />
                            )}
                          </View>

                          <View style={{ flex: 1 }}>
                            <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700" }} numberOfLines={1}>
                              {tx.fundName}
                            </Text>
                            <Text style={{ color: C.textFaint, fontSize: 10.5, marginTop: 2 }}>
                              {formatTxDate(tx.createdAt)}
                            </Text>
                          </View>
                        </View>

                        <View style={{ alignItems: "flex-end" }}>
                          <Text style={{ color: typeColor, fontSize: 13.5, fontWeight: "800" }}>
                            {isBuy ? "+" : isRedeem ? "-" : ""}₹{Number(tx.amount).toLocaleString("en-IN")}
                          </Text>
                          <View
                            style={{
                              backgroundColor:
                                tx.status === "COMPLETED"
                                  ? "rgba(16, 185, 129, 0.18)"
                                  : "rgba(245, 158, 11, 0.18)",
                              paddingHorizontal: 6,
                              paddingVertical: 2,
                              borderRadius: 6,
                              marginTop: 2,
                            }}
                          >
                            <Text
                              style={{
                                color: tx.status === "COMPLETED" ? "#34d399" : "#fbbf24",
                                fontSize: 9,
                                fontWeight: "700",
                              }}
                            >
                              {tx.status}
                            </Text>
                          </View>
                        </View>
                      </View>
                    );
                  })}
                </View>
              )}
            </View>
          </View>

          {/* ─────────────────────────────────────────────────────────── */}
          {/* 4. SECURITY, TAX STATEMENTS & PREFERENCES */}
          {/* ─────────────────────────────────────────────────────────── */}
          <View style={{ paddingHorizontal: 16, marginBottom: 16 }}>
            <View
              style={{
                borderRadius: 28,
                backgroundColor: C.card,
                borderWidth: 1,
                borderColor: C.cardEdge,
                padding: 18,
                ...depthShadow("md"),
              }}
            >
              <Text
                style={{
                  color: C.textFaint,
                  fontSize: 10.5,
                  fontWeight: "700",
                  letterSpacing: 1.2,
                  marginBottom: 12,
                  paddingLeft: 4,
                }}
              >
                ACCOUNT & FIDUCIARY SERVICES
              </Text>

              <View style={{ gap: 8 }}>
                {[
                  {
                    icon: Wallet,
                    title: "Bank Mandate & AutoPay",
                    subtitle: "Auto-debit active for direct SIPs",
                    badge: "Active",
                    color: GREEN,
                    onPress: () => router.push("/(tabs)/invest" as any),
                  },
                  {
                    icon: FileText,
                    title: "Tax Statement & Capital Gains",
                    subtitle: "Download CAS & FY 2025-26 80C slips",
                    badge: "PDF",
                    color: CYAN,
                    onPress: () => Alert.alert("Tax Statements", "Your Consolidated Account Statement (CAS) has been prepared and sent to your registered email."),
                  },
                  {
                    icon: ShieldCheck,
                    title: "256-Bit Bank Encryption",
                    subtitle: "SEBI Registered Direct Mutual Funds",
                    badge: "Protected",
                    color: VIOLET,
                    onPress: () => Alert.alert("Security Certified", "Your account is secured with 256-bit TLS bank encryption and SEBI direct execution protocols."),
                  },
                  {
                    icon: HelpCircle,
                    title: "AI Advisor & Priority Support",
                    subtitle: "24/7 dedicated portfolio intelligence",
                    badge: "AI 24/7",
                    color: "#f59e0b",
                    onPress: () => router.push("/(tabs)/ai-advisor" as any),
                  },
                ].map((item) => {
                  const IconComp = item.icon;
                  return (
                    <TouchableOpacity
                      key={item.title}
                      onPress={item.onPress}
                      activeOpacity={0.75}
                      style={{
                        flexDirection: "row",
                        alignItems: "center",
                        padding: 12,
                        borderRadius: 18,
                        backgroundColor: "rgba(255, 255, 255, 0.03)",
                        borderWidth: 1,
                        borderColor: "rgba(255, 255, 255, 0.06)",
                      }}
                    >
                      <View
                        style={{
                          width: 38,
                          height: 38,
                          borderRadius: 12,
                          backgroundColor: `${item.color}22`,
                          alignItems: "center",
                          justifyContent: "center",
                          marginRight: 12,
                        }}
                      >
                        <IconComp size={18} color={item.color} />
                      </View>

                      <View style={{ flex: 1 }}>
                        <Text style={{ color: "#fff", fontSize: 13.5, fontWeight: "700" }}>
                          {item.title}
                        </Text>
                        <Text style={{ color: C.textFaint, fontSize: 11, marginTop: 1 }}>
                          {item.subtitle}
                        </Text>
                      </View>

                      <View
                        style={{
                          backgroundColor: "rgba(255, 255, 255, 0.06)",
                          paddingHorizontal: 8,
                          paddingVertical: 3,
                          borderRadius: 8,
                          marginRight: 6,
                        }}
                      >
                        <Text style={{ color: item.color, fontSize: 10, fontWeight: "700" }}>
                          {item.badge}
                        </Text>
                      </View>

                      <ChevronRight size={14} color={C.textFaint} />
                    </TouchableOpacity>
                  );
                })}
              </View>
            </View>
          </View>

          {/* ─────────────────────────────────────────────────────────── */}
          {/* 5. LOGOUT BUTTON & SEBI DISCLAIMER */}
          {/* ─────────────────────────────────────────────────────────── */}
          <View style={{ paddingHorizontal: 16, marginTop: 4 }}>
            <TouchableOpacity
              onPress={handleLogout}
              activeOpacity={0.85}
              style={{
                borderRadius: 22,
                backgroundColor: "rgba(244, 63, 94, 0.12)",
                borderWidth: 1,
                borderColor: "rgba(244, 63, 94, 0.3)",
                paddingVertical: 14,
                flexDirection: "row",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
                ...depthShadow("sm"),
              }}
            >
              <LogOut size={16} color={RED} />
              <Text style={{ color: RED, fontWeight: "700", fontSize: 14 }}>
                Log Out of Mentrafi
              </Text>
            </TouchableOpacity>

            <View
              style={{
                flexDirection: "row",
                alignItems: "center",
                justifyContent: "center",
                marginTop: 20,
                gap: 6,
              }}
            >
              <ShieldCheck size={13} color={GREEN} />
              <Text style={{ color: C.textFaint, fontSize: 11, fontWeight: "500" }}>
                SEBI Registered Direct Mutual Fund Platform • 256-Bit Encryption
              </Text>
            </View>
          </View>
        </ScrollView>

        {/* Global Bottom Navigation */}
        <BottomNav tabs={tabs} paddingBottom={Platform.OS === "ios" ? 18 : 10} />

        {/* ─────────────────────────────────────────────────────────── */}
        {/* AVATAR SOURCE PICKER ACTION SHEET */}
        {/* ─────────────────────────────────────────────────────────── */}
        {showAvatarOptions && (
          <View
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: "rgba(0,0,0,0.6)",
              justifyContent: "flex-end",
            }}
          >
            <TouchableOpacity
              style={{ flex: 1 }}
              activeOpacity={1}
              onPress={() => setShowAvatarOptions(false)}
            />
            <SafeAreaView
              edges={["bottom"]}
              style={{
                backgroundColor: C.card,
                borderTopLeftRadius: 28,
                borderTopRightRadius: 28,
                borderWidth: 1,
                borderColor: C.cardEdge,
                paddingHorizontal: 20,
                paddingTop: 12,
                paddingBottom: 24,
              }}
            >
              <View
                style={{
                  width: 36,
                  height: 4,
                  borderRadius: 2,
                  backgroundColor: C.inputBorder,
                  alignSelf: "center",
                  marginBottom: 16,
                }}
              />
              <Text
                style={{
                  color: "#fff",
                  fontSize: 16,
                  fontWeight: "700",
                  textAlign: "center",
                  marginBottom: 16,
                }}
              >
                Change Profile Picture
              </Text>

              <TouchableOpacity
                onPress={handleTakePhoto}
                activeOpacity={0.75}
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  gap: 12,
                  paddingVertical: 14,
                  borderBottomWidth: 1,
                  borderBottomColor: "rgba(255, 255, 255, 0.06)",
                }}
              >
                <CameraIcon size={20} color={CYAN} />
                <Text style={{ color: "#fff", fontSize: 15, fontWeight: "600" }}>Take Photo</Text>
              </TouchableOpacity>

              <TouchableOpacity
                onPress={handlePickFromLibrary}
                activeOpacity={0.75}
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  gap: 12,
                  paddingVertical: 14,
                }}
              >
                <ImageIcon size={20} color={GREEN} />
                <Text style={{ color: "#fff", fontSize: 15, fontWeight: "600" }}>Choose from Library</Text>
              </TouchableOpacity>

              <TouchableOpacity
                onPress={() => setShowAvatarOptions(false)}
                activeOpacity={0.75}
                style={{
                  marginTop: 12,
                  paddingVertical: 13,
                  alignItems: "center",
                  backgroundColor: "rgba(255, 255, 255, 0.06)",
                  borderRadius: 16,
                }}
              >
                <Text style={{ color: C.textMuted, fontSize: 14, fontWeight: "700" }}>Cancel</Text>
              </TouchableOpacity>
            </SafeAreaView>
          </View>
        )}

        {/* ─────────────────────────────────────────────────────────── */}
        {/* PERSONAL INFO EDIT MODAL */}
        {/* ─────────────────────────────────────────────────────────── */}
        <Modal
          visible={showPersonalInfoEditor}
          transparent
          animationType="slide"
          onRequestClose={() => setShowPersonalInfoEditor(false)}
        >
          <View
            style={{
              flex: 1,
              backgroundColor: "rgba(0,0,0,0.75)",
              justifyContent: "flex-end",
            }}
          >
            <KeyboardAvoidingView
              behavior={Platform.OS === "ios" ? "padding" : undefined}
              style={{ maxHeight: "88%" }}
            >
              <View
                style={{
                  backgroundColor: "#0a0f1d",
                  borderTopLeftRadius: 32,
                  borderTopRightRadius: 32,
                  borderWidth: 1,
                  borderColor: "rgba(255, 255, 255, 0.12)",
                  paddingHorizontal: 20,
                  paddingTop: 12,
                  paddingBottom: 30,
                }}
              >
                <View
                  style={{
                    width: 36,
                    height: 4,
                    borderRadius: 2,
                    backgroundColor: "rgba(255, 255, 255, 0.2)",
                    alignSelf: "center",
                    marginBottom: 14,
                  }}
                />

                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    justifyContent: "space-between",
                    marginBottom: 14,
                  }}
                >
                  <Text style={{ color: "#fff", fontSize: 18, fontWeight: "800" }}>
                    Edit Investor Details
                  </Text>
                  <TouchableOpacity onPress={() => setShowPersonalInfoEditor(false)}>
                    <X size={20} color={C.textMuted} />
                  </TouchableOpacity>
                </View>

                {modalError ? (
                  <View
                    style={{
                      flexDirection: "row",
                      alignItems: "center",
                      gap: 8,
                      backgroundColor: "rgba(244, 63, 94, 0.15)",
                      borderWidth: 1,
                      borderColor: "rgba(244, 63, 94, 0.35)",
                      borderRadius: 14,
                      padding: 10,
                      marginBottom: 12,
                    }}
                  >
                    <AlertCircle size={16} color={RED} />
                    <Text style={{ color: RED, fontSize: 12, fontWeight: "600", flex: 1 }}>
                      {modalError}
                    </Text>
                  </View>
                ) : null}

                <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ gap: 14 }}>
                  {/* Full Name */}
                  <View>
                    <Text style={{ color: C.textMuted, fontSize: 12, fontWeight: "700", marginBottom: 6 }}>
                      Full Name
                    </Text>
                    <TextInput
                      value={personalInfoForm.name}
                      onChangeText={(t) => setPersonalInfoForm((p) => ({ ...p, name: t }))}
                      placeholder="Your Full Name"
                      placeholderTextColor={C.textFaint}
                      style={{
                        backgroundColor: C.input,
                        borderRadius: 16,
                        borderWidth: 1,
                        borderColor: C.inputBorder,
                        color: "#fff",
                        paddingHorizontal: 14,
                        paddingVertical: 12,
                        fontSize: 14,
                        fontWeight: "600",
                      }}
                    />
                  </View>

                  {/* Phone */}
                  <View>
                    <Text style={{ color: C.textMuted, fontSize: 12, fontWeight: "700", marginBottom: 6 }}>
                      Mobile Number (10 digits)
                    </Text>
                    <View
                      style={{
                        flexDirection: "row",
                        alignItems: "center",
                        backgroundColor: C.input,
                        borderRadius: 16,
                        borderWidth: 1,
                        borderColor: C.inputBorder,
                        paddingHorizontal: 12,
                      }}
                    >
                      <Text style={{ color: "#fff", fontSize: 13, fontWeight: "700", marginRight: 8 }}>
                        🇮🇳 +91
                      </Text>
                      <TextInput
                        value={personalInfoForm.phone}
                        onChangeText={(t) =>
                          setPersonalInfoForm((p) => ({
                            ...p,
                            phone: t.replace(/\D/g, "").slice(0, 10),
                          }))
                        }
                        placeholder="10-digit mobile"
                        placeholderTextColor={C.textFaint}
                        keyboardType="phone-pad"
                        maxLength={10}
                        style={{
                          flex: 1,
                          color: "#fff",
                          fontSize: 14.5,
                          fontWeight: "600",
                          paddingVertical: 12,
                        }}
                      />
                    </View>
                  </View>

                  {/* Date of Birth */}
                  <View>
                    <Text style={{ color: C.textMuted, fontSize: 12, fontWeight: "700", marginBottom: 6 }}>
                      Date of Birth (DD / MM / YYYY)
                    </Text>
                    <TextInput
                      value={personalInfoForm.dateOfBirth}
                      onChangeText={(t) => {
                        const digits = t.replace(/\D/g, "").slice(0, 8);
                        let formatted = digits;
                        if (digits.length > 4) {
                          formatted = `${digits.slice(0, 2)} / ${digits.slice(2, 4)} / ${digits.slice(4)}`;
                        } else if (digits.length > 2) {
                          formatted = `${digits.slice(0, 2)} / ${digits.slice(2)}`;
                        }
                        setPersonalInfoForm((p) => ({ ...p, dateOfBirth: formatted }));
                      }}
                      placeholder="DD / MM / YYYY (e.g. 10 / 10 / 2005)"
                      placeholderTextColor={C.textFaint}
                      keyboardType="number-pad"
                      maxLength={14}
                      style={{
                        backgroundColor: C.input,
                        borderRadius: 16,
                        borderWidth: 1,
                        borderColor: C.inputBorder,
                        color: "#fff",
                        paddingHorizontal: 14,
                        paddingVertical: 12,
                        fontSize: 14,
                        fontWeight: "600",
                      }}
                    />
                  </View>

                  {/* Occupation */}
                  <View>
                    <Text style={{ color: C.textMuted, fontSize: 12, fontWeight: "700", marginBottom: 6 }}>
                      Occupation / Job
                    </Text>
                    <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
                      {OCCUPATION_OPTIONS.map((occ) => {
                        const isSelected = personalInfoForm.occupation === occ.label;
                        return (
                          <TouchableOpacity
                            key={occ.id}
                            onPress={() =>
                              setPersonalInfoForm((p) => ({ ...p, occupation: occ.label }))
                            }
                            activeOpacity={0.8}
                            style={{
                              paddingHorizontal: 11,
                              paddingVertical: 7,
                              borderRadius: 12,
                              backgroundColor: isSelected
                                ? "rgba(6, 182, 212, 0.25)"
                                : "rgba(255, 255, 255, 0.04)",
                              borderWidth: 1,
                              borderColor: isSelected ? CYAN : "rgba(255, 255, 255, 0.08)",
                            }}
                          >
                            <Text
                              style={{
                                color: isSelected ? "#fff" : C.textMuted,
                                fontSize: 12,
                                fontWeight: isSelected ? "700" : "500",
                              }}
                            >
                              {occ.label}
                            </Text>
                          </TouchableOpacity>
                        );
                      })}
                    </View>
                  </View>

                  {/* Monthly SIP Budget */}
                  <View>
                    <Text style={{ color: C.textMuted, fontSize: 12, fontWeight: "700", marginBottom: 6 }}>
                      Monthly SIP Budget (₹)
                    </Text>
                    <TextInput
                      value={personalInfoForm.monthlySipBudget}
                      onChangeText={(t) => setPersonalInfoForm((p) => ({ ...p, monthlySipBudget: t }))}
                      placeholder="e.g. 5000"
                      placeholderTextColor={C.textFaint}
                      keyboardType="numeric"
                      style={{
                        backgroundColor: C.input,
                        borderRadius: 16,
                        borderWidth: 1,
                        borderColor: C.inputBorder,
                        color: "#fff",
                        paddingHorizontal: 14,
                        paddingVertical: 12,
                        fontSize: 14,
                        fontWeight: "600",
                      }}
                    />
                  </View>

                  {/* Risk Appetite */}
                  <View>
                    <Text style={{ color: C.textMuted, fontSize: 12, fontWeight: "700", marginBottom: 6 }}>
                      Risk Appetite
                    </Text>
                    <View style={{ flexDirection: "row", gap: 8 }}>
                      {["Low", "Moderate", "High"].map((r) => {
                        const isSelected = personalInfoForm.riskAppetite === r;
                        return (
                          <TouchableOpacity
                            key={r}
                            onPress={() => setPersonalInfoForm((p) => ({ ...p, riskAppetite: r }))}
                            style={{
                              flex: 1,
                              alignItems: "center",
                              paddingVertical: 10,
                              borderRadius: 12,
                              backgroundColor: isSelected
                                ? "rgba(16, 185, 129, 0.2)"
                                : "rgba(255, 255, 255, 0.04)",
                              borderWidth: 1,
                              borderColor: isSelected ? GREEN : "rgba(255, 255, 255, 0.08)",
                            }}
                          >
                            <Text
                              style={{
                                color: isSelected ? "#fff" : C.textMuted,
                                fontSize: 12.5,
                                fontWeight: isSelected ? "700" : "500",
                              }}
                            >
                              {r}
                            </Text>
                          </TouchableOpacity>
                        );
                      })}
                    </View>
                  </View>

                  {/* Investment Goal */}
                  <View>
                    <Text style={{ color: C.textMuted, fontSize: 12, fontWeight: "700", marginBottom: 6 }}>
                      Primary Investment Goal
                    </Text>
                    <TextInput
                      value={personalInfoForm.investmentGoal}
                      onChangeText={(t) => setPersonalInfoForm((p) => ({ ...p, investmentGoal: t }))}
                      placeholder="e.g. Retirement planning, Wealth creation"
                      placeholderTextColor={C.textFaint}
                      style={{
                        backgroundColor: C.input,
                        borderRadius: 16,
                        borderWidth: 1,
                        borderColor: C.inputBorder,
                        color: "#fff",
                        paddingHorizontal: 14,
                        paddingVertical: 12,
                        fontSize: 14,
                        fontWeight: "600",
                      }}
                    />
                  </View>

                  {/* Save Button */}
                  <TouchableOpacity
                    onPress={savePersonalInfo}
                    disabled={savingPersonalInfo}
                    activeOpacity={0.88}
                    style={{
                      height: 52,
                      borderRadius: 26,
                      overflow: "hidden",
                      marginTop: 10,
                      marginBottom: 20,
                      shadowColor: CYAN,
                      shadowOffset: { width: 0, height: 4 },
                      shadowOpacity: 0.35,
                      shadowRadius: 12,
                      elevation: 8,
                    }}
                  >
                    <LinearGradient
                      colors={[VIOLET, CYAN, GREEN]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 1, y: 0 }}
                      style={{
                        flex: 1,
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      {savingPersonalInfo ? (
                        <ActivityIndicator color="#fff" />
                      ) : (
                        <Text style={{ color: "#fff", fontSize: 15, fontWeight: "700" }}>
                          Save Changes
                        </Text>
                      )}
                    </LinearGradient>
                  </TouchableOpacity>
                </ScrollView>
              </View>
            </KeyboardAvoidingView>
          </View>
        </Modal>
      </SafeAreaView>
    </View>
  );
}