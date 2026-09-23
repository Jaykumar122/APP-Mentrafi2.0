import dotenv from "dotenv";
dotenv.config();

const MENTRAFIAI_URL = process.env.MENTRAFIAI_URL || "http://localhost:8000";

export interface UserProfileContext {
  age?: number;
  name?: string;
  monthly_sip_budget?: number;
  risk_appetite?: string;
  investment_goal?: string;
  investment_horizon?: number;
  occupation?: string;
}

export interface FundCardData {
  id: string;
  schemeCode?: number;
  name: string;
  category: string;
  subcategory: string;
  rating: number;
  oneYearReturn?: number | null;
  fiveYearReturn?: number | null;
  nav?: number | null;
  monthlyAmount?: number;
  allocationPercent?: number;
  planType?: string;
}

export interface AIChatResponse {
  reply: string;
  recommendedFunds: FundCardData[];
  intent?: string;
  risk?: string;
}

/**
 * Checks whether the MentraFiAI Python service is healthy
 */
export async function isMentraFiAIOnline(): Promise<boolean> {
  try {
    const res = await fetch(`${MENTRAFIAI_URL}/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

export interface ChatHistoryItem {
  role: string;
  content: string;
}

/**
 * Non-streaming query to MentraFiAI Neural Engine
 */
export async function getAIRecommendation(
  message: string,
  profile?: UserProfileContext,
  userId?: number,
  history?: ChatHistoryItem[]
): Promise<AIChatResponse> {
  const response = await fetch(`${MENTRAFIAI_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, profile, user_id: userId, history }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`MentraFiAI server error (${response.status}): ${errorText}`);
  }

  return (await response.json()) as AIChatResponse;
}

/**
 * Streaming query to MentraFiAI Neural Engine (Server-Sent Events)
 */
export async function getAIRecommendationStream(
  message: string,
  profile: UserProfileContext | undefined,
  userId: number | undefined,
  onChunk: (chunk: string) => void,
  onFunds?: (funds: FundCardData[]) => void,
  signal?: AbortSignal,
  history?: ChatHistoryItem[]
): Promise<void> {
  const response = await fetch(`${MENTRAFIAI_URL}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, profile, user_id: userId, stream: true, history }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`MentraFiAI stream connection failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const jsonStr = trimmed.slice(5).trim();
      if (jsonStr === "[DONE]") return;

      try {
        const parsed = JSON.parse(jsonStr);
        if (parsed.recommendedFunds && onFunds) {
          onFunds(parsed.recommendedFunds);
        }
        if (typeof parsed.chunk === "string") {
          onChunk(parsed.chunk);
        }
      } catch {
        // Skip malformed chunk
      }
    }
  }
}
