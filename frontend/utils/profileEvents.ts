/**
 * Profile Event Bus for Cross-Screen Real-Time Synchronization
 * Allows instant event propagation when user saves/updates their profile
 * in Profile or ProfileSetup screens to AI Advisor and other active screens.
 */

type ProfileUpdateListener = () => void;

const listeners: Set<ProfileUpdateListener> = new Set();

export function notifyProfileUpdated(): void {
  listeners.forEach((listener) => {
    try {
      listener();
    } catch (err) {
      console.warn("Profile update listener error:", err);
    }
  });
}

export function subscribeProfileUpdates(listener: ProfileUpdateListener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
