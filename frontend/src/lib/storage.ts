// localStorage that tolerates being unavailable (private windows, blocked
// site data): reads fall back to null and writes are skipped.

export function load(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function save(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Not persisted; the app still works for this visit.
  }
}
