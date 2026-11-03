export function percent(value: number | null | undefined): string {
  return `${Math.round((value ?? 0) * 100)}%`;
}

export function fileSize(bytes: number | null | undefined): string {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return "";
  // The API sends ISO 8601 with an offset; a bare "YYYY-MM-DD HH:MM:SS" is UTC.
  const then = new Date(iso.includes("T") || iso.endsWith("Z") ? iso : `${iso.replace(" ", "T")}Z`);
  const seconds = Math.round((Date.now() - then.getTime()) / 1000);
  if (Number.isNaN(seconds)) return "";
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export function documentUrl(domain: string, filename: string): string {
  return `/api/documents/${encodeURIComponent(domain)}/${encodeURIComponent(filename)}`;
}
