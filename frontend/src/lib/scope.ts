// A chat about one course document, opened from Course Materials: the link
// carries the document, and the chat attaches it so every answer in that
// conversation draws on it alone.

export interface ScopedDocument {
  domain: string;
  filename: string;
  title: string;
}

/** The chat link for one document; `quiz` asks for a practice quiz on it straight away. */
export function chatAbout(doc: ScopedDocument, { quiz = false } = {}): string {
  const params = new URLSearchParams({ domain: doc.domain, file: doc.filename, title: doc.title });
  if (quiz) params.set("quiz", "1");
  return `/?${params}`;
}

/** The document a chat link carries, or null for an ordinary visit. */
export function scopeFrom(params: Pick<URLSearchParams, "get">): (ScopedDocument & { quiz: boolean }) | null {
  const domain = params.get("domain");
  const filename = params.get("file");
  if (!domain || !filename) return null;
  return { domain, filename, title: params.get("title") || filename, quiz: params.get("quiz") === "1" };
}
