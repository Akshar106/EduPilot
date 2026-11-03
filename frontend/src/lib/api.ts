// Thin client for the FastAPI backend. Same origin in production (FastAPI
// serves this app) and in development (`next dev` proxies /api).

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
    readonly requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Turn a failed response into an ApiError, reading the backend's error envelope. */
async function toError(response: Response): Promise<ApiError> {
  let message = response.statusText || `Request failed (${response.status})`;
  let code: string | undefined;
  let requestId: string | undefined;
  try {
    const body = await response.json();
    const err = body?.error ?? body;
    if (typeof err?.message === "string") message = err.message;
    else if (typeof body?.detail === "string") message = body.detail;
    code = err?.code;
    requestId = err?.request_id;
  } catch {
    // Not JSON — keep the status text.
  }
  return new ApiError(message, response.status, code, requestId);
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, { credentials: "same-origin", ...init });
  if (!response.ok) throw await toError(response);
  return (await response.json()) as T;
}

export const api = {
  get: <T>(path: string) => apiFetch<T>(path),
  post: <T>(path: string, body?: unknown) =>
    apiFetch<T>(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  del: <T>(path: string) => apiFetch<T>(path, { method: "DELETE" }),
  upload: <T>(path: string, form: FormData) => apiFetch<T>(path, { method: "POST", body: form }),
};

export interface StreamEvent {
  event: string;
  data: Record<string, unknown>;
}

/**
 * Split a server-sent-event byte stream into events.
 *
 * Frames end with a blank line; a frame's `data:` lines are joined. Frames
 * can arrive split across chunks, so text is buffered until a frame is whole.
 */
export async function* readEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<StreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let end: number;
      while ((end = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        const parsed = parseFrame(frame);
        if (parsed) yield parsed;
      }
    }
    const last = parseFrame(buffer);
    if (last) yield last;
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(frame: string): StreamEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return null;
  return { event, data: JSON.parse(data.join("\n")) };
}

/**
 * POST to a streaming endpoint. `onEvent` receives every event except the
 * final `result`, which the promise resolves with; an `error` event rejects.
 */
export async function streamPost<T>(
  path: string,
  body: unknown,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await toError(response);
  if (!response.body) throw new ApiError("The server sent no stream.", response.status);

  for await (const ev of readEvents(response.body)) {
    if (ev.event === "result") return ev.data as T;
    if (ev.event === "error") {
      const d = ev.data as { message?: string; code?: string; request_id?: string };
      throw new ApiError(d.message ?? "The request failed.", 500, d.code, d.request_id);
    }
    onEvent(ev);
  }
  throw new ApiError("The connection closed before the answer finished.", 0);
}
