"use client";

import { GraduationCap, PanelLeft } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useEffectEvent, useRef, useState } from "react";

import { AttachDialog } from "@/components/chat/attach-dialog";
import { type Attachment, Composer } from "@/components/chat/composer";
import { AssistantMessage, type ChatItem, PendingMessage, UserMessage } from "@/components/chat/messages";
import { ChatSidebar } from "@/components/chat/sidebar";
import { useApp } from "@/components/providers";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { ApiError, api, streamPost } from "@/lib/api";
import { progressLabel } from "@/lib/progress";
import { scopeFrom } from "@/lib/scope";
import { load, save } from "@/lib/storage";
import type { ChatResult, SessionSummary, StoredMessage } from "@/lib/types";

const SESSION_KEY = "ep_session";

const SAMPLES = [
  "Explain the bias-variance tradeoff.",
  "Why is merge sort O(n log n)?",
  "How does dynamic programming solve rod cutting?",
  "How does attention work in transformers?",
  "Quiz me on greedy algorithms",
  "Make flashcards on transformers",
];

let counter = 0;
const key = () => `k${Date.now().toString(36)}-${counter++}`;

/** A stored message as a chat item. Sources are not stored; practice data is. */
function fromStored(sessionId: string, m: StoredMessage): ChatItem {
  if (m.role === "user") return { kind: "user", key: `m${m.id}`, text: m.content, msgId: m.id };
  const practice = m.pipeline_meta?.practice ?? null;
  return {
    kind: "assistant",
    key: `m${m.id}`,
    msgId: m.id,
    result: {
      session_id: sessionId,
      final_answer: m.content,
      detected_domains: m.detected_domains ?? [],
      grounding_score: m.quality_score ?? null,
      sources: [],
      practice,
      task: practice?.type ?? "answer",
      debug: m.pipeline_meta ?? undefined,
    },
  };
}

/** "A, B and C" */
function joinNames(names: string[]): string {
  return names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}` : (names[0] ?? "");
}

function Welcome({ onPick }: { onPick: (q: string) => void }) {
  const { config } = useApp();
  const courses = joinNames(Object.values(config?.domains ?? {}).map((d) => d.name));
  return (
    <div className="flex flex-col items-center gap-6 py-10 text-center">
      <div className="flex size-14 items-center justify-center rounded-2xl bg-primary text-primary-foreground">
        <GraduationCap className="size-7" />
      </div>
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold">Ask anything about your courses</h1>
        <p className="mx-auto max-w-lg text-muted-foreground">
          Answers come from the course materials{courses ? ` for ${courses}` : ""} — every claim cited, and a refusal
          rather than a guess.
        </p>
      </div>
      <div className="grid w-full max-w-2xl gap-2 sm:grid-cols-2">
        {SAMPLES.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => onPick(s)}
            className="rounded-xl border bg-card px-4 py-3 text-left text-sm transition-colors hover:border-primary/40 hover:bg-muted"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

/** The conversation this browser was last in, or a new one. */
function initialSession(): string {
  if (typeof window === "undefined") return "";
  return load(SESSION_KEY) ?? crypto.randomUUID();
}

export function ChatPage() {
  const { config, settings } = useApp();
  const colors = Object.fromEntries(Object.entries(config?.domains ?? {}).map(([k, d]) => [k, d.color]));
  // "Practice these" on the progress page links here with a prompt; "Ask" and
  // "Quiz me" on Course Materials, with one document for a new conversation.
  const params = useSearchParams();
  const [scope] = useState(() => scopeFrom(params));

  const [sessionId, setSessionId] = useState(() => (scope ? crypto.randomUUID() : initialSession()));
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [items, setItems] = useState<ChatItem[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [input, setInput] = useState(() => params.get("ask") ?? "");
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState<{ msgId: number; index: number } | null>(null);
  const [attachments, setAttachments] = useState<Attachment[]>(() =>
    scope ? [{ domain: scope.domain, filename: scope.filename, title: scope.title }] : [],
  );
  const [attachOpen, setAttachOpen] = useState(false);
  const [navOpen, setNavOpen] = useState(false);

  const scroller = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  const refreshSessions = useCallback(async () => {
    try {
      setSessions((await api.get<{ sessions: SessionSummary[] }>("/api/sessions")).sessions);
    } catch {
      // The list is a convenience; the conversation itself still works.
    }
  }, []);

  // Load the open conversation whenever it changes.
  useEffect(() => {
    if (!sessionId) return;
    save(SESSION_KEY, sessionId);
    let current = true;
    api
      .get<{ messages: StoredMessage[] }>(`/api/sessions/${sessionId}`)
      .then((data) => data.messages.map((m) => fromStored(sessionId, m)))
      .catch((err) => {
        // A session id this browser made up has no messages until its first turn.
        if (!(err instanceof ApiError && err.status === 404)) console.warn(err);
        return [] as ChatItem[];
      })
      .then((loadedItems) => {
        if (!current) return;
        stickToBottom.current = true;
        setItems(loadedItems);
        setLoaded(true);
      });
    return () => {
      current = false;
    };
  }, [sessionId]);

  useEffect(() => {
    api
      .get<{ sessions: SessionSummary[] }>("/api/sessions")
      .then((r) => setSessions(r.sessions))
      .catch(() => undefined);
  }, []);

  const openSession = (id: string) => {
    setSessionId(id);
    setEditing(null);
    setNavOpen(false);
  };

  useEffect(() => {
    const el = scroller.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [items]);

  const newSession = () => {
    openSession(crypto.randomUUID());
    setAttachments([]);
  };

  const deleteSession = async (id: string) => {
    try {
      await api.del(`/api/sessions/${id}`);
    } catch (err) {
      console.warn(err);
    }
    if (id === sessionId) newSession();
    void refreshSessions();
  };

  const send = async (text?: string) => {
    const query = (text ?? input).trim();
    if (!query || busy || !sessionId) return;
    setInput("");
    setBusy(true);
    stickToBottom.current = true;

    let base = items;
    if (editing) {
      // The server forgets the edited turn and everything after it.
      await api.del(`/api/sessions/${sessionId}/messages/${editing.msgId}`).catch(() => undefined);
      base = items.slice(0, editing.index);
      setEditing(null);
    }

    const userKey = key();
    const pendingKey = key();
    setItems([
      ...base,
      { kind: "user", key: userKey, text: query },
      { kind: "pending", key: pendingKey, text: "", status: "Thinking…" },
    ]);

    // Tokens arrive faster than the screen refreshes: render once per frame.
    let streamed = "";
    let status = "Thinking…";
    let frame = 0;
    const flush = () => {
      frame = 0;
      setItems((prev) => prev.map((it) => (it.key === pendingKey ? { ...it, text: streamed, status } : it)));
    };

    try {
      const result = await streamPost<ChatResult>(
        "/api/chat/stream",
        {
          query,
          session_id: sessionId,
          model: settings.model,
          top_k: settings.topK,
          rerank_top_k: settings.rerankTopK,
          enable_verification: settings.verify,
          manual_domains: attachments.length ? [...new Set(attachments.map((a) => a.domain))] : null,
          attached_filenames: attachments.length ? attachments.map((a) => a.filename) : null,
        },
        (ev) => {
          if (ev.event === "token") streamed += String(ev.data.text ?? "");
          else if (ev.event === "reset") streamed = "";
          else status = progressLabel(ev.event, ev.data) ?? status;
          if (!frame) frame = requestAnimationFrame(flush);
        },
      );
      cancelAnimationFrame(frame);
      setItems((prev) =>
        prev.map((it) => {
          if (it.key === userKey) return { ...it, msgId: result.user_message_id };
          if (it.key === pendingKey) return { kind: "assistant", key: pendingKey, result, msgId: result.assistant_message_id };
          return it;
        }),
      );
      void refreshSessions();
    } catch (err) {
      cancelAnimationFrame(frame);
      const message = err instanceof Error ? err.message : String(err);
      setItems((prev) =>
        prev.map((it) =>
          it.key === pendingKey
            ? { kind: "assistant", key: pendingKey, result: { session_id: sessionId, final_answer: `⚠️ ${message}`, sources: [] } }
            : it,
        ),
      );
    } finally {
      setBusy(false);
    }
  };

  // A link from Course Materials is used once: a reload must not open another
  // conversation or ask for another quiz.
  const quizPending = useRef(scope?.quiz ?? false);
  const askForQuiz = useEffectEvent((title: string) => void send(`Quiz me on ${title}`));
  useEffect(() => {
    if (scope) window.history.replaceState(null, "", "/");
  }, [scope]);
  useEffect(() => {
    if (!loaded || !quizPending.current || !scope) return;
    quizPending.current = false;
    askForQuiz(scope.title);
  }, [loaded, scope]);

  const sidebar = (
    <ChatSidebar
      sessions={sessions}
      activeId={sessionId}
      onSelect={openSession}
      onNew={newSession}
      onDelete={(id) => void deleteSession(id)}
    />
  );

  return (
    <div className="flex h-full">
      <aside className="hidden w-72 shrink-0 border-r lg:block">{sidebar}</aside>
      <Sheet open={navOpen} onOpenChange={setNavOpen}>
        <SheetContent side="left" className="w-80 p-0 pt-10">
          <SheetTitle className="sr-only">Conversations and settings</SheetTitle>
          {sidebar}
        </SheetContent>
      </Sheet>

      <section className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-2 border-b px-3 py-1.5 lg:hidden">
          <Button variant="ghost" size="icon" onClick={() => setNavOpen(true)} aria-label="Conversations and settings">
            <PanelLeft />
          </Button>
          <span className="text-sm text-muted-foreground">Conversations &amp; settings</span>
        </div>

        <div
          ref={scroller}
          onScroll={(e) => {
            const el = e.currentTarget;
            stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
          }}
          className="min-h-0 flex-1 overflow-y-auto"
        >
          <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
            {loaded && items.length === 0 && <Welcome onPick={(q) => void send(q)} />}
            {items.map((it, index) => {
              if (it.kind === "user") {
                return (
                  <UserMessage
                    key={it.key}
                    text={it.text}
                    editable={!busy && it.msgId != null}
                    onEdit={() => {
                      setEditing({ msgId: it.msgId!, index });
                      setInput(it.text);
                    }}
                  />
                );
              }
              if (it.kind === "pending") return <PendingMessage key={it.key} text={it.text} status={it.status} />;
              return (
                <AssistantMessage
                  key={it.key}
                  result={it.result}
                  msgId={it.msgId}
                  colors={colors}
                  showDebug={settings.debug}
                />
              );
            })}
          </div>
        </div>

        <div className="border-t bg-background">
          <div className="mx-auto max-w-3xl px-4 py-3">
            <Composer
              value={input}
              onChange={setInput}
              onSend={() => void send()}
              busy={busy}
              placeholder={
                attachments.length === 1
                  ? `Ask about ${attachments[0].title ?? attachments[0].filename}…`
                  : "Ask a question about your coursework…"
              }
              maxLength={config?.defaults.max_query_chars}
              attachments={attachments}
              onRemoveAttachment={(f) => setAttachments((prev) => prev.filter((a) => a.filename !== f))}
              onAttach={() => setAttachOpen(true)}
              editing={editing != null}
              onCancelEdit={() => {
                setEditing(null);
                setInput("");
              }}
            />
          </div>
        </div>
      </section>

      <AttachDialog
        open={attachOpen}
        onOpenChange={setAttachOpen}
        onAttached={(added) =>
          setAttachments((prev) => [...prev.filter((a) => !added.some((n) => n.filename === a.filename)), ...added])
        }
      />
    </div>
  );
}
