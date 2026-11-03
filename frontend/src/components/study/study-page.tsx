"use client";

import { BookOpen, FileText, Loader2, PanelLeft, Plus, Trash2, Upload } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";

import { AnswerBody } from "@/components/answer/answer-body";
import { Feedback } from "@/components/answer/feedback";
import { Composer } from "@/components/chat/composer";
import { PendingMessage, UserMessage } from "@/components/chat/messages";
import { useApp } from "@/components/providers";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { Switch } from "@/components/ui/switch";
import { api } from "@/lib/api";
import { fileSize, relativeTime } from "@/lib/format";
import { summarize, waitForJobs } from "@/lib/jobs";
import { load, save } from "@/lib/storage";
import type { Diagnostics, Practice, QueuedUpload, StoredMessage, StudyDocument, StudyResult, StudySessionSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

const ACTIVE_KEY = "ep_ss_session";
const ACCEPTED = /\.(pdf|txt|md|docx)$/i;

type StudyItem =
  | { kind: "user"; key: string; text: string }
  | { kind: "assistant"; key: string; result: StudyResult; msgId?: number }
  | { kind: "pending"; key: string }
  | { kind: "clarify"; key: string; query: string };

let counter = 0;
const key = () => `s${Date.now().toString(36)}-${counter++}`;

/** A broad question over several documents: worth asking which ones to search. */
function isGeneralQuery(query: string, docs: StudyDocument[]): boolean {
  const q = query.toLowerCase();
  if (docs.some((d) => q.includes(d.filename.toLowerCase().replace(/\.[^.]+$/, "")))) return false;
  const broad = ["all", "any", "every", "across", "materials", "documents", "files", "uploaded", "everything", "overall", "summary", "overview", "important", "key"];
  return broad.some((w) => q.includes(w)) || query.split(/\s+/).length < 8;
}

function fromStored(m: StoredMessage): StudyItem {
  if (m.role === "user") return { kind: "user", key: `m${m.id}`, text: m.content };
  const meta = (m.pipeline_meta ?? {}) as Diagnostics;
  const practice = (meta.practice ?? null) as Practice | null;
  return {
    kind: "assistant",
    key: `m${m.id}`,
    msgId: m.id,
    result: { final_answer: m.content, grounding_score: m.quality_score ?? null, sources: [], practice, task: practice?.type },
  };
}

function CreateDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated: (id: string) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);

  const create = async () => {
    if (!name.trim()) return;
    setBusy(true);
    try {
      const res = await api.post<{ ss_session_id: string }>("/api/self-study/sessions", {
        name: name.trim(),
        description: description.trim() || null,
      });
      setName("");
      setDescription("");
      onOpenChange(false);
      onCreated(res.ss_session_id);
    } catch (err) {
      toast.error((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New study session</DialogTitle>
          <DialogDescription>Upload your own documents to it and chat with them privately.</DialogDescription>
        </DialogHeader>
        <form
          className="space-y-3"
          onSubmit={(e) => {
            e.preventDefault();
            void create();
          }}
        >
          <div className="space-y-1.5">
            <Label htmlFor="ss-name">Name</Label>
            <Input id="ss-name" value={name} maxLength={60} placeholder="e.g. ML revision" onChange={(e) => setName(e.target.value)} autoFocus />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="ss-desc">Description (optional)</Label>
            <Input id="ss-desc" value={description} maxLength={120} placeholder="What are you studying?" onChange={(e) => setDescription(e.target.value)} />
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={busy || !name.trim()}>
              {busy && <Loader2 className="animate-spin" />} Create session
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function DocumentStatus({ doc }: { doc: StudyDocument }) {
  if (doc.status === "queued" || doc.status === "indexing") {
    return (
      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        <Loader2 className="size-3 animate-spin" /> {doc.status === "queued" ? "Waiting to be indexed…" : "Indexing…"}
      </div>
    );
  }
  if (doc.status === "failed") {
    return <div className="text-xs text-destructive">{doc.error || "Could not be indexed"}</div>;
  }
  return (
    <div className={cn("text-xs", doc.chunk_count ? "text-muted-foreground" : "text-amber-600")}>
      {doc.chunk_count
        ? `${doc.chunk_count} chunks${doc.file_size_bytes ? ` · ${fileSize(doc.file_size_bytes)}` : ""}`
        : "No text extracted — the file may be scanned or encrypted"}
    </div>
  );
}

export function StudyPage() {
  const { settings } = useApp();
  const [sessions, setSessions] = useState<StudySessionSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(() => (typeof window === "undefined" ? null : load(ACTIVE_KEY)));
  const [docs, setDocs] = useState<StudyDocument[]>([]);
  const [items, setItems] = useState<StudyItem[]>([]);
  const [filter, setFilter] = useState<string[] | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [verify, setVerify] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [navOpen, setNavOpen] = useState(false);

  const active = sessions.find((s) => s.ss_session_id === activeId);
  const ready = docs.filter((d) => d.chunk_count > 0);
  const selected = filter ?? ready.map((d) => d.filename);

  const fetchSessions = useCallback(
    () => api.get<{ sessions: StudySessionSummary[] }>("/api/self-study/sessions").then((r) => r.sessions),
    [],
  );
  const fetchSession = useCallback(
    (id: string) =>
      api.get<{ documents: StudyDocument[]; messages: StoredMessage[] }>(`/api/self-study/sessions/${id}`),
    [],
  );

  useEffect(() => {
    fetchSessions()
      .then(setSessions)
      .catch(() => undefined);
  }, [fetchSessions]);

  useEffect(() => {
    if (!activeId) return;
    save(ACTIVE_KEY, activeId);
    let current = true;
    fetchSession(activeId)
      .then((data) => {
        if (!current) return;
        setDocs(data.documents);
        setItems(data.messages.map(fromStored));
        setFilter(null);
      })
      .catch(() => {
        if (current) setActiveId(null);
      });
    return () => {
      current = false;
    };
  }, [activeId, fetchSession]);

  const reload = async () => {
    setSessions(await fetchSessions());
    if (activeId) setDocs((await fetchSession(activeId)).documents);
  };

  // Files still being indexed (say, after a page reload mid-upload): refresh
  // until they settle.
  const indexing = docs.some((d) => d.status === "queued" || d.status === "indexing");
  useEffect(() => {
    if (!indexing || !activeId || uploading) return;
    const timer = setInterval(() => {
      fetchSession(activeId)
        .then((data) => setDocs(data.documents))
        .catch(() => undefined);
    }, 2000);
    return () => clearInterval(timer);
  }, [indexing, activeId, uploading, fetchSession]);

  const upload = async (files: File[]) => {
    const valid = files.filter((f) => ACCEPTED.test(f.name));
    if (!activeId || !valid.length) {
      if (files.length) toast.error("Upload PDF, TXT, MD or DOCX files.");
      return;
    }
    setUploading(true);
    const form = new FormData();
    for (const f of valid) form.append("files", f);
    try {
      const res = await api.upload<{ queued: QueuedUpload[] }>(`/api/self-study/sessions/${activeId}/upload`, form);
      // The files are listed (as queued) straight away; the worker indexes them.
      await reload();
      const { ok, errors } = summarize(await waitForJobs(res.queued.map((q) => q.job_id)));
      for (const e of errors) toast.error(e);
      if (ok) toast.success(ok);
      await reload();
    } catch (err) {
      toast.error((err as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const removeDoc = async (doc: StudyDocument) => {
    if (!activeId) return;
    try {
      await api.del(`/api/self-study/sessions/${activeId}/documents/${doc.id}`);
      await reload();
    } catch (err) {
      toast.error((err as Error).message);
    }
  };

  const deleteSession = async () => {
    if (!activeId) return;
    try {
      await api.del(`/api/self-study/sessions/${activeId}`);
    } catch (err) {
      toast.error((err as Error).message);
    }
    setActiveId(null);
    setItems([]);
    setDocs([]);
    setSessions(await fetchSessions().catch(() => []));
  };

  const ask = async (query: string, sourceFilter: string[] | null) => {
    if (!activeId) return;
    setBusy(true);
    const pendingKey = key();
    setItems((prev) => [...prev.filter((it) => it.kind !== "clarify"), { kind: "pending", key: pendingKey }]);
    const history = items
      .filter((it): it is Extract<StudyItem, { kind: "user" | "assistant" }> => it.kind === "user" || it.kind === "assistant")
      .slice(-10)
      .map((it) => ({ role: it.kind, content: it.kind === "user" ? it.text : it.result.final_answer }));
    try {
      const result = await api.post<StudyResult>("/api/self-study/chat", {
        query,
        ss_session_id: activeId,
        model: settings.model,
        top_k: settings.topK,
        rerank_top_k: settings.rerankTopK,
        enable_verification: verify,
        chat_history: history,
        source_filter: sourceFilter,
      });
      setItems((prev) =>
        prev.map((it) => (it.key === pendingKey ? { kind: "assistant", key: pendingKey, result, msgId: result.assistant_message_id } : it)),
      );
    } catch (err) {
      setItems((prev) =>
        prev.map((it) =>
          it.key === pendingKey
            ? { kind: "assistant", key: pendingKey, result: { final_answer: `⚠️ ${(err as Error).message}`, sources: [] } }
            : it,
        ),
      );
    } finally {
      setBusy(false);
    }
  };

  const send = () => {
    const query = input.trim();
    if (!query || busy || !activeId) return;
    setInput("");
    const allSelected = selected.length === ready.length;
    setItems((prev) => [...prev, { kind: "user", key: key(), text: query }]);
    if (ready.length >= 2 && allSelected && isGeneralQuery(query, ready)) {
      setItems((prev) => [...prev, { kind: "clarify", key: key(), query }]);
      return;
    }
    void ask(query, allSelected ? null : selected);
  };

  const sessionList = (
    <div className="flex h-full flex-col gap-2 p-3">
      <Button onClick={() => setCreateOpen(true)}>
        <Plus /> New session
      </Button>
      <div className="min-h-0 flex-1 space-y-1 overflow-y-auto">
        {sessions.length === 0 && <p className="px-1 text-sm text-muted-foreground">No sessions yet.</p>}
        {sessions.map((s) => (
          <button
            key={s.ss_session_id}
            type="button"
            onClick={() => {
              setActiveId(s.ss_session_id);
              setNavOpen(false);
            }}
            className={cn(
              "w-full rounded-lg border px-3 py-2 text-left transition-colors hover:bg-muted",
              s.ss_session_id === activeId && "border-primary/40 bg-muted",
            )}
          >
            <div className="truncate text-sm font-medium">{s.name}</div>
            <div className="text-xs text-muted-foreground">
              {s.doc_count} file{s.doc_count === 1 ? "" : "s"} · {s.total_chunks.toLocaleString()} chunks · {relativeTime(s.updated_at)}
            </div>
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="flex h-full">
      <aside className="hidden w-64 shrink-0 border-r lg:block">{sessionList}</aside>
      <Sheet open={navOpen} onOpenChange={setNavOpen}>
        <SheetContent side="left" className="w-80 p-0 pt-10">
          <SheetTitle className="sr-only">Study sessions</SheetTitle>
          {sessionList}
        </SheetContent>
      </Sheet>
      <CreateDialog open={createOpen} onOpenChange={setCreateOpen} onCreated={(id) => void fetchSessions().then((s) => { setSessions(s); setActiveId(id); })} />

      <section className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-2 border-b px-3 py-2">
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setNavOpen(true)} aria-label="Study sessions">
            <PanelLeft />
          </Button>
          {active ? (
            <>
              <div className="min-w-0 flex-1">
                <div className="truncate font-medium">{active.name}</div>
                <div className="truncate text-xs text-muted-foreground">
                  {docs.length} file{docs.length === 1 ? "" : "s"} · {ready.reduce((n, d) => n + d.chunk_count, 0).toLocaleString()} chunks
                  {active.description ? ` · ${active.description}` : ""}
                </div>
              </div>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="ghost" size="sm" className="text-destructive">
                    <Trash2 /> Delete
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Delete “{active.name}”?</AlertDialogTitle>
                    <AlertDialogDescription>Its documents, their index and the conversation are removed permanently.</AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction variant="destructive" onClick={() => void deleteSession()}>
                      Delete
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </>
          ) : (
            <div className="flex-1 text-sm text-muted-foreground">Self Study</div>
          )}
        </div>

        {!active ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-4 p-6 text-center">
            <BookOpen className="size-10 text-primary" />
            <div className="space-y-1">
              <h1 className="text-xl font-semibold">Study your own material</h1>
              <p className="max-w-md text-muted-foreground">
                Create a session, upload any PDFs or notes, and ask questions — answered only from your files, kept
                separate from the course knowledge base.
              </p>
            </div>
            <Button onClick={() => setCreateOpen(true)}>
              <Plus /> Create a study session
            </Button>
          </div>
        ) : (
          <div className="grid min-h-0 flex-1 lg:grid-cols-[300px_1fr]">
            {/* Documents */}
            <div className="flex max-h-72 min-h-0 flex-col gap-2 overflow-y-auto border-b p-3 lg:max-h-none lg:border-r lg:border-b-0">
              <label
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  void upload(Array.from(e.dataTransfer.files));
                }}
                className={cn(
                  "flex cursor-pointer flex-col items-center gap-1 rounded-xl border border-dashed p-4 text-center text-sm text-muted-foreground transition-colors hover:bg-muted/50",
                  dragging && "border-primary bg-primary/5",
                )}
              >
                {uploading ? <Loader2 className="size-5 animate-spin" /> : <Upload className="size-5" />}
                {uploading ? "Indexing…" : "Drop files or click to upload"}
                <span className="text-xs">PDF · TXT · MD · DOCX</span>
                <input
                  type="file"
                  multiple
                  accept=".pdf,.txt,.md,.docx"
                  className="sr-only"
                  disabled={uploading}
                  onChange={(e) => {
                    void upload(Array.from(e.target.files ?? []));
                    e.target.value = "";
                  }}
                />
              </label>
              {docs.map((d) => (
                <div key={d.id} className="rounded-lg border p-2 text-sm">
                  <div className="flex items-start gap-2">
                    <FileText className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate" title={d.filename}>
                        {d.filename}
                      </div>
                      <DocumentStatus doc={d} />
                    </div>
                    <Button variant="ghost" size="icon-xs" aria-label={`Remove ${d.filename}`} onClick={() => void removeDoc(d)}>
                      <Trash2 />
                    </Button>
                  </div>
                </div>
              ))}
            </div>

            {/* Chat */}
            <div className="flex min-h-0 flex-col">
              <div className="min-h-0 flex-1 overflow-y-auto">
                <div className="mx-auto max-w-3xl space-y-5 px-4 py-5">
                  {items.length === 0 && (
                    <p className="py-10 text-center text-muted-foreground">
                      {ready.length ? "Ask anything about your documents." : "Upload a document to start."}
                    </p>
                  )}
                  {items.map((it) => {
                    if (it.kind === "user") return <UserMessage key={it.key} text={it.text} editable={false} />;
                    if (it.kind === "pending") return <PendingMessage key={it.key} text="" status="Reading your documents…" />;
                    if (it.kind === "clarify") {
                      return (
                        <div key={it.key} className="space-y-2 rounded-xl border bg-muted/40 p-3 text-sm">
                          <p>Which documents should I search?</p>
                          <div className="flex flex-wrap gap-1.5">
                            {ready.map((d) => (
                              <Button key={d.id} variant="outline" size="sm" onClick={() => void ask(it.query, [d.filename])}>
                                {d.filename}
                              </Button>
                            ))}
                            <Button size="sm" onClick={() => void ask(it.query, null)}>
                              All documents
                            </Button>
                          </div>
                        </div>
                      );
                    }
                    const score = it.result.grounding_score;
                    return (
                      <div key={it.key} className="space-y-2">
                        {score != null && (
                          <span className="inline-flex rounded-md bg-muted px-1.5 py-0.5 text-xs font-medium">
                            Grounded {Math.round(score * 100)}%
                          </span>
                        )}
                        <AnswerBody text={it.result.final_answer} practice={it.result.practice} />
                        {it.result.sources.length > 0 && (
                          <div className="flex flex-wrap gap-1.5">
                            {it.result.sources.map((s, i) => (
                              <span key={i} className="rounded-md bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                                {s.citation_label}
                              </span>
                            ))}
                          </div>
                        )}
                        <Feedback kind="self_study" messageId={it.msgId} />
                      </div>
                    );
                  })}
                </div>
              </div>
              <div className="space-y-2 border-t px-4 py-3">
                <div className="mx-auto max-w-3xl space-y-2">
                  {ready.length >= 2 && (
                    <div className="flex flex-wrap items-center gap-1.5 text-xs">
                      <span className="text-muted-foreground">Search:</span>
                      {ready.map((d) => {
                        const on = selected.includes(d.filename);
                        return (
                          <button
                            key={d.id}
                            type="button"
                            onClick={() => {
                              const next = on ? selected.filter((f) => f !== d.filename) : [...selected, d.filename];
                              setFilter(next.length ? next : null);
                            }}
                            className={cn("rounded-full border px-2 py-0.5", on ? "border-primary/50 bg-primary/10 text-foreground" : "text-muted-foreground")}
                          >
                            {d.filename}
                          </button>
                        );
                      })}
                    </div>
                  )}
                  <Composer
                    value={input}
                    onChange={setInput}
                    onSend={send}
                    busy={busy || !ready.length}
                    placeholder={ready.length ? "Ask anything about your documents…" : "Upload a document first"}
                  />
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Switch checked={verify} onCheckedChange={setVerify} size="sm" /> Verify answers
                  </label>
                </div>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
