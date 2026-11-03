"use client";

import { GraduationCap, Loader2, Pencil } from "lucide-react";
import { useState } from "react";

import { AnswerBody } from "@/components/answer/answer-body";
import { DebugPanel } from "@/components/answer/debug-panel";
import { Feedback } from "@/components/answer/feedback";
import { Sources } from "@/components/answer/sources";
import { Markdown } from "@/components/markdown";
import { Button } from "@/components/ui/button";
import type { ChatResult } from "@/lib/types";
import { cn } from "@/lib/utils";

export type ChatItem =
  | { kind: "user"; key: string; text: string; msgId?: number }
  | { kind: "assistant"; key: string; result: ChatResult; msgId?: number }
  | { kind: "pending"; key: string; text: string; status: string };

function Avatar() {
  return (
    <div className="flex size-8 shrink-0 items-center justify-center rounded-full border bg-muted">
      <GraduationCap className="size-4 text-primary" />
    </div>
  );
}

export function UserMessage({
  text,
  onEdit,
  editable,
}: {
  text: string;
  onEdit?: () => void;
  editable: boolean;
}) {
  return (
    <div className="group flex justify-end gap-2">
      {editable && onEdit && (
        <Button
          variant="ghost"
          size="icon-xs"
          onClick={onEdit}
          aria-label="Edit and resend"
          className="self-center opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
        >
          <Pencil />
        </Button>
      )}
      <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 whitespace-pre-wrap text-primary-foreground">
        {text}
      </div>
    </div>
  );
}

function Chip({ children, className, style }: { children: React.ReactNode; className?: string; style?: React.CSSProperties }) {
  return (
    <span className={cn("inline-flex items-center rounded-md px-1.5 py-0.5 text-xs font-medium", className)} style={style}>
      {children}
    </span>
  );
}

export function AssistantMessage({
  result,
  msgId,
  colors,
  showDebug,
}: {
  result: ChatResult;
  msgId?: number;
  colors: Record<string, string>;
  showDebug: boolean;
}) {
  const [focus, setFocus] = useState<number | null>(null);
  // Re-trigger the scroll when the same citation is clicked twice.
  const cite = (n: number) => {
    setFocus(null);
    requestAnimationFrame(() => setFocus(n));
  };
  const score = result.grounding_score;
  const pct = score == null ? null : Math.round(score * 100);

  return (
    <div className="flex gap-3">
      <Avatar />
      <div className="min-w-0 flex-1 space-y-2">
        <div className="flex flex-wrap items-center gap-1.5">
          {(result.detected_domains ?? []).map((d) => (
            <Chip key={d} style={{ background: colors[d] ?? "#888", color: "white" }}>
              {d}
            </Chip>
          ))}
          {pct != null && (
            <Chip
              className={cn(
                pct >= 70 ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400" : pct >= 40 ? "bg-amber-500/15 text-amber-700 dark:text-amber-400" : "bg-destructive/15 text-destructive",
              )}
            >
              Grounded {pct}%
            </Chip>
          )}
          {result.verification_revised && <Chip className="bg-muted">Revised after verification</Chip>}
          {result.task === "quiz" && <Chip className="bg-muted">Practice quiz</Chip>}
          {result.task === "flashcards" && <Chip className="bg-muted">Flashcards</Chip>}
          {result.debug?.cache?.hit && <Chip className="bg-muted">Cached answer</Chip>}
        </div>

        <AnswerBody text={result.final_answer} practice={result.practice} onCite={cite} />

        {result.awaiting_clarification && (
          <p className="text-xs text-muted-foreground">Reply below and I’ll continue from here.</p>
        )}

        <Sources sources={result.sources ?? []} colors={colors} focus={focus} />

        <div className="flex items-center gap-2">
          <Feedback kind="chat" messageId={msgId} />
        </div>

        {showDebug && result.debug && <DebugPanel debug={result.debug} />}
      </div>
    </div>
  );
}

export function PendingMessage({ text, status }: { text: string; status: string }) {
  return (
    <div className="flex gap-3">
      <Avatar />
      <div className="min-w-0 flex-1 space-y-2">
        {text ? (
          <div className="streaming-caret">
            <Markdown text={text} />
          </div>
        ) : null}
        <div className="flex items-center gap-2 text-sm text-muted-foreground" aria-live="polite">
          <Loader2 className="size-4 animate-spin" />
          {status}
        </div>
      </div>
    </div>
  );
}
