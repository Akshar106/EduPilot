"use client";

import { ArrowUp, Paperclip, X } from "lucide-react";
import { useEffect, useRef } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

export interface Attachment {
  filename: string;
  domain: string;
  /** How to show it; the file name when absent. */
  title?: string;
}

interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  busy: boolean;
  placeholder: string;
  maxLength?: number;
  attachments?: Attachment[];
  onRemoveAttachment?: (filename: string) => void;
  onAttach?: () => void;
  editing?: boolean;
  onCancelEdit?: () => void;
}

export function Composer({
  value,
  onChange,
  onSend,
  busy,
  placeholder,
  maxLength,
  attachments = [],
  onRemoveAttachment,
  onAttach,
  editing,
  onCancelEdit,
}: ComposerProps) {
  const ref = useRef<HTMLTextAreaElement>(null);

  // Grow with the text, up to a cap. CSS `field-sizing: content` does this
  // natively; measuring is only the fallback for browsers without it, since
  // a measurement taken before layout settles locks in the wrong height.
  useEffect(() => {
    const el = ref.current;
    if (!el || CSS.supports("field-sizing", "content")) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  useEffect(() => {
    if (editing) ref.current?.focus();
  }, [editing]);

  const canSend = !busy && value.trim().length > 0;

  return (
    <div className="space-y-2">
      {editing && (
        <div className="flex items-center gap-2 text-xs text-primary">
          Editing — sending replaces this message and everything after it.
          <button type="button" className="underline" onClick={onCancelEdit}>
            Cancel
          </button>
        </div>
      )}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {attachments.map((a) => (
            <span key={a.filename} className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-1 text-xs">
              {a.title ?? a.filename} <span className="text-muted-foreground">({a.domain})</span>
              <button type="button" aria-label={`Remove ${a.filename}`} onClick={() => onRemoveAttachment?.(a.filename)}>
                <X className="size-3" />
              </button>
            </span>
          ))}
        </div>
      )}
      <div className="flex items-end gap-2 rounded-2xl border bg-background p-2 shadow-sm focus-within:ring-2 focus-within:ring-ring/40">
        {onAttach && (
          <Button variant="ghost" size="icon" onClick={onAttach} aria-label="Attach a document">
            <Paperclip />
          </Button>
        )}
        <Textarea
          ref={ref}
          rows={1}
          value={value}
          maxLength={maxLength}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              if (canSend) onSend();
            }
          }}
          className="max-h-[200px] min-h-9 flex-1 resize-none border-0 bg-transparent px-1 py-1.5 shadow-none focus-visible:ring-0 dark:bg-transparent"
        />
        <Button size="icon" onClick={onSend} disabled={!canSend} aria-label="Send">
          <ArrowUp />
        </Button>
      </div>
    </div>
  );
}
