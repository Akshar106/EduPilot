"use client";

import { Calculator, ChevronDown, Download, FileText } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { documentUrl } from "@/lib/format";
import type { Source } from "@/lib/types";
import { cn } from "@/lib/utils";

interface SourcesProps {
  sources: Source[];
  colors: Record<string, string>;
  /** Source number to open and scroll to, set when a citation is clicked. */
  focus: number | null;
}

/** The excerpts behind an answer, so a student can check each citation. */
export function Sources({ sources, colors, focus }: SourcesProps) {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState<number | null>(null);
  const refs = useRef(new Map<number, HTMLDivElement>());

  // A clicked citation opens the list at that source.
  const [seenFocus, setSeenFocus] = useState(focus);
  if (focus !== seenFocus) {
    setSeenFocus(focus);
    if (focus != null) {
      setOpen(true);
      setExpanded(focus);
    }
  }

  useEffect(() => {
    if (focus == null) return;
    // After the list has opened.
    requestAnimationFrame(() =>
      refs.current.get(focus)?.scrollIntoView({ behavior: "smooth", block: "nearest" }),
    );
  }, [focus]);

  if (!sources.length) return null;

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-xl border">
      <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground">
        <FileText className="size-4" />
        <span>
          {sources.length} source{sources.length === 1 ? "" : "s"}
        </span>
        <span className="hidden text-xs sm:inline">— open to check the citations</span>
        <ChevronDown className={cn("ml-auto size-4 transition-transform", open && "rotate-180")} />
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-1.5 border-t p-2">
        {sources.map((s) => {
          const color = colors[s.domain] ?? "#888";
          const isOpen = expanded === s.source_num;
          return (
            <div
              key={`${s.domain}-${s.source_num}`}
              ref={(el) => {
                if (el) refs.current.set(s.source_num, el);
                else refs.current.delete(s.source_num);
              }}
              className={cn(
                "rounded-lg border transition-colors",
                focus === s.source_num && "border-primary/50 bg-primary/5",
              )}
            >
              <div className="flex items-center gap-2 px-2.5 py-1.5 text-sm">
                <button
                  type="button"
                  onClick={() => setExpanded(isOpen ? null : s.source_num)}
                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                  aria-expanded={isOpen}
                >
                  <span className="shrink-0 rounded bg-muted px-1.5 text-xs font-semibold">{s.source_num}</span>
                  <span
                    className="shrink-0 rounded px-1.5 text-xs font-medium"
                    style={{ background: `${color}22`, color }}
                  >
                    {s.domain}
                  </span>
                  <span className="truncate" title={s.citation_label}>
                    {s.citation_label}
                  </span>
                </button>
                {s.filename && s.source_type !== "book" ? (
                  <a
                    href={documentUrl(s.domain, s.filename)}
                    download
                    className="flex shrink-0 items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                    title={`Download ${s.filename}`}
                  >
                    <Download className="size-3.5" />
                    <span className="hidden sm:inline">PDF</span>
                  </a>
                ) : s.kind === "calculation" ? (
                  <span className="flex shrink-0 items-center gap-1 text-xs text-muted-foreground">
                    <Calculator className="size-3.5" /> calculated
                  </span>
                ) : null}
              </div>
              {isOpen && s.text && (
                <p className="border-t px-3 py-2 text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground">
                  {s.text}
                </p>
              )}
            </div>
          );
        })}
      </CollapsibleContent>
    </Collapsible>
  );
}
