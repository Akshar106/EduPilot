"use client";

import { ChevronLeft, ChevronRight, Shuffle } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import type { FlashcardItem } from "@/lib/types";
import { cn } from "@/lib/utils";

interface FlashcardsProps {
  title: string;
  items: FlashcardItem[];
  onCite?: (n: number) => void;
}

/** One card at a time: click or press Space to flip, arrow keys to move. */
export function Flashcards({ title, items, onCite }: FlashcardsProps) {
  const [order, setOrder] = useState(() => items.map((_, i) => i));
  const [position, setPosition] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const card = items[order[position]];

  const go = (delta: number) => {
    setPosition((p) => (p + delta + items.length) % items.length);
    setFlipped(false);
  };
  const shuffle = () => {
    const next = [...order];
    for (let i = next.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [next[i], next[j]] = [next[j], next[i]];
    }
    setOrder(next);
    setPosition(0);
    setFlipped(false);
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-semibold">{title}</h3>
        <span className="text-sm text-muted-foreground">
          Card {position + 1} of {items.length}
        </span>
      </div>

      <button
        type="button"
        onClick={() => setFlipped((f) => !f)}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") go(1);
          else if (e.key === "ArrowLeft") go(-1);
        }}
        aria-label={flipped ? "Show the front" : "Show the answer"}
        className={cn(
          "flex min-h-40 w-full flex-col items-center justify-center rounded-xl border p-6 text-center transition-colors",
          flipped ? "border-primary/40 bg-primary/5" : "bg-card hover:bg-muted/60",
        )}
      >
        <span className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">
          {flipped ? "Answer" : "Term"}
        </span>
        <span className={cn("text-balance", flipped ? "text-sm leading-relaxed" : "text-lg font-semibold")}>
          {flipped ? card.back : card.front}
        </span>
        {!flipped && <span className="mt-3 text-xs text-muted-foreground">Click or press Space to flip</span>}
      </button>

      <div className="flex flex-wrap items-center gap-2">
        <Button variant="outline" size="sm" onClick={() => go(-1)} aria-label="Previous card">
          <ChevronLeft />
        </Button>
        <Button variant="outline" size="sm" onClick={() => go(1)} aria-label="Next card">
          <ChevronRight />
        </Button>
        <Button variant="ghost" size="sm" onClick={shuffle}>
          <Shuffle /> Shuffle
        </Button>
        {flipped && card.source && (
          <button
            type="button"
            onClick={() => onCite?.(card.source!)}
            className="ml-auto text-xs font-semibold text-primary hover:underline"
          >
            Source {card.source}
          </button>
        )}
      </div>
    </div>
  );
}
