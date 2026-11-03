"use client";

import { Check, RotateCcw, X } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import type { QuizItem } from "@/lib/types";
import { cn } from "@/lib/utils";

const LETTERS = "ABCDEF";

interface QuizProps {
  title: string;
  items: QuizItem[];
  onCite?: (n: number) => void;
}

/** Multiple-choice practice: pick an answer, see whether it was right and why. */
export function Quiz({ title, items, onCite }: QuizProps) {
  // Chosen option per question; undefined until answered.
  const [answers, setAnswers] = useState<(number | undefined)[]>(() => items.map(() => undefined));
  const answered = answers.filter((a) => a !== undefined).length;
  const correct = answers.filter((a, i) => a === items[i].answer_index).length;
  const done = answered === items.length;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-semibold">{title}</h3>
        <span className="text-sm text-muted-foreground" aria-live="polite">
          {done ? `Score: ${correct} / ${items.length}` : `${answered} of ${items.length} answered`}
        </span>
      </div>

      <ol className="space-y-3">
        {items.map((item, qi) => {
          const chosen = answers[qi];
          const locked = chosen !== undefined;
          return (
            <li key={qi} className="rounded-xl border bg-card p-3">
              <p className="mb-2 font-medium">
                {qi + 1}. {item.question}
              </p>
              <div className="grid gap-1.5">
                {item.options.map((option, oi) => {
                  const isAnswer = oi === item.answer_index;
                  const isChosen = oi === chosen;
                  return (
                    <button
                      key={oi}
                      type="button"
                      disabled={locked}
                      onClick={() => setAnswers((prev) => prev.map((a, i) => (i === qi ? oi : a)))}
                      className={cn(
                        "flex items-start gap-2 rounded-lg border px-3 py-2 text-left text-sm transition-colors",
                        !locked && "hover:border-primary/50 hover:bg-muted",
                        locked && isAnswer && "border-emerald-500/60 bg-emerald-500/10",
                        locked && isChosen && !isAnswer && "border-destructive/60 bg-destructive/10",
                        locked && !isAnswer && !isChosen && "opacity-60",
                      )}
                    >
                      <span className="font-semibold text-muted-foreground">{LETTERS[oi]}.</span>
                      <span className="flex-1">{option}</span>
                      {locked && isAnswer && <Check className="size-4 shrink-0 text-emerald-600" aria-label="correct answer" />}
                      {locked && isChosen && !isAnswer && <X className="size-4 shrink-0 text-destructive" aria-label="your answer" />}
                    </button>
                  );
                })}
              </div>
              {locked && (
                <p className="mt-2 text-sm text-muted-foreground">
                  <span className={cn("font-medium", chosen === item.answer_index ? "text-emerald-600" : "text-destructive")}>
                    {chosen === item.answer_index ? "Correct. " : `Not quite — the answer is ${LETTERS[item.answer_index]}. `}
                  </span>
                  {item.explanation}
                  {item.source && (
                    <button
                      type="button"
                      onClick={() => onCite?.(item.source!)}
                      className="ml-1 text-xs font-semibold text-primary hover:underline"
                    >
                      Source {item.source}
                    </button>
                  )}
                </p>
              )}
            </li>
          );
        })}
      </ol>

      {answered > 0 && (
        <Button variant="outline" size="sm" onClick={() => setAnswers(items.map(() => undefined))}>
          <RotateCcw /> Try again
        </Button>
      )}
    </div>
  );
}
