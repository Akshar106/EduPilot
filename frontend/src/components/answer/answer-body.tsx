"use client";

import { Markdown } from "@/components/markdown";
import { Flashcards } from "@/components/practice/flashcards";
import { Quiz } from "@/components/practice/quiz";
import type { Practice } from "@/lib/types";

interface AnswerBodyProps {
  text: string;
  practice?: Practice | null;
  onCite?: (n: number) => void;
}

/** An answer: interactive practice when the tutor wrote one, markdown otherwise. */
export function AnswerBody({ text, practice, onCite }: AnswerBodyProps) {
  if (practice?.type === "quiz" && practice.items.length) {
    return <Quiz title={practice.title} items={practice.items} onCite={onCite} />;
  }
  if (practice?.type === "flashcards" && practice.items.length) {
    return <Flashcards title={practice.title} items={practice.items} onCite={onCite} />;
  }
  return <Markdown text={text} onCite={onCite} />;
}
