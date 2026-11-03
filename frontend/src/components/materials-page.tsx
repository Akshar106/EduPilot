"use client";

import { BookOpen, Eye, ListChecks, MessageSquare } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { documentUrl } from "@/lib/format";
import { chatAbout } from "@/lib/scope";
import { load, save } from "@/lib/storage";
import { cn } from "@/lib/utils";

interface Material {
  filename: string;
  title: string;
  /** The course's own number for a lecture ("6", "1-2"); null when it has none. */
  number?: string | null;
}

interface CourseMaterials {
  name: string;
  color: string;
  lectures: Material[];
  books: Material[];
}

const COURSE_KEY = "edupilot.materials.course";

export function MaterialsPage() {
  const [courses, setCourses] = useState<Record<string, CourseMaterials> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(() => load(COURSE_KEY));

  useEffect(() => {
    api
      .get<Record<string, CourseMaterials>>("/api/kb/materials")
      .then(setCourses)
      .catch((err: Error) => setError(err.message));
  }, []);

  const codes = Object.keys(courses ?? {});
  const code = selected && courses?.[selected] ? selected : codes[0];
  const course = code ? courses?.[code] : undefined;

  const choose = (next: string) => {
    setSelected(next);
    save(COURSE_KEY, next);
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
        <div>
          <h1 className="text-2xl font-semibold">Course materials</h1>
          <p className="text-muted-foreground">What the tutor answers from, lecture by lecture.</p>
        </div>

        {error && <p className="text-sm text-destructive">Could not load the course materials: {error}</p>}
        {!courses && !error && <p className="text-sm text-muted-foreground">Loading…</p>}

        {codes.length > 1 && (
          <div className="flex flex-wrap gap-2" role="group" aria-label="Course">
            {codes.map((c) => (
              <button
                key={c}
                type="button"
                aria-pressed={c === code}
                onClick={() => choose(c)}
                className={cn(
                  "flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm transition-colors",
                  c === code ? "border-primary/50 bg-primary/5" : "hover:bg-muted",
                )}
              >
                <span
                  className="rounded px-1.5 text-xs font-semibold text-white"
                  style={{ background: courses![c].color }}
                >
                  {c}
                </span>
                <span className="hidden sm:inline">{courses![c].name}</span>
              </button>
            ))}
          </div>
        )}

        {course && code && (
          <>
            <section className="space-y-2">
              <h2 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                Lectures · {course.lectures.length}
              </h2>
              {course.lectures.length ? (
                <ul className="divide-y rounded-lg border">
                  {course.lectures.map((doc) => (
                    <li key={doc.filename} className="flex items-center gap-3 px-3 py-2 text-sm">
                      <span className="w-8 shrink-0 text-xs text-muted-foreground tabular-nums">{doc.number}</span>
                      <span className="min-w-0 flex-1 truncate" title={doc.title}>
                        {doc.title}
                      </span>
                      <div className="flex shrink-0 gap-0.5">
                        <Button asChild variant="ghost" size="sm">
                          <a href={documentUrl(code, doc.filename)} target="_blank" rel="noopener noreferrer" aria-label={`Open ${doc.title}`}>
                            <Eye /> <span className="hidden sm:inline">Open</span>
                          </a>
                        </Button>
                        <Button asChild variant="ghost" size="sm">
                          <Link href={chatAbout({ domain: code, ...doc })} aria-label={`Ask about ${doc.title}`}>
                            <MessageSquare /> <span className="hidden sm:inline">Ask</span>
                          </Link>
                        </Button>
                        <Button asChild variant="ghost" size="sm">
                          <Link href={chatAbout({ domain: code, ...doc }, { quiz: true })} aria-label={`Quiz me on ${doc.title}`}>
                            <ListChecks /> <span className="hidden sm:inline">Quiz me</span>
                          </Link>
                        </Button>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">No lectures for this course yet.</p>
              )}
            </section>

            {course.books.length > 0 && (
              <section className="space-y-2">
                <h2 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                  Textbooks · used for depth, cited by page
                </h2>
                <ul className="divide-y rounded-lg border">
                  {course.books.map((doc) => (
                    <li key={doc.filename} className="flex items-center gap-3 px-3 py-2 text-sm">
                      <BookOpen className="size-4 shrink-0 text-muted-foreground" />
                      <span className="min-w-0 flex-1 truncate" title={doc.title}>
                        {doc.title}
                      </span>
                      <Button asChild variant="ghost" size="sm">
                        <Link href={chatAbout({ domain: code, ...doc })} aria-label={`Ask about ${doc.title}`}>
                          <MessageSquare /> <span className="hidden sm:inline">Ask</span>
                        </Link>
                      </Button>
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </>
        )}
      </div>
    </div>
  );
}
