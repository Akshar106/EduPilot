// What the agent is doing, in words, from a stream event. Mirrors the
// stages emitted by agents/graph.py (`stage`) and its branches (`progress`).

type Ev = Record<string, unknown>;

const plural = (n: unknown, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

export function progressLabel(event: string, ev: Ev): string | null {
  if (event === "progress") {
    switch (ev.stage) {
      case "retrieving":
        return Number(ev.attempt) > 1
          ? `Searching ${ev.domain} materials again…`
          : `Searching ${ev.domain} materials…`;
      case "graded":
        return `${ev.relevant} of ${ev.total} ${ev.domain} excerpts address your question…`;
      case "rewriting":
        return `Rephrasing the ${ev.domain} search as “${ev.query}”…`;
      case "answering":
        return `Writing the ${ev.domain} answer from ${plural(ev.chunks, "source")}…`;
      case "calculating":
        return `Calculating ${ev.expression} = ${ev.result}…`;
      case "revising":
        return "Revising the answer to remove unsupported claims…";
      case "writing":
        return `Writing ${ev.count} ${ev.task === "flashcards" ? "flashcards" : "practice questions"}…`;
      default:
        return null;
    }
  }
  if (event !== "stage") return null;
  switch (ev.stage) {
    case "memory":
      return "Understanding your question…";
    case "route": {
      if (ev.is_course_related === false) return "Finishing up…";
      const domains = (ev.domains as string[] | undefined) ?? [];
      return `Routing to ${domains.join(", ") || "the course materials"}…`;
    }
    case "plan": {
      const parts = (ev.sub_questions as unknown[] | undefined) ?? [];
      return parts.length > 1 ? `Splitting into ${parts.length} parts…` : "Searching course materials…";
    }
    case "research":
      return `Finished ${ev.domain}…`;
    case "synthesize":
      return "Checking the answer against its sources…";
    case "verify":
      return ev.needs_revision ? "Some claims need fixing — revising…" : "Validating citations…";
    case "guardrails":
      return "Finishing up…";
    default:
      return null;
  }
}
