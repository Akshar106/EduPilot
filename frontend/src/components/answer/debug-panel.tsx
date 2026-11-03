"use client";

import { ChevronDown, ExternalLink, Wrench } from "lucide-react";
import { useState } from "react";

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { percent } from "@/lib/format";
import type { Diagnostics, VerifierScores } from "@/lib/types";
import { cn } from "@/lib/utils";

function Row({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-0.5">
      <div className="text-[0.7rem] font-semibold tracking-wide text-muted-foreground uppercase">{title}</div>
      <div className="text-xs leading-relaxed">{children}</div>
    </div>
  );
}

/**
 * The scores an audit measured. The claim check scores grounding only, so
 * quality and coverage are left out rather than shown as 0%.
 */
function verificationSummary(v: VerifierScores): string {
  const scores = [
    v.quality_score != null && `Quality ${percent(v.quality_score)}`,
    v.coverage_score != null && `Coverage ${percent(v.coverage_score)}`,
    v.grounding_score != null && `Grounding ${percent(v.grounding_score)}`,
  ].filter(Boolean);
  const by = v.auditor === "claim check" ? "Claim check" : "Model verifier";
  return [by, ...scores].join(" · ");
}

/** What the pipeline did for one answer: routing, retrieval, verification, cost. */
export function DebugPanel({ debug }: { debug: Diagnostics }) {
  const [open, setOpen] = useState(false);
  const verifier = debug.verifier ?? debug.verification;
  const audits = debug.audits ?? [];

  const rows: React.ReactNode[] = [];
  if (debug.router) {
    rows.push(
      <Row key="route" title="Routing">
        {debug.router.intent_type} · {(debug.router.domains ?? []).join(", ") || "no domain"}
        {debug.router.reasoning && <div className="text-muted-foreground">{debug.router.reasoning}</div>}
      </Row>,
    );
  }
  if (debug.sub_questions?.length) {
    rows.push(
      <Row key="plan" title="Decomposition">
        <ol className="list-decimal pl-4">
          {debug.sub_questions.map((sq, i) => (
            <li key={i}>
              [{sq.domain}] {sq.question}
            </li>
          ))}
        </ol>
      </Row>,
    );
  }
  if (debug.retrieval?.length) {
    rows.push(
      <Row key="retrieval" title="Retrieval, grading & rewrites">
        {debug.retrieval.map((r, i) => (
          <div key={i}>
            <b>{r.domain}</b>: {r.chunks} source{r.chunks === 1 ? "" : "s"}
            {!!r.grades?.length && ` · graded ${r.grades.map((g) => `${g.relevant}/${g.total}`).join(" → ")}`}
            {!!r.rewrites?.length && <div className="pl-3 text-muted-foreground">rephrased as “{r.rewrites.join("”, “")}”</div>}
            {!!r.calculations?.length && <div className="pl-3 text-muted-foreground">calculated {r.calculations.join(", ")}</div>}
          </div>
        ))}
      </Row>,
    );
  }
  if (verifier && !verifier.skipped) {
    rows.push(
      <Row key="verify" title="Verification">
        {verificationSummary(verifier)}
        {audits.length > 1 && (
          <div className="text-muted-foreground">
            Revised: grounding {audits.map((a) => percent(a.grounding_score)).join(" → ")}
          </div>
        )}
      </Row>,
    );
  }
  if (debug.profile?.weak_topics?.length) {
    rows.push(
      <Row key="profile" title="Learning profile">
        Topics to reinforce: {debug.profile.weak_topics.join(", ")}
      </Row>,
    );
  }
  if (debug.usage?.calls) {
    rows.push(
      <Row key="usage" title="Cost">
        {debug.usage.calls} model call{debug.usage.calls === 1 ? "" : "s"} · {debug.usage.input_tokens ?? 0} in /{" "}
        {debug.usage.output_tokens ?? 0} out tokens
        {!!debug.usage.fallbacks && ` · ${debug.usage.fallbacks} fallback${debug.usage.fallbacks === 1 ? "" : "s"}`}
        {debug.total_latency_ms != null && ` · ${(debug.total_latency_ms / 1000).toFixed(1)}s`}
      </Row>,
    );
  }
  if (debug.trace?.traced && debug.trace.run_id) {
    rows.push(
      <Row key="trace" title="Trace">
        <a
          href={`/api/trace/${encodeURIComponent(debug.trace.run_id)}`}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-primary hover:underline"
        >
          Open every step in LangSmith <ExternalLink className="size-3" />
        </a>
      </Row>,
    );
  }
  if (!rows.length) return null;

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-xl border border-dashed">
      <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground">
        <Wrench className="size-3.5" /> Pipeline details
        <ChevronDown className={cn("ml-auto size-3.5 transition-transform", open && "rotate-180")} />
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-2.5 border-t border-dashed p-3">{rows}</CollapsibleContent>
    </Collapsible>
  );
}
