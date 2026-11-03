// Uploads are indexed in the background: the upload answers 202 with a job per
// file, and these helpers wait for the jobs to finish.

import { api } from "./api";
import type { Job } from "./types";

const DONE = new Set(["done", "failed"]);

/** Poll each job until it is done or failed, then return the final states. */
export async function waitForJobs(ids: number[], intervalMs = 1000, timeoutMs = 10 * 60_000): Promise<Job[]> {
  const deadline = Date.now() + timeoutMs;
  const finished = new Map<number, Job>();
  while (finished.size < ids.length && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
    const pending = ids.filter((id) => !finished.has(id));
    const states = await Promise.all(pending.map((id) => api.get<Job>(`/api/jobs/${id}`).catch(() => null)));
    for (const job of states) if (job && DONE.has(job.status)) finished.set(job.job_id, job);
  }
  return ids.map((id) => finished.get(id)).filter((j): j is Job => !!j);
}

/** "Indexed 2 files, 31 chunks." — and a toast-ready error per failed file. */
export function summarize(jobs: Job[]): { ok: string | null; errors: string[] } {
  const done = jobs.filter((j) => j.status === "done");
  const chunks = done.reduce((n, j) => n + (j.chunks_indexed ?? 0), 0);
  return {
    ok: done.length ? `Indexed ${done.length} file${done.length === 1 ? "" : "s"}, ${chunks} chunks.` : null,
    errors: jobs.filter((j) => j.status === "failed").map((j) => `${j.filename}: ${j.error ?? "indexing failed"}`),
  };
}
