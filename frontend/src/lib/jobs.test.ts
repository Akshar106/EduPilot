import { afterEach, describe, expect, it, vi } from "vitest";

import { summarize, waitForJobs } from "./jobs";
import type { Job } from "./types";

const job = (id: number, status: Job["status"], extra: Partial<Job> = {}): Job => ({
  job_id: id,
  status,
  filename: `f${id}.pdf`,
  chunks_indexed: null,
  error: null,
  ...extra,
});

afterEach(() => vi.unstubAllGlobals());

describe("waitForJobs", () => {
  it("polls until every job has finished", async () => {
    const replies: Record<number, Job[]> = {
      1: [job(1, "queued"), job(1, "running"), job(1, "done", { chunks_indexed: 12 })],
      2: [job(2, "failed", { error: "no text" })],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        const id = Number(url.split("/").pop());
        const next = replies[id].length > 1 ? replies[id].shift()! : replies[id][0];
        return new Response(JSON.stringify(next), { status: 200 });
      }),
    );

    const jobs = await waitForJobs([1, 2], 0);
    expect(jobs.map((j) => j.status)).toEqual(["done", "failed"]);
  });
});

describe("summarize", () => {
  it("counts indexed files and lists the failures", () => {
    const out = summarize([
      job(1, "done", { chunks_indexed: 10 }),
      job(2, "done", { chunks_indexed: 5 }),
      job(3, "failed", { error: "encrypted" }),
    ]);
    expect(out.ok).toBe("Indexed 2 files, 15 chunks.");
    expect(out.errors).toEqual(["f3.pdf: encrypted"]);
  });
});
