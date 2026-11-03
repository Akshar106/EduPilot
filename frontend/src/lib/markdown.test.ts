import { describe, expect, it } from "vitest";

import { citedSources, prepareMarkdown } from "./markdown";
import { progressLabel } from "./progress";

describe("prepareMarkdown", () => {
  it("links citations and converts bracket maths, leaving code alone", () => {
    const out = prepareMarkdown("Var is \\(\\sigma^2\\) [Source 2]. `\\(x\\) [Source 9]`");
    expect(out).toContain("$\\sigma^2$");
    expect(out).toContain("[Source 2](#source-2)");
    expect(out).toContain("`\\(x\\) [Source 9]`");
  });

  it("turns display brackets into a maths block", () => {
    expect(prepareMarkdown("\\[ a + b \\]")).toContain("$$\na + b\n$$");
  });

  it("lists cited sources once each, in order", () => {
    expect(citedSources("[Source 3] and [Source 1] and [Source 3]")).toEqual([1, 3]);
  });
});

describe("progressLabel", () => {
  it("describes branch progress and graph stages", () => {
    expect(progressLabel("progress", { stage: "rewriting", domain: "ALGO", query: "merge sort running time" }))
      .toBe("Rephrasing the ALGO search as “merge sort running time”…");
    expect(progressLabel("stage", { stage: "plan", sub_questions: [{}, {}] })).toBe("Splitting into 2 parts…");
    expect(progressLabel("token", {})).toBeNull();
  });
});
