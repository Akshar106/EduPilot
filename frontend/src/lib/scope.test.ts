import { describe, expect, it } from "vitest";

import { chatAbout, scopeFrom } from "./scope";

const lecture = { domain: "AML", filename: "lec6_Regularization & co (1).pdf", title: "Regularization and polynomial regression" };

describe("chat links for one document", () => {
  it("carry the document through the query string, special characters included", () => {
    const url = new URL(chatAbout(lecture), "http://localhost");
    expect(url.pathname).toBe("/");
    expect(scopeFrom(url.searchParams)).toEqual({ ...lecture, quiz: false });
  });

  it("ask for a quiz only when told to", () => {
    const url = new URL(chatAbout(lecture, { quiz: true }), "http://localhost");
    expect(scopeFrom(url.searchParams)?.quiz).toBe(true);
  });

  it("leave an ordinary visit alone", () => {
    expect(scopeFrom(new URLSearchParams("ask=Quiz%20me"))).toBeNull();
  });
});
