import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { DebugPanel } from "./debug-panel";

afterEach(cleanup);

const open = () => fireEvent.click(screen.getByRole("button", { name: /Pipeline details/ }));

describe("the verification row", () => {
  it("shows what the claim check measured, and not 0% for what it didn't", () => {
    render(<DebugPanel debug={{ verifier: { auditor: "claim check", grounding_score: 0.83, coverage_score: null } }} />);
    open();
    expect(screen.getByText("Claim check · Grounding 83%")).toBeTruthy();
    expect(screen.queryByText(/Coverage|Quality/)).toBeNull();
  });

  it("shows every score the model verifier gave", () => {
    render(<DebugPanel debug={{ verifier: { grounding_score: 0.9, coverage_score: 0.8, quality_score: 0.7 } }} />);
    open();
    expect(screen.getByText("Model verifier · Quality 70% · Coverage 80% · Grounding 90%")).toBeTruthy();
  });
});
