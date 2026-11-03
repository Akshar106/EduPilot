import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Flashcards } from "./flashcards";
import { Quiz } from "./quiz";

const QUESTIONS = [
  {
    question: "What does variance measure?",
    options: ["Spread around the mean", "The middle value"],
    answer_index: 0,
    explanation: "It is the average squared deviation.",
    source: 1,
  },
  {
    question: "Why square the deviations?",
    options: ["To keep them positive", "To shrink them"],
    answer_index: 0,
    explanation: "Squaring stops them cancelling.",
    source: null,
  },
];

describe("Quiz", () => {
  it("locks a question once answered and keeps score", () => {
    const onCite = vi.fn();
    render(<Quiz title="Practice quiz" items={QUESTIONS} onCite={onCite} />);
    expect(screen.getByText("0 of 2 answered")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Spread around the mean"));
    expect(screen.getByText(/Correct\./)).toBeInTheDocument();
    expect(screen.getByText("The middle value").closest("button")).toBeDisabled();

    fireEvent.click(screen.getByText("To shrink them"));
    expect(screen.getByText(/the answer is A/)).toBeInTheDocument();
    expect(screen.getByText("Score: 1 / 2")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Source 1"));
    expect(onCite).toHaveBeenCalledWith(1);

    fireEvent.click(screen.getByText("Try again"));
    expect(screen.getByText("0 of 2 answered")).toBeInTheDocument();
  });
});

describe("Flashcards", () => {
  const cards = [
    { front: "Variance", back: "Average squared deviation", source: 2 },
    { front: "Standard deviation", back: "Square root of the variance", source: null },
  ];

  it("flips a card and moves through the deck", () => {
    render(<Flashcards title="Flashcards" items={cards} />);
    expect(screen.getByText("Card 1 of 2")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Show the answer" }));
    expect(screen.getByText("Average squared deviation")).toBeInTheDocument();
    expect(screen.getByText("Source 2")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Next card" }));
    expect(screen.getByText("Card 2 of 2")).toBeInTheDocument();
    expect(screen.getByText("Standard deviation")).toBeInTheDocument(); // a new card starts face up
  });
});
