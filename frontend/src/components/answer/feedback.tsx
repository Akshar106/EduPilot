"use client";

import { ThumbsDown, ThumbsUp } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

interface FeedbackProps {
  kind: "chat" | "self_study";
  messageId?: number;
}

/** 👍/👎 on an answer. Rating again replaces the earlier vote. */
export function Feedback({ kind, messageId }: FeedbackProps) {
  const [rating, setRating] = useState<1 | -1 | null>(null);
  if (!messageId) return null;

  const rate = async (value: 1 | -1) => {
    const previous = rating;
    setRating(value);
    try {
      await api.post("/api/feedback", { kind, message_id: messageId, rating: value });
    } catch (err) {
      setRating(previous);
      toast.error(`Could not save your rating: ${(err as Error).message}`);
    }
  };

  return (
    <div className="flex items-center gap-0.5">
      {([1, -1] as const).map((value) => {
        const Icon = value === 1 ? ThumbsUp : ThumbsDown;
        return (
          <Button
            key={value}
            variant="ghost"
            size="icon-xs"
            aria-label={value === 1 ? "Helpful" : "Not helpful"}
            aria-pressed={rating === value}
            onClick={() => rate(value)}
            className={cn("text-muted-foreground", rating === value && "bg-muted text-foreground")}
          >
            <Icon className={cn(rating === value && "fill-current")} />
          </Button>
        );
      })}
    </div>
  );
}
