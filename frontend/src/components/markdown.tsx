"use client";

import { memo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import { prepareMarkdown } from "@/lib/markdown";
import { cn } from "@/lib/utils";

/**
 * Answers are model output grounded in uploaded documents, so they are
 * untrusted: raw HTML is parsed (quizzes fold answers into <details>) and
 * then sanitized, before maths and code highlighting add their own markup.
 */
const schema = {
  ...defaultSchema,
  tagNames: [...(defaultSchema.tagNames ?? []), "details", "summary"],
  attributes: {
    ...defaultSchema.attributes,
    code: [["className", /^language-./, "math-inline", "math-display"]],
  },
};

interface MarkdownProps {
  text: string;
  className?: string;
  /** Called with N when a `[Source N]` chip is clicked. */
  onCite?: (n: number) => void;
}

function MarkdownImpl({ text, className, onCite }: MarkdownProps) {
  const components: Components = {
    a({ href, children }) {
      const cite = href?.match(/^#source-(\d+)$/);
      if (cite) {
        const n = Number(cite[1]);
        return (
          <button
            type="button"
            onClick={() => onCite?.(n)}
            className="mx-0.5 inline-flex h-5 items-center rounded-md bg-primary/10 px-1.5 align-baseline text-[0.7rem] font-semibold text-primary hover:bg-primary/20"
            title={`Show source ${n}`}
          >
            {n}
          </button>
        );
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer">
          {children}
        </a>
      );
    },
    table({ children }) {
      return (
        <div className="my-3 overflow-x-auto">
          <table>{children}</table>
        </div>
      );
    },
  };

  return (
    <div className={cn("prose-chat", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, [rehypeSanitize, schema], rehypeKatex, rehypeHighlight]}
        components={components}
      >
        {prepareMarkdown(text)}
      </ReactMarkdown>
    </div>
  );
}

export const Markdown = memo(MarkdownImpl);
