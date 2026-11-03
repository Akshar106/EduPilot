// Text preparation before markdown rendering.

const CITATION = /\[Source (\d+)\]/g;

/**
 * Normalize maths delimiters and turn citations into in-page links.
 *
 * Models write maths as \( \) and \[ \] as often as $ $ and $$ $$, but
 * remark-math only understands the dollar forms. `[Source N]` becomes a link
 * to `#source-N`, which the renderer turns into a chip that opens source N.
 * Code spans and fences are left alone.
 */
export function prepareMarkdown(text: string): string {
  return text
    .split(/(```[\s\S]*?```|`[^`\n]*`)/g)
    .map((part, i) => {
      if (i % 2 === 1) return part; // code
      return part
        .replace(/\\\[([\s\S]+?)\\\]/g, (_, m) => `\n$$\n${m.trim()}\n$$\n`)
        .replace(/\\\(([\s\S]+?)\\\)/g, (_, m) => `$${m.trim()}$`)
        .replace(CITATION, (_, n) => `[Source ${n}](#source-${n})`);
    })
    .join("");
}

export function citedSources(text: string): number[] {
  return [...new Set([...text.matchAll(CITATION)].map((m) => Number(m[1])))].sort((a, b) => a - b);
}
