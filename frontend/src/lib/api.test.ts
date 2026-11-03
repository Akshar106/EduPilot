import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, apiFetch, readEvents, streamPost } from "./api";

function streamOf(...chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c));
      controller.close();
    },
  });
}

async function collect(body: ReadableStream<Uint8Array>) {
  const out = [];
  for await (const ev of readEvents(body)) out.push(ev);
  return out;
}

afterEach(() => vi.unstubAllGlobals());

describe("readEvents", () => {
  it("reassembles frames split across chunks", async () => {
    const events = await collect(
      streamOf('event: stage\ndata: {"stage":"ro', 'ute"}\n\nevent: token\ndata: {"text":"Hi"}\n', "\n"),
    );
    expect(events).toEqual([
      { event: "stage", data: { stage: "route" } },
      { event: "token", data: { text: "Hi" } },
    ]);
  });

  it("accepts CRLF line endings and a final frame without a blank line", async () => {
    const events = await collect(streamOf('event: result\r\ndata: {"ok":true}'));
    expect(events).toEqual([{ event: "result", data: { ok: true } }]);
  });
});

describe("streamPost", () => {
  it("passes progress through and resolves with the result", async () => {
    const body = streamOf(
      'event: token\ndata: {"text":"a"}\n\n',
      'event: result\ndata: {"final_answer":"done"}\n\n',
    );
    vi.stubGlobal("fetch", vi.fn(async () => new Response(body, { status: 200 })));
    const seen: string[] = [];
    const result = await streamPost<{ final_answer: string }>("/api/chat/stream", {}, (ev) => seen.push(ev.event));
    expect(seen).toEqual(["token"]);
    expect(result.final_answer).toBe("done");
  });

  it("rejects with the server's message on an error event", async () => {
    const body = streamOf('event: error\ndata: {"message":"Upstream busy","code":"upstream_quota","request_id":"r1"}\n\n');
    vi.stubGlobal("fetch", vi.fn(async () => new Response(body, { status: 200 })));
    await expect(streamPost("/x", {}, () => {})).rejects.toMatchObject({
      message: "Upstream busy",
      code: "upstream_quota",
      requestId: "r1",
    });
  });
});

describe("apiFetch", () => {
  it("reads the error envelope", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json({ error: { code: "not_found", message: "No such session", request_id: "r2" } }, { status: 404 }),
      ),
    );
    const err = await apiFetch("/api/sessions/x").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 404, code: "not_found", message: "No such session" });
  });
});
