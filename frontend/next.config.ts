import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

/**
 * Production builds are a static export (`out/`), copied into the Python
 * package and served by FastAPI — one container, one origin, no Node server.
 *
 * `next dev` instead proxies /api to the FastAPI dev server, so the browser
 * talks to one origin in development too (the anonymous-identity cookie and
 * server-sent events need no CORS setup). Rewrites cannot be combined with a
 * static export, hence the two shapes.
 */
const shared: NextConfig = {
  trailingSlash: true,
  images: { unoptimized: true },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default function config(phase: string): NextConfig {
  if (phase === PHASE_DEVELOPMENT_SERVER) {
    const api = process.env.EDUPILOT_API ?? "http://127.0.0.1:8000";
    return {
      ...shared,
      async rewrites() {
        return [{ source: "/api/:path*", destination: `${api}/api/:path*` }];
      },
    };
  }
  return { ...shared, output: "export" };
}
