// Copy the static export into the Python package, where FastAPI serves it
// (edupilot.core.config.STATIC_DIR). The destination is generated: it is
// replaced wholesale on every build and ignored by git.
import { cp, rm } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const out = fileURLToPath(new URL("../out/", import.meta.url));
const dest = fileURLToPath(new URL("../../src/edupilot/web/static/", import.meta.url));

await rm(dest, { recursive: true, force: true });
await cp(out, dest, { recursive: true });
console.log(`copied the static export to ${dest}`);
