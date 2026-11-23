// Static guard: the dashboard front end only ever issues GET requests and
// contains no call that could change data or start a run.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const SRC = join(fileURLToPath(new URL(".", import.meta.url)), "..");
const files = (dir) =>
  readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? files(p) : /\.(jsx?|css)$/.test(n) && !n.endsWith(".test.js") ? [p] : [];
  });

test("no source file sets a non-GET HTTP method or uses XHR/sendBeacon", () => {
  for (const f of files(SRC)) {
    const text = readFileSync(f, "utf8");
    assert.doesNotMatch(text, /method\s*:\s*["'`](?!GET)/i, f);
    assert.doesNotMatch(text, /\b(XMLHttpRequest|sendBeacon)\b/, f);
  }
});

test("the API client exposes no delete/run/start/write helper", () => {
  const text = readFileSync(join(SRC, "api", "client.js"), "utf8");
  assert.doesNotMatch(text, /\b(delete|remove|start|launch|execute|post|put)\w*\s*:/i);
});

test("every fetch goes through the single GET helper", () => {
  const hits = files(SRC).filter((f) => /\bfetch\s*\(/.test(readFileSync(f, "utf8")));
  assert.deepEqual(hits.map((f) => f.split("/").pop()), ["client.js"]);
});
