#!/usr/bin/env node
"use strict";

/**
 * Thin launcher for the Python implementation of `aeh`.
 *
 * The harness itself lives on PyPI as `aehcli`. This npm package exists so that
 * JavaScript projects can call `npx aehcli ...` from the same toolchain as the
 * rest of their repo. It does not bundle Python; it delegates to it.
 */

const { spawnSync } = require("node:child_process");

const CANDIDATES = [
  { cmd: "aeh", args: [] },
  { cmd: "python3", args: ["-m", "agent_eval_harness"] },
  { cmd: "python", args: ["-m", "agent_eval_harness"] },
];

function tryRun(candidate, passthrough) {
  const result = spawnSync(candidate.cmd, [...candidate.args, ...passthrough], {
    stdio: "inherit",
  });
  return result;
}

function main() {
  const passthrough = process.argv.slice(2);

  for (const candidate of CANDIDATES) {
    const result = tryRun(candidate, passthrough);
    if (result.error && result.error.code === "ENOENT") continue;
    if (result.error) {
      continue;
    }
    return process.exit(typeof result.status === "number" ? result.status : 1);
  }

  console.error(
    [
      "aeh: could not find a working Python installation of aehcli.",
      "",
      "Install it first:",
      "  pip install aehcli",
      "",
      "Then re-run:",
      "  npx aehcli --help",
    ].join("\n"),
  );
  process.exit(127);
}

main();
