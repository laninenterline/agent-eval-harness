# aehcli (npm wrapper)

**[aehcli](https://pypi.org/project/aehcli/)** is a framework-neutral, task-level
evaluation and regression harness for AI agents. It is implemented in Python.

This package is a thin launcher so that JavaScript/TypeScript projects can call
`aeh` from the same toolchain as the rest of their repo:

```bash
npx aehcli --help
npx aehcli run --cases cases.jsonl --traces ./traces
npx aehcli diff before.json after.json
```

## Requirements

`aehcli` delegates to the Python implementation, which must be installed:

```bash
pip install aehcli        # or: pipx install aehcli
```

The launcher looks for, in order:

1. an `aeh` executable on `PATH`
2. `python3 -m agent_eval_harness`
3. `python -m agent_eval_harness`

## Why a wrapper instead of a rewrite

Agent traces are OpenTelemetry data; parsing them in two languages would double
the surface area for the subtle semantic-convention bugs this project exists to
prevent. One implementation, two entry points.

## Links

- Python package: https://pypi.org/project/aehcli/
- Source: https://github.com/laninenterline/agent-eval-harness
- Benchmark: https://github.com/laninenterline/agent-eval-harness/blob/main/bench/README.md
