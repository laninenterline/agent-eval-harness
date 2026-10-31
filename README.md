# agent-eval-harness

**Task-level evaluation and regression testing for AI agents — framework-neutral.**

Agents fail in ways unit tests never catch: they call the right tool with the
wrong argument, finish with a plausible but wrong answer, or quietly double
their token cost after a model swap. `aeh` treats an agent run as a **test
case**: it ingests OpenTelemetry GenAI traces, applies a rubric, scores the
outcome, and **fails CI when an agent regresses**.

```
trace (OTel GenAI)  ->  rubric checks  ->  judge  ->  report  ->  git diff-able JSON
```

## Why another eval tool

Most agent evaluation today is either (a) a closed dashboard, or (b) a
framework-specific SDK. `aeh` is deliberately the opposite:

- **Framework-neutral** — it reads traces, not framework internals. LangGraph,
  NVIDIA NeMo Agent Toolkit, OpenAI Agents SDK, or your own loop all produce
  spans; `aeh` does not care which one produced them.
- **Trace-first** — evaluation input is a real production trace, not a
  hand-written prompt. This is what makes regression testing possible.
- **CI-native** — `aeh run` exits non-zero on failure; `aeh diff` exits non-zero
  on regression. It is a test runner, not a dashboard.
- **Zero heavy dependencies** — one package, stdlib only by default. An LLM
  judge is opt-in via any OpenAI-compatible endpoint.

## Quick start

```bash
pip install -e .

# run the example suite
aeh run --cases examples/cases.jsonl --traces examples/traces/otel_sample.json

# compare two runs (exit code 1 on regression)
aeh diff before.json after.json
```

## How a case is defined

```json
{"id": "weather-agent-001",
 "task": "Find the weather in Shanghai and summarize it",
 "rubric": {"expect_tools": ["get_weather"],
            "output_contains": ["Shanghai"],
            "max_total_tokens": 2000},
 "judge_threshold": 0.5}
```

The `id` is matched against the trace's `aeh.task.id` attribute (falling back to
the trace id). Three kinds of checks run:

1. **Deterministic rubric** — expected tools called, output contains, budget caps,
   no error spans.
2. **Judge** — pluggable. A zero-dependency `RuleJudge` ships by default;
   `LLMJudge` talks to any OpenAI-compatible endpoint (`OPENAI_API_KEY`).
3. **Aggregate** — pass rate, per-case latency and token cost, captured in a
   JSON report that is stable enough to diff.

## Supported trace conventions

| Convention | Produced by | Handling |
|---|---|---|
| OTel GenAI (`gen_ai.*`) | OpenAI Agents SDK, LangGraph exporters, most OTel setups | native |
| OpenInference (`openinference.span.kind`, `llm.token_count.*`, `output.value`) | **NVIDIA NeMo Agent Toolkit** (`nat.plugins.opentelemetry`) | normalized in `semconv.py` |

Both are folded into one normalized `gen_ai.*` view before evaluation, so a case
written once works across frameworks. The NAT mapping was verified against
`nvidia-nat 1.9.0`'s `span_converter.py`.

## Where this fits


The gap it fills is narrow on purpose: **business-level acceptance of an agent
run**, not GPU profiling and not model benchmarking.

| Layer | Who owns it today |
|---|---|
| GPU / kernel profiling | NVIDIA Nsight |
| Model benchmarks | NeMo Evaluator, MLflow |
| Trace transport | OpenTelemetry GenAI semconv |
| **Task-level accept/regress** | **nobody, yet** -> this project |

## Benchmark

120-task synthetic model-swap dataset, 40 real regressions (30 structural, 10
semantic):

| detector | recall | precision | false alarms |
|---|---|---|---|
| deterministic rubric only | 75% (30/40) | 100% | 0 |
| rubric + `deepseek-v4-pro` judge | **100% (40/40)** | 100% | 0 |
| rubric + `glm-5.3` / `glm-5.3-flash` judge | **100% (40/40)** | 100% | 0 |
| rubric + `deepseek-flash` judge | 98% (39/40) | 100% | 0 |
| rubric + `glm-5.3-flash` judge | **100% (40/40)** | 100% | 0 |
| rubric + `gemini-3.1-flash-lite` judge | 85% (34/40) | 100% | 0 |
| rubric + `kimi-k2.6` judge (subset, rate-limited) | 10/10 semantic | — | 0/10 |
| naive "output non-empty" | 0% | — | 0 |

Naive checks catch nothing. The deterministic layer is free and precise. The
judge buys the semantic tail with zero false alarms across four judge models
(DeepSeek and GLM). Note: reasoning models
consume `max_tokens` with chain-of-thought, so a small budget returns empty
content that looks like failure — see [`bench/README.md`](bench/README.md).

## Status


v0.1 — OTLP JSON ingestion, deterministic rubric, rule judge, JSON/Markdown
reports, regression diff, GitHub Action example, one adapter (NVIDIA NAT).

## License

Apache-2.0.
