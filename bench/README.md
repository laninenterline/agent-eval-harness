# Benchmark: detecting regressions after a model swap

## Question

When you swap the model behind an agent (or change a prompt, or upgrade a tool),
how many of the resulting behaviour changes does a detector actually catch —
without drowning the team in false alarms?

## Method

`generate_dataset.py` builds 120 synthetic agent tasks and two runs of the same
agent:

| Run | Behaviour |
|---|---|
| `baseline.json` | correct tool, expected fact in output, within token/latency budget |
| `degraded.json` | 30 tasks fail **structurally**, 10 fail **semantically**, 80 unchanged |

Structural failures (deterministic checks can see them):

| mode | what changed |
|---|---|
| `missing_tool` | the required tool was never called |
| `wrong_tool` | a different tool was called |
| `missing_fact` | output no longer contains the required fact |
| `token_explosion` | token budget exceeded |
| `latency` | latency budget exceeded |
| `error_span` | a span errored |

Semantic failure (deterministic checks cannot see it):

| mode | what changed |
|---|---|
| `plausible_wrong` | the fact is still present, but the answer is hedged: *"the source was unavailable so this may be stale"* |

Ground truth is constructed, so recall and false alarms are exactly measurable.

## Results

Measured with real judges on DeepSeek (210 judge calls per model, 3 workers,
`AEH_JUDGE_MAX_TOKENS=1500`):

```
tasks=120  true_regressions=40 (structural=30, semantic=10)

detector                        detected  false_alarms  missed   recall  precision
----------------------------------------------------------------------------------
aeh (deterministic rubric)            30             0      10      75%       100%
aeh (rubric + deepseek-v4-pro)        40             0       0     100%       100%
aeh (rubric + deepseek-flash)         39             0       1      98%       100%
naive: output non-empty                0             0      40       0%         0%
```

### Judge model comparison

| model | provider | recall | precision | false alarms | tokens | p50 | suite |
|---|---|---|---|---|---|---|---|
| `deepseek-v4-pro` | DeepSeek | 100% | 100% | 0 | 94,951 | 2,200 ms | full (210 calls) |
| `deepseek-flash` | DeepSeek | 98% | 100% | 0 | 87,273 | **1,560 ms** | full (210 calls) |
| `glm-5.3` | Zhipu | 100% | 100% | 0 | 90,704 | 3,294 ms | full (210 calls) |
| `glm-5.3-flash` | Zhipu | 100% | 100% | 0 | **83,999** | 3,121 ms | full (210 calls) |
| `kimi-k2.6` | Moonshot | 10/10 semantic | — | 0/10 | 14,707 | 19,295 ms | subset (20 calls) |

All five judges reach **100% precision with zero false alarms** on the cases they
ran. They differ on recall of the semantic tail and, heavily, on operational fit:

- **`deepseek-flash`** — best default: fastest, 98% recall, zero false alarms.
- **`deepseek-v4-pro` / `glm-5.3` / `glm-5.3-flash`** — full recall; `glm-5.3-flash`
  is the most token-efficient. Fine for nightly runs or second-opinion gating.
- **`kimi-k2.6`** — quality is perfect on the subset (0/10 false alarms, 10/10
  semantic caught) but the account is capped at **3 requests/minute** and each
  call takes **~19–30 s**. The full 210-call suite would take ~70 minutes, so only
  a 20-call subset was measured. Kimi is also the only model here that rejects
  `temperature != 1`, so its scores are non-deterministic run to run.

**Operational lesson:** judge selection is not only about accuracy. Rate limits,
latency and determinism decide whether a judge can live in CI. A model that is
perfect on quality but costs 20 s and 3 RPM is a manual tool, not a CI gate.

### A trap worth documenting

Both DeepSeek models are **reasoning models**: reasoning tokens are billed
against `max_tokens`. With `max_tokens=200` the entire budget is consumed by
chain-of-thought, the API returns `finish_reason="length"` with **empty
content**, and a naive judge client silently records a score of 0 — which looks
exactly like a regression. This produced a fake "85% precision, 11 false alarms"
result before it was caught.

Two fixes are now in the client:

1. `AEH_JUDGE_MAX_TOKENS` defaults to **1500**, not 200.
2. `finish_reason == "length"` is treated as a retryable error, never as a score.

## Reading the result

1. **Deterministic rubric: 75% recall, 100% precision, zero false alarms.** It
   catches every structural regression and never flags a healthy run. This is the
   layer you always turn on, in CI, for free.
2. **The naive detectors catch nothing.** "Output is non-empty / not much
   shorter" is blind to wrong tools, missing facts and budget blowups.
3. **The remaining 10 are semantic.** A substring check cannot tell a confident
   answer from a hedged one. That is exactly the bucket the judge layer exists
   for. With a real judge: **100% recall at 85% precision** (7 false alarms on
   40 flagged, 11 on the 120 healthy baseline).

   That ~9% false-alarm rate is the honest price of catching the semantic tail.
   In production you would tune the threshold, add majority voting over 3 judge
   calls, or gate the judge layer to PRs that change a model/prompt.

The rule judge shipped by default is keyword overlap; it is a placeholder for
tests and offline runs, not a real semantic judge. Run the benchmark with
`--llm-judge` (needs `OPENAI_API_KEY`) to measure what the judge adds.

## Honest limitations

- The dataset is synthetic. The failure *shapes* mirror real model swaps, but
  the numbers are not a claim about any specific model.
- Next validation step: replay real traces captured before/after a model swap
  and compare against human labels.

## Reproduce

```bash
python bench/generate_dataset.py
python bench/run_benchmark.py
```
