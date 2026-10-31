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

| model | provider | recall | precision | false alarms | semantic | tokens | p50 | errors |
|---|---|---|---|---|---|---|---|---|
| `deepseek-v4-pro` | DeepSeek | 100% | 100% | 0 | 10/10 | 94,951 | 2,200 ms | 0 |
| `deepseek-flash` | DeepSeek | 98% | 100% | 0 | 9/10 | 87,273 | **1,560 ms** | 0 |
| `glm-5.3` | Zhipu | 100% | 100% | 0 | 10/10 | 90,704 | 3,294 ms | 0 |
| `glm-5.3-flash` | Zhipu | 100% | 100% | 0 | 10/10 | **83,999** | 3,121 ms | 0 |
| `gemini-3.1-flash-lite` | Google | 85% | 100% | 0 | 4/10 | **47,508** | 2,860 ms | 15 |
| `kimi-k2.6` | Moonshot | subset | — | 0/10 | 10/10 | 14,707 | 19,295 ms | 0 (20 calls) |

Findings:

- **Zero false alarms everywhere.** No judge, on any provider, flagged a healthy
  run. The differences are entirely about how much of the *semantic* tail each
  model notices.
- **`deepseek-flash`** — best default: fastest, 98% recall, zero false alarms.
- **`glm-5.3-flash`** — full recall and the most token-efficient of the full-suite
  models. Good for nightly runs.
- **`gemini-3.1-flash-lite`** — cheapest by tokens, but only catches 4/10 semantic
  regressions. It is *too forgiving*: a perfectly calibrated judge that never
  fires is useless. Note: the account also hit rate limits (15 failed calls were
  excluded from scoring, not counted as failures — see below).
- **`kimi-k2.6`** — quality is perfect on the subset (0/10 false alarms, 10/10
  semantic) but the account is capped at **3 requests/minute** and each call
  takes **~19–30 s**; the full suite would take ~70 min. Kimi also rejects
  `temperature != 1`, so its verdicts are non-deterministic.
- **`gemini-3.8-flash`** (newest) timed out at >180 s per call in this
  environment and could not be evaluated at all.

**Operational lesson:** judge selection is accuracy × reliability, not accuracy
alone. A judge that is perfect but rate-limited to 3 RPM is a manual tool; a
judge that is cheap but blind to the semantic tail is worse than no judge,
because it creates false confidence.

### Three traps worth documenting

Both DeepSeek models are **reasoning models**: reasoning tokens are billed
against `max_tokens`. With `max_tokens=200` the entire budget is consumed by
chain-of-thought, the API returns `finish_reason="length"` with **empty
content**, and a naive judge client silently records a score of 0 — which looks
exactly like a regression. This produced a fake "85% precision, 11 false alarms"
result before it was caught.

Fixes now in the client:

1. `AEH_JUDGE_MAX_TOKENS` defaults to **1500**, not 200.
2. `finish_reason == "length"` is treated as a retryable error, never as a score.

**Trap 2 — a judge error is not a regression.** When a call fails (429, timeout),
the old client returned `score=0.0`, which the benchmark counted as "the agent
regressed". Gemini's first run reported 80% precision purely from 25 failed
calls. `Verdict.error` now carries the failure explicitly and the benchmark
excludes errored verdicts from detection metrics while reporting them separately.

**Trap 3 — provider quirks are part of the API.** Moonshot rejects any
`temperature != 1`; Gemini's OpenAI-compatible endpoint requires `models/` in the
model id; both DeepSeek and Kimi are reasoning models whose thinking tokens are
billed against `max_tokens`. A judge abstraction that assumes one uniform
convention will produce silently wrong numbers on at least two of the four
providers tested here.

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
