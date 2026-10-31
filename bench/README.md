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

| model | provider | recall | precision | false alarms | semantic | tokens | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| **`nvidia/nemotron-3.5-bolt-e4b`** | NVIDIA | **100%** | **100%** | 0 | 10/10 | 52,588 | **1,449 ms** | 3,541 ms |
| `deepseek-flash` | DeepSeek | 98% | 100% | 0 | 9/10 | 87,273 | 1,560 ms | 3,011 ms |
| `deepseek-v4-pro` | DeepSeek | 100% | 100% | 0 | 10/10 | 94,951 | 2,200 ms | 3,798 ms |
| `gemini-3.1-flash-lite` | Google | 85% | 100% | 0 | 4/10 | 47,508 | 2,860 ms | 24,023 ms |
| `glm-5.3-flash` | Zhipu | 100% | 100% | 0 | 10/10 | 83,999 | 3,121 ms | 6,501 ms |
| `glm-5.3` | Zhipu | 100% | 100% | 0 | 10/10 | 90,704 | 3,294 ms | 6,075 ms |
| `kimi-k2.6` | Moonshot | subset | — | 0/10 | 10/10 | 14,707 | 19,295 ms | 25,672 ms |

Findings:

- **The winner is NVIDIA's own model.** `nvidia/nemotron-3.5-bolt-e4b` gets
  **100% recall, 100% precision, zero false alarms, at the lowest latency
  (1,449 ms p50) and second-lowest token cost**. On this benchmark, the best
  judge available is hosted by the company this project is aimed at.
- **Zero false alarms everywhere.** No judge flagged a healthy run. The
  differences are entirely about how much of the *semantic* tail each model
  notices.
- **`deepseek-flash`** — best fallback: fastest of the DeepSeek pair, 98% recall.
- **`glm-5.3-flash` / `glm-5.3`** — full recall, moderate cost; good for nightly
  or second-opinion runs.
- **`gemini-3.1-flash-lite`** — cheapest tokens but only **4/10** semantic
  recall. Perfectly calibrated judges that never fire are worse than none: they
  manufacture false confidence. (Its account also hit rate limits; the 15 failed
  calls were excluded, not scored as failures.)
- **`kimi-k2.6`** — perfect quality on the subset, but **3 RPM** and **~19–30 s**
  per call make it a manual tool, not a CI gate. It also rejects
  `temperature != 1`, so verdicts are non-deterministic.
- **`gemini-3.8-flash`** timed out at >180 s per call and could not be evaluated.

### Same model, two hosts

`deepseek-flash` is also served by NVIDIA NIM as
`deepseek-ai/deepseek-v4.1-flash`. Same nominal model, two hosts:

| host | model id | recall | precision | false alarms | tokens | p50 | p95 | errors |
|---|---|---|---|---|---|---|---|---|
| DeepSeek direct | `deepseek-flash` | 98% | 100% | 0 | 87,273 | **1,560 ms** | 3,011 ms | 0 |
| NVIDIA NIM | `nvidia/deepseek-v4.1-flash` | **100%** | 100% | 0 | **51,265** | 6,278 ms | 21,603 ms | 8 |

Hosting changes the numbers materially: **4x the latency**, **~40% fewer tokens**
(more concise reasoning), and a small (8/210) rate-limit error count. Both reach
zero false alarms.

Caveat: this is not a clean isolation. The two endpoints expose different
serving configurations (the NIM variant also has the JSON-mode quirk below), so
the token difference may reflect generation settings rather than the host. It is
evidence that "same model name" does not mean "same judge behaviour", not a
controlled benchmark of hosting quality.

**Operational lesson:** judge selection is accuracy × reliability × cost, not
accuracy alone. A perfect judge at 3 RPM is a manual tool; a cheap judge that
misses 60% of semantic regressions is worse than no judge.

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
model id; DeepSeek and Kimi are reasoning models whose thinking tokens are
billed against `max_tokens`; and the NVIDIA-hosted `deepseek-v4.1-flash`
returns the answer in `reasoning_content` with `content: null` when JSON mode is
on. The client now falls back to `reasoning_content` before declaring an empty
response. A judge abstraction that assumes one uniform convention will produce
silently wrong numbers on most of the providers tested here.

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
