# [P] I benchmarked 8 LLM judges on agent regressions — accuracy was the easy part

**Subreddits:** r/MachineLearning (use `[P]`), r/LocalLLaMA (drop the tag)

**Link:** https://laninenterline.github.io/agent-eval-harness/

**Body:**

I built a small harness that turns an agent run into a test case: ingest the
OpenTelemetry trace, apply a deterministic rubric (expected tools, output facts,
token/latency budgets, error spans), and optionally call an LLM judge for what a
rubric can't see.

Benchmark: 120 synthetic tasks, 40 real regressions — 30 structural (missing
tool, wrong tool, missing fact, token blowup, latency, error span) and 10
semantic (the fact is present, but the answer is hedged: "source unavailable,
may be stale").

**Results** (210 judge calls per model, same prompts):

| model | recall | precision | false alarms | tokens | p50 |
|---|---|---|---|---|---|
| nvidia/nemotron-3.5-bolt-e4b | 100% | 100% | 0 | 52,588 | 1,449 ms |
| deepseek-v4-pro | 100% | 100% | 0 | 94,951 | 2,200 ms |
| glm-5.3-flash | 100% | 100% | 0 | 83,999 | 3,121 ms |
| deepseek-flash | 98% | 100% | 0 | 87,273 | 1,560 ms |
| gemini-3.1-flash-lite | 85% | 100% | 0 | 47,508 | 2,860 ms |

Deterministic rubric alone: 75% recall / 100% precision / **0 false alarms**.
Naive "output is non-empty": 0% recall.

**Four things that were silently wrong before they were fixed:**

1. Reasoning models bill thinking tokens against `max_tokens`. With a 200-token
   budget the API returns `finish_reason: "length"` and empty content; my client
   recorded `score: 0.0`, which the benchmark read as "the agent regressed."
2. A 429 or timeout also landed as `score: 0.0` — a judge *error* counted as a
   *regression*.
3. Provider dialects: Moonshot rejects `temperature != 1`; Gemini needs `models/`
   in the model id; NVIDIA's JSON mode puts the answer in `reasoning_content`
   with `content: null`.
4. The cheapest judge was the worst: Gemini flash-lite caught 4/10 semantic
   regressions. A judge that never fires looks perfectly calibrated.

**Caveat I'd most like to be challenged on:** the dataset is synthetic. The
failure *shapes* mirror real model swaps, but none of this is a claim about a
particular model's quality. If you think the benchmark is unfair, I want to hear
it.

- Write-up: https://laninenterline.github.io/agent-eval-harness/
- Code: https://github.com/laninenterline/agent-eval-harness
- `pip install aehcli && aeh demo`
