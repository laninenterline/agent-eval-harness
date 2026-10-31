# Show HN: I benchmarked 8 LLM judges on agent regressions

**URL:** https://github.com/laninenterline/agent-eval-harness/blob/main/blog/2026-10-31-your-agent-eval-is-silently-wrong.md

**Text:**

I built a small harness that treats an agent run as a test case: it ingests
OpenTelemetry traces, applies a deterministic rubric (expected tools, output
facts, token/latency budgets), and optionally calls an LLM judge for the things
a rubric can't see.

Then I measured it. 120 synthetic tasks, 40 real regressions: 30 structural
(missing tool, wrong tool, missing fact, token blowup, latency, error span) and
10 semantic (the fact is present, but the answer is hedged: "source unavailable,
may be stale").

- deterministic rubric alone: **75% recall, 100% precision, 0 false alarms**
- + LLM judge: **100% recall** (8 judges across 5 providers benchmarked)
- naive "output is non-empty" check: **0% recall**

The numbers were the least interesting part. Four ways the eval was silently
wrong:

1. **Reasoning models spend `max_tokens` on thinking.** With a 200-token budget,
   the model returned `finish_reason: "length"` and empty content. My client
   recorded `score: 0.0`, which the benchmark read as "the agent regressed".
   Nothing had regressed.
2. **A judge error is not a regression.** A 429 or a timeout also landed as
   `score: 0.0`. Gemini's first run reported 80% precision purely from 25 failed
   calls.
3. **Every provider has a dialect.** Moonshot rejects `temperature != 1`. Gemini's
   OpenAI-compatible endpoint needs `models/` in the model id. NVIDIA's hosted
   `deepseek-v4.1-flash` returns the answer in `reasoning_content` with
   `content: null` when JSON mode is on — `finish_reason: "stop"`, looks healthy,
   parses as empty.
4. **The cheapest judge was the worst judge.** `gemini-3.1-flash-lite` used the
   fewest tokens and caught 4/10 semantic regressions. A judge that never fires
   manufactures confidence.

The best judge on this dataset was `nvidia/nemotron-3.5-bolt-e4b`: 100% recall,
zero false alarms, 1.4s p50, ~52k tokens for 210 calls.

- Repo: https://github.com/laninenterline/agent-eval-harness
- PyPI: `pip install aehcli`
- npm: `npx aehcli --help`
- Benchmark details: `bench/README.md`

Caveat: the dataset is synthetic. The failure *shapes* mirror real model swaps,
but none of this is a claim about a specific model's quality. Happy to hear
where the benchmark is unfair — that's the part I'd most like to be wrong about.
