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

Measured with a real judge (`deepseek-v4-pro`, 210 judge calls, 3 workers):

```
tasks=120  true_regressions=40 (structural=30, semantic=10)

detector                        detected  false_alarms  missed   recall  precision
----------------------------------------------------------------------------------
aeh (deterministic rubric)            30             0      10      75%       100%
aeh (rubric + LLM judge)              40             7       0     100%        85%
naive: output non-empty                0             0      40       0%         0%

baseline false alarms (healthy runs): 11/120 (9%)
semantic regressions caught by judge: 10/10
```

Without a judge (rule judge only, keyword overlap), the numbers are identical to
the deterministic row: a keyword judge cannot separate a confident answer from a
hedged one, which is the whole point of the semantic bucket.

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
