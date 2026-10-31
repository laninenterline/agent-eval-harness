# Your agent eval is silently wrong

*A judge that returns nothing. A 429 that looks like a regression. Eight model
configs benchmarked so you don't have to.*

---

## The bug that started this

I was benchmarking an LLM-as-judge for agent regression testing. The first
results looked reasonable: 85% precision, a handful of false alarms. Then I ran
the baseline — the run where *nothing* changed — and 11 out of 120 healthy agent
runs were flagged as failures.

The judge was not wrong. The judge was **silent**.

DeepSeek's `deepseek-v4-pro` is a reasoning model. Reasoning tokens are billed
against `max_tokens`. I had set `max_tokens` to 200. The model spent all 200
tokens thinking, returned `finish_reason: "length"`, and gave me an empty
`content` string:

```json
{
  "finish_reason": "length",
  "content": "",
  "usage": {"completion_tokens_details": {"reasoning_tokens": 200}}
}
```

My client failed to parse that, and on parse failure it returned `score: 0.0`.
The downstream code read `0.0 < 0.5` and concluded: *the agent regressed*.

Nothing regressed. I had just built a machine that turns "I don't know" into
"it's broken."

That is the failure mode this post is about. Agent evaluation has a long tail of
ways to be silently wrong, and almost all of them look like real signal.

---

## What I was actually building

An agent run is hard to test with unit tests. It can call the right tool with the
wrong argument, finish with a plausible but wrong answer, or double its token
cost after a model swap — and your test suite stays green.

I built a small harness that treats an agent run as a test case:

```
OTel trace  ->  deterministic rubric  ->  judge  ->  report  ->  git-diffable JSON
```

The important design choice: **it reads traces, not framework internals**. If your
agent emits OpenTelemetry spans, it works. NVIDIA's NeMo Agent Toolkit emits
OpenInference attributes rather than `gen_ai.*` — a normalization layer maps both
into one view, so a case written once survives a framework migration.

Everything below comes from measuring that harness against a constructed
regression set.

---

## The benchmark

- **120 synthetic agent tasks**, each producing a baseline trace and a
  "degraded model" trace.
- **30 structural regressions**: missing tool call, wrong tool, missing fact,
  token blowup, latency blowup, error span.
- **10 semantic regressions**: the required fact is still present, but the answer
  is hedged — *"the source was unavailable so this may be stale."* A substring
  check cannot see the difference.
- **80 unchanged** runs, to measure false alarms.

Ground truth is constructed, so recall and precision are exactly measurable.

The deterministic layer alone:

| detector | recall | precision | false alarms |
|---|---|---|---|
| deterministic rubric | **75%** (30/40) | **100%** | **0** |
| naive "output non-empty" | 0% | — | 0 |
| naive "output got shorter" | 0% | — | 0 |

The deterministic layer catches every structural regression, never flags a
healthy run, and costs nothing in tokens. That is the layer you always turn on in
CI. The 10 remaining cases are the semantic tail — and that is exactly where an
LLM judge earns its keep.

So I benchmarked eight of them.

---

## Trap #1: the judge that scored zero because of a token budget

The bug from the intro. Two fixes:

```python
# 1. Give reasoning models room to think AND answer
AEH_JUDGE_MAX_TOKENS=1500   # not 200

# 2. Never treat truncation as a verdict
if choice["finish_reason"] == "length":
    raise ValueError("judge response truncated by max_tokens")
```

After the fix, the "85% precision, 11 false alarms" result disappeared. The
number was never about judge quality. It was a token-budget artifact wearing a
verdict's clothes.

**Rule:** a judge client that cannot distinguish *"the model said no"* from
*"the model said nothing"* is not a judge. It is a random number generator with
good manners.

---

## Trap #2: a failed judge is not a regression

Second benchmark run. Gemini reported 80% precision and 14 false alarms on the
healthy baseline.

Again, not the model. The free-tier account was rate-limiting; 25 calls returned
429 or timed out. My client caught the exception and returned `score: 0.0` —
which the benchmark counted as *"the agent regressed."*

The fix is a type change:

```python
@dataclass
class Verdict:
    score: float
    rationale: str = ""
    error: bool = False        # <- the important field
```

Errored verdicts are excluded from detection metrics and reported separately.
A judge error is a **scheduling problem**, not a **quality signal**. Conflating
them is the same bug as Trap #1, one layer up.

---

## Trap #3: every provider has a dialect

Four of the providers I tested disagree about things a "standard" OpenAI-compatible
client assumes:

| Provider | Quirk | Consequence |
|---|---|---|
| Moonshot (`kimi-k2.6`) | rejects `temperature != 1` | 400 on every call, or non-deterministic verdicts |
| Google Gemini | model id must keep the `models/` prefix | 404 |
| NVIDIA `deepseek-v4.1-flash` | JSON mode returns the answer in `reasoning_content`, `content` is `null` | silently parses as empty |
| DeepSeek / Kimi | thinking tokens billed against `max_tokens` | empty content (Trap #1) |

The NVIDIA one is my favorite, because it fails *while reporting success*:

```json
{
  "finish_reason": "stop",
  "message": {
    "content": null,
    "reasoning_content": "{\"score\": 0, \"rationale\": \"...\"}"
  }
}
```

`finish_reason: "stop"` looks healthy. The verdict is right there. A client that
only reads `content` will conclude the judge returned nothing and score it 0.0.

The fix is one line — fall back to `reasoning_content` — but you only find it by
running the benchmark on that provider.

**Rule:** "OpenAI-compatible" means "accepts the same JSON shape". It does not
mean "behaves the same."

---

## The results: eight judges

Full suite = 210 judge calls per model (120 baseline + 90 degraded cases that
passed deterministic checks). Latency measured from one Termux/Android
environment over the public internet, so treat it as a relative signal.

| model | provider | recall | precision | false alarms | semantic | tokens | p50 |
|---|---|---|---|---|---|---|---|
| **`nvidia/nemotron-3.5-bolt-e4b`** | NVIDIA | **100%** | **100%** | 0 | 10/10 | 52,588 | **1,449 ms** |
| `deepseek-flash` | DeepSeek | 98% | 100% | 0 | 9/10 | 87,273 | 1,560 ms |
| `deepseek-v4-pro` | DeepSeek | 100% | 100% | 0 | 10/10 | 94,951 | 2,200 ms |
| `gemini-3.1-flash-lite` | Google | 85% | 100% | 0 | 4/10 | 47,508 | 2,860 ms |
| `glm-5.3-flash` | Zhipu | 100% | 100% | 0 | 10/10 | 83,999 | 3,121 ms |
| `glm-5.3` | Zhipu | 100% | 100% | 0 | 10/10 | 90,704 | 3,294 ms |
| `nvidia/deepseek-v4.1-flash` | NVIDIA | 100% | 100% | 0 | 10/10 | 51,265 | 6,278 ms |
| `kimi-k2.6`* | Moonshot | subset | — | 0/10 | 10/10 | 14,707 | 19,295 ms |

\* Kimi's account is capped at **3 requests/minute** and each call takes ~19–30 s.
The full suite would have taken ~70 minutes, so only a 20-call subset was run.

Four things stood out.

### 1. The smallest model won

`nemotron-3.5-bolt-e4b` is one of the smallest models in the set and it got
**100% recall, 100% precision, zero false alarms, the lowest latency, and the
second-lowest token cost.**

Judge quality is not about parameter count. It is about whether the model does
what the prompt asks: read the rubric, check the output, return one number. A
small model that follows instructions beats a large model that reasons its way
into being charitable.

### 2. Zero false alarms everywhere

Not one model flagged a healthy run. The differences are entirely about how much
of the semantic tail each one notices. That is a much better place to be than
"everything is noisy" — it means the judge layer is safe to enable.

### 3. The cheapest judge was the worst judge

`gemini-3.1-flash-lite` used the fewest tokens and caught only **4 of 10**
semantic regressions. A perfectly calibrated judge that never fires is worse than
no judge at all: it manufactures confidence. If you only look at cost per call,
you will pick exactly the wrong model.

### 4. Rate limits disqualify judges faster than accuracy does

Kimi's quality was perfect on the subset. It is also unusable in CI: 3 RPM and
20 seconds per call. A judge that is right but cannot be called is a manual tool.

**Judge selection is accuracy × reliability × cost.** Optimizing only the first
term is how you end up with a CI gate nobody trusts.

---

## Bonus: does the host matter?

`deepseek-flash` is also served by NVIDIA NIM. Same nominal model, two hosts:

| host | recall | tokens | p50 | errors |
|---|---|---|---|---|
| DeepSeek direct | 98% | 87,273 | 1,560 ms | 0 |
| NVIDIA NIM | 100% | 51,265 | 6,278 ms | 8 |

Four times the latency, ~40% fewer tokens, same zero false alarms. This is not a
clean isolation — the two endpoints expose different serving configurations — but
it is enough to establish that **"the same model name" does not mean "the same
judge behaviour."** Benchmark the endpoint, not the model card.

---

## What this means for your eval

1. **Split the work.** Deterministic checks catch structural regressions for
   free and never false-alarm. Send only the cases that pass determinism to a
   judge. You get full recall at a fraction of the cost.
2. **Treat judge errors as errors.** `score=0.0` on an exception is a bug, not a
   verdict.
3. **Give reasoning models a real token budget**, and check `finish_reason`.
4. **Benchmark the endpoint, not the model.** Hosting, rate limits and JSON-mode
   quirks all change the numbers.
5. **Pick a small, instruction-following model** unless you have evidence that
   size buys you something on *your* dataset.

---

## The harness

`agent-eval-harness` is a small, dependency-free Python package that ingests
OpenTelemetry GenAI or OpenInference traces, applies a rubric, optionally calls a
judge from a provider registry, and emits a JSON report you can diff in CI.

```bash
pip install -e .
aeh run --cases examples/cases.jsonl --traces examples/traces \
        --judge-model nvidia/nemotron-3.5-bolt-e4b
aeh diff before.json after.json     # exits 1 on regression
```

- 25 unit tests
- `aeh judges` lists known providers
- NVIDIA NeMo Agent Toolkit traces supported via the OpenInference normalization
  layer (verified against `nvidia-nat` 1.9.0)

If you maintain an agent framework and want your traces supported, open an issue
with a sample export — the adapter is ~50 lines.

---

*The dataset is synthetic. The failure shapes mirror real model swaps, but the
numbers are not a claim about any specific model's quality. Reproduce it with
`python bench/generate_dataset.py && python bench/run_judge_compare.py`.*
