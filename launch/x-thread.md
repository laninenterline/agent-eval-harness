# X / Twitter thread

8 tweets. All under 280 characters (URLs count as 23 on X, so tweet 8 is
effectively shorter than shown).

**Images to attach**
- Tweet 1: terminal screenshot of `aeh demo` (PASS/FAIL table)
- Tweet 6: the benchmark table (screenshot or a rendered table image)
- Tweet 8: optional — the GitHub repo card will auto-expand from the link

**Posting**
1. Post tweet 1 alone, then reply with each subsequent one to form a thread.
2. Pin tweet 1 for the week.
3. Add alt text to both images (X rewards accessibility and it increases reach).

### Tweet 1/8 — `hook` [275 chars OK]

```
My agent eval flagged 11 regressions.

Nothing had regressed.

The judge had returned an empty response — reasoning tokens ate the whole max_tokens budget — and my client recorded score: 0.

I found 3 more ways an eval lies silently. Then fixed them and benchmarked 8 judges.
```

### Tweet 2/8 — `what` [239 chars OK]

```
What I built: aeh treats an agent run as a test case.

trace -> rubric -> judge -> report -> git diff

It reads OpenTelemetry traces, so it doesn't care which framework produced them.

aeh diff before.json after.json exits 1 on regression.
```

### Tweet 3/8 — `trap2` [213 chars OK]

```
Trap 2: a failed judge is not a regression.

A 429 or a timeout also landed as score 0.0 — which the benchmark read as "the agent regressed".

Gemini's first run reported 80% precision purely from 25 failed calls.
```

### Tweet 4/8 — `trap3` [243 chars OK]

```
Trap 3: every provider has a dialect.

- Moonshot rejects temperature != 1
- Gemini needs models/ in the model id
- NVIDIA JSON mode returns the answer in reasoning_content, content: null

finish_reason: "stop". Looks healthy. Parses as empty.
```

### Tweet 5/8 — `trap4` [175 chars OK]

```
Trap 4: the cheapest judge was the worst.

Gemini flash-lite used the fewest tokens and caught 4/10 semantic regressions.

A judge that never fires looks perfectly calibrated.
```

### Tweet 6/8 — `numbers` [214 chars OK]

```
120 tasks, 40 real regressions:

- deterministic rubric: 75% recall, 0 false alarms
- + best judge: 100% recall
- cheapest judge: 85% recall

Best overall: nemotron-3.5-bolt — 100% recall, 0 false alarms, 1.4s p50.
```

### Tweet 7/8 — `install` [218 chars OK]

```
pip install aehcli && aeh demo
npx aehcli --help
brew tap laninenterline/tap && brew install aehcli

Caveat: the dataset is synthetic. The failure shapes mirror real model swaps — not a claim about any model's quality.
```

### Tweet 8/8 — `cta` [193 chars OK]

```
If the benchmark is unfair, I want to hear it.

Write-up + full tables + the four traps:
https://laninenterline.github.io/agent-eval-harness/

Code: github.com/laninenterline/agent-eval-harness
```
