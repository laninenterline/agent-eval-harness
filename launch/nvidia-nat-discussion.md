# Note: NAT emits OpenInference attributes, not GenAI semconv

Hi all — I maintain a framework-neutral evaluation harness
([`aehcli`](https://pypi.org/project/aehcli/)) that turns OTel traces of agent
runs into pass/fail regression tests.

While integrating with **NVIDIA NeMo Agent Toolkit 1.9.0**, I hit something that
seems worth flagging: `nat/plugins/opentelemetry/span_converter.py` maps spans
onto **OpenInference** semantic conventions, not the OTel `gen_ai.*` family:

| NAT span | attribute emitted | generic OTel consumer expects |
|---|---|---|
| LLM call | `openinference.span.kind = LLM` | `gen_ai.operation.name = chat` |
| tool call | `openinference.span.kind = TOOL` | `gen_ai.operation.name = execute_tool` |
| token usage | `llm.token_count.prompt/completion` | `gen_ai.usage.input_tokens/output_tokens` |
| model | `llm.model_name` | `gen_ai.request.model` |
| final answer | `output.value` | `gen_ai.completion` |

The practical consequence: a generic OTel consumer that only reads `gen_ai.*`
will parse every NAT trace as **zero tool calls, zero tokens, no output** — and
then report the agent as failed, with no error anywhere. It fails silently while
reporting success.

I added a normalization layer that maps OpenInference onto `gen_ai.*` so one eval
case works across frameworks, and verified the mapping against the 1.9.0 source:

```python
# agent_eval_harness/semconv.py
_KIND_TO_OPERATION = {"LLM": "chat", "TOOL": "execute_tool", "AGENT": "invoke_agent", ...}
```

Two questions for the maintainers:

1. Is OpenInference the intended long-term output for NAT's OTel plugin, or is a
   `gen_ai.*` mode planned as well?
2. Would you be open to either (a) a docs note for people consuming NAT traces
   with generic OTel tooling, or (b) a small in-tree adapter/normalizer so both
   conventions are available?

I'm happy to send either as a PR. Also curious whether other folks here have
worked around this in their own pipelines.

- Harness: https://github.com/laninenterline/agent-eval-harness
- The mapping table also lives in `adapters/nvidia_nat.py`
