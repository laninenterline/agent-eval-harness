"""Command line interface for agent-eval-harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import diff as diff_module
from .judge import RuleJudge
from .providers import PROVIDERS, make_judge
from .models import Trace
from .otel import load_otlp_json
from .report import to_markdown, write_json
from .runner import EvalCase, run


def load_cases(path: Path) -> list[EvalCase]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [EvalCase.from_dict(json.loads(line)) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    items = data if isinstance(data, list) else data.get("cases", [])
    return [EvalCase.from_dict(item) for item in items]


def load_traces(path: Path) -> dict[str, Trace]:
    traces: list[Trace] = []
    if path.is_dir():
        for candidate in sorted(path.glob("*.json")):
            traces.extend(load_otlp_json(candidate))
    else:
        traces = load_otlp_json(path)
    keyed: dict[str, Trace] = {}
    for trace in traces:
        keyed[trace.task_id or trace.trace_id] = trace
    return keyed


def build_judge(args: argparse.Namespace):
    model = args.judge_model or (args.model if args.llm_judge else None)
    if not model:
        return RuleJudge()
    return make_judge(
        model,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
    )


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases))
    traces = load_traces(Path(args.traces))
    judge = build_judge(args)
    report = run(cases, traces, judge)
    output = Path(args.out)
    write_json(report, output)
    print(to_markdown(report))
    print(f"report written to {output}")
    return 0 if report.all_passed else 1


def cmd_diff(args: argparse.Namespace) -> int:
    before = diff_module.load_report(args.before)
    after = diff_module.load_report(args.after)
    entries = diff_module.diff_reports(before, after)
    print(diff_module.render(entries))
    return 1 if diff_module.has_regressions(entries) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aeh",
        description="Framework-neutral task-level evaluation and regression harness for AI agents",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run eval cases against traces")
    run_parser.add_argument("--cases", required=True, help="cases file (.jsonl or .json)")
    run_parser.add_argument("--traces", required=True, help="OTLP JSON file or directory")
    run_parser.add_argument("--out", default="aeh-report.json", help="report output path")
    run_parser.add_argument("--judge-model", default=None,
                            help="LLM judge model, e.g. deepseek-flash or nvidia/nemotron-3.5-bolt-e4b")
    run_parser.add_argument("--judge-temperature", type=float, default=None,
                            help="override the provider's default temperature")
    run_parser.add_argument("--judge-max-tokens", type=int, default=None,
                            help="override the provider's default token budget")
    run_parser.add_argument("--llm-judge", action="store_true", help="(compat) use an LLM judge")
    run_parser.add_argument("--model", default=None, help="(compat) judge model name")
    run_parser.set_defaults(func=cmd_run)

    diff_parser = subparsers.add_parser("diff", help="compare two reports, fail on regression")
    diff_parser.add_argument("before")
    diff_parser.add_argument("after")
    diff_parser.set_defaults(func=cmd_diff)

    subparsers.add_parser("judges", help="list known judge providers").set_defaults(
        func=cmd_judges
    )
    return parser


def cmd_judges(_args: argparse.Namespace) -> int:
    header = f"{'model':36s} {'base url':52s} {'key env':20s} {'temp':>4s} {'max_tokens':>10s}"
    print(header)
    print("-" * len(header))
    for provider in PROVIDERS.values():
        print(f"{provider.model:36s} {provider.base_url:52s} {provider.key_env:20s} "
              f"{provider.temperature:4.1f} {provider.max_tokens:10d}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
