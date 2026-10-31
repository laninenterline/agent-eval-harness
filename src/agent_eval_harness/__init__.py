"""agent-eval-harness: task-level evaluation and regression testing for AI agents."""

from .judge import Judge, LLMJudge, RuleJudge, Verdict
from .providers import PROVIDERS, Provider, make_judge, resolve
from .runner import EvalCase, RunReport, run

__version__ = "0.1.0"

__all__ = [
    "EvalCase",
    "Judge",
    "LLMJudge",
    "Provider",
    "PROVIDERS",
    "RuleJudge",
    "RunReport",
    "Verdict",
    "make_judge",
    "resolve",
    "run",
]
