"""Shared contract for every guardrail in the benchmark.

Every guardrail module exposes a class with a `name` attribute and a
`check(prompt) -> Verdict` method. The runner never needs to know which
kind of guardrail it is talking to.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class Verdict:
    # True = treated as an attack. None only when the call failed (see `error`).
    flagged: bool | None
    # Probability the prompt is an attack, in [0, 1]. None if the method
    # only gives yes/no (regex, Lakera).
    score: float | None
    # Wall time of the successful attempt only (retry back-off excluded).
    latency_ms: float
    # Actual billed USD for this call if known; 0.0 for local methods;
    # None if the provider doesn't report it.
    cost_usd: float | None = None
    # Short error string when the call failed after retries, or the reply
    # couldn't be parsed. Failed rows are kept and counted, never dropped.
    error: str | None = None
    # Raw provider response (JSON-serialisable) so analysis can be rerun
    # without paying again. Keep it small: no request echo, no headers.
    raw: dict[str, Any] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Guardrail(Protocol):
    name: str  # short id used in file names and tables, e.g. "jev"

    def check(self, prompt: str) -> Verdict: ...


# The definition of what to detect now lives in `tasks/<name>.toml` (see
# guardrails/task.py). These three names are kept as deprecated aliases for code
# that imported them: they are the prompt_injection task's definition, word for word.
_DEPRECATED = {
    "ATTACK_QUESTION": "question",
    "ATTACK_TRUE": "criteria_true",
    "ATTACK_FALSE": "criteria_false",
}


def __getattr__(name: str):
    if name in _DEPRECATED:
        from .task import default_task

        return getattr(default_task(), _DEPRECATED[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
