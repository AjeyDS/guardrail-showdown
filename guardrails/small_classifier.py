"""ProtectAI DeBERTa-v3 prompt-injection classifier, run locally.

Model: protectai/deberta-v3-base-prompt-injection-v2 (labels SAFE / INJECTION).
`score` is P(INJECTION), taken from the full label distribution rather than
the top label, so it is a true attack probability in [0, 1].
"""

from __future__ import annotations

import time
from typing import Any

from guardrails.base import Verdict

# Registry metadata (read statically by guardrails/registry.py: keep these literals).
METHOD = "protectai"
DISPLAY_NAME = "ProtectAI"
LOCAL = True
REQUIRES_KEYS: list[str] = []
TASKS: list[str] = ["prompt_injection"]  # built for this task only

MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"
ATTACK_LABEL = "INJECTION"
THRESHOLD = 0.5
MAX_LENGTH = 512


class ProtectAIClassifier:
    name = "protectai"

    def __init__(self) -> None:
        import torch
        from transformers import pipeline

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._pipe = pipeline(
            "text-classification",
            model=MODEL_ID,
            device=device,
            top_k=None,  # return every label so we can read P(INJECTION)
            truncation=True,
            max_length=MAX_LENGTH,
        )
        self.device = device
        # Warm-up so the first measured call is not inflated by lazy init.
        # Vary the length: MPS compiles kernels per input shape.
        for text in ("warm-up prompt", "warm-up prompt " * 20, "warm-up prompt " * 150):
            self._pipe(text, truncation=True, max_length=MAX_LENGTH)

    def _score(self, prompt: str) -> tuple[float, dict[str, Any]]:
        out = self._pipe(prompt, truncation=True, max_length=MAX_LENGTH)
        if out and isinstance(out[0], list):
            out = out[0]
        probs = {d["label"].upper(): float(d["score"]) for d in out}
        top = max(out, key=lambda d: d["score"])
        if ATTACK_LABEL in probs:
            p_attack = probs[ATTACK_LABEL]
        else:  # defensive: unknown label set -> use 1 - P(safe)
            p_attack = 1.0 - probs.get("SAFE", 0.0)
        return p_attack, {"label": top["label"], "top_score": float(top["score"])}

    def check(self, prompt: str) -> Verdict:
        t0 = time.perf_counter()
        try:
            p_attack, raw = self._score(prompt or "")
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return Verdict(flagged=p_attack >= THRESHOLD, score=p_attack,
                           latency_ms=latency_ms, cost_usd=0.0, raw=raw)
        except Exception as exc:  # contract: check() never raises
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return Verdict(flagged=None, score=None, latency_ms=latency_ms,
                           cost_usd=0.0, error=f"{type(exc).__name__}: {str(exc)[:120]}")


def make(task=None) -> ProtectAIClassifier:  # `task` is accepted and ignored
    return ProtectAIClassifier()
