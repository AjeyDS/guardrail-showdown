"""Jev (TypeSafe System One) via the OpenRouter Decisions API (alpha)."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from ._http import post_json
from .base import ATTACK_FALSE, ATTACK_QUESTION, ATTACK_TRUE, Verdict

load_dotenv()

# Registry metadata (read statically by guardrails/registry.py: keep these literals).
METHOD = "jev"
DISPLAY_NAME = "Jev"
LOCAL = False
REQUIRES_KEYS: list[str] = ["OPENROUTER_API_KEY"]

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"  # pinned; not jev-latest / jev-router


class Jev:
    name = "jev"

    def check(self, prompt: str) -> Verdict:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            return Verdict(None, None, 0.0, error="missing OPENROUTER_API_KEY")
        body = {
            "model": MODEL,
            "state": {"user_prompt": prompt},
            "questions": {
                "is_attack": {
                    "type": "noul",
                    "instructions": ATTACK_QUESTION,
                    "criteria": {"true": ATTACK_TRUE, "false": ATTACK_FALSE},
                }
            },
            "provider": {"data_collection": "deny"},
        }
        res = post_json(URL, {"Authorization": f"Bearer {key}"}, body)
        if res.error:
            return Verdict(None, None, res.latency_ms, error=res.error)
        data = res.json if isinstance(res.json, dict) else {}
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        raw: dict[str, Any] = {
            "model": data.get("model"),
            "answers": data.get("answers"),
            "usage": usage,
        }
        cost = usage.get("cost") if usage else None
        cost = float(cost) if isinstance(cost, (int, float)) else None
        try:
            score = float(data["answers"]["is_attack"]["noul"])
        except (KeyError, TypeError, ValueError):
            return Verdict(None, None, res.latency_ms, cost, "parse_error: no answers.is_attack.noul", raw)
        return Verdict(score >= 0.5, score, res.latency_ms, cost, None, raw)


def make() -> Jev:
    return Jev()
