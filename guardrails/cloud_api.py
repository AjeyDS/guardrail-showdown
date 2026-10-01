"""Lakera Guard v2. We only measure its prompt-attack detector."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from ._http import post_json
from .base import Verdict

load_dotenv()

# Registry metadata (read statically by guardrails/registry.py: keep these literals).
METHOD = "lakera"
DISPLAY_NAME = "Lakera Guard"
LOCAL = False
REQUIRES_KEYS: list[str] = ["LAKERA_API_KEY"]

URL = "https://api.lakera.ai/v2/guard"


class Lakera:
    name = "lakera"

    def check(self, prompt: str) -> Verdict:
        key = os.environ.get("LAKERA_API_KEY")
        if not key:
            return Verdict(None, None, 0.0, error="missing LAKERA_API_KEY")
        body: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "breakdown": True,
        }
        if os.environ.get("LAKERA_PROJECT_ID"):
            body["project_id"] = os.environ["LAKERA_PROJECT_ID"]
        res = post_json(URL, {"Authorization": f"Bearer {key}"}, body)
        if res.error:
            return Verdict(None, None, res.latency_ms, error=res.error)
        data = res.json if isinstance(res.json, dict) else {}
        breakdown = data.get("breakdown")
        attack = [
            d
            for d in (breakdown if isinstance(breakdown, list) else [])
            if isinstance(d, dict) and "prompt_attack" in str(d.get("detector_type", ""))
        ]
        raw: dict[str, Any] = {
            "top_level_flagged": data.get("flagged"),
            "prompt_attack": [
                {"detector_type": d.get("detector_type"), "detected": d.get("detected"), "result": d.get("result")}
                for d in attack
            ],
        }
        if attack:
            # Top-level `flagged` is always false in Detect mode and also
            # reflects PII/moderation detectors, so decide from prompt_attack only.
            flagged = any(d.get("detected") is True for d in attack)
        elif isinstance(data.get("flagged"), bool):
            raw["fallback"] = "no prompt_attack detector in breakdown; used top-level flagged"
            flagged = data["flagged"]
        else:
            return Verdict(None, None, res.latency_ms, None, "parse_error: no prompt_attack detector or flagged", raw)
        # Lakera gives only ordinal levels (e.g. l1_confident), no probability.
        return Verdict(flagged, None, res.latency_ms, None, None, raw)


def make() -> Lakera:
    return Lakera()
