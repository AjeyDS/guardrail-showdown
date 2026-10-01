"""Shared HTTP helper for the API guardrails.

`post_json` never raises. It retries on 429, 5xx, timeouts and connection
errors, and reports the latency of the final attempt only (back-off sleeps
and failed earlier attempts are not counted).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

TIMEOUT_S = 30
# One initial attempt plus three retries, waiting 1s, 2s, 4s before each retry.
BACKOFF_S = (1, 2, 4)


@dataclass
class HttpResult:
    status: int | None  # HTTP status of the last attempt, None if no response
    json: Any  # parsed body of the last response, None if absent/unparseable
    latency_ms: float  # wall time of the last attempt only
    error: str | None  # None on success (2xx), else a short reason


def post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float = TIMEOUT_S,
) -> HttpResult:
    result = HttpResult(None, None, 0.0, "no_attempt")
    for attempt in range(len(BACKOFF_S) + 1):
        if attempt:
            time.sleep(BACKOFF_S[attempt - 1])
        start = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        except requests.Timeout:
            result = HttpResult(None, None, _ms(start), "timeout")
            continue
        except requests.RequestException as exc:
            # Class name only: exception text can contain URLs/headers.
            result = HttpResult(None, None, _ms(start), f"request_error: {type(exc).__name__}")
            continue
        latency = _ms(start)
        try:
            data = resp.json()
        except ValueError:
            data = None
        if 200 <= resp.status_code < 300:
            return HttpResult(resp.status_code, data, latency, None)
        result = HttpResult(resp.status_code, data, latency, f"http_{resp.status_code}: {_snippet(resp)}")
        if resp.status_code != 429 and resp.status_code < 500:
            return result  # permanent client error: retrying will not help
        if resp.status_code == 429 and "quota" in (resp.text or "").lower():
            return result  # quota exhausted (not a rate limit): retrying will not help
    return result


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _snippet(resp: Any) -> str:
    try:
        return (resp.text or "")[:150].replace("\n", " ")
    except Exception:
        return ""
