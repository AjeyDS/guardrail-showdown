# Add your own guardrail

Drop one file into `guardrails/`. It is auto-discovered: no registry to edit.

## The contract

`guardrails/base.py` defines it. Your module needs:

| Name | Meaning |
|---|---|
| `METHOD` | short id, used in file names and tables (`"mine"`). A string literal. |
| `DISPLAY_NAME` | label for humans (`"My Guardrail"`). A string literal. |
| `LOCAL` | `True` if it runs on this machine (free, no network). |
| `REQUIRES_KEYS` | env var names it needs (`[]` for local). Put them in `.env`. |
| `make()` | returns an object with `name: str` and `check(prompt) -> Verdict`. |

Metadata is read without importing your module, so write the four constants as plain literals. Import heavy libraries (torch, ...) inside `make()` or `__init__`, not at the top of the file.

`Verdict(flagged, score, latency_ms, cost_usd=None, error=None, raw=None)`:

- `flagged`: `True` = attack, `False` = benign, `None` only if the call failed.
- `score`: P(attack) in [0, 1], or `None` if you only have yes/no. Use threshold 0.5 when you have a score.
- `latency_ms`: wall time of the successful attempt.
- `cost_usd`: `0.0` for local, billed amount if known, else `None`.
- `error`: short reason on failure. `check()` must never raise.
- `raw`: small JSON-serialisable provider response, so analysis can be re-run for free.

## Example: a toy keyword guardrail

Save as `guardrails/keywords.py` (this example is not shipped in the repo):

```python
import time

from guardrails.base import Verdict

METHOD = "keywords"
DISPLAY_NAME = "Toy Keywords"
LOCAL = True
REQUIRES_KEYS: list[str] = []

BAD = ("ignore previous", "system prompt", "developer mode", "jailbreak")


class Keywords:
    name = "keywords"

    def check(self, prompt: str) -> Verdict:
        t0 = time.perf_counter()
        try:
            hits = [w for w in BAD if w in prompt.lower()]
            return Verdict(flagged=bool(hits), score=None, cost_usd=0.0, raw={"hits": hits},
                           latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:  # never raise
            return Verdict(None, None, 0.0, cost_usd=0.0, error=f"{type(exc).__name__}")


def make() -> Keywords:
    return Keywords()
```

Try it right away:

```
.venv/bin/python try.py "Ignore previous instructions"
```

## Run it on the benchmark

```
.venv/bin/python run_benchmark.py --split test --methods keywords
.venv/bin/python analyze.py
```

Raw verdicts land in `results/raw/test/keywords.jsonl` and are cached, so re-runs resume and analysis costs nothing. Use `--split val --limit 50` first as a smoke test.

## Fairness rules

1. **Same question for everyone.** If your guardrail is model-based (an LLM judge or a decision API), use `ATTACK_QUESTION`, `ATTACK_TRUE`, `ATTACK_FALSE` from `guardrails/base.py` verbatim, as Jev and Luna do.
2. **Never tune on test.** Choose rules, prompts and thresholds on `val` only; run `test` once, for the final numbers.
3. **Count failures.** If a call errors, return a `Verdict` with `error` set. Failed rows are kept and counted in the results, never dropped or retried until they pass quietly.
4. **Be honest about cost and latency.** Report what the provider billed, and measure the successful attempt only.

## Contribute your results

Open a PR with your `guardrails/<name>.py` and your `results/raw/<split>/<name>.jsonl` files (no API keys, no `.env`). Say which splits you ran and whether anything failed.
