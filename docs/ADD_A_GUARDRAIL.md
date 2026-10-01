# Add your own guardrail

Drop one file into `guardrails/`. It is auto-discovered: no registry to edit.

Guardrails are evaluated on a *task* (what to detect: prompt injection, toxicity, ...). A task is a config file in `tasks/`; see [ADD_A_TASK.md](ADD_A_TASK.md). Your guardrail says which tasks it supports.

## The contract

`guardrails/base.py` defines it. Your module needs:

| Name | Meaning |
|---|---|
| `METHOD` | short id, used in file names and tables (`"mine"`). A string literal. |
| `DISPLAY_NAME` | label for humans (`"My Guardrail"`). A string literal. |
| `LOCAL` | `True` if it runs on this machine (free, no network). |
| `REQUIRES_KEYS` | env var names it needs (`[]` for local). Put them in `.env`. |
| `TASKS` | which tasks it can run on. A list of task names, or `["*"]` for any. A literal. If you leave it out it means `["prompt_injection"]`. |
| `make(task=None)` | returns an object with `name: str` and `check(prompt) -> Verdict`. `task` is the loaded `Task` (see below). |

Metadata is read without importing your module, so write the five constants as plain literals. Import heavy libraries (torch, ...) inside `make()` or `__init__`, not at the top of the file.

### Which value for `TASKS`

- **`["*"]`: task-agnostic.** Use this for a method that does what the task's definition says, such as an LLM judge or a decision API (Jev and Luna do). It gets run on every task.
- **`["prompt_injection"]` (or any named tasks): task-specific.** Use this for a method with a fixed purpose, such as a regex list, a trained classifier or a vendor's detector (regex, ProtectAI and Lakera do). It is run only on those tasks, and the scorecard for other tasks lists it under "Not applicable to this task".

### Using the task in `make(task)`

The runner and `try.py` call `make(task)` with a `guardrails.task.Task`. Task-specific modules can ignore it (`def make(task=None)`). Model-based, task-agnostic ones build their prompt from it:

| `Task` field | Use |
|---|---|
| `task.question`, `task.criteria_true`, `task.criteria_false` | the definition, to be sent word for word |
| `task.judge_role` | opening line for an LLM system prompt |
| `task.state_key`, `task.question_key` | key names for the text and the question (Jev style) |
| `task.positive_label`, `task.negative_label` | human wording ("attack", "safe prompt") |

See `guardrails/jev.py` and `guardrails/llm_judge.py`. A `make()` with no parameters still works: it is then called without the task, and you should use `TASKS = ["prompt_injection"]`.

`Verdict(flagged, score, latency_ms, cost_usd=None, error=None, raw=None)`:

- `flagged`: `True` = positive class (an attack, for prompt injection), `False` = negative, `None` only if the call failed.
- `score`: P(positive) in [0, 1], or `None` if you only have yes/no. Use threshold 0.5 when you have a score.
- `latency_ms`: wall time of the successful attempt.
- `cost_usd`: `0.0` for local, billed amount if known, else `None`.
- `error`: short reason on failure. `check()` must never raise.
- `raw`: small JSON-serialisable provider response, so analysis can be re-run for free.

## Example: a toy keyword guardrail

Save as `guardrails/keywords.py` (this example is not shipped in the repo). It is built for prompt injection, so it declares that task:

```python
import time

from guardrails.base import Verdict

METHOD = "keywords"
DISPLAY_NAME = "Toy Keywords"
LOCAL = True
REQUIRES_KEYS: list[str] = []
TASKS: list[str] = ["prompt_injection"]

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


def make(task=None) -> Keywords:  # the task is ignored: this method is task-specific
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

Add `--task <name>` to either command (and to `try.py`) to use another task. Raw verdicts land in `<results_dir>/raw/<split>/keywords.jsonl` (`results/raw/test/keywords.jsonl` for the default prompt-injection task) and are cached, so re-runs resume and analysis costs nothing. Use `--split val --limit 50` first as a smoke test.

## Fairness rules

1. **Same question for everyone.** If your guardrail is model-based (an LLM judge or a decision API), use the task's definition (`task.question`, `task.criteria_true`, `task.criteria_false`) verbatim, as Jev and Luna do. (`ATTACK_QUESTION`, `ATTACK_TRUE` and `ATTACK_FALSE` in `guardrails/base.py` still work as deprecated aliases for the prompt-injection task.)
2. **Never tune on test.** Choose rules, prompts and thresholds on `val` only; run `test` once, for the final numbers.
3. **Count failures.** If a call errors, return a `Verdict` with `error` set. Failed rows are kept and counted in the results, never dropped or retried until they pass quietly.
4. **Be honest about cost and latency.** Report what the provider billed, and measure the successful attempt only.

## Contribute your results

Open a PR with your `guardrails/<name>.py` and your `<results_dir>/raw/<split>/<name>.jsonl` files (no API keys, no `.env`). Say which task and splits you ran and whether anything failed.
