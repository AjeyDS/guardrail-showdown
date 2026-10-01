> **Historical:** this is the build brief used while the benchmark was being written. Contracts have since moved: the definition lives in `tasks/<name>.toml`, guardrails declare `TASKS` and take `make(task)`. See `docs/ADD_A_GUARDRAIL.md` and `docs/ADD_A_TASK.md` for the current ones.

# Build Brief (shared by all builders)

## Launch phase Goal Card (current)
Goal: Turn the finished benchmark into a public repo people can try in a minute, reuse (hard set, harness), and extend (add their own guardrail), with visuals a non-expert understands at a glance. It will be shared on GitHub, Medium and LinkedIn.
Principles:
- Every number shown must come from `results/` data; never hand-typed.
- Honest framing: caveats (small hard set, Lakera partial, disputed labels) stay visible; no visual should make any method look better than the data says.
- Zero-friction: things a visitor runs first must work with no API keys.
Non-goals: no new guardrails, datasets, web apps, or re-running paid benchmarks.
Autonomy: peer (adapt-and-advise), same rules as below.

Current facts: Lakera data is partial (332 val prompts) and is excluded from test/hard results. `results/results.csv` has all splits; Lakera runs are mostly quota errors; `analyze.py` leaves out any method with over 20% failed calls. The 50-prompt pre-fix Jev/Luna smoke results (narrow attack definition) are in `results/archive/val_{jev,luna}_narrow_def.jsonl`.

## Goal Card
Goal: Compare five prompt-injection guardrails (regex, ProtectAI small classifier, Jev, GPT-6 Luna as judge, Lakera Guard) on one public dataset and produce a simple scorecard plus a plain verdict on speed, cost, accuracy, and whether confidence scores can be trusted. Jev is the newcomer whose claims (fast, cheap, calibrated) are being checked.
Principles:
- Fairness over cleverness: every method gets the same prompts, the same "attack" definition, and failures are counted, never dropped.
- Easy to interpret over exhaustive: plain metrics, one table, a few charts.
- Cheap and reproducible: total API spend under $1; every raw API response cached so analysis reruns for free.
Non-goals:
- No extra guardrails, datasets, dashboards, or frameworks beyond the plan.
- No tuning any method on the test split.
Autonomy: peer (adapt-and-advise). You may adapt *how* when reality differs from this brief (log every deviation in your final report with a one-line reason). Anything that changes the shared contract below, the file layout, or scope: stop and report instead.

Full plan: `docs/PLAN.md`. Read it once.

## Environment
- Python venv at `.venv/` (Python 3.13). Run things with `.venv/bin/python` and `.venv/bin/pytest`. Dependencies are already installed from `requirements.txt`; if you truly need a new package, report it, don't install it.
- API keys are in `.env` (`OPENROUTER_API_KEY`, `LAKERA_API_KEY`, `OPENAI_API_KEY` unused). Load with `python-dotenv`. **Never print, log, or write key values anywhere.**
- Machine: Apple M5, 16 GB. Torch MPS is available.
- Not a place to commit: do not run git commands.

## Shared contract (already written; do not change)
`guardrails/base.py` defines:
- `Verdict(flagged, score, latency_ms, cost_usd=None, error=None, raw=None)` where `score` = P(attack) in [0,1] or None.
- `Guardrail` protocol: attribute `name: str`, method `check(prompt: str) -> Verdict`.
- `ATTACK_QUESTION`, `ATTACK_TRUE`, `ATTACK_FALSE`: the one shared definition of "attack". Jev and the LLM judge must both use these strings verbatim.

Rules every guardrail follows:
- `check()` never raises. On failure after retries, return `Verdict(flagged=None, score=None, latency_ms=<last attempt>, error="<short reason>")`.
- API guardrails retry up to 3 times on 429/5xx/timeouts with exponential back-off (1s, 2s, 4s); `latency_ms` measures the successful attempt only. Timeout 30 s per request.
- `flagged` uses threshold 0.5 on `score` when a score exists.
- Each guardrail module has a module-level factory `make() -> Guardrail` the runner calls.

## Data (prepared by `prepare_data.py`)
`data/val.csv`, `data/test.csv`, with columns: `row_id, text, label, category, source, severity, group_id, augmented, tags`.
`label`: 1 = attack (injection or jailbreak), 0 = benign. `tags` is `|`-joined.
Val split: for writing regex rules and choosing thresholds. Test split: final numbers only. **Never read test.csv contents to make design decisions.**
More splits may be added later (e.g. `data/hard.csv`) with the same columns; code must treat the split name generically (`data/<split>.csv`).

Facts about the data (from val; test is similar):
- ~941 rows, ~57% attacks. Prompts are short: median ~55 chars, max ~4,200 chars.
- Benign rows: `category == "benign"` (mostly short templated requests) or `category == "edge_case"` (hard negatives, ~10 rows).
- **is_hard_negative** = `label == 0 and (category == "edge_case" or tags contain "hard_negative" or "security_adjacent")`.
- Attack categories are a long tail (~25 values). Analysis groups them into families:
  - `direct_injection`: direct_injection, instruction_override, prompt_injection, control, payload_injection, output_manipulation, response_manipulation
  - `jailbreak_persona`: jailbreak, persona_replacement, multi_turn, many_shot
  - `command_exec`: adversarial, code_execution
  - `obfuscation`: encoding, encoding_obfuscation, token_smuggling, token_injection
  - `indirect_rag`: indirect_injection, rag_poisoning, context_confusion, agent_manipulation
  - `extraction`: prompt_extraction, system_extraction, training_extraction, model_fingerprinting, system_manipulation
  - anything unmapped → `other`
- "Hard attacks" for the scorecard = families `obfuscation` + `indirect_rag` + `jailbreak_persona`.

## Method → module map (runner registry)
| method | module | notes |
|---|---|---|
| `regex` | `guardrails/regex_filter.py` | local, score None |
| `protectai` | `guardrails/small_classifier.py` | local, `protectai/deberta-v3-base-prompt-injection-v2`, truncate to 512 tokens |
| `jev` | `guardrails/jev.py` | OpenRouter Decisions API |
| `luna` | `guardrails/llm_judge.py` | OpenRouter chat completions, `openai/gpt-6-luna` |
| `lakera` | `guardrails/cloud_api.py` | Lakera Guard v2, score None |

## Results layout (runner output, consumed by analysis)
- `results/raw/<split>/<method>.jsonl`: one JSON object per line: `{"row_id": int, **Verdict.to_dict()}`. The runner resumes by skipping row_ids already present without an error.
- `results/results.csv`: one row per (split, method, row_id) with columns:
  `split, method, row_id, label, category, is_hard_negative, flagged, score, latency_ms, cost_usd, error`
- Method names (exact): `regex`, `protectai`, `jev`, `luna`, `lakera`.
