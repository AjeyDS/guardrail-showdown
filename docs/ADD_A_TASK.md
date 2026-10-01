# Add a task

A task is what the guardrails are asked to detect: prompt injection (task #1), toxicity, PII, off-topic requests, and so on. A new task is one config file plus a CSV. No Python needed.

## In five steps

1. **Copy the template.** `cp tasks/_template.toml tasks/toxicity.toml` (use your own name; it must match the `name` line inside the file).
2. **Write the definition.** Edit the `[definition]` table (see "Why the definition matters" below). Set `positive_label` and `negative_label`, the words the scorecard uses for "the thing to flag" and "the thing to let through".
3. **Add data.** Put `val.csv` and `test.csv` in `data/toxicity/` with the columns `text,label` (see the data contract below).
4. **Try one prompt.** `python try.py --task toxicity "you are an idiot"` runs every guardrail that supports the task and has its key set.
5. **Run and score.**

```bash
python run_benchmark.py --task toxicity --split val  --methods jev,luna --limit 50   # cheap smoke test
python run_benchmark.py --task toxicity --split test --methods jev,luna
python analyze.py --task toxicity                                                     # scorecard
python make_visuals.py --task toxicity                                                # shareable charts
```

Raw verdicts are cached in `results/toxicity/raw/<split>/<method>.jsonl`, so reruns resume and `analyze.py` costs nothing. The scorecard lands in `results/toxicity/scorecard.md`. Change `data_dir` and `results_dir` in the task file if you want them elsewhere (they default to `data/<name>` and `results/<name>`).

## The data contract

A split CSV (`<data_dir>/<split>.csv`) needs two columns:

| Column | Meaning |
|---|---|
| `text` | the string the guardrail checks |
| `label` | `1` = should be flagged (the positive class), `0` = should pass |

Everything else is optional. `row_id` is generated from row order if you leave it out (so keep the row order stable once you have cached results, or add your own unique `row_id`). `category` and `tags` (`|`-joined) default to empty; they only matter for the optional breakdowns below. Bad labels, missing columns and duplicate `row_id` values stop the run with a message that names the file and the line.

Split names are a convention, not a rule: `val` is for choosing cut-offs and writing rules, `test` is for the final numbers (never tune on it), and `hard` is picked up by `make_visuals.py` if you have one. `analyze.py --split` picks the main split and `--threshold-split` (default `val`) the split used for the tuned cut-off.

## Why the definition matters

Two guardrails in the harness follow a definition you write instead of a built-in one:

- **Jev** gets your `question`, `criteria_true` and `criteria_false` as its decision question.
- **Luna**, the LLM judge, gets the same three strings inside its system prompt, plus your `judge_role` as the first line.

Both receive exactly the same words, so neither gets an easier question. This is the part of the setup that moves results most. In the prompt-injection benchmark, adding one sentence to the shared definition (covering command execution and forced output, which the dataset labels as attacks) moved Luna from 32% to 93% on a 50-prompt smoke test (see the README). Write the definition first, check it on a few dozen `val` prompts, and only then run `test`. Say what counts as positive, what does not, and mention your tricky negatives explicitly.

A few mechanics:

- `state_key` is the name the text travels under. Refer to it in the question with backticks (`` `message` ``), as the template does.
- `question_key` is the key Jev answers under (`is_toxic`). Luna's JSON answer uses the same name without the `is_` prefix (`toxic`).
- The definition is part of the experiment. The runner records a fingerprint of it next to each cached file (`<method>.definition.json`) and refuses to add answers under an edited definition to a cache made under the old one. Move the old `.jsonl` aside to rerun; keep it if you want a before/after comparison, as we did for the 32% to 93% chart.

## Which guardrails run on a task

Each guardrail declares the tasks it can handle (`TASKS` in its module, see [ADD_A_GUARDRAIL.md](ADD_A_GUARDRAIL.md)):

| Method | Tasks | Why |
|---|---|---|
| `jev`, `luna` | any (`["*"]`) | They follow whatever definition the task gives them. |
| `regex`, `protectai`, `lakera` | `prompt_injection` only | They were built and tuned for that one job: a regex list, a model trained on injection data, a vendor's prompt-attack detector. |

`run_benchmark.py` and `try.py` default to the methods that support the task and refuse the others with a clear message. The scorecard says which known methods were left out: "Not applicable to this task: ...". To add a method for your task, see the guardrail guide.

## Optional extras

Everything below can be left out. The scorecard drops the parts it cannot compute.

- `[families]` maps your `category` values to families and enables the category breakdown, plus `hard_families` (reported as "hard" positives) and the hard-negative rule (`hard_negative_categories`, `hard_negative_tags`) for tricky-but-fine examples.
- `[splits.<split>] note = "..."` is the dataset line in the scorecard footnotes.
- `[scorecard]` holds a few sentences of prose (title, intro, notes). See the commented lines in the template.

The prompt-injection task, `tasks/prompt_injection.toml`, uses all of them and is the best worked example.

## Cost and honesty checklist

- Jev and Luna are paid calls (a few cents per thousand prompts). Start with `--limit 50`.
- Failed calls count as "not flagged" and are reported, never dropped.
- Report the size of your test set. With a few hundred rows the confidence intervals in the scorecard are wide: differences smaller than the interval are ties.
