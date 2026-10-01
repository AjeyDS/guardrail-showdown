# Guardrail Showdown: scorecard (test split)

Five prompt-injection guardrails, one dataset. On the **test** split each method saw 42 prompts: 26 attacks and 16 safe ones (4 of the safe ones are tricky-but-safe). Regex = keyword rules we wrote; ProtectAI = small open classifier; Jev = decision model; Luna = GPT-6 as a judge; Lakera = managed security API.

## Scorecard

Numbers in square brackets are 95% confidence intervals. Differences smaller than the interval are ties.

| Method | Catch rate | False blocks | False blocks on tricky-but-safe (blocked/total) | Catch rate on hard attacks | Catch rate at 5% false blocks | ROC-AUC | Median latency (ms) | p95 latency (ms) | Cost per 1M checks (USD) | Calibration error (ECE) | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Regex | 73.1% [53.8, 88.5] | 43.8% [18.8, 68.8] | 100.0% (4/4) | 75.0% [50.0, 100.0] | n/a | n/a | 0.05 | 0.07 | $0 (local) | n/a | 0 (0.0%) |
| ProtectAI | 96.2% [88.5, 100.0] | 12.5% [0.0, 31.2] | 50.0% (2/4) | 91.7% [75.0, 100.0] | 100.0% [100.0, 100.0] | 0.971 | 36.1 | 63.3 | $0 (local) | 0.223 | 0 (0.0%) |
| Jev | 80.8% [65.4, 96.2] | 6.2% [0.0, 18.8] | 0.0% (0/4) | 75.0% [50.0, 100.0] | 69.2% [50.0, 84.6] | 0.936 | 282 | 487 | $10.00 | 0.216 | 2 (4.8%) |
| Luna | 65.4% [46.2, 84.6] | 31.2% [12.5, 56.2] | 100.0% (4/4) | 66.7% [41.7, 91.7] | 7.7% [0.0, 19.2] | 0.723 | 1,404 | 2,417 | $40.00 | 0.236 | 1 (2.4%) |
| Lakera | 65.4% [46.2, 84.6] | 0.0% [0.0, 0.0] | 0.0% (0/4) | 50.0% [25.0, 75.0] | n/a | n/a | 167 | 242 | free tier (quota hit after ~390 calls) | n/a | 0 (0.0%) |

How to read it:
- **Catch rate**: of all attacks, how many were blocked (higher is better).
- **False blocks**: of safe prompts, how many were wrongly blocked (lower is better). The tricky-but-safe subset (security homework, questions about SQL injection) is small, so read it as a hint, not a verdict.
- **Hard attacks**: obfuscated (encoding tricks), indirect (instructions hidden in documents) and persona/jailbreak attacks.
- **Catch rate at 5% false blocks**: scored methods only. The cut-off is picked on the **val** split as the lowest one that wrongly blocks at most 5% of safe prompts, then applied to test. (5%, not 1%: about 10 'safe' prompts in val are jailbreak-style openers from the WildGuard source that every scored method flags at ~100%, so a 1% budget is unreachable because of disputed labels, not guardrail skill.)
- **ROC-AUC**: how well the raw score ranks attacks above safe prompts (1.0 perfect, 0.5 coin flip).
- **Calibration error (ECE)**: when it says 90%, is it right about 90% of the time? 0 is perfect; lower is better. n/a for yes/no-only tools.
- **Cost**: projected from the average real cost per call; local methods are free.

Cut-offs chosen on val for the 5% column: ProtectAI 0.47 (blocks 12.5% of safe prompts on test); Jev 0.605 (blocks 0.0% of safe prompts on test); Luna 0.939 (blocks 0.0% of safe prompts on test).

## Category breakdown: catch rate per attack family

![Catch rate by attack family](figures/category_heatmap.png)

| Method | direct injection (n=8) | jailbreak persona (n=4) | command exec (n=2) | obfuscation (n=4) | indirect rag (n=4) | extraction (n=3) | other (n=1) |
|---|---|---|---|---|---|---|---|
| Regex | 75.0% | 100.0% | 50.0% | 100.0% | 25.0% | 66.7% | 100.0% |
| ProtectAI | 100.0% | 100.0% | 100.0% | 100.0% | 75.0% | 100.0% | 100.0% |
| Jev | 75.0% | 75.0% | 100.0% | 75.0% | 75.0% | 100.0% | 100.0% |
| Luna | 62.5% | 75.0% | 50.0% | 50.0% | 75.0% | 100.0% | 0.0% |
| Lakera | 75.0% | 50.0% | 50.0% | 50.0% | 50.0% | 100.0% | 100.0% |

Families with few attacks are noisy: one missed prompt can move the percentage a lot.

## Awards

| Award | Winner | Why (one line) |
|---|---|---|
| Catches the most attacks | Tie: ProtectAI, Jev, Regex | ProtectAI catches 96.2% [88.5, 100.0] of attacks; Jev catches 80.8% [65.4, 96.2] of attacks; Regex catches 73.1% [53.8, 88.5] of attacks (intervals overlap, so no clear winner) |
| Fewest false blocks | Tie: Lakera, Jev, ProtectAI | Lakera wrongly blocks 0.0% [0.0, 0.0] of safe prompts; Jev wrongly blocks 6.2% [0.0, 18.8] of safe prompts; ProtectAI wrongly blocks 12.5% [0.0, 31.2] of safe prompts (intervals overlap, so no clear winner) |
| Fastest | Regex | Regex median 0.05 ms (p95 0.07 ms); next best ProtectAI median 36.1 ms (p95 63.3 ms) |
| Cheapest | Tie: Regex, ProtectAI | Regex $0 (local) per 1M checks; ProtectAI $0 (local) per 1M checks (exactly equal) |
| Most trustworthy confidence | Jev | Jev calibration error 0.216 (0 = perfect); next best ProtectAI calibration error 0.223 (0 = perfect) |
| Best on hard attacks | Tie: ProtectAI, Regex, Jev, Luna, Lakera | ProtectAI catches 91.7% [75.0, 100.0] of hard attacks; Regex catches 75.0% [50.0, 100.0] of hard attacks; Jev catches 75.0% [50.0, 100.0] of hard attacks; Luna catches 66.7% [41.7, 91.7] of hard attacks; Lakera catches 50.0% [25.0, 75.0] of hard attacks (intervals overlap, so no clear winner) |

A rate award is a **tie** when the runner-up's confidence interval overlaps the leader's. Speed, cost and calibration just take the best value. Methods with no cost data are left out of 'Cheapest'; yes/no-only methods are left out of 'Most trustworthy confidence'.

## Use X if...

_To be written by a human. Not auto-generated._

- Use Regex if... TODO
- Use ProtectAI if... TODO
- Use Jev if... TODO
- Use Luna if... TODO
- Use Lakera if... TODO
- Jev's verdict (did it hold up on speed, cost and calibration, and where did it fall short?): TODO

## Charts

![Catch rate vs false blocks](figures/catch_vs_false_blocks.png)

![Calibration](figures/calibration.png)

![Latency](figures/latency.png)

## Footnotes

1. **Dataset**: neuralchemy/Prompt-injection-dataset, 'core' config, test split.
2. **Size**: test: 42 prompts per method (26 attacks, 16 safe, 4 tricky-but-safe); val: 42 prompts per method (26 attacks, 16 safe, 4 tricky-but-safe).
3. **Run date**: <DATE> (last modified time of the results file).
4. **Errors**: a failed call (timeout, refusal, API error) counts as **not flagged**, because a guardrail that fails open lets the attack through. So errors lower catch rate and never raise false blocks. ROC-AUC, calibration, the cut-off and latency use only calls that returned a result.
5. **Confidence intervals**: bootstrap, 1,000 resamples of the prompts, seed 0, 2.5th to 97.5th percentile.
6. **Latency** is measured on successful calls only. Cost is the mean of the calls that reported a cost.
