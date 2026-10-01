# Guardrail Showdown: scorecard (val split)

Five prompt-injection guardrails, one dataset. On the **val** split each method saw 42 prompts: 26 attacks and 16 safe ones (4 of the safe ones are tricky-but-safe). Regex = keyword rules we wrote; ProtectAI = small open classifier; Jev = decision model; Luna = GPT-6 as a judge; Lakera = managed security API.

## Scorecard

Numbers in square brackets are 95% confidence intervals. Differences smaller than the interval are ties.

| Method | Catch rate | False blocks | False blocks on tricky-but-safe (blocked/total) | Catch rate on hard attacks | Catch rate at 5% false blocks | ROC-AUC | Median latency (ms) | p95 latency (ms) | Cost per 1M checks (USD) | Calibration error (ECE) | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Regex | 73.1% [53.8, 88.5] | 25.0% [6.2, 43.8] | 100.0% (4/4) | 91.7% [75.0, 100.0] | n/a | n/a | 0.05 | 0.07 | $0 (local) | n/a | 0 (0.0%) |
| ProtectAI | 88.5% [76.9, 100.0] | 0.0% [0.0, 0.0] | 0.0% (0/4) | 83.3% [58.3, 100.0] | 96.2% [88.5, 100.0] | 0.995 | 40.0 | 65.6 | $0 (local) | 0.220 | 0 (0.0%) |
| Jev | 84.6% [69.2, 96.2] | 18.8% [0.0, 37.5] | 50.0% (2/4) | 83.3% [58.3, 100.0] | 80.8% [65.4, 96.2] | 0.957 | 268 | 450 | $10.00 | 0.192 | 2 (4.8%) |
| Luna | 96.2% [88.5, 100.0] | 37.5% [12.5, 62.5] | 100.0% (4/4) | 100.0% [100.0, 100.0] | 3.8% [0.0, 11.5] | 0.752 | 1,599 | 2,534 | $40.00 | 0.274 | 1 (2.4%) |
| Lakera | 65.4% [46.2, 84.6] | 0.0% [0.0, 0.0] | 0.0% (0/4) | 75.0% [50.0, 100.0] | n/a | n/a | 140 | 260 | free tier (quota hit after ~390 calls) | n/a | 0 (0.0%) |

How to read it:
- **Catch rate**: of all attacks, how many were blocked (higher is better).
- **False blocks**: of safe prompts, how many were wrongly blocked (lower is better). The tricky-but-safe subset (security homework, questions about SQL injection) is small, so read it as a hint, not a verdict.
- **Hard attacks**: obfuscated (encoding tricks), indirect (instructions hidden in documents) and persona/jailbreak attacks.
- **Catch rate at 5% false blocks**: scored methods only. The cut-off is picked on the **val** split as the lowest one that wrongly blocks at most 5% of safe prompts, then applied to val. (5%, not 1%: about 10 'safe' prompts in val are jailbreak-style openers from the WildGuard source that every scored method flags at ~100%, so a 1% budget is unreachable because of disputed labels, not guardrail skill.)
- **ROC-AUC**: how well the raw score ranks attacks above safe prompts (1.0 perfect, 0.5 coin flip).
- **Calibration error (ECE)**: when it says 90%, is it right about 90% of the time? 0 is perfect; lower is better. n/a for yes/no-only tools.
- **Cost**: projected from the average real cost per call; local methods are free.

Cut-offs chosen on val for the 5% column: ProtectAI 0.47 (blocks 0.0% of safe prompts on val); Jev 0.605 (blocks 0.0% of safe prompts on val); Luna 0.939 (blocks 0.0% of safe prompts on val).

## Category breakdown: catch rate per attack family

![Catch rate by attack family](figures/category_heatmap.png)

| Method | direct injection (n=8) | jailbreak persona (n=4) | command exec (n=2) | obfuscation (n=4) | indirect rag (n=4) | extraction (n=3) | other (n=1) |
|---|---|---|---|---|---|---|---|
| Regex | 75.0% | 75.0% | 50.0% | 100.0% | 100.0% | 0.0% | 100.0% |
| ProtectAI | 100.0% | 100.0% | 100.0% | 75.0% | 75.0% | 100.0% | 0.0% |
| Jev | 87.5% | 100.0% | 100.0% | 75.0% | 75.0% | 66.7% | 100.0% |
| Luna | 87.5% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Lakera | 50.0% | 75.0% | 0.0% | 100.0% | 50.0% | 100.0% | 100.0% |

Families with few attacks are noisy: one missed prompt can move the percentage a lot.

## Awards

| Award | Winner | Why (one line) |
|---|---|---|
| Catches the most attacks | Tie: Luna, ProtectAI, Jev, Regex | Luna catches 96.2% [88.5, 100.0] of attacks; ProtectAI catches 88.5% [76.9, 100.0] of attacks; Jev catches 84.6% [69.2, 96.2] of attacks; Regex catches 73.1% [53.8, 88.5] of attacks (intervals overlap, so no clear winner) |
| Fewest false blocks | Tie: ProtectAI, Lakera, Jev | ProtectAI wrongly blocks 0.0% [0.0, 0.0] of safe prompts; Lakera wrongly blocks 0.0% [0.0, 0.0] of safe prompts; Jev wrongly blocks 18.8% [0.0, 37.5] of safe prompts (intervals overlap, so no clear winner) |
| Fastest | Regex | Regex median 0.05 ms (p95 0.07 ms); next best ProtectAI median 40.0 ms (p95 65.6 ms) |
| Cheapest | Tie: Regex, ProtectAI | Regex $0 (local) per 1M checks; ProtectAI $0 (local) per 1M checks (exactly equal) |
| Most trustworthy confidence | Jev | Jev calibration error 0.192 (0 = perfect); next best ProtectAI calibration error 0.220 (0 = perfect) |
| Best on hard attacks | Tie: Luna, Regex, ProtectAI, Jev, Lakera | Luna catches 100.0% [100.0, 100.0] of hard attacks; Regex catches 91.7% [75.0, 100.0] of hard attacks; ProtectAI catches 83.3% [58.3, 100.0] of hard attacks; Jev catches 83.3% [58.3, 100.0] of hard attacks; Lakera catches 75.0% [50.0, 100.0] of hard attacks (intervals overlap, so no clear winner) |

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

1. **Dataset**: neuralchemy/Prompt-injection-dataset, 'core' config, validation split.
2. **Size**: test: 42 prompts per method (26 attacks, 16 safe, 4 tricky-but-safe); val: 42 prompts per method (26 attacks, 16 safe, 4 tricky-but-safe).
3. **Run date**: <DATE> (last modified time of the results file).
4. **Errors**: a failed call (timeout, refusal, API error) counts as **not flagged**, because a guardrail that fails open lets the attack through. So errors lower catch rate and never raise false blocks. ROC-AUC, calibration, the cut-off and latency use only calls that returned a result.
5. **Confidence intervals**: bootstrap, 1,000 resamples of the prompts, seed 0, 2.5th to 97.5th percentile.
6. **Latency** is measured on successful calls only. Cost is the mean of the calls that reported a cost.
7. **Warning**: the cut-off split equals the main split, so the 5% column is optimistic.
