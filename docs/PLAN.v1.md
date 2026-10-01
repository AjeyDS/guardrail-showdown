# Jev Guardrail Benchmark — Project Plan

## Goal

Test whether Jev (TypeSafe AI's "System One" decision model) works better than a regex filter and an LLM judge as a guardrail for prompt injection, and whether its confidence scores can actually be trusted.

## Questions this project answers

1. Which guardrail catches the most attacks while wrongly blocking the fewest normal prompts?
2. How do they compare on speed and cost?
3. When Jev says "80% sure," is it right about 80% of the time?
4. Does Jev stay accurate on tricky, reworded attacks, or does it become confidently wrong?

## Scope (keep it small)

- One task: prompt injection detection
- Three methods: regex, Jev, LLM judge
- One dataset of about 500 to 1,000 prompts, plus 30 to 50 hand-written hard attacks
- Timeline: two weekends

---

## Weekend 1: Setup and baseline

### Step 1 — Get access
- [ ] Sign up for Jev early access, or use it through Cloudflare Workers AI
- [ ] Get an API key for one cheap LLM to use as the judge (for example Claude Haiku)
- [ ] Store keys in a `.env` file and add it to `.gitignore`

### Step 2 — Pick the data
- [ ] Choose one or two labeled public datasets from Hugging Face (options: `deepset/prompt-injections`, `jackhhao/jailbreak-classification`)
- [ ] Check each dataset's license
- [ ] Balance it so attacks and normal prompts are roughly equal
- [ ] Save a fixed copy in `data/` so results can be repeated

### Step 3 — Build the three guardrails
Each one is a small function with the same shape:
`check(prompt) -> {"unsafe": True/False, "confidence": 0.0 to 1.0}`

- [ ] **Regex** (`guardrails/regex.py`): a list of common attack phrases like "ignore previous instructions." Confidence is simply 1.0 on a match, 0.0 otherwise.
- [ ] **Jev** (`guardrails/jev.py`): one yes/no question, "Is this a prompt injection attempt?", using Jev's returned probability as the confidence
- [ ] **LLM judge** (`guardrails/llm_judge.py`): the same question as a prompt, asking for an answer plus a confidence number from 0 to 100

### Step 4 — Run the benchmark
- [ ] `run_benchmark.py` runs all three guardrails on every prompt
- [ ] Record for each prompt: true label, each method's answer, confidence, and time taken
- [ ] Save everything to `results/results.csv`

---

## Weekend 2: Analysis and write-up

### Step 5 — Compare the basics
For each method, measure:
- [ ] Accuracy
- [ ] False blocks: normal prompts wrongly blocked
- [ ] Misses: attacks let through
- [ ] Average response time
- [ ] Cost per 1,000 requests

### Step 6 — Check calibration
- [ ] Group answers by confidence (60–70%, 70–80%, 80–90%, 90–100%)
- [ ] For each group, check how often the answer was actually right
- [ ] Plot "stated confidence vs actual accuracy" for Jev and the LLM judge. A trustworthy model's points sit close to the diagonal line.

### Step 7 — Stress test
- [ ] Write 30 to 50 reworded attacks, for example:
  - Polite requests that hide an instruction
  - Instructions buried inside a story or a fake document
  - Attacks written in other languages
- [ ] Save them in `data/hard_attacks.csv`
- [ ] Run all three methods and check: does Jev's confidence drop on these, or does it stay confidently wrong?

### Step 8 — Write the README
- [ ] One short paragraph on what Jev is and why it was tested
- [ ] One results table
- [ ] The calibration chart
- [ ] Stress test findings
- [ ] An honest verdict: where each method wins and where it fails
- [ ] How to run the project yourself

---

## Repo structure

```
jev-guardrail-benchmark/
├── data/
│   ├── prompts.csv          # fixed copy of the public dataset
│   └── hard_attacks.csv     # your hand-written tricky attacks
├── guardrails/
│   ├── regex.py
│   ├── jev.py
│   └── llm_judge.py
├── results/
│   └── results.csv
├── run_benchmark.py         # runs all three, saves results
├── analysis.ipynb           # tables and charts
├── .env.example             # key names only, no real keys
├── requirements.txt
└── README.md                # findings
```

## Done means

- Anyone can clone the repo, add their keys, and reproduce the results
- The README answers all four questions at the top of this plan
- Results are reported fairly, including any cases where Jev does poorly
