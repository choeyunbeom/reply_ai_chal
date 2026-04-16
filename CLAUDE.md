# CLAUDE.md — Reply Mirror

Context for Claude Code when working on this repo. Read this before making changes.

## What this is

A fraud detection system for the **Reply Mirror** competition (16 April 2026). We have **6 hours total** to submit across **5 levels** of increasing difficulty, with a total **$160 LLM budget** split per level. Each level provides a training dataset and an evaluation dataset; **only the first submission per level counts**.

The system must be **agent-based** (deterministic-only approaches are penalized) and must detect fraud patterns that **drift across levels** — new merchants, shifted time windows, new geographies, new amount patterns, new behavioral sequences.

## My role (the operator working with you)

I own **Role A — Pipeline & Orchestration**. My agents are the **Orchestrator** and the **Memory Agent**. My teammates own:

- **Role B** — Scorer (LightGBM) and Context agents
- **Role C** — Investigator (LLM) and Critic (L4-L5 only) agents

When I ask you to work on Scorer/Context/Investigator/Critic internals, **stop and confirm** — those are owned by B and C. You can edit interfaces, stubs, and integration glue freely.

## Architecture

Five cooperating agents. Same codebase for all 5 levels; only `LEVEL_CONFIG` changes behavior.

```
Transaction stream
    ↓
Orchestrator Agent ── routes, aggregates, decides
    ├─→ Scorer Agent (LightGBM risk score, fast, cheap)
    ├─→ Context Agent (user history, GPS, SMS/email evidence)
    └─→ Investigator Agent (LLM, gray-zone only)
           └─→ Critic Agent (L4-L5 only, verifies Investigator)
    ↓
Decision Fusion ── threshold + cost-aware
    ↓
Fraud IDs output
    ↕
Memory Agent ── cross-level pattern store
```

**Routing logic:**
```python
score = scorer.predict(tx)
if score < LOW:    verdict = legit
elif score > HIGH: verdict = fraud
else:
    ctx = context.build(tx)
    verdict = investigator.judge(tx, ctx)
    if LEVEL_CONFIG[level]["critic"]:
        verdict = critic.verify(verdict, ctx)
memory.update(tx, verdict)
```

## Level configuration

| Level | Budget | Gray zone | LLM model | Critic |
|-------|--------|-----------|-----------|--------|
| 1 | $8 | 0.30-0.70 | gpt-4o-mini | off |
| 2 | $12 | 0.30-0.70 | gpt-4o-mini | off |
| 3 | $15 | 0.25-0.75 | gpt-4o-mini | off |
| 4 | $55 | 0.15-0.85 | claude-haiku-4-5 | on |
| 5 | $60 | 0.10-0.90 | claude-sonnet-4-6 | on |

Levels 1-3 prioritize cost efficiency (ML does most of the work). Levels 4-5 widen the gray zone and add the Critic because drift becomes severe.

## Shared interfaces (team contract — do not change without flagging)

```python
scorer.predict(tx_df) -> np.ndarray          # risk scores in [0, 1]
context.build(tx_id) -> dict                  # evidence bundle for LLM
investigator.judge(tx, context) -> dict       # {fraud, confidence, reason}
critic.verify(verdict, context) -> dict       # {agree, reason}
memory.update(tx, verdict) -> None
memory.query(tx) -> dict                      # prior-level patterns
```

## Repo layout

```
reply-mirror/
├── main.py                  # entry: python main.py --level N
├── config.py                # LEVEL_CONFIG table
├── agents/
│   ├── orchestrator.py      # Role A
│   ├── memory.py            # Role A
│   ├── scorer.py            # Role B
│   ├── context.py           # Role B
│   ├── investigator.py      # Role C
│   └── critic.py            # Role C
├── utils/
│   ├── cost_tracker.py      # Role A — critical
│   └── validator.py         # Role A — critical
├── prompts/                 # Role C
├── data/level_{N}/
├── submissions/
└── requirements.txt
```

## Submission format

- ASCII text file, one suspected fraudulent `transaction_id` per line, newline-separated
- Filename: `submissions/level_{N}.txt`

**Invalid conditions (auto-reject):**
- Empty file
- All transactions flagged
- Fewer than 15% of actual fraudulent transactions identified

The first two are checkable; the third is not (we don't have ground truth at submission time). Keep flagging rates reasonable; aim for recall well above the threshold.

## Critical safeguards — do not remove or weaken

These exist because mistakes are unrecoverable (first submission per level is final).

1. **Cost tracker with auto-stop.** Every LLM call must go through `utils/cost_tracker.py`. At 90% of level budget, the Orchestrator narrows the gray zone. At 100%, LLM calls are blocked and remaining gray-zone cases fall back to Scorer threshold.

2. **Submission validator.** `utils/validator.py` must be called before writing the output file. It rejects empty / all-flagged / obviously broken submissions.

3. **Time-based validation split.** Hold out the last 20% of each level's training data. Never use random splits — they hide drift effects.

4. **Pre-submit checklist** (manual, human-in-the-loop):
   - Validator passed
   - Line count looks reasonable (not suspiciously round or tiny)
   - First and last 5 lines eyeballed
   - Teammates notified before pressing submit

## Per-level workflow (my routine)

```
1. Load Level N data (2 min)
2. Drift check vs Level N-1 (hour/amount/location distribution shifts)
3. Pass drift summary to Role B for feature tweaks
4. Pull relevant patterns from Memory, feed to Scorer
5. Run pipeline with live cost tracker
6. Validate submission
7. Pre-submit checklist with team
8. Submit
9. Memory.update() with this level's patterns for next level
```

## Drift handling

At the start of each level, run:

```python
def check_drift(prev_tx, curr_tx):
    return {
        "hour_shift": abs(prev_tx.hour.mean() - curr_tx.hour.mean()),
        "amount_shift": abs(prev_tx.amount.median() - curr_tx.amount.median()),
        "new_locations": set(curr_tx.location) - set(prev_tx.location),
        "new_merchants": set(curr_tx.recipient_id) - set(prev_tx.recipient_id),
    }
```

Pass this to Role B. Scorer should **retrain on current level's training data**, not reuse prior models.

## Memory Agent scope — keep simple

A Python dict is sufficient. No vector DB, no semantic search, no pruning heuristics in this competition. Just:

```python
memory = {
    "known_fraud_merchants": set(),
    "fraud_hour_distribution": dict,   # hour -> count
    "fraud_amount_ranges": list,
    "level_scores": dict,              # level -> our internal validation score
}
```

Memory is used two ways:
- **Feature injection into Scorer**: `is_known_fraud_merchant` boolean, etc.
- **Prompt context for Investigator**: "in prior levels, fraud concentrated around 2-4am"

## Coding conventions

- Python 3.11+, type hints where they clarify
- `pathlib.Path` for all file paths
- `logging` (not print) — cost tracker and drift info must be logged
- Small pure functions; agents are stateless classes where possible
- No hardcoded paths; use `config.py` for everything variable across levels

## Dependencies

Core: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `openai`, `anthropic`
Keep `requirements.txt` minimal — reviewers reproduce our runs.

## What to do when I ask for help

- **Default to editing Role A code** (Orchestrator, Memory, utils)
- **For Role B or C internals**, propose the change but flag ownership before committing
- **For interface changes**, draft both sides (caller + callee) and highlight the diff so I can circulate to the team
- **Never relax submission guards** (validator, cost tracker, 15% floor logic) without explicit instruction
- **Time pressure is real** — prefer minimal working changes over refactors during the competition

## Things to actively avoid

- Random train/test splits (breaks drift evaluation)
- LLM calls outside the cost tracker
- Storing full transaction text in Memory (too large, unnecessary)
- Over-engineered agent frameworks (LangGraph, AutoGen) — plain Python classes are enough for 6 hours
- Rewriting Scorer or Investigator internals without flagging Role B / Role C
- Calling the submit endpoint without running the validator first
