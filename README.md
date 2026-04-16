# Reply Mirror — Fraud Detection Pipeline

Agent-based fraud detection system for the Reply Mirror challenge.

## Architecture

Five cooperating agents:
- **Orchestrator** — routes transactions, aggregates decisions
- **Scorer** — Isolation Forest anomaly scorer (unsupervised, no labels needed)
- **Context** — builds evidence bundle (GPS, SMS, user history)
- **Investigator** — LLM-based gray-zone judge (Popperian falsification + Bayesian update)
- **Memory** — cross-level pattern store
- **Critic** — LLM verifier (Level 4-5 only)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your keys
```

Required environment variables (in `.env`):
```
OPENROUTER_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://challenges.reply.com/langfuse
```

## Running

```bash
# Dry run (no output file)
python main.py --level 1

# Generate submission file
python main.py --level 1 --submit
```

Submission file is written to `submissions/level_1.txt`.

## Data layout

```
data/
  level_1/
    The Truman Show - train/
      transactions.csv
      users.json
      locations.json
      sms.json
    The Truman Show - validation/
      transactions.csv
      users.json
      locations.json
      sms.json
  level_2/ ...
  level_3/ ...
  level_4/ ...
  level_5/ ...
```

## Dependencies

See `requirements.txt`. Key libraries: `pandas`, `numpy`, `scikit-learn`, `openai`, `langfuse`, `python-dotenv`.
