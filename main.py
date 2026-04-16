"""
main.py — Entry point.

Usage:
    python main.py --level N [--train] [--submit]

Flags:
    --level N    Level to run (1-5, required)
    --train      Fit the Scorer on level N training data before running
    --submit     Write submission file after pipeline completes
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from config import LEVEL_CONFIG, DATA_DIR, SUBMISSIONS_DIR
from agents.orchestrator import OrchestratorAgent, check_drift, time_split
from agents.memory import MemoryAgent
from agents.scorer import ScorerAgent
from agents.context import ContextAgent
from agents.investigator import InvestigatorAgent
from agents.critic import CriticAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data(level: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and eval CSVs for a level."""
    level_dir = DATA_DIR / f"level_{level}"
    train_path = level_dir / "train.csv"
    eval_path = level_dir / "eval.csv"

    if not train_path.exists():
        logger.error("Missing: %s", train_path)
        sys.exit(1)
    if not eval_path.exists():
        logger.error("Missing: %s", eval_path)
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    logger.info("Loaded level %d: train=%d eval=%d", level, len(train_df), len(eval_df))
    return train_df, eval_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Reply Mirror fraud pipeline")
    parser.add_argument("--level", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--train", action="store_true", help="Train scorer before running")
    parser.add_argument("--submit", action="store_true", help="Write submission file")
    args = parser.parse_args()

    level = args.level
    cfg = LEVEL_CONFIG[level]
    logger.info("=== Level %d | budget=$%.2f | model=%s ===", level, cfg["budget"], cfg["llm_model"])

    # Load data
    train_df, eval_df = load_data(level)

    # Drift check vs prior level (skip for level 1)
    if level > 1:
        prev_train_df, _ = load_data(level - 1)
        drift = check_drift(prev_train_df, train_df)
        print("\n" + "=" * 60)
        print(f"DRIFT SUMMARY (L{level-1} → L{level}) — share with Role B")
        print("=" * 60)
        for k, v in drift.items():
            print(f"  {k}: {v}")
        print("=" * 60 + "\n")

    # Time-based split for internal validation
    train_split, val_split = time_split(train_df)

    # Initialise agents
    memory = MemoryAgent()
    scorer = ScorerAgent()
    context_agent = ContextAgent()
    investigator = InvestigatorAgent()
    critic = CriticAgent() if cfg["critic"] else None

    orchestrator = OrchestratorAgent(
        level=level,
        scorer=scorer,
        context_agent=context_agent,
        investigator=investigator,
        critic=critic,
        memory=memory,
    )

    # Optionally train scorer
    if args.train:
        logger.info("Training scorer on level %d train split (%d rows)…", level, len(train_split))
        scorer.train(train_split)

    # Run pipeline on eval set
    fraud_ids = orchestrator.run(eval_df)
    logger.info("Pipeline complete — %d fraud IDs flagged", len(fraud_ids))

    # Write submission
    if args.submit:
        output_path = orchestrator.submit(fraud_ids, eval_df)
        logger.info("Submission → %s", output_path)

        # Pre-submit checklist reminder
        print("\n" + "=" * 60)
        print("PRE-SUBMIT CHECKLIST (manual)")
        print("=" * 60)
        print(f"  [ ] Validator passed (done automatically)")
        print(f"  [ ] Line count looks reasonable: {len(fraud_ids)}")
        print(f"  [ ] Eyeball first/last 5 lines of {output_path}")
        print(f"  [ ] Teammates notified")
        print(f"  [ ] Cost tracker: {orchestrator.cost_tracker.summary()}")
        print("=" * 60 + "\n")
    else:
        logger.info("Dry run — use --submit to write submission file")


if __name__ == "__main__":
    main()
