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
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

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
    cfg = LEVEL_CONFIG[level]
    level_dir = DATA_DIR / f"level_{level}"
    train_path = level_dir / cfg["train_folder"] / "transactions.csv"
    eval_path = level_dir / cfg["eval_folder"] / "transactions.csv"

    if not train_path.exists():
        logger.error("Missing: %s", train_path)
        sys.exit(1)
    if not eval_path.exists():
        logger.error("Missing: %s", eval_path)
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    # Derive hour from timestamp (real dataset uses ISO timestamp, not a pre-split hour column)
    for df in (train_df, eval_df):
        if "timestamp" in df.columns and "hour" not in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour

    logger.info("Loaded level %d: train=%d eval=%d", level, len(train_df), len(eval_df))
    return train_df, eval_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Reply Mirror fraud pipeline")
    parser.add_argument("--level", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--train", action="store_true", help="(disabled: no labels in dataset)")
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
    level_dir = DATA_DIR / f"level_{level}"
    train_dir = level_dir / cfg["train_folder"]

    logger.info("langfuse | tracing enabled → %s", os.environ["LANGFUSE_HOST"])

    # LLM client + Langfuse tracing — official challenge tutorial method
    import ulid as _ulid
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    from langchain_openai import ChatOpenAI

    team_name = os.environ.get("TEAM_NAME", "reply-mirror")
    session_id = f"{team_name}-{_ulid.new().str}"
    logger.info("langfuse | session_id=%s", session_id)
    print(f"\n{'='*60}\nLANGFUSE SESSION ID: {session_id}\n{'='*60}\n")

    _langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ["LANGFUSE_HOST"],
    )
    _langfuse_handler = LangfuseCallbackHandler()

    llm_client = ChatOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        model=cfg["llm_model"],
        temperature=0.2,
        max_tokens=1200,
    )
    # Attach langfuse handler + session_id so investigator can tag LLM calls
    llm_client._langfuse_handler = _langfuse_handler
    llm_client._langfuse_session_id = session_id

    memory = MemoryAgent()
    scorer = ScorerAgent(users_path=train_dir / "users.json")
    context_agent = ContextAgent(
        transactions_path=train_dir / "transactions.csv",
        users_path=train_dir / "users.json",
        locations_path=train_dir / "locations.json",
        sms_path=train_dir / "sms.json",
    )
    investigator = InvestigatorAgent(llm_client=llm_client)
    investigator._model = cfg["llm_model"]
    critic = CriticAgent() if cfg["critic"] else None

    orchestrator = OrchestratorAgent(
        level=level,
        scorer=scorer,
        context_agent=context_agent,
        investigator=investigator,
        critic=critic,
        memory=memory,
    )

    # --train disabled: dataset has no fraud labels, heuristic scorer used instead
    if args.train:
        logger.warning("--train ignored: no is_fraud labels in dataset, using heuristic scorer")

    # Fit Isolation Forest on training data (no labels needed)
    logger.info("scorer | fitting IsolationForest on train data (%d rows)", len(train_df))
    scorer.fit(train_df)

    # Step 4: cache training baseline for drift detection, load hypotheses
    memory.cache_training_baseline(train_df)
    if level >= 2:
        memory.refresh_hypotheses(level)
    logger.info("memory | baseline cached, hypotheses=%d", len(memory.hypotheses()))

    # Level 3+: audio call recordings available for Role C (STT-based evidence)
    if cfg.get("audio_folder"):
        train_audio = train_dir / cfg["audio_folder"]
        logger.info("audio | train recordings: %s (%d files)", train_audio,
                    len(list(train_audio.glob("*.mp3"))) if train_audio.exists() else 0)

    # Reload context_agent with eval data so build(tx_id) works on eval set
    eval_dir = level_dir / cfg["eval_folder"]
    context_agent.reload(
        transactions_path=eval_dir / "transactions.csv",
        users_path=eval_dir / "users.json" if (eval_dir / "users.json").exists() else None,
        locations_path=eval_dir / "locations.json" if (eval_dir / "locations.json").exists() else None,
        sms_path=eval_dir / "sms.json" if (eval_dir / "sms.json").exists() else None,
    )
    logger.info("context | reloaded with eval data")

    # Rebuild user stats from eval data — eval users are different from train users.
    # IF model stays train-fitted; only the per-user baselines (mean/std/max) update
    # so that amount_zscore/amount_vs_user_max are computed against each eval user's
    # own history rather than cold-start defaults.
    logger.info("scorer | rebuilding user stats from eval data (%d rows)", len(eval_df))
    scorer._stats.build(eval_df)
    import json as _json
    eval_users_path = eval_dir / "users.json"
    if eval_users_path.exists():
        with open(eval_users_path) as _f:
            _eval_users = _json.load(_f)
        from agents.scorer import build_home_cities as _bhc
        scorer._home_cities = _bhc(eval_df, _eval_users)

    # Run pipeline on eval set
    fraud_ids = orchestrator.run(eval_df)
    _langfuse.flush()
    logger.info("Pipeline complete — %d fraud IDs flagged", len(fraud_ids))

    # Step 9: record internal validation score for next level
    # Role B should pass val_score after evaluating on val_split
    # memory.record_level_score(level, val_score)
    logger.info("memory | TODO: call memory.record_level_score(%d, val_score) after Role B eval", level)

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
