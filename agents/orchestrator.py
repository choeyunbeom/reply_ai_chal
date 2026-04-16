"""
agents/orchestrator.py — Orchestrator Agent (Role A).

Routes each transaction through the pipeline:
  score → [gray zone] → context → investigator → [critic] → decision

Also owns:
  - cost-aware gray zone narrowing when budget is throttled
  - time-based train/eval split (last 20%)
  - drift check vs prior level
  - final submission assembly
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import LEVEL_CONFIG, DATA_DIR, SUBMISSIONS_DIR
from utils.cost_tracker import CostTracker, BudgetExhausted
from utils.validator import validate, write_submission
from agents.memory import MemoryAgent

logger = logging.getLogger(__name__)


def check_drift(prev_tx: pd.DataFrame, curr_tx: pd.DataFrame) -> dict:
    """
    Compute distribution shift between two levels.
    Pass the result to Role B for feature tweaks.
    """
    hour_col = "hour" if "hour" in curr_tx.columns else None
    amount_col = "amount" if "amount" in curr_tx.columns else None
    loc_col = "location" if "location" in curr_tx.columns else None
    merchant_col = (
        "merchant_id" if "merchant_id" in curr_tx.columns
        else "recipient_id" if "recipient_id" in curr_tx.columns
        else None
    )

    result: dict = {}

    if hour_col:
        result["hour_shift"] = float(
            abs(prev_tx[hour_col].mean() - curr_tx[hour_col].mean())
        )
    if amount_col:
        result["amount_shift"] = float(
            abs(prev_tx[amount_col].median() - curr_tx[amount_col].median())
        )
    if loc_col:
        result["new_locations"] = list(
            set(curr_tx[loc_col].astype(str)) - set(prev_tx[loc_col].astype(str))
        )
    if merchant_col:
        result["new_merchants"] = list(
            set(curr_tx[merchant_col].astype(str))
            - set(prev_tx[merchant_col].astype(str))
        )

    logger.info("orchestrator | drift check: %s", result)
    return result


def time_split(df: pd.DataFrame, holdout_frac: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split: train = first 80%, val = last 20%.
    NEVER use random splits — they hide drift effects.
    """
    n = len(df)
    cutoff = int(n * (1 - holdout_frac))
    train = df.iloc[:cutoff].reset_index(drop=True)
    val = df.iloc[cutoff:].reset_index(drop=True)
    logger.info("orchestrator | time_split: train=%d val=%d", len(train), len(val))
    return train, val


class OrchestratorAgent:
    def __init__(
        self,
        level: int,
        scorer,         # Role B — agents.scorer.ScorerAgent
        context_agent,  # Role B — agents.context.ContextAgent
        investigator,   # Role C — agents.investigator.InvestigatorAgent
        critic,         # Role C — agents.critic.CriticAgent (may be None)
        memory: MemoryAgent,
    ) -> None:
        self.level = level
        cfg = LEVEL_CONFIG[level]

        self.scorer = scorer
        self.context_agent = context_agent
        self.investigator = investigator
        self.critic = critic
        self.memory = memory

        self.cost_tracker = CostTracker(
            level=level,
            budget=cfg["budget"],
            throttle_at=cfg["throttle_at"],
        )

        self._gray_low: float = cfg["gray_low"]
        self._gray_high: float = cfg["gray_high"]
        self._model: str = cfg["llm_model"]
        self._use_critic: bool = cfg["critic"]

    # ------------------------------------------------------------------ #
    # Gray zone boundaries (throttle-aware)                               #
    # ------------------------------------------------------------------ #

    @property
    def gray_low(self) -> float:
        if self.cost_tracker.throttled:
            # Shrink gray zone when budget is tight → fewer LLM calls
            return self._gray_low + 0.10
        return self._gray_low

    @property
    def gray_high(self) -> float:
        if self.cost_tracker.throttled:
            return self._gray_high - 0.10
        return self._gray_high

    # ------------------------------------------------------------------ #
    # Main pipeline                                                        #
    # ------------------------------------------------------------------ #

    def run(self, eval_df: pd.DataFrame) -> list[str]:
        """
        Process all transactions in eval_df and return a list of fraud IDs.

        Parameters
        ----------
        eval_df:
            Evaluation dataset for this level (no labels).
        """
        tx_id_col = "transaction_id"
        logger.info(
            "orchestrator | level=%d transactions=%d", self.level, len(eval_df)
        )

        # Inject memory-derived features before scoring
        memory_features = eval_df.apply(
            lambda row: self.memory.query(row.to_dict()), axis=1
        )
        memory_df = pd.DataFrame(list(memory_features))
        enriched_df = pd.concat(
            [eval_df.reset_index(drop=True), memory_df], axis=1
        )

        # Batch score all transactions
        scores: np.ndarray = self.scorer.predict(enriched_df)

        fraud_ids: list[str] = []

        for idx, (_, row) in enumerate(eval_df.iterrows()):
            tx_id = str(row[tx_id_col])
            score = float(scores[idx])
            tx = row.to_dict()

            verdict = self._route(tx, tx_id, score)
            if verdict["fraud"]:
                fraud_ids.append(tx_id)

            # Update memory with this decision
            self.memory.update(tx, verdict)

        logger.info(
            "orchestrator | done — fraud_ids=%d  cost=%s",
            len(fraud_ids),
            self.cost_tracker.summary(),
        )
        return fraud_ids

    def _route(self, tx: dict, tx_id: str, score: float) -> dict:
        """
        Apply routing logic for a single transaction.
        Returns a verdict dict: {fraud, confidence, reason}
        """
        if score < self.gray_low:
            return {"fraud": False, "confidence": 1 - score, "reason": "below_gray_zone"}

        if score > self.gray_high:
            return {"fraud": True, "confidence": score, "reason": "above_gray_zone"}

        # Gray zone — call LLM agents
        try:
            ctx = self.context_agent.build(tx_id)
            verdict = self.investigator.judge(tx, ctx)

            if self._use_critic and self.critic is not None:
                critic_result = self.critic.verify(verdict, ctx)
                if not critic_result.get("agree"):
                    logger.info(
                        "orchestrator | critic disagreed on tx=%s: %s",
                        tx_id,
                        critic_result.get("reason"),
                    )
                    # Critic overrides — flip to safe side (not fraud)
                    verdict = {
                        "fraud": False,
                        "confidence": 0.5,
                        "reason": f"critic_override: {critic_result.get('reason')}",
                    }

            return verdict

        except BudgetExhausted:
            logger.warning(
                "orchestrator | budget exhausted, falling back to scorer for tx=%s score=%.3f",
                tx_id,
                score,
            )
            # Fallback: treat gray zone midpoint as threshold
            fraud = score >= (self.gray_low + self.gray_high) / 2
            return {
                "fraud": fraud,
                "confidence": score,
                "reason": "budget_exhausted_fallback",
            }

    # ------------------------------------------------------------------ #
    # Submission                                                           #
    # ------------------------------------------------------------------ #

    def submit(self, fraud_ids: list[str], eval_df: pd.DataFrame) -> Path:
        """
        Validate and write the submission file.
        ALWAYS call this instead of writing directly.
        """
        validate(fraud_ids, eval_df)

        output_path = SUBMISSIONS_DIR / f"level_{self.level}.txt"
        write_submission(fraud_ids, output_path)

        logger.info(
            "orchestrator | submission written → %s  (%d fraud IDs)",
            output_path,
            len(fraud_ids),
        )
        return output_path
