"""
agents/orchestrator.py — Orchestrator + Decision Fusion (Role A).

Pipeline flow:
  score → DecisionFusion → [gray zone] → context → investigator → [critic] → final verdict

Role A owns this file entirely (L1-5).
Do NOT modify without flagging Role A.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import LEVEL_CONFIG, SUBMISSIONS_DIR
from utils.cost_tracker import CostTracker, BudgetExhausted
from utils.validator import validate, write_submission

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def check_drift(prev_tx: pd.DataFrame, curr_tx: pd.DataFrame) -> dict:
    """
    Compute distribution shift between two levels.
    Pass the returned dict to Role B for feature tweaks before retraining.
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

    logger.info("drift | %s", result)
    return result


def time_split(df: pd.DataFrame, holdout_frac: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split: train = first 80%, val = last 20%.
    NEVER use random splits — they hide drift effects.
    """
    cutoff = int(len(df) * (1 - holdout_frac))
    train = df.iloc[:cutoff].reset_index(drop=True)
    val = df.iloc[cutoff:].reset_index(drop=True)
    logger.info("time_split | train=%d val=%d", len(train), len(val))
    return train, val


# ------------------------------------------------------------------ #
# Decision Fusion                                                      #
# ------------------------------------------------------------------ #

class DecisionFusion:
    """
    Combines scorer output and LLM verdicts into a final fraud decision.

    Routing logic:
      score < gray_low  → legit  (no LLM call)
      score > gray_high → fraud  (no LLM call)
      else              → pass to investigator (and critic if enabled)

    When budget is exhausted, falls back to scorer midpoint threshold.
    When budget is throttled (≥90%), gray zone narrows by ±0.10.
    """

    THROTTLE_MARGIN = 0.10

    def __init__(
        self,
        gray_low: float,
        gray_high: float,
        use_critic: bool,
        cost_tracker: CostTracker,
    ) -> None:
        self._gray_low = gray_low
        self._gray_high = gray_high
        self._use_critic = use_critic
        self.cost_tracker = cost_tracker

    @property
    def gray_low(self) -> float:
        if self.cost_tracker.throttled:
            return self._gray_low + self.THROTTLE_MARGIN
        return self._gray_low

    @property
    def gray_high(self) -> float:
        if self.cost_tracker.throttled:
            return self._gray_high - self.THROTTLE_MARGIN
        return self._gray_high

    def decide(
        self,
        tx: dict,
        tx_id: str,
        score: float,
        context_agent,
        investigator,
        critic,
    ) -> dict:
        """
        Return a verdict dict: {fraud: bool, confidence: float, reason: str}
        """
        if score < self.gray_low:
            return {"fraud": False, "confidence": 1 - score, "reason": "below_gray_zone"}

        if score > self.gray_high:
            return {"fraud": True, "confidence": score, "reason": "above_gray_zone"}

        # Gray zone → LLM path
        try:
            ctx = context_agent.build(tx_id)
            verdict = investigator.judge(tx, ctx)

            if self._use_critic and critic is not None:
                critic_result = critic.verify(verdict, ctx)
                if not critic_result.get("agree"):
                    logger.info(
                        "fusion | critic disagreed tx=%s reason=%s",
                        tx_id,
                        critic_result.get("reason"),
                    )
                    verdict = {
                        "fraud": False,
                        "confidence": 0.5,
                        "reason": f"critic_override: {critic_result.get('reason')}",
                    }

            return verdict

        except BudgetExhausted:
            logger.warning(
                "fusion | budget exhausted, scorer fallback tx=%s score=%.3f",
                tx_id, score,
            )
            midpoint = (self.gray_low + self.gray_high) / 2
            return {
                "fraud": score >= midpoint,
                "confidence": score,
                "reason": "budget_exhausted_fallback",
            }


# ------------------------------------------------------------------ #
# Orchestrator                                                         #
# ------------------------------------------------------------------ #

class OrchestratorAgent:
    """
    Top-level pipeline coordinator (Role A).

    Wires together: memory → scorer → decision fusion → submission.
    Does NOT own Scorer, Context, Investigator, Memory, or Critic internals.
    """

    def __init__(
        self,
        level: int,
        scorer,         # Role B — agents.scorer.ScorerAgent
        context_agent,  # Role B — agents.context.ContextAgent
        investigator,   # Role C — agents.investigator.InvestigatorAgent
        critic,         # Role C — agents.critic.CriticAgent (None for L1-3)
        memory,         # Role C — agents.memory.MemoryAgent
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

        self.fusion = DecisionFusion(
            gray_low=cfg["gray_low"],
            gray_high=cfg["gray_high"],
            use_critic=cfg["critic"],
            cost_tracker=self.cost_tracker,
        )

    # Expose gray zone for tests / logging
    @property
    def gray_low(self) -> float:
        return self.fusion.gray_low

    @property
    def gray_high(self) -> float:
        return self.fusion.gray_high

    # ------------------------------------------------------------------ #
    # Main pipeline                                                        #
    # ------------------------------------------------------------------ #

    def run(self, eval_df: pd.DataFrame) -> list[str]:
        """
        Run the full pipeline on eval_df. Returns list of fraud transaction IDs.
        """
        tx_id_col = "transaction_id"
        logger.info("orchestrator | level=%d tx=%d", self.level, len(eval_df))

        # Memory feature injection → enrich before batch scoring
        memory_features = eval_df.apply(
            lambda row: self.memory.query(row.to_dict()), axis=1
        )
        enriched_df = pd.concat(
            [eval_df.reset_index(drop=True), pd.DataFrame(list(memory_features))],
            axis=1,
        )

        scores: np.ndarray = self.scorer.predict(enriched_df)

        fraud_ids: list[str] = []

        for idx, (_, row) in enumerate(eval_df.iterrows()):
            tx_id = str(row[tx_id_col])
            tx = row.to_dict()
            score = float(scores[idx])

            verdict = self.fusion.decide(
                tx=tx,
                tx_id=tx_id,
                score=score,
                context_agent=self.context_agent,
                investigator=self.investigator,
                critic=self.critic,
            )

            if verdict["fraud"]:
                fraud_ids.append(tx_id)

            self.memory.update(tx, verdict)

        logger.info(
            "orchestrator | done fraud=%d cost=%s",
            len(fraud_ids),
            self.cost_tracker.summary(),
        )
        return fraud_ids

    # ------------------------------------------------------------------ #
    # Submission                                                           #
    # ------------------------------------------------------------------ #

    def submit(self, fraud_ids: list[str], eval_df: pd.DataFrame) -> Path:
        """
        Validate then write submission. ALWAYS use this — never write directly.
        """
        validate(fraud_ids, eval_df)
        output_path = SUBMISSIONS_DIR / f"level_{self.level}.txt"
        write_submission(fraud_ids, output_path)
        logger.info("orchestrator | submission → %s (%d IDs)", output_path, len(fraud_ids))
        return output_path
