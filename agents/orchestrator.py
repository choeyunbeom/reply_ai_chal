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

    Real dataset columns: timestamp, sender_id, recipient_id, amount,
    transaction_type, payment_method, location, balance_after, description.
    `hour` is derived in load_data() from timestamp.
    """
    hour_col = "hour" if "hour" in curr_tx.columns else None
    amount_col = "amount" if "amount" in curr_tx.columns else None
    loc_col = "location" if "location" in curr_tx.columns else None
    # recipient_id is the real column; merchant_id kept as fallback for older data
    merchant_col = (
        "recipient_id" if "recipient_id" in curr_tx.columns
        else "merchant_id" if "merchant_id" in curr_tx.columns
        else None
    )
    tx_type_col = "transaction_type" if "transaction_type" in curr_tx.columns else None
    pay_method_col = "payment_method" if "payment_method" in curr_tx.columns else None

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
        result["new_recipients"] = list(
            set(curr_tx[merchant_col].astype(str))
            - set(prev_tx[merchant_col].astype(str))
        )
    if tx_type_col:
        result["new_transaction_types"] = list(
            set(curr_tx[tx_type_col].astype(str)) - set(prev_tx[tx_type_col].astype(str))
        )
    if pay_method_col:
        result["new_payment_methods"] = list(
            set(curr_tx[pay_method_col].astype(str)) - set(prev_tx[pay_method_col].astype(str))
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
    When drift is detected, gray zone widens by ±(0.20 * drift_score).
    Drift widening is applied before throttle narrowing.
    """

    THROTTLE_MARGIN = 0.10
    DRIFT_MARGIN = 0.20

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
        self._drift_score: float = 0.0  # set via apply_drift()

    def apply_drift(self, drift_score: float) -> None:
        """
        Called once per run() with memory.drift_signal()["drift_score"].
        Widens gray zone proportionally to detected drift.
        """
        self._drift_score = float(np.clip(drift_score, 0.0, 1.0))
        logger.info(
            "fusion | drift_score=%.3f → gray zone base [%.2f, %.2f] → adjusted [%.2f, %.2f]",
            self._drift_score,
            self._gray_low, self._gray_high,
            self._gray_low - self.DRIFT_MARGIN * self._drift_score,
            self._gray_high + self.DRIFT_MARGIN * self._drift_score,
        )

    @property
    def gray_low(self) -> float:
        low = self._gray_low - self.DRIFT_MARGIN * self._drift_score
        if self.cost_tracker.throttled:
            low += self.THROTTLE_MARGIN
        return float(np.clip(low, 0.0, 0.5))

    @property
    def gray_high(self) -> float:
        high = self._gray_high + self.DRIFT_MARGIN * self._drift_score
        if self.cost_tracker.throttled:
            high -= self.THROTTLE_MARGIN
        return float(np.clip(high, 0.5, 1.0))

    def decide(
        self,
        tx: dict,
        tx_id: str,
        score: float,
        context_agent,
        investigator,
        critic,
        memory=None,
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
            verdict = investigator.judge(tx, ctx, memory_handle=memory)

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

        # Drift-aware gray zone adjustment (Role C proposal, agreed L1+)
        try:
            drift = self.memory.drift_signal()
            self.fusion.apply_drift(drift.get("drift_score", 0.0))
        except NotImplementedError:
            logger.debug("fusion | drift_signal() not implemented yet — skipping")

        # Memory feature injection → enrich before batch scoring
        # memory.query() returns patterns including "known_fraud_merchants" set.
        # ScorerAgent.update_memory_features() handles injection into the DataFrame.
        memory_patterns = self.memory.query({})  # batch-level patterns
        # ScorerAgent.update_memory_features() expects "known_fraud_merchants" set;
        # MemoryAgent exposes it via fraud_merchants.fraud_merchants (Role C internals).
        # We extract it safely here to avoid coupling Scorer to Memory internals.
        if hasattr(self.memory, "fraud_merchants"):
            memory_patterns.setdefault(
                "known_fraud_merchants",
                self.memory.fraud_merchants.fraud_merchants,
            )
        if hasattr(self.scorer, "update_memory_features"):
            enriched_df = self.scorer.update_memory_features(eval_df, memory_patterns)
        else:
            enriched_df = eval_df.copy()

        scores: np.ndarray = self.scorer.predict(enriched_df)

        # Context-aware score boost: if context_agent has phishing signals for a tx,
        # bump score into gray zone so LLM investigator handles it.
        # This compensates for heuristic scorer being blind to SMS/email signals.
        boosted_scores = scores.copy()
        if hasattr(self.context_agent, "build"):
            for idx, (_, row) in enumerate(eval_df.iterrows()):
                try:
                    ctx_quick = self.context_agent.build(str(row[tx_id_col]))
                    sms = ctx_quick.get("sms_fraud_signals", {})
                    phishing_hits = sms.get("phishing_hits", 0) or 0
                    _fk = sms.get("fraud_keywords", 0)
                    fraud_kw = len(_fk) if isinstance(_fk, list) else int(_fk)
                    if phishing_hits > 0 or fraud_kw >= 2:
                        # Push into gray zone minimum
                        boosted_scores[idx] = max(
                            boosted_scores[idx],
                            self.fusion.gray_low + 0.01,
                        )
                except Exception:
                    pass
        scores = boosted_scores

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
                memory=self.memory,
            )

            if verdict["fraud"]:
                fraud_ids.append(tx_id)

            verdict_str = "fraud" if verdict.get("fraud") else "legit"
            self.memory.update(tx, verdict_str)

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
