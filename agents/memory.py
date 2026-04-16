"""
agents/memory.py — Memory Agent (Role A).

Cross-level pattern store. Intentionally simple: a plain Python dict.
No vector DB, no semantic search, no pruning heuristics.

Two consumers:
  - Scorer: boolean feature injection (is_known_fraud_merchant, etc.)
  - Investigator: prose summary injected into prompt context
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryAgent:
    # Known fraud merchants accumulated across levels
    known_fraud_merchants: set[str] = field(default_factory=set)

    # hour (0-23) -> count of fraud transactions seen at that hour
    fraud_hour_distribution: dict[int, int] = field(default_factory=dict)

    # list of (low, high) amount ranges observed in fraud
    fraud_amount_ranges: list[tuple[float, float]] = field(default_factory=list)

    # level -> internal validation score we achieved
    level_scores: dict[int, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Interface (team contract — do not rename without flagging)           #
    # ------------------------------------------------------------------ #

    def update(self, tx: dict, verdict: dict) -> None:
        """
        Record a transaction and its verdict into memory.

        Parameters
        ----------
        tx:
            Transaction dict with at minimum:
            - transaction_id, merchant_id (or recipient_id), amount, hour
        verdict:
            {fraud: bool, confidence: float, reason: str}
        """
        if not verdict.get("fraud"):
            return

        merchant = tx.get("merchant_id") or tx.get("recipient_id")
        if merchant:
            self.known_fraud_merchants.add(str(merchant))

        hour = tx.get("hour")
        if hour is not None:
            self.fraud_hour_distribution[int(hour)] = (
                self.fraud_hour_distribution.get(int(hour), 0) + 1
            )

        amount = tx.get("amount")
        if amount is not None:
            self.fraud_amount_ranges.append((float(amount), float(amount)))

        logger.debug("memory | updated with fraud tx=%s", tx.get("transaction_id"))

    def query(self, tx: dict) -> dict:
        """
        Return a dict of memory-derived features for a single transaction.

        Used by:
          - Scorer: feature injection (boolean / numeric flags)
          - Investigator: context summary (call as_prompt_context())
        """
        merchant = tx.get("merchant_id") or tx.get("recipient_id")
        hour = tx.get("hour")
        amount = tx.get("amount")

        is_known_fraud_merchant = (
            str(merchant) in self.known_fraud_merchants if merchant else False
        )

        hour_fraud_count = self.fraud_hour_distribution.get(int(hour), 0) if hour is not None else 0

        amount_in_fraud_range = False
        if amount is not None and self.fraud_amount_ranges:
            amount_in_fraud_range = any(
                lo * 0.8 <= float(amount) <= hi * 1.2
                for lo, hi in self.fraud_amount_ranges
            )

        return {
            "is_known_fraud_merchant": is_known_fraud_merchant,
            "hour_fraud_count": hour_fraud_count,
            "amount_in_fraud_range": amount_in_fraud_range,
        }

    def record_level_score(self, level: int, score: float) -> None:
        self.level_scores[level] = score
        logger.info("memory | level %d internal score = %.4f", level, score)

    def as_prompt_context(self) -> str:
        """
        Return a human-readable summary for injection into Investigator prompts.
        """
        top_hours = sorted(
            self.fraud_hour_distribution.items(), key=lambda kv: -kv[1]
        )[:5]
        hour_str = ", ".join(f"{h}:00 ({c}x)" for h, c in top_hours) or "none"

        merchants_sample = list(self.known_fraud_merchants)[:10]
        merchant_str = ", ".join(merchants_sample) or "none"

        return (
            f"Prior-level fraud patterns:\n"
            f"- Known fraud merchants (sample): {merchant_str}\n"
            f"- Peak fraud hours: {hour_str}\n"
            f"- Amount ranges flagged: {len(self.fraud_amount_ranges)} prior instances\n"
            f"- Level scores: {self.level_scores}"
        )
