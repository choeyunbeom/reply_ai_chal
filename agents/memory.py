"""
agents/memory.py — Memory Agent (Role C).

STUB — implementation owned by Role C.
Do NOT modify internals without flagging Role C.

Cross-level pattern store. Intentionally simple: a plain Python dict.
No vector DB, no semantic search, no pruning heuristics.

Two consumers:
  - Scorer (Role B): boolean feature injection via query()
  - Investigator (Role C): prose summary via as_prompt_context()

Interface (team contract — do not rename without flagging Role A):
    memory.update(tx: dict, verdict: dict) -> None
    memory.query(tx: dict) -> dict
        Returns at minimum:
            is_known_fraud_merchant: bool
            hour_fraud_count:        int
            amount_in_fraud_range:   bool
    memory.record_level_score(level: int, score: float) -> None
    memory.as_prompt_context() -> str

Suggested internal store (Role C may extend):
    known_fraud_merchants:   set[str]
    fraud_hour_distribution: dict[int, int]   # hour -> count
    fraud_amount_ranges:     list[tuple[float, float]]
    level_scores:            dict[int, float]
"""

import logging

logger = logging.getLogger(__name__)


class MemoryAgent:
    def update(self, tx: dict, verdict: dict) -> None:
        """
        Record a transaction and its verdict.
        Called by Orchestrator after every decision.
        Role C owns this method.
        """
        raise NotImplementedError("Role C: implement MemoryAgent.update()")

    def query(self, tx: dict) -> dict:
        """
        Return memory-derived feature flags for a single transaction.
        Called by Orchestrator before scoring (feature injection).
        Role C owns this method.

        Must return a dict containing at minimum:
            is_known_fraud_merchant: bool
            hour_fraud_count:        int
            amount_in_fraud_range:   bool
        """
        raise NotImplementedError("Role C: implement MemoryAgent.query()")

    def record_level_score(self, level: int, score: float) -> None:
        """Store internal validation score for a completed level."""
        raise NotImplementedError("Role C: implement MemoryAgent.record_level_score()")

    def as_prompt_context(self) -> str:
        """
        Return a human-readable summary for injection into Investigator prompts.
        Role C owns this method.
        """
        raise NotImplementedError("Role C: implement MemoryAgent.as_prompt_context()")
