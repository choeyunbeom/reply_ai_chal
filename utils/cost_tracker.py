"""
utils/cost_tracker.py — Role A (critical safeguard).

Every LLM call must go through spend() before the call is made.
At 90% of level budget → Orchestrator narrows gray zone.
At 100% of level budget → LLM calls are blocked (raises BudgetExhausted).
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class BudgetExhausted(Exception):
    """Raised when spend() is called after the level budget is used up."""


# Approximate cost per 1K tokens (USD). Update as needed.
MODEL_COST_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
}


@dataclass
class CostTracker:
    level: int
    budget: float
    throttle_at: float = 0.90

    _spent: float = field(default=0.0, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self._spent)

    @property
    def fraction_used(self) -> float:
        return self._spent / self.budget if self.budget > 0 else 1.0

    @property
    def throttled(self) -> bool:
        """True when the Orchestrator should narrow the gray zone."""
        return self.fraction_used >= self.throttle_at

    @property
    def exhausted(self) -> bool:
        return self._spent >= self.budget

    def estimate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Return estimated cost in USD without recording it."""
        rates = MODEL_COST_PER_1K.get(model, {"input": 0.01, "output": 0.03})
        return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000

    def spend(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        label: str = "",
    ) -> float:
        """
        Record an LLM call. Raises BudgetExhausted if the budget is already gone.
        Returns the cost of this call.
        """
        if self.exhausted:
            raise BudgetExhausted(
                f"Level {self.level} budget ${self.budget:.2f} exhausted "
                f"(spent ${self._spent:.4f})"
            )

        cost = self.estimate(model, input_tokens, output_tokens)
        self._spent += cost

        logger.info(
            "cost_tracker | label=%s model=%s tokens_in=%d tokens_out=%d "
            "cost=$%.4f spent=$%.4f / $%.2f (%.1f%%)",
            label or "—",
            model,
            input_tokens,
            output_tokens,
            cost,
            self._spent,
            self.budget,
            self.fraction_used * 100,
        )

        if self.throttled:
            logger.warning(
                "cost_tracker | THROTTLE ACTIVE — %.0f%% of budget used. "
                "Orchestrator should narrow gray zone.",
                self.fraction_used * 100,
            )

        return cost

    def summary(self) -> dict:
        return {
            "level": self.level,
            "budget": self.budget,
            "spent": round(self._spent, 6),
            "remaining": round(self.remaining, 6),
            "fraction_used_pct": round(self.fraction_used * 100, 2),
            "throttled": self.throttled,
            "exhausted": self.exhausted,
        }
