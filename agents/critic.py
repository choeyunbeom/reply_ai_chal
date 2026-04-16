"""
agents/critic.py — Critic Agent (Role C, L4-L5 only).

STUB — implementation owned by Role C.
Do NOT modify internals without flagging Role C.

Interface (team contract):
    critic.verify(verdict: dict, context: dict) -> dict
        Returns {agree: bool, reason: str}

Notes for Role C:
  - Only active when LEVEL_CONFIG[level]["critic"] is True (levels 4 and 5).
  - All LLM calls MUST go through the CostTracker.
  - If critic disagrees, the Orchestrator flips the verdict to not-fraud
    (safe-side fallback). Role C may propose a different override strategy
    but must flag Role A before changing orchestrator.py.
"""

import logging

logger = logging.getLogger(__name__)


class CriticAgent:
    def verify(self, verdict: dict, context: dict) -> dict:
        """
        Verify the Investigator's verdict.
        Role C owns this method.

        Returns
        -------
        dict with keys:
            agree:  bool
            reason: str
        """
        raise NotImplementedError("Role C: implement CriticAgent.verify()")
