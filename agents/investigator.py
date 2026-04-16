"""
agents/investigator.py — Investigator Agent (Role C).

STUB — implementation owned by Role C.
Do NOT modify internals without flagging Role C.

Interface (team contract):
    investigator.judge(tx: dict, context: dict) -> dict
        Returns {fraud: bool, confidence: float, reason: str}

Notes for Role C:
  - All LLM calls MUST go through the CostTracker passed in at construction.
  - The model to use is determined by LEVEL_CONFIG[level]["llm_model"].
  - Only called for gray-zone transactions (Orchestrator guarantees this).
  - Memory context (as_prompt_context()) is available via the memory agent;
    the Orchestrator will inject it into context if Role C requests it.
"""

import logging

logger = logging.getLogger(__name__)


class InvestigatorAgent:
    def judge(self, tx: dict, context: dict) -> dict:
        """
        LLM-based fraud judgment for a gray-zone transaction.
        Role C owns this method.

        Returns
        -------
        dict with keys:
            fraud:      bool
            confidence: float in [0, 1]
            reason:     str
        """
        raise NotImplementedError("Role C: implement InvestigatorAgent.judge()")
