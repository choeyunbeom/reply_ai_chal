"""
agents/context.py — Context Agent (Role B).

STUB — implementation owned by Role B.
Do NOT modify internals without flagging Role B.

Interface (team contract):
    context.build(tx_id: str) -> dict
        Returns an evidence bundle for the Investigator / Critic LLM.

Expected keys in the returned dict (Role B decides final schema):
    user_history:   list of recent transactions for this user
    location_match: bool — GPS/location consistency
    sms_email_flag: bool — any SMS/email evidence of fraud
    merchant_info:  dict — merchant metadata
    (+ any additional evidence Role B deems useful)
"""

import logging

logger = logging.getLogger(__name__)


class ContextAgent:
    def build(self, tx_id: str) -> dict:
        """
        Assemble evidence bundle for a transaction.
        Role B owns this method.
        """
        raise NotImplementedError("Role B: implement ContextAgent.build()")
