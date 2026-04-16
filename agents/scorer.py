"""
agents/scorer.py — Scorer Agent (Role B).

STUB — implementation owned by Role B.
Do NOT modify internals without flagging Role B.

Interface (team contract):
    scorer.predict(tx_df: pd.DataFrame) -> np.ndarray
        Returns risk scores in [0, 1] for each row in tx_df.

Notes for Role B:
  - Retrain on current level's training data each level (do not reuse prior models).
  - tx_df may include memory-derived columns injected by the Orchestrator:
      is_known_fraud_merchant (bool)
      hour_fraud_count (int)
      amount_in_fraud_range (bool)
  - Use time_split() from orchestrator for validation, never random split.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScorerAgent:
    def __init__(self) -> None:
        self._model = None  # Role B: replace with trained LightGBM model

    def train(self, train_df: pd.DataFrame, label_col: str = "label") -> None:
        """
        Train the LightGBM scorer on training data.
        Role B owns this method.
        """
        raise NotImplementedError("Role B: implement ScorerAgent.train()")

    def predict(self, tx_df: pd.DataFrame) -> np.ndarray:
        """
        Return risk scores in [0, 1] for every row in tx_df.
        Role B owns this method.
        """
        raise NotImplementedError("Role B: implement ScorerAgent.predict()")
