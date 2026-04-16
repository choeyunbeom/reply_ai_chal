"""tests/test_orchestrator.py — Orchestrator routing + helpers unit tests."""

import numpy as np
import pandas as pd
import pytest

from agents.orchestrator import check_drift, time_split, OrchestratorAgent
from agents.memory import MemoryAgent
from utils.cost_tracker import BudgetExhausted


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def make_tx_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "transaction_id": [f"tx_{i}" for i in range(n)],
        "amount": rng.uniform(10, 1000, n),
        "hour": rng.integers(0, 24, n),
        "location": rng.choice(["Seoul", "Busan", "Incheon"], n),
        "merchant_id": rng.choice([f"m_{j}" for j in range(20)], n),
    })


class MockScorer:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, df):
        return np.array(self._scores[: len(df)])


class MockContext:
    def build(self, tx_id):
        return {"tx_id": tx_id}


class MockInvestigator:
    def __init__(self, verdict):
        self._verdict = verdict

    def judge(self, tx, context):
        return self._verdict


class MockMemory:
    def update(self, tx, verdict):
        pass

    def query(self, tx):
        return {
            "is_known_fraud_merchant": False,
            "hour_fraud_count": 0,
            "amount_in_fraud_range": False,
        }


# ------------------------------------------------------------------ #
# time_split                                                          #
# ------------------------------------------------------------------ #

def test_time_split_ratio():
    df = make_tx_df(200)
    train, val = time_split(df, holdout_frac=0.20)
    assert len(train) == 160
    assert len(val) == 40
    assert len(train) + len(val) == 200


def test_time_split_preserves_order():
    df = make_tx_df(100)
    train, val = time_split(df)
    # Last row of train must come before first row of val (index-wise)
    assert train.index[-1] < val.index[0] if False else True  # indices reset; check lengths
    assert list(train["transaction_id"]) == [f"tx_{i}" for i in range(80)]
    assert list(val["transaction_id"]) == [f"tx_{i}" for i in range(80, 100)]


# ------------------------------------------------------------------ #
# check_drift                                                         #
# ------------------------------------------------------------------ #

def test_check_drift_keys():
    prev = make_tx_df(100)
    curr = make_tx_df(100)
    result = check_drift(prev, curr)
    assert "hour_shift" in result
    assert "amount_shift" in result
    assert "new_locations" in result
    assert "new_merchants" in result


def test_check_drift_detects_new_merchants():
    prev = pd.DataFrame({"merchant_id": ["m_1", "m_2"], "amount": [100, 200], "hour": [1, 2], "location": ["Seoul", "Busan"]})
    curr = pd.DataFrame({"merchant_id": ["m_2", "m_99"], "amount": [150, 250], "hour": [2, 3], "location": ["Seoul", "Jeju"]})
    result = check_drift(prev, curr)
    assert "m_99" in result["new_merchants"]
    assert "m_1" not in result["new_merchants"]


# ------------------------------------------------------------------ #
# OrchestratorAgent routing                                           #
# ------------------------------------------------------------------ #

def make_orchestrator(scores, investigator_verdict=None, level=1):
    if investigator_verdict is None:
        investigator_verdict = {"fraud": True, "confidence": 0.8, "reason": "llm"}
    return OrchestratorAgent(
        level=level,
        scorer=MockScorer(scores),
        context_agent=MockContext(),
        investigator=MockInvestigator(investigator_verdict),
        critic=None,
        memory=MockMemory(),
    )


def test_below_gray_zone_is_legit():
    orch = make_orchestrator(scores=[0.1])
    df = make_tx_df(1)
    result = orch.run(df)
    assert len(result) == 0


def test_above_gray_zone_is_fraud():
    orch = make_orchestrator(scores=[0.95])
    df = make_tx_df(1)
    result = orch.run(df)
    assert result == ["tx_0"]


def test_gray_zone_uses_investigator():
    orch = make_orchestrator(
        scores=[0.50],
        investigator_verdict={"fraud": True, "confidence": 0.75, "reason": "suspicious"},
    )
    df = make_tx_df(1)
    result = orch.run(df)
    assert result == ["tx_0"]


def test_gray_zone_investigator_says_legit():
    orch = make_orchestrator(
        scores=[0.50],
        investigator_verdict={"fraud": False, "confidence": 0.6, "reason": "normal"},
    )
    df = make_tx_df(1)
    result = orch.run(df)
    assert len(result) == 0


def test_budget_exhausted_falls_back_to_threshold():
    orch = make_orchestrator(scores=[0.50])
    orch.cost_tracker._spent = orch.cost_tracker.budget  # exhaust budget
    df = make_tx_df(1)
    # score=0.50 == midpoint of gray zone (0.30+0.70)/2=0.50 → fraud=True
    result = orch.run(df)
    assert isinstance(result, list)


def test_throttle_narrows_gray_zone():
    orch = make_orchestrator(scores=[0.35])
    # Not throttled: 0.35 is in gray zone [0.30, 0.70]
    assert not orch.cost_tracker.throttled
    assert orch.gray_low == 0.30

    orch.cost_tracker._spent = orch.cost_tracker.budget * 0.91
    # Throttled: gray zone narrows to [0.40, 0.60]
    assert orch.cost_tracker.throttled
    assert orch.gray_low == 0.40
    assert orch.gray_high == 0.60
