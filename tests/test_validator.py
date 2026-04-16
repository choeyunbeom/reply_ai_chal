"""tests/test_validator.py — validator unit tests."""

import pytest
import pandas as pd
from utils.validator import validate, SubmissionError


def make_eval(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({"transaction_id": [f"tx_{i}" for i in range(n)]})


def test_valid_submission():
    eval_df = make_eval(100)
    fraud_ids = [f"tx_{i}" for i in range(10)]
    validate(fraud_ids, eval_df)  # should not raise


def test_empty_submission_raises():
    eval_df = make_eval(100)
    with pytest.raises(SubmissionError, match="empty"):
        validate([], eval_df)


def test_all_flagged_raises():
    eval_df = make_eval(100)
    fraud_ids = [f"tx_{i}" for i in range(100)]
    with pytest.raises(SubmissionError, match="flags"):
        validate(fraud_ids, eval_df)


def test_duplicate_ids_raises():
    eval_df = make_eval(100)
    fraud_ids = ["tx_0", "tx_0", "tx_1"]
    with pytest.raises(SubmissionError, match="duplicate"):
        validate(fraud_ids, eval_df)


def test_unknown_ids_raises():
    eval_df = make_eval(100)
    fraud_ids = ["tx_999", "tx_1000"]
    with pytest.raises(SubmissionError, match="not found"):
        validate(fraud_ids, eval_df)


def test_empty_eval_raises():
    eval_df = make_eval(0)
    with pytest.raises(SubmissionError, match="empty"):
        validate(["tx_0"], eval_df)
