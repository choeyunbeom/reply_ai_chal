"""tests/test_cost_tracker.py — CostTracker unit tests."""

import pytest
from utils.cost_tracker import CostTracker, BudgetExhausted


def make_tracker(budget: float = 10.0, throttle_at: float = 0.90) -> CostTracker:
    return CostTracker(level=1, budget=budget, throttle_at=throttle_at)


def test_initial_state():
    ct = make_tracker()
    assert ct.spent == 0.0
    assert ct.remaining == 10.0
    assert not ct.throttled
    assert not ct.exhausted


def test_spend_records_cost():
    ct = make_tracker()
    cost = ct.spend("gpt-4o-mini", input_tokens=1000, output_tokens=200)
    assert cost > 0
    assert ct.spent == pytest.approx(cost)


def test_throttle_activates_at_90pct():
    ct = make_tracker(budget=10.0, throttle_at=0.90)
    # spend just under 90%
    ct._spent = 8.99
    assert not ct.throttled
    ct._spent = 9.00
    assert ct.throttled


def test_exhausted_blocks_spend():
    ct = make_tracker(budget=0.001)
    # exhaust the budget
    ct._spent = 0.001
    assert ct.exhausted
    with pytest.raises(BudgetExhausted):
        ct.spend("gpt-4o-mini", input_tokens=100, output_tokens=100)


def test_remaining_never_negative():
    ct = make_tracker(budget=1.0)
    ct._spent = 999.0
    assert ct.remaining == 0.0


def test_summary_keys():
    ct = make_tracker()
    s = ct.summary()
    assert {"level", "budget", "spent", "remaining", "fraction_used_pct", "throttled", "exhausted"} == set(s)


def test_estimate_does_not_record():
    ct = make_tracker()
    ct.estimate("gpt-4o-mini", 1000, 500)
    assert ct.spent == 0.0
