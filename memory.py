"""
Memory Agent for Reply Mirror fraud detection system.

Four internal components operating at three timescales:
    - FraudMerchantTracker: per-transaction, merchants seen in confirmed fraud
    - AccountGraph:          per-transaction, sender-recipient relational state
    - DriftMonitor:          per-batch, distribution shift vs. training data
    - HypothesisGenerator:   per-level, LLM-generated attack-evolution hypotheses

The MemoryAgent class is a thin facade exposing a small public interface to
Orchestrator, Scorer, and Investigator. Hot-path queries are O(1); expensive
work (drift computation, hypothesis generation) is lazy.

Public interface (see MemoryAgent docstring for details):
    update(tx, verdict)            - per-transaction state update
    query(tx)                      - per-transaction feature lookup
    drift_signal()                 - distribution shift signal
    cache_training_baseline(df)    - level-start baseline caching
    refresh_hypotheses(level)      - trigger per-level LLM call
    hypotheses()                   - retrieve cached hypotheses
    save(path) / load(path)        - cross-level persistence
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tx_get(tx: Any, field_name: str, default=None):
    """
    Uniform field access for transactions.

    Accepts dicts, pandas Series, namedtuples, or dataclasses. Returns default
    on missing fields. Transactions arrive in different shapes throughout the
    pipeline; this normalises access without committing to one type.
    """
    if isinstance(tx, dict):
        return tx.get(field_name, default)
    if hasattr(tx, field_name):
        val = getattr(tx, field_name)
        # pandas Series returns NaN for missing; treat as default
        if val is None:
            return default
        try:
            if pd.isna(val):
                return default
        except (TypeError, ValueError):
            pass
        return val
    try:
        # pandas Series supports dict-like access
        return tx[field_name] if field_name in tx else default
    except (KeyError, TypeError):
        return default


def _parse_timestamp(ts: Any) -> datetime | None:
    """Parse a timestamp field into datetime, returning None on failure."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime()
    try:
        return pd.to_datetime(ts).to_pydatetime()
    except (ValueError, TypeError):
        return None


def _extract_merchant(tx: Any) -> str | None:
    """
    Extract a merchant identity from a transaction.

    Strategy depends on transaction type:
      - in-person / e-commerce: use the location field (merchant name)
      - bank transfer / direct debit: use recipient IBAN
      - withdrawal: no merchant concept, return None

    Kept as a single function so Scorer and Investigator see identical
    merchant identities.
    """
    tx_type = _tx_get(tx, "transaction_type") or _tx_get(tx, "type") or ""
    tx_type = str(tx_type).lower()

    if tx_type in ("in-person payment", "e-commerce", "in_person_payment"):
        location = _tx_get(tx, "location")
        if location:
            return str(location)

    if "transfer" in tx_type or "direct_debit" in tx_type or tx_type == "direct debit":
        recipient_iban = _tx_get(tx, "recipient_iban")
        if recipient_iban:
            return str(recipient_iban)

    return None


# ---------------------------------------------------------------------------
# Component 1: FraudMerchantTracker
# ---------------------------------------------------------------------------

@dataclass
class FraudMerchantTracker:
    """
    Tracks merchants involved in confirmed fraud.

    This is the explicit Memory->Scorer integration: Scorer consumes
    is_known_fraud_merchant via memory.query() and injects it as a feature
    via scorer.update_memory_features().

    Data structure: set for O(1) membership check on the hot path.
    Counts tracked separately for forward-compatibility (future weighting).
    """
    fraud_merchants: set[str] = field(default_factory=set)
    fraud_merchant_counts: dict[str, int] = field(default_factory=dict)

    def record_fraud(self, tx: Any) -> None:
        """Called when a transaction is confirmed fraud."""
        merchant = _extract_merchant(tx)
        if not merchant:
            return
        self.fraud_merchants.add(merchant)
        self.fraud_merchant_counts[merchant] = (
            self.fraud_merchant_counts.get(merchant, 0) + 1
        )

    def is_known_fraud_merchant(self, tx: Any) -> bool:
        """O(1) membership check. Called every transaction by Scorer."""
        merchant = _extract_merchant(tx)
        if not merchant:
            return False
        return merchant in self.fraud_merchants

    def count(self, tx: Any) -> int:
        """How many times this merchant has been seen in fraud. 0 if never."""
        merchant = _extract_merchant(tx)
        if not merchant:
            return 0
        return self.fraud_merchant_counts.get(merchant, 0)


# ---------------------------------------------------------------------------
# Component 2: AccountGraph
# ---------------------------------------------------------------------------

@dataclass
class AccountGraph:
    """
    Cross-user relational state: sender-recipient edges, degrees, account ages.

    Memory's unique contribution. Neither Scorer nor Context sees the graph.

    Design decisions:
      - edges as set[tuple] for O(1) pair existence check
      - degrees updated incrementally (never rebuilt)
      - fraud_connected is one-hop propagation; full union-find is a stretch
        goal if component-level queries prove useful
      - no per-edge timestamps (add if recency weighting becomes necessary)
    """
    edges: set[tuple[str, str]] = field(default_factory=set)
    in_degree: dict[str, int] = field(default_factory=dict)     # recipient -> distinct senders
    out_degree: dict[str, int] = field(default_factory=dict)    # sender -> distinct recipients
    account_first_seen: dict[str, datetime] = field(default_factory=dict)
    fraud_connected: set[str] = field(default_factory=set)

    def update(self, tx: Any, verdict: str | None = None) -> None:
        sender = _tx_get(tx, "sender_id")
        recipient = _tx_get(tx, "recipient_id")
        if not sender or not recipient:
            return

        sender, recipient = str(sender), str(recipient)

        # Edge and degree updates (only count distinct pairs)
        if (sender, recipient) not in self.edges:
            self.edges.add((sender, recipient))
            self.in_degree[recipient] = self.in_degree.get(recipient, 0) + 1
            self.out_degree[sender] = self.out_degree.get(sender, 0) + 1

        # Account first-seen tracking
        ts = _parse_timestamp(_tx_get(tx, "timestamp"))
        if ts is not None:
            for account in (sender, recipient):
                if account not in self.account_first_seen:
                    self.account_first_seen[account] = ts

        # Fraud connectivity (one-hop propagation)
        if verdict == "fraud":
            self.fraud_connected.add(sender)
            self.fraud_connected.add(recipient)

    def query(self, tx: Any) -> dict[str, Any]:
        """Returns graph-derived features for the given transaction."""
        sender = _tx_get(tx, "sender_id")
        recipient = _tx_get(tx, "recipient_id")

        sender_s = str(sender) if sender else ""
        recipient_s = str(recipient) if recipient else ""

        return {
            "is_new_counterparty": (sender_s, recipient_s) not in self.edges,
            "recipient_in_degree": self.in_degree.get(recipient_s, 0),
            "sender_out_degree": self.out_degree.get(sender_s, 0),
            "recipient_age_days": self._age_days(recipient_s, tx),
            "recipient_fraud_connected": recipient_s in self.fraud_connected,
        }

    def _age_days(self, account: str, tx: Any) -> float:
        """Days between account's first-seen timestamp and current tx time."""
        if account not in self.account_first_seen:
            return 0.0
        tx_ts = _parse_timestamp(_tx_get(tx, "timestamp"))
        if tx_ts is None:
            return 0.0
        delta = tx_ts - self.account_first_seen[account]
        return max(0.0, delta.total_seconds() / 86400.0)


# ---------------------------------------------------------------------------
# Component 3: DriftMonitor
# ---------------------------------------------------------------------------

@dataclass
class DriftMonitor:
    """
    Rolling-window distribution shift detection.

    Called by Orchestrator for gray-zone widening. Four components combined
    into a single drift_score in [0, 1]:
      - hour_shift:              Wasserstein distance on hour-of-day (norm by 12)
      - amount_shift:            log-amount median delta (clipped at 2.0)
      - new_location_fraction:   recent locations not in training
      - new_counterparty_fraction: recent (sender, recipient) pairs not in training

    Design decisions:
      - deque(maxlen) for O(1) append with automatic eviction
      - training baseline cached once, not recomputed per query
      - drift computation is lazy (only when drift_signal() called)
    """
    window_size: int = 500
    recent: deque = field(default_factory=lambda: deque(maxlen=500))
    training_baseline: dict[str, Any] | None = None
    samples_seen: int = 0

    def __post_init__(self):
        # Ensure deque respects configured window_size (dataclass default is 500)
        if self.recent.maxlen != self.window_size:
            self.recent = deque(maxlen=self.window_size)

    def cache_training_baseline(self, training_df: pd.DataFrame) -> None:
        """
        Cache training distribution at level start.

        Robust to missing columns - degrades gracefully rather than crashing.
        """
        if training_df is None or len(training_df) == 0:
            self.training_baseline = None
            return

        baseline = {}

        # Hour distribution
        if "timestamp" in training_df.columns:
            hours = pd.to_datetime(training_df["timestamp"], errors="coerce").dt.hour
            baseline["hour_hist"] = self._hour_histogram(hours.dropna().tolist())
        else:
            baseline["hour_hist"] = None

        # Log-amount median
        if "amount" in training_df.columns:
            amounts = training_df["amount"].dropna()
            if len(amounts) > 0:
                baseline["log_amount_median"] = float(np.log1p(amounts.median()))
            else:
                baseline["log_amount_median"] = None
        else:
            baseline["log_amount_median"] = None

        # Locations seen
        if "location" in training_df.columns:
            baseline["locations"] = set(training_df["location"].dropna().astype(str))
        else:
            baseline["locations"] = set()

        # Counterparty pairs seen
        if "sender_id" in training_df.columns and "recipient_id" in training_df.columns:
            pairs = training_df[["sender_id", "recipient_id"]].dropna()
            baseline["counterparty_pairs"] = set(
                zip(pairs["sender_id"].astype(str), pairs["recipient_id"].astype(str))
            )
        else:
            baseline["counterparty_pairs"] = set()

        self.training_baseline = baseline

    def update(self, tx: Any) -> None:
        """Append tx summary to rolling window. O(1)."""
        self.recent.append({
            "hour": self._extract_hour(tx),
            "amount": _tx_get(tx, "amount"),
            "location": _tx_get(tx, "location"),
            "sender_id": _tx_get(tx, "sender_id"),
            "recipient_id": _tx_get(tx, "recipient_id"),
        })
        self.samples_seen += 1

    def drift_signal(self) -> dict[str, Any]:
        """
        Lazy drift computation. Returns {drift_score, detail, cold_start, ...}.

        Cold start (fewer samples than window_size, or no baseline cached)
        returns drift_score=0.0 with cold_start=True.
        """
        if (
            self.training_baseline is None
            or self.samples_seen < self.window_size
        ):
            return {
                "drift_score": 0.0,
                "detail": {},
                "window_size": self.window_size,
                "samples_seen": self.samples_seen,
                "cold_start": True,
            }

        hour_shift = self._compute_hour_shift()
        amount_shift = self._compute_amount_shift()
        new_loc = self._compute_new_location_fraction()
        new_cp = self._compute_new_counterparty_fraction()

        drift_score = 0.25 * (hour_shift + amount_shift + new_loc + new_cp)
        drift_score = float(np.clip(drift_score, 0.0, 1.0))

        return {
            "drift_score": drift_score,
            "detail": {
                "hour_shift": hour_shift,
                "amount_shift": amount_shift,
                "new_location_fraction": new_loc,
                "new_counterparty_fraction": new_cp,
            },
            "window_size": self.window_size,
            "samples_seen": self.samples_seen,
            "cold_start": False,
        }

    # -- component-specific computations ------------------------------------

    def _compute_hour_shift(self) -> float:
        """Wasserstein distance on hour distribution, normalised to [0, 1]."""
        if self.training_baseline.get("hour_hist") is None:
            return 0.0
        recent_hours = [r["hour"] for r in self.recent if r["hour"] is not None]
        if not recent_hours:
            return 0.0

        recent_hist = self._hour_histogram(recent_hours)
        train_hist = self.training_baseline["hour_hist"]

        # 1D Wasserstein distance via cumulative distributions
        recent_cdf = np.cumsum(recent_hist)
        train_cdf = np.cumsum(train_hist)
        w_distance = float(np.sum(np.abs(recent_cdf - train_cdf)))

        # Normalise: max possible W-distance on [0, 23] is 23 * 0.5 ~ 11.5;
        # divide by 12 and clip to [0, 1]
        return float(np.clip(w_distance / 12.0, 0.0, 1.0))

    def _compute_amount_shift(self) -> float:
        """Absolute log-amount median delta, normalised to [0, 1] via clip at 2.0."""
        if self.training_baseline.get("log_amount_median") is None:
            return 0.0
        recent_amounts = [
            r["amount"] for r in self.recent
            if r["amount"] is not None and r["amount"] > 0
        ]
        if not recent_amounts:
            return 0.0
        recent_log_median = float(np.log1p(np.median(recent_amounts)))
        delta = abs(recent_log_median - self.training_baseline["log_amount_median"])
        return float(np.clip(delta / 2.0, 0.0, 1.0))

    def _compute_new_location_fraction(self) -> float:
        """Fraction of recent transactions at locations unseen in training."""
        train_locations = self.training_baseline.get("locations", set())
        recent_locations = [
            str(r["location"]) for r in self.recent if r["location"] is not None
        ]
        if not recent_locations:
            return 0.0
        new_count = sum(1 for loc in recent_locations if loc not in train_locations)
        return new_count / len(recent_locations)

    def _compute_new_counterparty_fraction(self) -> float:
        """Fraction of recent transactions with counterparty pairs unseen in training."""
        train_pairs = self.training_baseline.get("counterparty_pairs", set())
        recent_pairs = [
            (str(r["sender_id"]), str(r["recipient_id"]))
            for r in self.recent
            if r["sender_id"] is not None and r["recipient_id"] is not None
        ]
        if not recent_pairs:
            return 0.0
        new_count = sum(1 for pair in recent_pairs if pair not in train_pairs)
        return new_count / len(recent_pairs)

    # -- utilities ----------------------------------------------------------

    @staticmethod
    def _hour_histogram(hours: list[int]) -> np.ndarray:
        """Normalised 24-bin histogram over hours."""
        hist = np.zeros(24)
        for h in hours:
            if 0 <= int(h) < 24:
                hist[int(h)] += 1
        total = hist.sum()
        if total > 0:
            hist = hist / total
        return hist

    @staticmethod
    def _extract_hour(tx: Any) -> int | None:
        ts = _parse_timestamp(_tx_get(tx, "timestamp"))
        return ts.hour if ts is not None else None


# ---------------------------------------------------------------------------
# Component 4: HypothesisGenerator
# ---------------------------------------------------------------------------

@dataclass
class HypothesisGenerator:
    """
    Per-level LLM call producing hypotheses about attack evolution.

    Idempotent: re-calling refresh() for the same level doesn't re-invoke LLM.
    Graceful degradation: on LLM failure, retains previous hypotheses rather
    than wiping, so Investigator always has *something* to work with.
    """
    hypotheses: list[str] = field(default_factory=list)
    last_level_generated: int = -1
    _llm_client: Any = None

    def set_llm_client(self, client: Any) -> None:
        self._llm_client = client

    def refresh(self, level: int, fraud_summary: list[dict]) -> None:
        """
        Trigger hypothesis generation for a new level.

        fraud_summary: list of per-level summaries of confirmed fraud so far.
        """
        if level <= self.last_level_generated:
            return  # idempotent

        if self._llm_client is None:
            logger.info("HypothesisGenerator: no LLM client, skipping refresh")
            return

        if not fraud_summary:
            logger.info("HypothesisGenerator: no fraud summary yet, skipping")
            return

        try:
            prompt = self._build_prompt(level, fraud_summary)
            response = self._call_llm(prompt)
            parsed = self._parse_response(response)
            if parsed:
                self.hypotheses = parsed
                self.last_level_generated = level
            else:
                logger.warning("HypothesisGenerator: parse returned empty")
        except Exception as e:
            logger.warning(f"HypothesisGenerator.refresh failed: {e}")
            # Preserve previous hypotheses; don't wipe

    def get(self) -> list[str]:
        """Defensive copy to prevent external mutation."""
        return list(self.hypotheses)

    # -- internals ----------------------------------------------------------

    def _build_prompt(self, level: int, fraud_summary: list[dict]) -> str:
        summary_text = "\n".join(
            f"  Level {s.get('level', '?')}: {s.get('description', '')}"
            for s in fraud_summary
        )
        return (
            f"You are analysing fraud-pattern evolution in a financial system.\n"
            f"Fraud patterns observed in previous levels:\n{summary_text}\n\n"
            f"We are now entering level {level}. Fraudsters adapt between levels by "
            f"shifting merchants, times, geographies, amounts, and behavioural sequences.\n\n"
            f"Based on the trajectory of shifts so far, propose 3-5 short hypotheses "
            f"about how fraud patterns might evolve in level {level}. "
            f"Each hypothesis should be one sentence, specific and testable.\n\n"
            f"Format: return each hypothesis on its own line, prefixed with '- '."
        )

    def _call_llm(self, prompt: str) -> str:
        """Abstracted LLM call. Adapt to whatever client the team wires in."""
        # Support multiple LLM client interfaces; the team can wire any of them.
        if hasattr(self._llm_client, "generate"):
            return self._llm_client.generate(prompt, max_tokens=500)
        if hasattr(self._llm_client, "complete"):
            return self._llm_client.complete(prompt, max_tokens=500)
        if callable(self._llm_client):
            return self._llm_client(prompt)
        raise RuntimeError("LLM client has no recognised interface")

    @staticmethod
    def _parse_response(response: str) -> list[str]:
        """Extract bullet-pointed hypotheses from LLM response."""
        if not response:
            return []
        lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
        hypotheses = []
        for ln in lines:
            # Strip common bullet prefixes
            for prefix in ("- ", "* ", "• "):
                if ln.startswith(prefix):
                    ln = ln[len(prefix):]
                    break
            # Strip numbered prefixes like "1. " or "1) "
            if len(ln) > 2 and ln[0].isdigit() and ln[1] in ".)":
                ln = ln[2:].strip()
            if ln and len(ln) > 10:  # skip fragments
                hypotheses.append(ln)
        return hypotheses[:5]  # cap at 5


# ---------------------------------------------------------------------------
# Facade: MemoryAgent
# ---------------------------------------------------------------------------

class MemoryAgent:
    """
    Memory Agent facade. Four internal components, six public methods.

    Interface contract (stable):
        update(tx, verdict)             Orchestrator, per-transaction
        query(tx) -> dict               Scorer/Investigator, per-transaction
        drift_signal() -> dict          Orchestrator, per-batch
        cache_training_baseline(df)     Orchestrator, per-level start
        refresh_hypotheses(level)       Orchestrator, per-level boundary
        hypotheses() -> list[str]       Investigator, per-prompt-build
        save(path) / load(path)         Cross-level persistence

    Usage:
        memory = MemoryAgent(llm_client=my_client)
        memory.cache_training_baseline(train_df)
        for tx in stream:
            features = memory.query(tx)
            verdict = pipeline_decide(tx, features)
            memory.update(tx, verdict)
        drift = memory.drift_signal()
        memory.save('memory_level_1.pkl')
    """

    def __init__(self, llm_client: Any = None, drift_window: int = 500,
                 fraud_confidence_threshold: float = 0.7):
        self.fraud_merchants = FraudMerchantTracker()
        self.graph = AccountGraph()
        self.drift = DriftMonitor(window_size=drift_window)
        self.hypotheses_gen = HypothesisGenerator(_llm_client=llm_client)

        # Track confirmed fraud for hypothesis generation (cross-level memory)
        self._fraud_summary: list[dict] = []
        self._current_level: int = 0
        self._fraud_confidence_threshold = fraud_confidence_threshold

    # -- update / query (hot path) ------------------------------------------

    def update(self, tx: Any, verdict: str | None = None,
               confidence: float = 1.0) -> None:
        """
        Called every transaction by the Orchestrator after decision is made.

        verdict: 'fraud', 'legit', or None. All states update graph and drift.
        confidence: Investigator's confidence in the verdict, in [0, 1].

        Only high-confidence fraud verdicts (>= fraud_confidence_threshold)
        are recorded in the merchant tracker and fraud summary. This prevents
        low-confidence Investigator guesses from contaminating Memory and
        cascading errors through the Scorer via the feedback loop:
            Investigator verdict → Memory → Scorer features → future verdicts

        Graph and drift updates are unconditional — they need every transaction
        regardless of verdict confidence.
        """
        self.graph.update(tx, verdict)
        self.drift.update(tx)

        if verdict == "fraud" and confidence >= self._fraud_confidence_threshold:
            self.fraud_merchants.record_fraud(tx)
            self._fraud_summary.append(self._summarise_fraud(tx))

    def query(self, tx: Any) -> dict[str, Any]:
        """
        Per-transaction feature lookup for Scorer and Investigator.

        Returns:
            is_known_fraud_merchant: bool  (per-transaction, for Investigator)
            known_fraud_merchants:   set   (raw set, for Scorer.update_memory_features)
            is_new_counterparty:     bool
            recipient_in_degree:     int
            sender_out_degree:       int
            recipient_age_days:      float
            recipient_fraud_connected: bool
        """
        out = self.graph.query(tx)
        out["is_known_fraud_merchant"] = self.fraud_merchants.is_known_fraud_merchant(tx)
        # Also expose the raw set — Scorer.update_memory_features() consumes this
        out["known_fraud_merchants"] = set(self.fraud_merchants.fraud_merchants)
        return out

    def known_fraud_merchants(self) -> set[str]:
        """
        Raw set of merchants seen in confirmed fraud.

        Dedicated accessor for Scorer.update_memory_features(), which applies
        the set across a batch of transactions rather than per-tx.
        """
        return set(self.fraud_merchants.fraud_merchants)

    # -- drift / hypotheses (warm path) -------------------------------------

    def drift_signal(self) -> dict[str, Any]:
        """Distribution shift signal for Orchestrator gray-zone widening."""
        return self.drift.drift_signal()

    def cache_training_baseline(self, training_df: pd.DataFrame) -> None:
        """Call at level start with that level's training data."""
        self.drift.cache_training_baseline(training_df)

    def refresh_hypotheses(self, level: int) -> None:
        """
        Call at level boundaries. Triggers one LLM call if conditions are met.

        No-op at level 1 (nothing to hypothesise about yet).
        """
        self._current_level = level
        if level >= 2:
            self.hypotheses_gen.refresh(level, self._fraud_summary)

    def hypotheses(self) -> list[str]:
        """Current hypotheses about attack evolution. Consumed by Investigator."""
        return self.hypotheses_gen.get()

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle full state for cross-level persistence."""
        # Strip LLM client before pickling (not guaranteed to be picklable)
        llm_backup = self.hypotheses_gen._llm_client
        self.hypotheses_gen._llm_client = None
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        finally:
            self.hypotheses_gen._llm_client = llm_backup

    @classmethod
    def load(cls, path: str, llm_client: Any = None) -> "MemoryAgent":
        """Restore from pickle. Re-inject the LLM client after load."""
        with open(path, "rb") as f:
            instance = pickle.load(f)
        if llm_client is not None:
            instance.hypotheses_gen.set_llm_client(llm_client)
        return instance

    # -- internal -----------------------------------------------------------

    def _summarise_fraud(self, tx: Any) -> dict[str, Any]:
        """Compact summary of a confirmed fraud tx for hypothesis generation."""
        ts = _parse_timestamp(_tx_get(tx, "timestamp"))
        return {
            "level": self._current_level,
            "tx_type": _tx_get(tx, "transaction_type") or _tx_get(tx, "type"),
            "amount": _tx_get(tx, "amount"),
            "hour": ts.hour if ts else None,
            "merchant": _extract_merchant(tx),
            "description": (
                f"type={_tx_get(tx, 'transaction_type') or _tx_get(tx, 'type')}, "
                f"amount={_tx_get(tx, 'amount')}, "
                f"hour={ts.hour if ts else '?'}, "
                f"merchant={_extract_merchant(tx) or 'n/a'}"
            ),
        }
