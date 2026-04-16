"""
agents/scorer.py
----------------
Role B — Scorer Agent.

Team contract (do not change signature without flagging Role A + B):
    scorer.predict(tx_df) -> np.ndarray   # float32, shape (N,), in [0, 1]

Model: Isolation Forest (unsupervised anomaly detection).
Rationale: training data has no fraud labels, so supervised learning is
impossible. IsolationForest scores each transaction by how "isolated"
(anomalous) it is in feature space — no labels needed.

Key design decisions:
  - fit() trains on the full training set without labels
  - predict() converts IF's raw anomaly score to [0, 1]:
      negate (IF: low = anomalous) → scale using training bounds → clip
  - Score conversion uses training-data bounds so extreme eval anomalies
    correctly saturate at 1.0 rather than breaking the [0, 1] contract
  - predict() accepts BOTH DataFrame and pd.Series (Orchestrator passes rows
    as Series in the routing loop)
  - All features have safe fallbacks — missing columns never crash predict()
  - Unseen users, transaction types, payment methods all handled gracefully

Per CLAUDE.md:
  - logging not print
  - pathlib.Path for all file access
  - Retrain at each level start — never reuse a prior model
  - Time-based train/val split is caller's responsibility
"""

import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk tables — safe defaults for unseen values
# ---------------------------------------------------------------------------

PAYMENT_METHOD_RISK: dict[str, float] = {
    "cash": 0.10,
    "direct debit": 0.10,
    "bank transfer": 0.15,
    "debit card": 0.25,
    "smartwatch": 0.35,
    "apple pay": 0.35,
    "google pay": 0.35,
    "paypal": 0.40,
    "credit card": 0.45,
    "crypto": 0.85,
}
PAYMENT_METHOD_DEFAULT = 0.30

TX_TYPE_RISK: dict[str, float] = {
    "direct debit": 0.05,
    "transfer": 0.20,
    "in-person payment": 0.30,
    "e-commerce": 0.50,
    "atm withdrawal": 0.60,
    "international transfer": 0.75,
    "crypto": 0.90,
}
TX_TYPE_DEFAULT = 0.35

ID_CITY_HINTS: dict[str, str] = {
    "HAM": "Hampstead",
    "DIE": "Dietzenbach",
    "AUD": "Audincourt",
}

# Features fed to Isolation Forest — all numeric, no NaN allowed
IF_FEATURE_COLS = [
    "amount",
    "hour_of_day",
    "balance_after",
    "amount_zscore",
    "amount_vs_user_max",
    "balance_ratio",
    "is_night",
    "tx_type_risk",
    "payment_method_risk",
    "is_new_merchant",
    "days_since_last_tx",
    "is_known_fraud_merchant",
]

COLUMN_DEFAULTS: dict[str, object] = {
    "location": None,
    "payment_method": None,
    "description": None,
    "balance_after": 0.0,
    "sender_iban": "",
    "recipient_iban": "",
    "recipient_id": None,
    "is_known_fraud_merchant": 0.0,
}


# ---------------------------------------------------------------------------
# Per-user statistics
# ---------------------------------------------------------------------------

class UserStats:
    """Per-user stats. Always returns safe cold-start defaults for unseen users."""

    _COLD_START = {
        "mean": 500.0, "std": 300.0, "max": 1000.0,
        "count": 0, "merchants": set(), "timestamps": [],
    }

    def __init__(self) -> None:
        self._stats: dict[str, dict] = {}

    def build(self, df: pd.DataFrame) -> None:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        df = df.sort_values("timestamp")
        for uid, grp in df.groupby("sender_id"):
            amounts = grp["amount"].values
            self._stats[str(uid)] = {
                "mean": float(np.mean(amounts)),
                "std": float(np.std(amounts)) if len(amounts) > 1
                       else max(float(np.mean(amounts)) * 0.20, 1.0),
                "max": float(np.max(amounts)),
                "count": int(len(amounts)),
                "merchants": set(grp["location"].dropna().tolist()),
                "timestamps": grp["timestamp"].tolist(),
            }

    def get(self, uid: str) -> dict:
        return self._stats.get(uid, self._COLD_START)

    def recent_count(self, uid: str, before: pd.Timestamp, days: int = 30) -> int:
        cutoff = before - pd.Timedelta(days=days)
        return sum(1 for t in self.get(uid)["timestamps"] if cutoff <= t < before)

    def days_since_last(self, uid: str, before: pd.Timestamp) -> float:
        prior = [t for t in self.get(uid)["timestamps"] if t < before]
        return min((before - max(prior)).total_seconds() / 86400.0, 180.0) if prior else 180.0


# ---------------------------------------------------------------------------
# Home city mapping
# ---------------------------------------------------------------------------

def _city_from_id(uid: str) -> str:
    for part in uid.upper().split("-"):
        if part in ID_CITY_HINTS:
            return ID_CITY_HINTS[part]
    return ""


def build_home_cities(df: pd.DataFrame, users: list[dict]) -> dict[str, str]:
    iban_to_city = {u["iban"]: u["residence"]["city"] for u in users}
    result: dict[str, str] = {}
    for _, row in df.drop_duplicates("sender_id").iterrows():
        uid = str(row["sender_id"])
        iban = str(row.get("sender_iban", "") or "")
        result[uid] = iban_to_city.get(iban, "") or _city_from_id(uid)
    return result


# ---------------------------------------------------------------------------
# Column safety
# ---------------------------------------------------------------------------

def _coerce_to_dataframe(tx) -> pd.DataFrame:
    """Accept DataFrame or pd.Series — Orchestrator passes rows as Series."""
    if isinstance(tx, pd.Series):
        return tx.to_frame().T.reset_index(drop=True)
    return tx.copy()


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, default in COLUMN_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    stats: UserStats,
    home_cities: dict[str, str],
) -> pd.DataFrame:
    """
    Build the numeric feature matrix for Isolation Forest.
    Every feature has an explicit safe fallback — no NaN reaches the model.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    out = pd.DataFrame(index=df.index)

    # Raw values (IF uses these directly as anomaly signals)
    out["amount"] = df["amount"].fillna(0.0)
    out["balance_after"] = df["balance_after"].fillna(0.0)

    # Temporal
    out["hour_of_day"] = df["timestamp"].dt.hour.astype(float)
    out["is_night"] = out["hour_of_day"].between(0, 5).astype(float)

    # Amount vs this user's own history
    def _z(row):
        s = stats.get(str(row["sender_id"]))
        return (row["amount"] - s["mean"]) / (s["std"] + 1e-9)

    def _vs_max(row):
        s = stats.get(str(row["sender_id"]))
        return row["amount"] / (s["max"] + 1e-9)

    out["amount_zscore"] = df.apply(_z, axis=1).clip(-5, 5)
    out["amount_vs_user_max"] = df.apply(_vs_max, axis=1).clip(0, 5)
    out["balance_ratio"] = (
        df["amount"] / (df["balance_after"].replace(0, np.nan) + df["amount"])
    ).fillna(0.5).clip(0, 1)

    # Activity
    out["days_since_last_tx"] = df.apply(
        lambda r: stats.days_since_last(str(r["sender_id"]), r["timestamp"]), axis=1
    )

    # New merchant flag
    def _new_merchant(row):
        loc = row.get("location")
        if pd.isna(loc) or loc == "":
            return 0.0
        return 0.0 if loc in stats.get(str(row["sender_id"]))["merchants"] else 1.0

    out["is_new_merchant"] = df.apply(_new_merchant, axis=1)

    # Risk encodings for categorical fields
    out["tx_type_risk"] = df["transaction_type"].apply(
        lambda x: TX_TYPE_RISK.get(str(x).lower().strip(), TX_TYPE_DEFAULT)
    )
    out["payment_method_risk"] = df["payment_method"].apply(
        lambda x: 0.0 if (pd.isna(x) or x == "")
                  else PAYMENT_METHOD_RISK.get(str(x).lower().strip(), PAYMENT_METHOD_DEFAULT)
    )

    # Memory Agent feature
    out["is_known_fraud_merchant"] = df.get("is_known_fraud_merchant", 0.0).fillna(0.0)

    # Final safety net: fill any remaining NaN with 0
    return out[IF_FEATURE_COLS].fillna(0.0)


# ---------------------------------------------------------------------------
# Score conversion: IF raw → [0, 1] fraud score
# ---------------------------------------------------------------------------

def _if_to_fraud_score(
    raw_scores: np.ndarray,
    train_score_min: float,
    train_score_max: float,
) -> np.ndarray:
    """
    Convert Isolation Forest score_samples() output to a [0, 1] fraud score.

    IF convention:  more negative  = more anomalous = more likely fraud
    Our convention: higher value   = more anomalous = more likely fraud

    Uses training-data bounds for scaling so eval anomalies more extreme than
    anything seen in training correctly saturate at 1.0.
    """
    neg = -raw_scores
    neg_min = -train_score_max
    neg_max = -train_score_min
    scaled = (neg - neg_min) / (neg_max - neg_min + 1e-9)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Heuristic fallback (only used if fit() was never called)
# ---------------------------------------------------------------------------

def _heuristic_fallback(row: pd.Series) -> float:
    s = 0.0
    s += 0.20 * float(np.clip(row["amount_zscore"] / 5.0, 0, 1))
    s += 0.15 * row["is_night"]
    s += 0.12 * row["is_new_merchant"]
    s += 0.12 * row["payment_method_risk"]
    s += 0.08 * row["tx_type_risk"]
    s += 0.07 * float(np.clip(row["amount_vs_user_max"], 0, 1))
    s += 0.07 * row["balance_ratio"]
    s += 0.05 * row["is_known_fraud_merchant"]
    return float(np.clip(s, 0.0, 1.0))


# ---------------------------------------------------------------------------
# ScorerAgent
# ---------------------------------------------------------------------------

class ScorerAgent:
    """
    Role B — Scorer Agent.

    Team contract:
        scorer.predict(tx) -> np.ndarray   # shape (N,), float32, in [0, 1]

    Uses Isolation Forest (unsupervised) — no fraud labels needed.
    fit() trains on the full training set. predict() scores by anomaly.

    Per-level workflow:
        scorer = ScorerAgent(users_path=...)
        scorer.fit(level_train_df)           # retrain every level, no labels needed
        scores = scorer.predict(eval_df)     # or predict(single_row_as_Series)
    """

    def __init__(self, users_path: str | Path = "data/users.json") -> None:
        self.model: IsolationForest | None = None
        self._train_score_min: float = -1.0
        self._train_score_max: float = -0.3
        self._stats = UserStats()
        self._home_cities: dict[str, str] = {}
        self._users: list[dict] = []

        p = Path(users_path)
        if p.exists():
            with open(p) as f:
                self._users = json.load(f)
            logger.info("ScorerAgent: loaded %d users from %s", len(self._users), p)
        else:
            logger.warning(
                "ScorerAgent: users_path not found (%s) — home city mapping disabled.", p
            )

    def fit(self, train_df: pd.DataFrame, label_col: str | None = None) -> None:
        """
        Train Isolation Forest on the full training set.
        No fraud labels required — label_col accepted but ignored.
        Must be called at the start of each level. Never reuse across levels.
        """
        df = _ensure_columns(_coerce_to_dataframe(train_df))
        self._stats.build(df)
        self._home_cities = build_home_cities(df, self._users)

        X = engineer_features(df, self._stats, self._home_cities).values

        self.model = IsolationForest(
            n_estimators=200,
            contamination="auto",   # unknown fraud rate → let IF decide
            max_features=1.0,
            bootstrap=False,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X)

        # Store training score bounds for consistent conversion at predict time
        train_scores = self.model.score_samples(X)
        self._train_score_min = float(train_scores.min())
        self._train_score_max = float(train_scores.max())

        logger.info(
            "ScorerAgent: IsolationForest trained — %d samples, %d features, "
            "score range [%.4f, %.4f]",
            len(X), X.shape[1],
            self._train_score_min, self._train_score_max,
        )

    def predict(self, tx) -> np.ndarray:
        """
        Score transactions by anomaly — higher = more anomalous = more likely fraud.

        Parameters
        ----------
        tx : pd.DataFrame or pd.Series
            One or more transactions. Series (single row) is accepted —
            the Orchestrator routing loop passes rows one at a time.

        Returns
        -------
        np.ndarray, float32, shape (N,), values in [0, 1].
        """
        df = _ensure_columns(_coerce_to_dataframe(tx))

        if not self._stats._stats:
            logger.warning("ScorerAgent.predict() called before fit() — cold-start mode.")
            self._stats.build(df)
        if not self._home_cities:
            self._home_cities = build_home_cities(df, self._users)

        feats = engineer_features(df, self._stats, self._home_cities)

        if self.model is not None:
            raw = self.model.score_samples(feats.values)
            return _if_to_fraud_score(raw, self._train_score_min, self._train_score_max)

        logger.warning("ScorerAgent: no model fitted — using heuristic fallback.")
        return feats.apply(_heuristic_fallback, axis=1).values.astype(np.float32)

    def update_memory_features(self, tx, memory_patterns: dict) -> pd.DataFrame:
        """
        Inject Memory Agent signals before calling predict().

        Parameters
        ----------
        tx : pd.DataFrame or pd.Series
        memory_patterns : dict
            Return value of memory.query(tx).
            Reads key: "known_fraud_merchants": set[str]

        Returns
        -------
        pd.DataFrame with is_known_fraud_merchant column added.
        """
        df = _coerce_to_dataframe(tx)
        fraud_merchants: set = memory_patterns.get("known_fraud_merchants", set())
        df["is_known_fraud_merchant"] = df["location"].apply(
            lambda loc: 1.0 if (pd.notna(loc) and loc in fraud_merchants) else 0.0
        )
        logger.debug(
            "ScorerAgent: memory injection — %d known fraud merchants, %d rows flagged.",
            len(fraud_merchants), int(df["is_known_fraud_merchant"].sum()),
        )
        return df

    def save(self, path: str | Path = "scorer_model.pkl") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "train_score_min": self._train_score_min,
                "train_score_max": self._train_score_max,
                "stats": self._stats,
                "home_cities": self._home_cities,
                "users": self._users,
            }, f)
        logger.info("ScorerAgent: saved → %s", path)

    def load(self, path: str | Path = "scorer_model.pkl") -> None:
        with open(Path(path), "rb") as f:
            p = pickle.load(f)
        self.model = p["model"]
        self._train_score_min = p["train_score_min"]
        self._train_score_max = p["train_score_max"]
        self._stats = p["stats"]
        self._home_cities = p["home_cities"]
        self._users = p.get("users", [])
        logger.info("ScorerAgent: loaded ← %s", path)

    def feature_importance(self) -> None:
        """Isolation Forest has no feature importances — returns None."""
        return None
