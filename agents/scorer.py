"""
agents/scorer.py
----------------
Role B — Scorer Agent.

Team contract (do not change signature without flagging Role A + B):
    scorer.predict(tx_df) -> np.ndarray   # float32, shape (N,), in [0, 1]

Key design decisions:
  - predict() accepts BOTH a DataFrame and a single pd.Series (Orchestrator
    passes one row at a time in the routing loop)
  - Heuristic mode works WITHOUT training labels but is intentionally conservative:
    most legitimate transactions score below 0.3. Fraud signals in validation
    data (new merchants, night-time, location mismatch, high z-score) will push
    scores up into the gray zone. This is correct behaviour — do not "fix" it.
  - fit() MUST be called at the start of each level with that level's training
    data before scoring the eval set. Retraining per level is mandatory per
    CLAUDE.md. Never reuse a model from a prior level.
  - Memory Agent signals injected via update_memory_features() before predict()

Per CLAUDE.md:
  - logging not print
  - pathlib.Path for all file access
  - Time-based train/val split (caller's responsibility — scorer never splits)
"""

import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("lightgbm not installed — heuristic fallback active.")


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
PAYMENT_METHOD_DEFAULT = 0.30   # unseen methods — moderate, never zero

TX_TYPE_RISK: dict[str, float] = {
    "direct debit": 0.05,
    "transfer": 0.20,
    "in-person payment": 0.30,
    "e-commerce": 0.50,
    "atm withdrawal": 0.60,
    "international transfer": 0.75,
    "crypto": 0.90,
}
TX_TYPE_DEFAULT = 0.35           # unseen types — moderate

# City hints from user ID suffix (e.g. BRCH-TRCY-802-HAM-1 → Hampstead)
ID_CITY_HINTS: dict[str, str] = {
    "HAM": "Hampstead",
    "DIE": "Dietzenbach",
    "AUD": "Audincourt",
}

FEATURE_COLS = [
    "amount_zscore",
    "amount_vs_user_max",
    "balance_ratio",
    "is_night",
    "hour_of_day",
    "day_of_week",
    "location_match",
    "user_tx_frequency_30d",
    "is_new_merchant",
    "tx_type_risk",
    "payment_method_risk",
    "days_since_last_tx",
    "amount_round_number",
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
    """
    Accept either a DataFrame or a single pd.Series (one transaction row).
    The Orchestrator routing loop passes rows as Series — we handle both.
    """
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
    Build features from raw transactions. Every feature has an explicit fallback.

    Note on heuristic calibration:
      Legitimate transactions (rent, salary, subscriptions) naturally score low.
      Fraud signals — new merchants, night-time, location mismatch, amount spike,
      known fraud merchant — will push scores into the gray zone on eval data.
      Do not tune heuristic weights to artificially inflate training scores.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    out = pd.DataFrame(index=df.index)

    out["hour_of_day"] = df["timestamp"].dt.hour.astype(float)
    out["is_night"] = out["hour_of_day"].between(0, 5).astype(float)
    out["day_of_week"] = df["timestamp"].dt.dayofweek.astype(float)

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

    def _loc(row):
        loc = row.get("location")
        if pd.isna(loc) or loc == "":
            return 0.5   # neutral: no location (transfer/debit)
        home = home_cities.get(str(row["sender_id"]), "")
        if not home:
            return 0.5   # unknown user: neutral
        return 1.0 if home.lower() in str(loc).lower() else 0.0

    out["location_match"] = df.apply(_loc, axis=1)

    out["user_tx_frequency_30d"] = df.apply(
        lambda r: float(stats.recent_count(str(r["sender_id"]), r["timestamp"])), axis=1
    )
    out["days_since_last_tx"] = df.apply(
        lambda r: stats.days_since_last(str(r["sender_id"]), r["timestamp"]), axis=1
    )

    def _new_merchant(row):
        loc = row.get("location")
        if pd.isna(loc) or loc == "":
            return 0.0
        return 0.0 if loc in stats.get(str(row["sender_id"]))["merchants"] else 1.0

    out["is_new_merchant"] = df.apply(_new_merchant, axis=1)
    out["tx_type_risk"] = df["transaction_type"].apply(
        lambda x: TX_TYPE_RISK.get(str(x).lower().strip(), TX_TYPE_DEFAULT)
    )
    out["payment_method_risk"] = df["payment_method"].apply(
        lambda x: 0.0 if (pd.isna(x) or x == "")
                  else PAYMENT_METHOD_RISK.get(str(x).lower().strip(), PAYMENT_METHOD_DEFAULT)
    )
    out["amount_round_number"] = (df["amount"] % 50 == 0).astype(float)
    out["is_known_fraud_merchant"] = df.get("is_known_fraud_merchant", 0.0).fillna(0.0)

    return out[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Heuristic scorer
# ---------------------------------------------------------------------------

def _heuristic_score(row: pd.Series) -> float:
    """
    Interpretable fallback when LightGBM is unavailable or untrained.

    Calibration intent:
      - Normal transactions (salary, rent, subscriptions): score 0.05–0.25
      - Mildly suspicious (new merchant, unusual hour): 0.25–0.45
      - Clearly suspicious (GPS mismatch + night + new merchant): 0.45–0.70+
      - Known fraud merchant: hard push toward fraud threshold

    The low baseline on training data is correct — training data is clean.
    """
    s = 0.0
    s += 0.20 * float(np.clip(row["amount_zscore"] / 5.0, 0, 1))
    s += 0.15 * row["is_night"]
    s += 0.12 * row["is_new_merchant"]
    s += 0.12 * row["payment_method_risk"]
    s += 0.10 * (1.0 - row["location_match"])
    s += 0.08 * row["tx_type_risk"]
    s += 0.07 * float(np.clip(row["amount_vs_user_max"], 0, 1))
    s += 0.07 * row["balance_ratio"]
    s += 0.05 * row["is_known_fraud_merchant"]
    s += 0.03 * row["amount_round_number"]
    s += 0.01 * (1.0 - float(np.clip(row["user_tx_frequency_30d"] / 10.0, 0, 1)))
    return float(np.clip(s, 0.0, 1.0))


# ---------------------------------------------------------------------------
# ScorerAgent
# ---------------------------------------------------------------------------

class ScorerAgent:
    """
    Role B — Scorer Agent.

    Team contract:
        scorer.predict(tx_df) -> np.ndarray   # shape (N,), float32, in [0, 1]

    predict() accepts either a DataFrame or a single pd.Series — both are safe.
    The Orchestrator routing loop passes individual rows as Series.

    Per-level workflow:
        scorer = ScorerAgent(users_path=...)
        scorer.fit(level_train_df)                           # retrain every level
        tx_df = scorer.update_memory_features(eval_df, memory.query(tx))
        scores = scorer.predict(tx_df)                       # or predict(tx_row)
    """

    def __init__(self, users_path: str | Path = "data/users.json") -> None:
        self.model: "lgb.LGBMClassifier | None" = None
        self._stats = UserStats()
        self._home_cities: dict[str, str] = {}
        self._users: list[dict] = []

        p = Path(users_path)
        if p.exists():
            with open(p) as f:
                self._users = json.load(f)
            logger.info("ScorerAgent: loaded %d users from %s", len(self._users), p)
        else:
            logger.warning("ScorerAgent: users_path not found (%s) — home city mapping disabled.", p)

    def fit(self, train_df: pd.DataFrame, label_col: str = "is_fraud") -> None:
        """
        Train LightGBM. Must be called at the start of each level.
        Caller is responsible for time-based train/val split (last 20% held out).
        Never call fit() with random splits — hides drift effects.
        """
        if label_col not in train_df.columns:
            raise ValueError(
                f"Column '{label_col}' not in train_df. "
                "Provide fraud labels before calling fit()."
            )
        df = _ensure_columns(_coerce_to_dataframe(train_df))
        self._stats.build(df)
        self._home_cities = build_home_cities(df, self._users)

        X = engineer_features(df, self._stats, self._home_cities)
        y = df[label_col].astype(int)

        if not HAS_LGB:
            logger.warning("ScorerAgent: LightGBM unavailable — heuristic fallback.")
            return

        fraud_rate = max(float(y.mean()), 1e-3)
        self.model = lgb.LGBMClassifier(
            objective="binary", metric="auc",
            learning_rate=0.05, num_leaves=31, min_child_samples=5,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            scale_pos_weight=(1.0 - fraud_rate) / fraud_rate,
            verbose=-1, n_estimators=300,
        )
        self.model.fit(X, y, eval_set=[(X, y)],
                       callbacks=[lgb.early_stopping(30, verbose=False)])
        logger.info(
            "ScorerAgent: trained — %d samples, %d fraud (%.1f%%)",
            len(X), int(y.sum()), fraud_rate * 100,
        )

    def predict(self, tx) -> np.ndarray:
        """
        Score transactions.

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

        if self.model is not None and HAS_LGB:
            return self.model.predict_proba(feats)[:, 1].astype(np.float32)
        return feats.apply(_heuristic_score, axis=1).values.astype(np.float32)

    def update_memory_features(
        self,
        tx,
        memory_patterns: dict,
    ) -> pd.DataFrame:
        """
        Inject Memory Agent signals before calling predict().

        Parameters
        ----------
        tx : pd.DataFrame or pd.Series
            Transactions to augment.
        memory_patterns : dict
            Return value of memory.query(tx). Reads key:
              "known_fraud_merchants": set[str]

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
                "model": self.model, "stats": self._stats,
                "home_cities": self._home_cities, "users": self._users,
            }, f)
        logger.info("ScorerAgent: saved → %s", path)

    def load(self, path: str | Path = "scorer_model.pkl") -> None:
        with open(Path(path), "rb") as f:
            p = pickle.load(f)
        self.model = p["model"]
        self._stats = p["stats"]
        self._home_cities = p["home_cities"]
        self._users = p.get("users", [])
        logger.info("ScorerAgent: loaded ← %s", path)

    def feature_importance(self) -> "pd.Series | None":
        if self.model is not None and HAS_LGB:
            return pd.Series(
                self.model.feature_importances_, index=FEATURE_COLS
            ).sort_values(ascending=False)
        return None
