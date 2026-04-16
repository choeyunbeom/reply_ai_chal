"""
agents/context.py
-----------------
Role B — Context Agent.

Team contract (do not change signature without flagging Role A + B):
    context.build(tx_id: str) -> dict   # evidence bundle for Investigator LLM

Robustness guarantees:
  - Works for any city in locations.json — not hardcoded to training cities
  - SMS phone extraction falls back to first-name scan if To: line is absent
  - bit.ly shorteners only flagged when co-occurring with fraud keywords
  - Users with no prior transactions return safe "no history" summaries
  - Unknown users (not in users.json) return neutral profiles, not errors
  - reload() allows swapping in validation-level data files without reinstantiating

Per CLAUDE.md:
  - Use logging, not print
  - pathlib.Path for all file access
  - No LLM calls — pure Python evidence assembly only
"""

import json
import logging
import math
import re
from datetime import timedelta
from pathlib import Path
from sys import flags

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fraud signal patterns
# ---------------------------------------------------------------------------

# These domains are unambiguously suspicious regardless of message context.
# Typosquatting pattern: number substitutions (paypa1, amaz0n, netfl1x, ub3r).
KNOWN_PHISHING_FRAGMENTS = [
    "paypa1-secure",
    "amaz0n-verify",
    "netfl1x-bill",
    "ub3r-verify",
    "paypa1",
    "amaz0n",
    "micros0ft",
    "app1e",
    "g00gle",
]

# URL shorteners: suspicious ONLY when co-occurring with a fraud keyword.
# Legitimate city-council and event SMS also use bit.ly — do not count alone.
URL_SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "rb.gy", "is.gd", "ow.ly"]

# Fraud keywords: strong signal when any of these appear in the message body.
FRAUD_KEYWORDS = [
    "verify now", "verify your account", "verify identity",
    "suspicious login", "suspicious sign-in", "unusual login",
    "account will be locked", "account suspended", "prevent lock",
    "update payment", "billing information", "payment details",
    "urgent", "immediately", "act now",
]

EMAIL_FRAUD_KEYWORDS = [
    "wire transfer", "wiring instructions", "change of bank details",
    "updated payment information", "new account number",
    "invoice attached", "overdue invoice", "past due",
    "please remit", "remittance",
    "confidential", "confidential matter", "do not discuss",
    "ceo", "director approval", "urgent request from",
    "gift card", "itunes card", "amazon card",
    "tax refund", "hmrc", "irs",
]

SALARY_BANDS = [
    (0,        20_000,       "low"),
    (20_000,   40_000,       "lower-middle"),
    (40_000,   70_000,       "middle"),
    (70_000,   120_000,      "upper-middle"),
    (120_000,  float("inf"), "high"),
]


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _median_city_coords(city: str, locations: list[dict]) -> tuple[float, float] | None:
    """
    Median lat/lng for any city in the GPS log.
    Works for training cities AND any new city in validation data.
    Returns None if city not found in logs.
    """
    pts = [(e["lat"], e["lng"]) for e in locations
           if e.get("city", "").lower() == city.lower()]
    if not pts:
        return None
    lats = sorted(p[0] for p in pts)
    lngs = sorted(p[1] for p in pts)
    mid = len(pts) // 2
    return (lats[mid], lngs[mid])


def _nearest_gps(
    timestamp: str,
    user_id: str,
    locations: list[dict],
    window_hours: int = 6,
) -> dict | None:
    """
    Closest GPS ping for a user within window_hours of the transaction.
    Biotag field in locations.json must match sender_id exactly.
    """
    ts = pd.Timestamp(timestamp)
    window = timedelta(hours=window_hours)
    best: dict | None = None
    best_delta = timedelta.max

    for entry in locations:
        if entry.get("biotag") != user_id:
            continue
        try:
            entry_ts = pd.Timestamp(entry["timestamp"])
        except Exception:
            continue
        delta = abs(ts - entry_ts)
        if delta <= window and delta < best_delta:
            best_delta = delta
            best = entry

    return best


# ---------------------------------------------------------------------------
# SMS fraud signal extraction
# ---------------------------------------------------------------------------

def _extract_phone(first_name: str, sms_data: list[dict]) -> str | None:
    """
    Find user's phone number from SMS data.
    Tries To: header first, then falls back to From: lines near name mentions.
    Returns None if not found — caller falls back to name-based filtering.
    """
    for entry in sms_data:
        raw = entry.get("sms", "")
        if first_name not in raw:
            continue
        m = re.search(r"To: (\+\d{7,15})", raw)
        if m:
            return m.group(1)
    return None


def _sms_fraud_signals(
    phone: str | None,
    first_name: str,
    sms_data: list[dict],
) -> dict:
    """
    Scan SMS messages for phishing and fraud signals.

    Rules:
      - Known phishing domains (paypa1, amaz0n, etc.) always count.
      - URL shorteners (bit.ly etc.) only count when a fraud keyword is
        also present in the same message — avoids false positives from
        legitimate council / event SMS.
      - Filtering: prefer phone-number match; fall back to first-name match.
    """
    phishing_hits = 0
    suspicious_domains: list[str] = []
    fraud_kw_found: list[str] = []

    for entry in sms_data:
        raw = entry.get("sms", "")
        is_relevant = (phone and phone in raw) or first_name in raw
        if not is_relevant:
            continue

        raw_lower = raw.lower()

        # Always-suspicious domains
        for frag in KNOWN_PHISHING_FRAGMENTS:
            if frag in raw_lower:
                phishing_hits += 1
                suspicious_domains.append(frag)

        # Fraud keywords
        for kw in FRAUD_KEYWORDS:
            if kw in raw_lower:
                fraud_kw_found.append(kw)

        # Shorteners: only flag when fraud keywords also present
        has_fraud_kw = any(kw in raw_lower for kw in FRAUD_KEYWORDS)
        if has_fraud_kw:
            for shortener in URL_SHORTENERS:
                if shortener in raw_lower:
                    phishing_hits += 1
                    suspicious_domains.append(shortener)

    return {
        "phishing_hits": phishing_hits,
        "suspicious_domains": list(set(suspicious_domains)),
        "fraud_keywords": list(set(fraud_kw_found)),
    }

def _email_fraud_signals(
    email_addr: str | None,
    first_name: str,
    email_data: list[dict],
) -> dict:
    """
    Scan email messages for phishing and fraud signals.

    Mirrors _sms_fraud_signals but:
      - scans BOTH SMS and EMAIL keyword sets (BEC/invoice language)
      - matches on email address first, falls back to first-name match
      - expects entries shaped like {"mail": "...full thread..."}
    """
    phishing_hits = 0
    suspicious_domains: list[str] = []
    fraud_kw_found: list[str] = []

    combined_keywords = FRAUD_KEYWORDS + EMAIL_FRAUD_KEYWORDS

    for entry in email_data:
        raw = entry.get("mail", "")
        is_relevant = (email_addr and email_addr in raw) or first_name in raw
        if not is_relevant:
            continue

        raw_lower = raw.lower()

        # Always-suspicious domains
        for frag in KNOWN_PHISHING_FRAGMENTS:
            if frag in raw_lower:
                phishing_hits += 1
                suspicious_domains.append(frag)

        # Fraud keywords (SMS + email-specific)
        for kw in combined_keywords:
            if kw in raw_lower:
                fraud_kw_found.append(kw)

        # Shorteners: only flag when fraud keywords also present
        has_fraud_kw = any(kw in raw_lower for kw in combined_keywords)
        if has_fraud_kw:
            for shortener in URL_SHORTENERS:
                if shortener in raw_lower:
                    phishing_hits += 1
                    suspicious_domains.append(shortener)

    return {
        "phishing_hits": phishing_hits,
        "suspicious_domains": list(set(suspicious_domains)),
        "fraud_keywords": list(set(fraud_kw_found)),
    }
# ---------------------------------------------------------------------------
# ContextAgent
# ---------------------------------------------------------------------------

class ContextAgent:
    """
    Role B — Context Agent.

    Team contract:
        context.build(tx_id: str) -> dict   # evidence bundle

    Instantiate once per level, then call build() per gray-zone transaction.
    Call reload() if you need to swap in a different data file mid-run.
    """

    def __init__(
        self,
        transactions_path: str | Path = "data/transactions.csv",
        users_path: str | Path = "data/users.json",
        locations_path: str | Path = "data/locations.json",
        sms_path: str | Path = "data/sms.json",
        email_path: str | Path = "data/messages.json",
    ) -> None:
        self._transactions_path = Path(transactions_path)
        self._users_path = Path(users_path)
        self._locations_path = Path(locations_path)
        self._sms_path = Path(sms_path)
        self._email_path = Path(email_path)  
        self._load_all()

    def _load_all(self) -> None:
        """Load or reload all data sources from disk."""
        self._df = pd.read_csv(self._transactions_path)
        self._df["timestamp"] = pd.to_datetime(self._df["timestamp"])

        self._users_by_iban: dict[str, dict] = {}
        if self._users_path.exists():
            with open(self._users_path) as f:
                users = json.load(f)
            self._users_by_iban = {u["iban"]: u for u in users}

        self._locations: list[dict] = []
        if self._locations_path.exists():
            with open(self._locations_path) as f:
                self._locations = json.load(f)

        self._sms: list[dict] = []
        if self._sms_path.exists():
            with open(self._sms_path) as f:
                self._sms = json.load(f)

        self._emails: list[dict] = []
        if self._email_path.exists():
            with open(self._email_path) as f:
                self._emails = json.load(f)

        logger.info(
            "ContextAgent loaded: %d txs | %d users | %d gps pings | %d sms | %d emails",
            len(self._df), len(self._users_by_iban),
            len(self._locations), len(self._sms), len(self._emails),
        )

    def reload(
        self,
        transactions_path: str | Path | None = None,
        users_path: str | Path | None = None,
        locations_path: str | Path | None = None,
        sms_path: str | Path | None = None,
        email_path: str | Path | None = None,
    ) -> None:
        """
        Hot-swap data sources — useful when validation data arrives mid-run.
        Only updates paths that are explicitly provided.
        """
        if transactions_path:
            self._transactions_path = Path(transactions_path)
        if users_path:
            self._users_path = Path(users_path)
        if locations_path:
            self._locations_path = Path(locations_path)
        if sms_path:
            self._sms_path = Path(sms_path)
        if email_path:
            self._email_path = Path(email_path)
        self._load_all()

    # ── Team contract ─────────────────────────────────────────────────────

    def build(self, tx_id: str) -> dict:
        """
        Build an evidence bundle for a gray-zone transaction.

        Returns
        -------
        dict with keys:
            tx_id, user_id, user_profile, transaction,
            recent_tx_summary, gps_location_match,
            sms_fraud_signals, amount_context, risk_flags

        risk_flags is a list[str] of plain-English signals ready
        to embed directly into the Investigator prompt.
        """
        rows = self._df[self._df["transaction_id"] == tx_id]
        if rows.empty:
            raise ValueError(f"Transaction '{tx_id}' not found in loaded data.")
        tx = rows.iloc[0]

        user = self._users_by_iban.get(str(tx.get("sender_iban", "")) or "", None)

        bundle: dict = {
            "tx_id": tx_id,
            "user_id": str(tx["sender_id"]),
            "user_profile": self._user_profile(user),
            "transaction": self._tx_dict(tx),
            "recent_tx_summary": self._recent_summary(tx),
            "gps_location_match": self._gps_check(tx),
            "sms_fraud_signals": self._sms_signals(user),
            "email_fraud_signals": self._email_signals(user),
            "amount_context": self._amount_context(tx, user),
            "risk_flags": [],
        }
        bundle["risk_flags"] = self._derive_flags(bundle)
        return bundle

    # ── Internal builders ─────────────────────────────────────────────────

    def _user_profile(self, user: dict | None) -> dict:
        if not user:
            return {
                "name": "unknown", "job": "unknown",
                "home_city": "unknown", "salary_band": "unknown",
                "annual_salary": None,
            }
        salary = user.get("salary", 0)
        band = next((b for lo, hi, b in SALARY_BANDS if lo <= salary < hi), "unknown")
        return {
            "name": f"{user.get('first_name', '')} {user.get('last_name', '')}".strip(),
            "job": user.get("job", "unknown"),
            "home_city": user.get("residence", {}).get("city", "unknown"),
            "salary_band": band,
            "annual_salary": salary,
        }

    def _tx_dict(self, tx: pd.Series) -> dict:
        def _safe(v):
            return v if pd.notna(v) else None
        return {
            "amount": float(tx["amount"]),
            "type": tx["transaction_type"],
            "location": _safe(tx.get("location")),
            "payment_method": _safe(tx.get("payment_method")),
            "description": _safe(tx.get("description")),
            "timestamp": str(tx["timestamp"]),
            "balance_after": float(tx["balance_after"]),
        }

    def _recent_summary(self, tx: pd.Series) -> dict:
        uid = tx["sender_id"]
        cutoff = tx["timestamp"] - timedelta(days=30)
        hist = self._df[
            (self._df["sender_id"] == uid)
            & (self._df["timestamp"] < tx["timestamp"])
            & (self._df["timestamp"] >= cutoff)
        ]
        if hist.empty:
            return {
                "count": 0, "total_amount": 0.0, "avg_amount": 0.0,
                "transaction_types": {}, "merchants_visited": [],
            }
        return {
            "count": len(hist),
            "total_amount": round(float(hist["amount"].sum()), 2),
            "avg_amount": round(float(hist["amount"].mean()), 2),
            "transaction_types": hist["transaction_type"].value_counts().to_dict(),
            "merchants_visited": hist["location"].dropna().unique().tolist(),
        }

    def _gps_check(self, tx: pd.Series) -> dict:
        loc = tx.get("location")
        uid = str(tx["sender_id"])

        if pd.isna(loc) or not loc:
            return {
                "tx_location": None, "gps_city_at_tx_time": None,
                "distance_km": None, "match": None,
                "note": "No location field — transfer/debit, GPS check not applicable.",
            }

        ping = _nearest_gps(str(tx["timestamp"]), uid, self._locations)
        if ping is None:
            return {
                "tx_location": loc, "gps_city_at_tx_time": None,
                "distance_km": None, "match": None,
                "note": "No GPS ping within 6h of transaction.",
            }

        gps_city = ping.get("city", "unknown")

        # Extract city guess from location string (works for any city format)
        # "Dietzenbach - Coffee House" → "Dietzenbach"
        # "SwiftCart Online" → "SwiftCart Online" (online merchant, no coords)
        tx_city = loc.split(" - ")[0].strip() if " - " in loc else loc
        tx_coords = _median_city_coords(tx_city, self._locations)

        if tx_coords:
            dist_km = _haversine_km(ping["lat"], ping["lng"], tx_coords[0], tx_coords[1])
            match = dist_km < 50.0
        else:
            # Online merchant or unknown city — fall back to string comparison
            dist_km = None
            match = gps_city.lower() in loc.lower()

        return {
            "tx_location": loc,
            "gps_city_at_tx_time": gps_city,
            "gps_timestamp": ping["timestamp"],
            "distance_km": round(dist_km, 1) if dist_km is not None else None,
            "match": match,
        }

    def _sms_signals(self, user: dict | None) -> dict:
        if not user:
            return {"phishing_hits": 0, "suspicious_domains": [], "fraud_keywords": []}
        first_name = user.get("first_name", "")
        phone = _extract_phone(first_name, self._sms)
        return _sms_fraud_signals(phone, first_name, self._sms)
    
    def _email_signals(self, user: dict | None) -> dict:
        """
        Scan the user's email threads for phishing and fraud signals.
        Matches on the user's email address if present in the users.json record;
        falls back to first-name matching if not.
        """
        if not user:
            return {"phishing_hits": 0, "suspicious_domains": [], "fraud_keywords": []}
        first_name = user.get("first_name", "")
        email_addr = user.get("email", "")
        return _email_fraud_signals(email_addr, first_name, self._emails)

    def _amount_context(self, tx: pd.Series, user: dict | None) -> dict:
        uid = tx["sender_id"]
        hist = self._df[
            (self._df["sender_id"] == uid)
            & (self._df["timestamp"] < tx["timestamp"])
        ]["amount"]

        z = float((tx["amount"] - hist.mean()) / (hist.std() + 1e-9)) if len(hist) > 1 else 0.0
        salary = user.get("salary", 0) if user else 0
        pct_monthly = (tx["amount"] / (salary / 12)) if salary > 0 else 0.0

        known_locs = set(self._df[self._df["sender_id"] == uid]["location"].dropna().tolist())
        loc = tx.get("location")
        is_new = bool(pd.notna(loc) and loc and loc not in known_locs)

        return {
            "z_score": round(z, 2),
            "pct_of_monthly_salary": round(pct_monthly, 2),
            "is_new_merchant": is_new,
        }

    def _derive_flags(self, b: dict) -> list[str]:
        """
        Translate evidence into plain-English flags for the Investigator prompt.
        Keep flags concrete and non-redundant.
        """
        flags: list[str] = []
        ac = b["amount_context"]
        gps = b["gps_location_match"]
        sms = b["sms_fraud_signals"]
        email = b.get("email_fraud_signals", {})

        if ac["z_score"] > 2.0:
            flags.append(f"Amount is {ac['z_score']:.1f} std devs above this user's mean.")
        if ac["pct_of_monthly_salary"] > 0.8:
            flags.append(
                f"Amount is {ac['pct_of_monthly_salary']*100:.0f}% of estimated monthly salary."
            )
        if ac["is_new_merchant"]:
            flags.append("First-ever transaction at this merchant for this user.")

        if gps["match"] is False:
            dist = f" ({gps['distance_km']} km away)" if gps.get("distance_km") else ""
            flags.append(
                f"GPS mismatch: user was in '{gps['gps_city_at_tx_time']}' "
                f"but transaction at '{gps['tx_location']}'{dist}."
            )
        elif gps["match"] is None and gps["tx_location"]:
            flags.append("GPS match inconclusive — no recent ping available.")

        if sms["phishing_hits"] > 0:
            domains = ", ".join(sms["suspicious_domains"][:3])
            flags.append(
                f"User's SMS history contains {sms['phishing_hits']} phishing signal(s): {domains}."
            )
        if sms["fraud_keywords"]:
            flags.append(
                f"Fraud keywords in SMS: {', '.join(sms['fraud_keywords'][:4])}."
            )
        if email.get("phishing_hits", 0) > 0:
            domains = ", ".join(email["suspicious_domains"][:3])
            flags.append(
                f"User's email history contains {email['phishing_hits']} phishing signal(s): {domains}."
            )
        if email.get("fraud_keywords"):
            flags.append(
                f"Fraud keywords in email: {', '.join(email['fraud_keywords'][:4])}."
            )

        # Cross-channel signal: fraud keywords in BOTH SMS and email is unusually strong
        if sms.get("fraud_keywords") and email.get("fraud_keywords"):
            flags.append(
        "Fraud keywords present in BOTH SMS and email history — "
        "cross-channel social engineering indicator."
            )
        
        if b["recent_tx_summary"]["count"] == 0:
            flags.append("No transaction activity in the 30 days prior to this transaction.")

        return flags
