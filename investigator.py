"""
Investigator Agent for Reply Mirror fraud detection system.

Methodology: Popperian falsification + Bayesian updating.
    1. Generate competing hypotheses (H0 legitimate, H1..Hn fraud types).
    2. Assign priors informed by Scorer risk score, drift, memory hypotheses.
    3. State falsifiable predictions for each hypothesis.
    4. Gather evidence aimed at refutation, not confirmation.
    5. Bayesian update posteriors based on which predictions survived.
    6. Return highest-posterior hypothesis with full audit trail.

Only called for gray-zone transactions. Consumes Context Agent's evidence
bundle as primary input; optionally calls tools for deeper investigation.

Public interface:
    judge(tx, context, memory_handle=None) -> dict   # structured verdict

The verdict dict includes:
    fraud: bool
    confidence: float
    verdict_hypothesis: str
    hypotheses_tested: list of {hypothesis, prior, posterior, predictions, ...}
    tools_called: list[str]
    reason: str                  # human-readable summary

Design notes:
    - Falsifiability validation is non-negotiable. Predictions that forbid
      nothing are rejected. Without this, the methodology collapses into
      narrative reasoning.
    - Budget-aware: if the LLM call fails or times out, falls back to a
      conservative rule-based verdict so the pipeline never blocks.
    - Tool calls bound to refutation_target fields for auditability.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """A falsifiable prediction attached to a hypothesis."""
    prediction: str                  # "sender's history should show consistent pattern"
    refutation_target: str           # which data field/tool would refute
    tested: bool = False
    survived: bool | None = None     # None if not tested
    evidence: str = ""
    diagnostic_weight: float = 1.0   # how informative this test is

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "refutation_target": self.refutation_target,
            "tested": self.tested,
            "survived": self.survived,
            "evidence": self.evidence,
            "diagnostic_weight": self.diagnostic_weight,
        }


@dataclass
class Hypothesis:
    """A competing explanation for the transaction."""
    name: str                                # "legitimate transaction", "social engineering"
    prior: float = 0.0
    posterior: float = 0.0
    predictions: list[Prediction] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hypothesis": self.name,
            "prior": self.prior,
            "posterior": self.posterior,
            "refutation_predictions": [p.to_dict() for p in self.predictions],
        }


# ---------------------------------------------------------------------------
# Default hypothesis set
# ---------------------------------------------------------------------------

DEFAULT_HYPOTHESES = [
    "legitimate transaction",
    "social engineering / account takeover",
    "mule account routing",
    "synthetic identity fraud",
    "unauthorised use of payment method",
]


# ---------------------------------------------------------------------------
# Falsifiability checks (non-LLM)
# ---------------------------------------------------------------------------

# Phrases that typically indicate an unfalsifiable prediction
UNFALSIFIABLE_PATTERNS = [
    r"\bsomething\s+suspicious\b",
    r"\bmight\s+(look|seem|be|feel)\s+(strange|odd|unusual|weird)\b",
    r"\bcould\s+be\s+(anything|something)\b",
    r"\bpotentially\s+(suspicious|fraudulent)\b",
    r"\bmay\s+indicate\b(?!\s+\w+\s+(more\s+than|greater\s+than|less\s+than|above|below))",
    r"\bsome(thing|times)\s+(odd|off|weird)\b",
]

# Tokens that signal specific testable content
FALSIFIABLE_KEYWORDS = [
    "amount", "hour", "time", "location", "merchant", "recipient", "sender",
    "balance", "frequency", "degree", "age", "distance", "gps", "sms", "email",
    "keyword", "phishing", "domain", "match", "deviation", "zscore", "fraction",
    "count", "rate", "ratio", "greater than", "less than", "above", "below",
    "exceed", "within", "outside", "contains", "includes", "more than", "at least",
]


def is_falsifiable(prediction: str) -> bool:
    """
    Check whether a prediction is falsifiable (forbids something specific).

    This is the methodological core. A prediction that can't be refuted by
    any observation isn't scientific under Popper - it's narrative. We
    reject it and require restatement.

    Heuristic (deliberately strict):
      - reject if matches any unfalsifiable pattern
      - require at least one falsifiable keyword (data field or comparator)
    """
    if not prediction or len(prediction.strip()) < 15:
        return False

    lower = prediction.lower()

    for pattern in UNFALSIFIABLE_PATTERNS:
        if re.search(pattern, lower):
            return False

    # Must reference at least one specific, testable concept
    has_falsifiable_content = any(kw in lower for kw in FALSIFIABLE_KEYWORDS)
    return has_falsifiable_content


# ---------------------------------------------------------------------------
# Prior assignment
# ---------------------------------------------------------------------------

def assign_priors(
    scorer_risk_score: float,
    drift_score: float,
    memory_hypotheses: list[str] | None,
    hypothesis_names: list[str],
) -> dict[str, float]:
    """
    Assign prior probabilities across hypotheses.

    Strategy:
      - Scorer risk score controls split between "legitimate" and fraud hypotheses.
      - Under high drift, flatten the distribution (less anchoring to history).
      - If memory_hypotheses name a specific fraud type, bias toward it.

    Returns normalised dict: {hypothesis_name: prior}.
    """
    risk = max(0.0, min(1.0, scorer_risk_score))
    drift = max(0.0, min(1.0, drift_score))

    # Base split: legit gets (1 - risk), fraud types share risk equally
    legit_name = hypothesis_names[0]  # convention: H0 is always legitimate
    fraud_names = hypothesis_names[1:]

    priors = {legit_name: 1.0 - risk}
    if fraud_names:
        per_fraud = risk / len(fraud_names)
        for name in fraud_names:
            priors[name] = per_fraud

    # Drift flattening: pull toward uniform as drift rises
    if drift > 0:
        uniform = 1.0 / len(hypothesis_names)
        for name in priors:
            priors[name] = (1 - drift) * priors[name] + drift * uniform

    # Memory-hypothesis bias: if a memory hypothesis textually matches
    # a fraud type, add weight to it
    if memory_hypotheses and fraud_names:
        boost = 0.0
        for mh in memory_hypotheses:
            mh_lower = mh.lower()
            for name in fraud_names:
                if any(keyword in mh_lower for keyword in name.lower().split()):
                    priors[name] += 0.05
                    boost += 0.05
        # Renormalise to keep sum = 1
        if boost > 0:
            total = sum(priors.values())
            priors = {k: v / total for k, v in priors.items()}

    return priors


# ---------------------------------------------------------------------------
# Bayesian update
# ---------------------------------------------------------------------------

def bayesian_update(
    priors: dict[str, float],
    hypotheses: list[Hypothesis],
) -> dict[str, float]:
    """
    Update posteriors based on survived/failed predictions.

    Simple multiplicative likelihood model:
      - Each survived prediction: likelihood multiplier of 1.0 + w
      - Each failed prediction: likelihood multiplier of 1.0 - w
      where w = diagnostic_weight (bounded in [0.0, 0.9] for stability).

    This is intentionally simple. A full implementation would model
    conditional likelihoods P(evidence | hypothesis) explicitly, but that
    requires data we don't have in a 6-hour challenge. The multiplicative
    heuristic captures the core intuition: hypotheses whose predictions
    survive gain posterior mass; those whose predictions fail lose it.
    """
    posteriors = dict(priors)

    for h in hypotheses:
        likelihood = 1.0
        for pred in h.predictions:
            if not pred.tested:
                continue
            w = max(0.0, min(0.9, pred.diagnostic_weight))
            if pred.survived:
                likelihood *= (1.0 + w)
            else:
                likelihood *= (1.0 - w)
        posteriors[h.name] = priors.get(h.name, 0.0) * likelihood

    # Normalise
    total = sum(posteriors.values())
    if total > 0:
        posteriors = {k: v / total for k, v in posteriors.items()}
    return posteriors


# ---------------------------------------------------------------------------
# Rule-based fallback (if LLM unavailable or fails)
# ---------------------------------------------------------------------------

def rule_based_verdict(context: dict, scorer_risk: float) -> dict:
    """
    Conservative fallback when LLM is unavailable.

    Uses Context Agent's risk_flags (plain-English signals) plus Scorer's
    risk score to produce a verdict. No hypothesis-testing, just weighted
    evidence. Pipeline never blocks even if LLM is broken.
    """
    risk_flags = context.get("risk_flags", []) if context else []
    flag_count = len(risk_flags)

    sms = context.get("sms_fraud_signals", {}) if context else {}
    phishing_hits = sms.get("phishing_hits", 0) if sms else 0
    fraud_keywords = sms.get("fraud_keywords", 0) if sms else 0

    gps = context.get("gps_location_match", {}) if context else {}
    gps_mismatch = gps.get("match") is False if gps else False

    amount_ctx = context.get("amount_context", {}) if context else {}
    high_zscore = abs(amount_ctx.get("z_score", 0) or 0) > 3

    # Score composition (conservative — bias toward legit unless clear signals)
    score = 0.0
    score += 0.25 * min(1.0, flag_count / 4.0)
    score += 0.2 if phishing_hits > 0 else 0.0
    score += 0.15 if fraud_keywords >= 2 else 0.0
    score += 0.15 if gps_mismatch else 0.0
    score += 0.1 if high_zscore else 0.0
    score += 0.15 * scorer_risk  # trust some of the Scorer's signal

    is_fraud = score > 0.5

    return {
        "fraud": is_fraud,
        "confidence": float(min(0.75, max(0.5, score))),  # cap confidence for fallback
        "verdict_hypothesis": "fraud (rule-based)" if is_fraud else "legitimate (rule-based)",
        "hypotheses_tested": [],
        "tools_called": [],
        "reason": (
            f"Rule-based fallback: risk_flags={flag_count}, phishing={phishing_hits}, "
            f"fraud_keywords={fraud_keywords}, gps_mismatch={gps_mismatch}, "
            f"high_zscore={high_zscore}, scorer_risk={scorer_risk:.2f}"
        ),
        "fallback": True,
    }


# ---------------------------------------------------------------------------
# Main Investigator
# ---------------------------------------------------------------------------

class InvestigatorAgent:
    """
    LLM-based fraud investigator using Popperian falsification with Bayesian
    updating.

    Usage:
        investigator = InvestigatorAgent(llm_client=my_client)
        verdict = investigator.judge(tx, context, memory_handle=memory)
    """

    def __init__(
        self,
        llm_client: Any = None,
        hypothesis_set: list[str] | None = None,
        max_tool_calls: int = 3,
        confidence_threshold: float = 0.7,
    ):
        self._llm = llm_client
        self.hypothesis_set = hypothesis_set or list(DEFAULT_HYPOTHESES)
        self.max_tool_calls = max_tool_calls
        self.confidence_threshold = confidence_threshold

    # -- main entry point ---------------------------------------------------

    def judge(
        self,
        tx: Any,
        context: dict,
        memory_handle: Any = None,
    ) -> dict:
        """
        Investigate a single gray-zone transaction.

        Args:
            tx: the transaction (dict / Series / etc.)
            context: Context Agent's evidence bundle (see Context spec)
            memory_handle: optional MemoryAgent for drift/hypotheses/tools

        Returns: structured verdict dict (see module docstring).
        """
        scorer_risk = self._extract_scorer_risk(context)
        drift_score = self._extract_drift(memory_handle)
        mem_hypotheses = self._extract_memory_hypotheses(memory_handle)

        # If no LLM configured, go straight to rule-based fallback
        if self._llm is None:
            logger.info("Investigator: no LLM client, using rule-based fallback")
            return rule_based_verdict(context, scorer_risk)

        # Step 1: priors
        priors = assign_priors(
            scorer_risk, drift_score, mem_hypotheses, self.hypothesis_set
        )

        # Step 2: generate hypotheses with falsifiable predictions
        try:
            hypotheses = self._generate_hypotheses_with_predictions(
                tx, context, priors, mem_hypotheses
            )
        except Exception as e:
            logger.warning(f"Hypothesis generation failed ({e}), falling back")
            return rule_based_verdict(context, scorer_risk)

        if not hypotheses:
            logger.warning("No valid hypotheses generated, falling back")
            return rule_based_verdict(context, scorer_risk)

        # If no hypothesis has any predictions at all, the LLM output was
        # unusable — fall back to rules rather than running a no-evidence
        # Bayesian update that just returns the priors.
        total_predictions = sum(len(h.predictions) for h in hypotheses)
        if total_predictions == 0:
            logger.warning("All hypotheses had zero predictions, falling back")
            return rule_based_verdict(context, scorer_risk)

        # Step 3: test predictions against context (and tools if available)
        tools_called: list[str] = []
        self._test_predictions(
            hypotheses, context, tx, memory_handle, tools_called
        )

        # Step 4: Bayesian update
        posteriors = bayesian_update(priors, hypotheses)
        for h in hypotheses:
            h.posterior = posteriors.get(h.name, 0.0)

        # Step 5: verdict
        winner = max(hypotheses, key=lambda h: h.posterior)
        is_fraud = winner.name != self.hypothesis_set[0]  # H0 is legitimate

        # Confidence = posterior weighted by test rigour
        tested_count = sum(1 for p in winner.predictions if p.tested)
        rigour_factor = min(1.0, tested_count / 3.0)
        confidence = float(winner.posterior * (0.6 + 0.4 * rigour_factor))

        return {
            "fraud": is_fraud,
            "confidence": confidence,
            "verdict_hypothesis": winner.name,
            "hypotheses_tested": [h.to_dict() for h in hypotheses],
            "tools_called": tools_called,
            "reason": self._build_reason(winner, hypotheses),
            "fallback": False,
        }

    # -- hypothesis generation ---------------------------------------------

    def _generate_hypotheses_with_predictions(
        self,
        tx: Any,
        context: dict,
        priors: dict[str, float],
        memory_hypotheses: list[str],
    ) -> list[Hypothesis]:
        """
        Call LLM to generate falsifiable predictions for each hypothesis.

        Falsifiability validation is applied to every prediction. Rejected
        predictions are dropped; a hypothesis with zero valid predictions is
        still retained but will only move via prior.
        """
        prompt = self._build_generation_prompt(tx, context, priors, memory_hypotheses)
        response = self._call_llm(prompt, max_tokens=1200)
        parsed = self._parse_hypothesis_response(response)

        hypotheses = []
        for name in self.hypothesis_set:
            predictions_raw = parsed.get(name, [])
            predictions = []
            for p_raw in predictions_raw:
                pred_text = p_raw.get("prediction", "") if isinstance(p_raw, dict) else str(p_raw)
                target = p_raw.get("refutation_target", "context") if isinstance(p_raw, dict) else "context"
                weight = p_raw.get("diagnostic_weight", 0.5) if isinstance(p_raw, dict) else 0.5

                if is_falsifiable(pred_text):
                    predictions.append(Prediction(
                        prediction=pred_text,
                        refutation_target=target,
                        diagnostic_weight=float(weight),
                    ))
                else:
                    logger.debug(f"Rejected unfalsifiable prediction: {pred_text}")

            hypotheses.append(Hypothesis(
                name=name,
                prior=priors.get(name, 0.0),
                predictions=predictions,
            ))

        return hypotheses

    # -- prediction testing -------------------------------------------------

    def _test_predictions(
        self,
        hypotheses: list[Hypothesis],
        context: dict,
        tx: Any,
        memory_handle: Any,
        tools_called: list[str],
    ) -> None:
        """
        Test each prediction against available evidence.

        Strategy:
          1. First pass: test against Context bundle (free).
          2. Second pass: for untested high-weight predictions, call tools
             (bounded by max_tool_calls).

        Each test sets pred.tested, pred.survived, pred.evidence in place.
        """
        # Pass 1: test against context
        for h in hypotheses:
            for pred in h.predictions:
                result = self._test_against_context(pred, context, h.name)
                if result is not None:
                    pred.tested = True
                    pred.survived = result["survived"]
                    pred.evidence = result["evidence"]

        # Pass 2: call tools for untested diagnostic predictions
        tool_budget = self.max_tool_calls
        for h in hypotheses:
            if tool_budget <= 0:
                break
            for pred in sorted(h.predictions, key=lambda p: -p.diagnostic_weight):
                if tool_budget <= 0:
                    break
                if pred.tested:
                    continue
                tool_result = self._test_with_tools(pred, tx, memory_handle, h.name)
                if tool_result is not None:
                    pred.tested = True
                    pred.survived = tool_result["survived"]
                    pred.evidence = tool_result["evidence"]
                    tools_called.extend(tool_result.get("tools", []))
                    tool_budget -= 1

    def _test_against_context(
        self, pred: Prediction, context: dict, hypothesis_name: str
    ) -> dict | None:
        """
        Test a prediction against the Context Agent's evidence bundle.

        Matches prediction text against known context fields. Returns None
        if we can't test from context alone (caller will try tools next).

        This is intentionally conservative — when in doubt, return None and
        let the tool-use pass handle it.
        """
        if not context:
            return None

        pred_lower = pred.prediction.lower()
        target_lower = pred.refutation_target.lower()
        is_fraud_hyp = hypothesis_name != "legitimate transaction"

        # GPS mismatch tests
        if "gps" in pred_lower or "location" in target_lower:
            gps = context.get("gps_location_match") or {}
            if gps.get("match") is not None:
                mismatch = gps["match"] is False
                # For fraud hypotheses: mismatch supports prediction (survives)
                # For legit hypothesis: mismatch refutes it
                survived = mismatch if is_fraud_hyp else (not mismatch)
                return {
                    "survived": survived,
                    "evidence": f"GPS match={gps.get('match')}, distance={gps.get('distance_km')}",
                }

        # SMS / phishing tests
        if any(k in pred_lower for k in ["sms", "phishing", "keyword", "message", "email"]):
            sms = context.get("sms_fraud_signals") or {}
            has_signals = (
                sms.get("phishing_hits", 0) > 0
                or sms.get("fraud_keywords", 0) > 0
            )
            survived = has_signals if is_fraud_hyp else (not has_signals)
            return {
                "survived": survived,
                "evidence": f"phishing={sms.get('phishing_hits', 0)}, keywords={sms.get('fraud_keywords', 0)}",
            }

        # Amount deviation tests
        if any(k in pred_lower for k in ["amount", "zscore", "deviation", "z-score", "z score"]):
            amount = context.get("amount_context") or {}
            z = abs(amount.get("z_score", 0) or 0)
            high_dev = z > 2.5
            survived = high_dev if is_fraud_hyp else (not high_dev)
            return {
                "survived": survived,
                "evidence": f"z_score={amount.get('z_score')}, pct_of_salary={amount.get('pct_of_monthly_salary')}",
            }

        # New merchant tests
        if any(k in pred_lower for k in ["merchant", "new merchant", "first-time"]):
            amount = context.get("amount_context") or {}
            is_new = amount.get("is_new_merchant", False)
            survived = is_new if is_fraud_hyp else (not is_new)
            return {
                "survived": survived,
                "evidence": f"is_new_merchant={is_new}",
            }

        # Recent activity tests
        if any(k in pred_lower for k in ["recent", "history", "frequency", "consistent"]):
            recent = context.get("recent_tx_summary") or {}
            count = recent.get("count", 0)
            # Low recent activity supports fraud hypotheses; high supports legit
            low_activity = count < 3
            survived = low_activity if is_fraud_hyp else (not low_activity)
            return {
                "survived": survived,
                "evidence": f"recent_count={count}, avg_amount={recent.get('avg_amount')}",
            }

        # Risk flags catch-all
        if "flag" in pred_lower or "risk" in pred_lower:
            flags = context.get("risk_flags", [])
            has_flags = len(flags) > 0
            survived = has_flags if is_fraud_hyp else (not has_flags)
            return {
                "survived": survived,
                "evidence": f"risk_flags count={len(flags)}",
            }

        return None  # Can't test from context alone

    def _test_with_tools(
        self,
        pred: Prediction,
        tx: Any,
        memory_handle: Any,
        hypothesis_name: str,
    ) -> dict | None:
        """
        Test a prediction using Memory's tools.

        Only a small set of tool integrations for now. Pipeline will add more
        if Context doesn't already cover them.
        """
        if memory_handle is None:
            return None

        pred_lower = pred.prediction.lower()
        is_fraud_hyp = hypothesis_name != "legitimate transaction"
        tools_used = []

        # Counterparty / graph tests via memory.query()
        if any(k in pred_lower for k in [
            "counterparty", "recipient", "sender", "graph", "degree", "new pair"
        ]):
            try:
                mem_features = memory_handle.query(tx)
                tools_used.append("memory.query")
                is_new = mem_features.get("is_new_counterparty", False)
                fraud_connected = mem_features.get("recipient_fraud_connected", False)
                in_degree = mem_features.get("recipient_in_degree", 0)

                suspicious = is_new or fraud_connected or in_degree > 20
                survived = suspicious if is_fraud_hyp else (not suspicious)
                return {
                    "survived": survived,
                    "evidence": (
                        f"is_new_counterparty={is_new}, "
                        f"fraud_connected={fraud_connected}, "
                        f"in_degree={in_degree}"
                    ),
                    "tools": tools_used,
                }
            except Exception as e:
                logger.debug(f"memory.query failed: {e}")

        # Drift tests
        if "drift" in pred_lower or "shift" in pred_lower:
            try:
                drift = memory_handle.drift_signal()
                tools_used.append("memory.drift_signal")
                high_drift = drift.get("drift_score", 0) > 0.5
                survived = high_drift if is_fraud_hyp else (not high_drift)
                return {
                    "survived": survived,
                    "evidence": f"drift_score={drift.get('drift_score'):.2f}",
                    "tools": tools_used,
                }
            except Exception as e:
                logger.debug(f"memory.drift_signal failed: {e}")

        return None

    # -- LLM interaction ----------------------------------------------------

    def _build_generation_prompt(
        self,
        tx: Any,
        context: dict,
        priors: dict[str, float],
        memory_hypotheses: list[str],
    ) -> str:
        """
        Build the prompt for hypothesis + prediction generation.

        Deliberately prescriptive: the LLM must produce predictions that
        reference specific data fields, not narrative suspicion.
        """
        hypothesis_list = "\n".join(
            f"  - {name} (prior={priors.get(name, 0.0):.2f})"
            for name in self.hypothesis_set
        )

        memory_hints = ""
        if memory_hypotheses:
            memory_hints = (
                "\n\nHYPOTHESES ABOUT RECENT ATTACK EVOLUTION (from memory):\n"
                + "\n".join(f"  - {h}" for h in memory_hypotheses[:5])
            )

        context_summary = self._summarise_context_for_prompt(context)

        return f"""You are a fraud investigator using Popperian falsification.
You do NOT look for evidence that CONFIRMS fraud. You test hypotheses by
trying to REFUTE them. A hypothesis that survives refutation attempts is
provisionally credible.

HYPOTHESES TO TEST:
{hypothesis_list}
{memory_hints}

TRANSACTION:
{self._summarise_tx_for_prompt(tx)}

CONTEXT BUNDLE:
{context_summary}

TASK: For each hypothesis, generate 2-3 predictions that would REFUTE it
if true. Each prediction must:
  - reference a SPECIFIC data field (amount, hour, gps, sms, merchant,
    counterparty, etc.) or a COMPARATOR (greater than, less than, within,
    outside, contains)
  - be falsifiable: it must be possible to say "this prediction failed"
    based on observed data
  - avoid vague phrases like "something suspicious" or "might seem unusual"

Return JSON in exactly this format:
{{
  "legitimate transaction": [
    {{"prediction": "...", "refutation_target": "gps_location_match", "diagnostic_weight": 0.7}},
    ...
  ],
  "social engineering / account takeover": [...],
  "mule account routing": [...],
  "synthetic identity fraud": [...],
  "unauthorised use of payment method": [...]
}}

Return ONLY valid JSON. No markdown, no preamble.
"""

    def _summarise_tx_for_prompt(self, tx: Any) -> str:
        """Compact transaction representation for prompt."""
        fields = ["transaction_id", "sender_id", "recipient_id", "amount",
                  "transaction_type", "location", "payment_method", "timestamp"]
        lines = []
        for f in fields:
            val = None
            if isinstance(tx, dict):
                val = tx.get(f)
            elif hasattr(tx, f):
                val = getattr(tx, f)
            else:
                try:
                    val = tx[f]
                except (KeyError, TypeError):
                    pass
            if val is not None:
                lines.append(f"  {f}: {val}")
        return "\n".join(lines) if lines else "  (no fields available)"

    def _summarise_context_for_prompt(self, context: dict) -> str:
        """Compact context bundle for prompt. Preserves risk_flags verbatim."""
        if not context:
            return "  (no context available)"

        parts = []

        if context.get("user_profile"):
            up = context["user_profile"]
            parts.append(
                f"  user: job={up.get('job')}, home={up.get('home_city')}, "
                f"salary_band={up.get('salary_band')}"
            )

        if context.get("gps_location_match"):
            gps = context["gps_location_match"]
            parts.append(
                f"  gps: match={gps.get('match')}, "
                f"distance={gps.get('distance_km')}, "
                f"note={gps.get('note')}"
            )

        if context.get("sms_fraud_signals"):
            sms = context["sms_fraud_signals"]
            parts.append(
                f"  sms: phishing={sms.get('phishing_hits')}, "
                f"keywords={sms.get('fraud_keywords')}, "
                f"domains={sms.get('suspicious_domains')}"
            )

        if context.get("amount_context"):
            ac = context["amount_context"]
            parts.append(
                f"  amount: z_score={ac.get('z_score')}, "
                f"pct_salary={ac.get('pct_of_monthly_salary')}, "
                f"new_merchant={ac.get('is_new_merchant')}"
            )

        if context.get("recent_tx_summary"):
            rc = context["recent_tx_summary"]
            parts.append(
                f"  recent: count={rc.get('count')}, "
                f"avg={rc.get('avg_amount')}, "
                f"types={rc.get('transaction_types')}"
            )

        if context.get("risk_flags"):
            flags = context["risk_flags"]
            parts.append(f"  risk_flags: {flags}")

        return "\n".join(parts) if parts else "  (empty bundle)"

    def _call_llm(self, prompt: str, max_tokens: int = 1200) -> str:
        """Abstracted LLM call. Supports multiple client interfaces."""
        if hasattr(self._llm, "generate"):
            return self._llm.generate(prompt, max_tokens=max_tokens)
        if hasattr(self._llm, "complete"):
            return self._llm.complete(prompt, max_tokens=max_tokens)
        if callable(self._llm):
            return self._llm(prompt)
        raise RuntimeError("LLM client has no recognised interface")

    @staticmethod
    def _parse_hypothesis_response(response: str) -> dict[str, list]:
        """
        Parse LLM JSON response. Robust to fenced code blocks and prose wrapping.
        """
        if not response:
            return {}

        # Strip markdown fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (may be ```json or ```)
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        # Try direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fallback: find first { and last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM hypothesis response as JSON")

        return {}

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _extract_scorer_risk(context: dict) -> float:
        """Pull Scorer's risk score from context if the pipeline attached it."""
        if not context:
            return 0.5
        risk = context.get("scorer_risk_score")
        if risk is None:
            risk = context.get("risk_score")
        try:
            return max(0.0, min(1.0, float(risk)))
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _extract_drift(memory_handle: Any) -> float:
        if memory_handle is None:
            return 0.0
        try:
            signal = memory_handle.drift_signal()
            return float(signal.get("drift_score", 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _extract_memory_hypotheses(memory_handle: Any) -> list[str]:
        if memory_handle is None:
            return []
        try:
            return list(memory_handle.hypotheses())
        except Exception:
            return []

    @staticmethod
    def _build_reason(winner: Hypothesis, all_hypotheses: list[Hypothesis]) -> str:
        """Human-readable summary of the verdict reasoning."""
        survived = [p for p in winner.predictions if p.tested and p.survived]
        failed = [p for p in winner.predictions if p.tested and not p.survived]

        parts = [
            f"Verdict: {winner.name} (posterior={winner.posterior:.2f}, "
            f"prior={winner.prior:.2f}). "
        ]
        if survived:
            parts.append(f"Survived {len(survived)} refutation attempts.")
        if failed:
            parts.append(f"Failed {len(failed)} refutations.")

        # Runner-up
        sorted_hyps = sorted(all_hypotheses, key=lambda h: -h.posterior)
        if len(sorted_hyps) > 1:
            runner_up = sorted_hyps[1]
            parts.append(
                f" Closest alternative: {runner_up.name} "
                f"(posterior={runner_up.posterior:.2f})."
            )

        return " ".join(parts)
