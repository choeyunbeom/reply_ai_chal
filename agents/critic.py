"""
agents/critic.py
----------------
Role C — Critic Agent (L4-L5 only).

Examines the Investigator's reasoning structure for internal contradictions
and evidential gaps. Does NOT re-investigate — it reads the Investigator's
existing structured output and checks whether the argument supports its own
conclusion.

Three checks (each grounded in a distinct epistemic concern):

1. LOGICAL VALIDITY — does the verdict follow from the evidence?
   The winning hypothesis should have the highest posterior given the
   test results. The reason field should not contradict the numbers.

2. EVIDENTIAL SUFFICIENCY — is the case strong enough?
   A verdict based on one low-weight prediction is technically valid
   but evidentially thin. The Critic enforces a minimum rigour threshold.

3. UNDERCUTTING DEFEATERS — is there unexamined evidence?
   If the context bundle contains strong signals (phishing, GPS mismatch)
   that appear nowhere in the hypothesis audit trail, the Investigator
   fixated on partial evidence and missed something relevant.

Feedback destinations:
    - Verdict: adjusted_confidence (affects Memory's confidence-gated recording)
    - Memory: reasoning weakness patterns (cross-level learning)
    - Orchestrator: agree/disagree signal (operational monitoring)

Public interface:
    critic.verify(verdict, context) -> dict

Per CLAUDE.md:
    - logging not print
    - pathlib.Path for all file access
    - Activated at L4-L5 only (LEVEL_CONFIG["critic"] == True)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# High-diagnostic evidence types (same list as Investigator prompt)
# ---------------------------------------------------------------------------

HIGH_DIAGNOSTIC_EVIDENCE = {
    "sms_fraud_signals": [
        lambda ctx: (ctx.get("sms_fraud_signals") or {}).get("phishing_hits", 0) > 0,
        "SMS phishing signals present",
    ],
    "email_fraud_signals": [
        lambda ctx: (ctx.get("email_fraud_signals") or {}).get("phishing_hits", 0) > 0,
        "Email phishing signals present",
    ],
    "audio_fraud_signals": [
        lambda ctx: (ctx.get("audio_fraud_signals") or {}).get("phishing_hits", 0) > 0,
        "Audio fraud signals present",
    ],
    "gps_mismatch": [
        lambda ctx: (
            (ctx.get("gps_location_match") or {}).get("match") is False
            and ((ctx.get("gps_location_match") or {}).get("distance_km") or 0) > 100
        ),
        "GPS mismatch with distance > 100km",
    ],
    "high_amount_deviation": [
        lambda ctx: abs((ctx.get("amount_context") or {}).get("z_score", 0) or 0) > 3.0,
        "Amount z-score > 3.0",
    ],
    "no_recent_activity": [
        lambda ctx: (ctx.get("recent_tx_summary") or {}).get("count", 1) == 0,
        "No transaction activity in prior 30 days",
    ],
}


# ---------------------------------------------------------------------------
# Rule-based checks (no LLM needed)
# ---------------------------------------------------------------------------

def check_logical_validity(verdict: dict) -> dict | None:
    """
    Check 1: does the conclusion follow from the premises?

    Examines whether the winning hypothesis actually has the highest
    posterior, and whether the prior-to-posterior movement is consistent
    with the test results.

    Returns a contradiction dict if found, None if valid.
    """
    hypotheses = verdict.get("hypotheses_tested", [])
    if not hypotheses:
        return None

    verdict_name = verdict.get("verdict_hypothesis", "")

    # Find the actual highest-posterior hypothesis
    best = max(hypotheses, key=lambda h: h.get("posterior", 0))
    best_name = best.get("hypothesis", "")

    # Check 1a: verdict matches highest posterior
    if best_name != verdict_name:
        return {
            "type": "verdict_posterior_mismatch",
            "detail": (
                f"Verdict claims '{verdict_name}' but highest posterior "
                f"is '{best_name}' ({best.get('posterior', 0):.2f} vs "
                f"{_find_posterior(hypotheses, verdict_name):.2f})"
            ),
        }

    # Check 1b: posterior should move in the direction of test results
    winner = _find_hypothesis(hypotheses, verdict_name)
    if winner:
        predictions = winner.get("refutation_predictions", [])
        tested = [p for p in predictions if p.get("tested")]
        survived = [p for p in tested if p.get("survived")]
        failed = [p for p in tested if not p.get("survived")]

        prior = winner.get("prior", 0)
        posterior = winner.get("posterior", 0)

        # If more predictions failed than survived, posterior should have
        # decreased — if it increased, the update logic is inconsistent
        if len(failed) > len(survived) and posterior > prior * 1.1:
            return {
                "type": "posterior_direction_inconsistency",
                "detail": (
                    f"'{verdict_name}' had {len(failed)} failed vs "
                    f"{len(survived)} survived predictions, but posterior "
                    f"increased from {prior:.2f} to {posterior:.2f}"
                ),
            }

    return None


def check_evidential_sufficiency(verdict: dict) -> dict | None:
    """
    Check 2: is the evidence strong enough to warrant the conclusion?

    A fraud verdict needs minimum rigour: at least 2 tested predictions,
    at least one with diagnostic_weight >= 0.5. A legit verdict is the
    conservative default and needs less rigour.

    Returns an insufficiency dict if found, None if sufficient.
    """
    is_fraud = verdict.get("fraud", False)
    if not is_fraud:
        # Legit is the default — lower bar for sufficiency
        return None

    hypotheses = verdict.get("hypotheses_tested", [])
    verdict_name = verdict.get("verdict_hypothesis", "")
    winner = _find_hypothesis(hypotheses, verdict_name)

    if not winner:
        return {
            "type": "no_winning_hypothesis_found",
            "detail": f"Verdict names '{verdict_name}' but it's not in hypotheses_tested",
        }

    predictions = winner.get("refutation_predictions", [])
    tested = [p for p in predictions if p.get("tested")]
    survived = [p for p in tested if p.get("survived")]
    high_weight_survived = [
        p for p in survived if p.get("diagnostic_weight", 0) >= 0.5
    ]

    # Minimum rigour for fraud: at least 2 tested, at least 1 high-weight survived
    if len(tested) < 2:
        return {
            "type": "insufficient_testing",
            "detail": (
                f"Fraud verdict based on only {len(tested)} tested predictions "
                f"(minimum 2 required for sufficient rigour)"
            ),
        }

    if len(high_weight_survived) == 0:
        return {
            "type": "no_high_weight_evidence",
            "detail": (
                f"Fraud verdict has no survived predictions with "
                f"diagnostic_weight >= 0.5. Evidence is too weak to "
                f"support a fraud conclusion."
            ),
        }

    # Check margin: winning hypothesis should lead runner-up by meaningful gap
    posteriors = sorted(
        [h.get("posterior", 0) for h in hypotheses], reverse=True
    )
    if len(posteriors) >= 2:
        margin = posteriors[0] - posteriors[1]
        if margin < 0.05:
            return {
                "type": "razor_thin_margin",
                "detail": (
                    f"Winning hypothesis leads by only {margin:.3f}. "
                    f"Too close to call with confidence."
                ),
            }

    return None


def check_undercutting_defeaters(verdict: dict, context: dict) -> dict | None:
    """
    Check 3: is there unexamined evidence that should have been considered?

    Scans the context bundle for high-diagnostic signals that are present
    but don't appear in any prediction's refutation_target or evidence field.
    If the Investigator ignored strong evidence, the verdict is suspect.

    Returns a defeater dict if found, None if no gaps.
    """
    if not context:
        return None

    # Collect all evidence fields the Investigator actually referenced
    hypotheses = verdict.get("hypotheses_tested", [])
    referenced_evidence: set[str] = set()

    for h in hypotheses:
        for pred in h.get("refutation_predictions", []):
            target = pred.get("refutation_target", "").lower()
            evidence = pred.get("evidence", "").lower()
            referenced_evidence.add(target)
            # Also extract field names mentioned in the evidence string
            for field_name in HIGH_DIAGNOSTIC_EVIDENCE:
                if field_name.lower() in evidence:
                    referenced_evidence.add(field_name.lower())

    # Check each high-diagnostic evidence type
    gaps: list[str] = []
    for field_name, (check_fn, description) in HIGH_DIAGNOSTIC_EVIDENCE.items():
        try:
            signal_present = check_fn(context)
        except Exception:
            continue

        if signal_present:
            # Is this evidence type referenced in any prediction?
            field_lower = field_name.lower()
            # Check both exact field name and partial matches
            was_referenced = any(
                field_lower in ref or ref in field_lower
                for ref in referenced_evidence
            )
            # Also check common abbreviations
            abbreviations = {
                "sms_fraud_signals": ["sms", "phishing"],
                "email_fraud_signals": ["email"],
                "audio_fraud_signals": ["audio", "voice", "call"],
                "gps_mismatch": ["gps", "location"],
                "high_amount_deviation": ["amount", "zscore", "z_score"],
                "no_recent_activity": ["recent", "history", "frequency"],
            }
            for abbr in abbreviations.get(field_name, []):
                if any(abbr in ref for ref in referenced_evidence):
                    was_referenced = True
                    break

            if not was_referenced:
                gaps.append(description)

    if gaps:
        return {
            "type": "undercutting_defeater",
            "detail": (
                f"High-diagnostic evidence present but not examined: "
                f"{'; '.join(gaps)}"
            ),
        }

    return None


# ---------------------------------------------------------------------------
# LLM-based verification (optional, for deeper analysis at L5)
# ---------------------------------------------------------------------------

def build_critic_prompt(verdict: dict, context: dict) -> str:
    """
    Build a prompt for LLM-based critic verification.

    Used only when rule-based checks are inconclusive and budget allows.
    The LLM examines the full reasoning chain for subtle contradictions
    that rule-based checks can't catch.
    """
    # Compact representation of the verdict's reasoning
    hyp_summary = []
    for h in verdict.get("hypotheses_tested", []):
        preds = h.get("refutation_predictions", [])
        tested = [p for p in preds if p.get("tested")]
        survived = sum(1 for p in tested if p.get("survived"))
        failed = len(tested) - survived
        hyp_summary.append(
            f"  {h['hypothesis']}: prior={h.get('prior', 0):.2f}, "
            f"posterior={h.get('posterior', 0):.2f}, "
            f"tested={len(tested)}, survived={survived}, failed={failed}"
        )

    risk_flags = context.get("risk_flags", []) if context else []
    flags_str = "\n  ".join(risk_flags) if risk_flags else "(none)"

    return f"""You are auditing a fraud investigation verdict for internal consistency.

VERDICT: {verdict.get('verdict_hypothesis')} (fraud={verdict.get('fraud')}, confidence={verdict.get('confidence', 0):.2f})
REASON GIVEN: {verdict.get('reason', '')}

HYPOTHESIS AUDIT TRAIL:
{chr(10).join(hyp_summary)}

CONTEXT RISK FLAGS:
  {flags_str}

Check for:
1. Does the verdict match the posterior rankings?
2. Is there enough tested evidence to support the conclusion?
3. Are any context risk flags ignored in the reasoning?

Return JSON only:
{{
  "agree": true/false,
  "contradiction": "specific contradiction found, or null",
  "evidence_gap": "specific evidence ignored, or null",
  "insufficiency": "why evidence is too thin, or null",
  "adjusted_confidence": float (0-1, same or lower than original)
}}
"""


# ---------------------------------------------------------------------------
# Memory feedback structure
# ---------------------------------------------------------------------------

@dataclass
class CriticFeedback:
    """
    Accumulated feedback for Memory's cross-level learning.

    Tracks patterns in Investigator failures so the system can adapt:
    which evidence types get ignored, which hypothesis types produce
    weak verdicts, how often the Critic disagrees.
    """
    total_reviewed: int = 0
    total_disagreements: int = 0
    contradiction_types: dict[str, int] = field(default_factory=dict)
    evidence_gaps: dict[str, int] = field(default_factory=dict)
    insufficiency_types: dict[str, int] = field(default_factory=dict)

    def record(self, result: dict) -> None:
        """Record a single critic verification result."""
        self.total_reviewed += 1
        if not result.get("agree", True):
            self.total_disagreements += 1

        contradiction = result.get("contradiction_found")
        if contradiction:
            ctype = contradiction.get("type", "unknown")
            self.contradiction_types[ctype] = (
                self.contradiction_types.get(ctype, 0) + 1
            )

        gap = result.get("evidence_gap")
        if gap:
            gtype = gap.get("type", "unknown")
            self.evidence_gaps[gtype] = (
                self.evidence_gaps.get(gtype, 0) + 1
            )

        insufficiency = result.get("insufficiency")
        if insufficiency:
            itype = insufficiency.get("type", "unknown")
            self.insufficiency_types[itype] = (
                self.insufficiency_types.get(itype, 0) + 1
            )

    def disagreement_rate(self) -> float:
        if self.total_reviewed == 0:
            return 0.0
        return self.total_disagreements / self.total_reviewed

    def summary(self) -> dict:
        """Summary for Memory to store and for Orchestrator to monitor."""
        return {
            "total_reviewed": self.total_reviewed,
            "total_disagreements": self.total_disagreements,
            "disagreement_rate": round(self.disagreement_rate(), 3),
            "top_contradictions": dict(
                sorted(self.contradiction_types.items(),
                       key=lambda x: -x[1])[:5]
            ),
            "top_evidence_gaps": dict(
                sorted(self.evidence_gaps.items(),
                       key=lambda x: -x[1])[:5]
            ),
            "top_insufficiencies": dict(
                sorted(self.insufficiency_types.items(),
                       key=lambda x: -x[1])[:5]
            ),
        }

    def investigator_guidance(self) -> list[str]:
        """
        Generate plain-English guidance for the Investigator prompt.

        Memory can inject this into the Investigator's system prompt
        on subsequent calls to address recurring weaknesses.
        """
        guidance = []

        # Flag commonly ignored evidence
        for gap_type, count in sorted(
            self.evidence_gaps.items(), key=lambda x: -x[1]
        ):
            if count >= 2:
                guidance.append(
                    f"CRITIC NOTE: the Investigator has repeatedly ignored "
                    f"'{gap_type}' evidence ({count} times). "
                    f"Ensure predictions reference this evidence type."
                )

        # Flag common insufficiencies
        for itype, count in sorted(
            self.insufficiency_types.items(), key=lambda x: -x[1]
        ):
            if count >= 2:
                guidance.append(
                    f"CRITIC NOTE: verdicts have been insufficiently supported "
                    f"({itype}, {count} times). Test more predictions before "
                    f"concluding."
                )

        # General disagreement warning
        if self.disagreement_rate() > 0.3 and self.total_reviewed >= 5:
            guidance.append(
                f"CRITIC NOTE: the Critic has disagreed with "
                f"{self.disagreement_rate()*100:.0f}% of verdicts. "
                f"Exercise additional rigour in testing predictions."
            )

        return guidance


# ---------------------------------------------------------------------------
# Critic Agent
# ---------------------------------------------------------------------------

class CriticAgent:
    """
    Role C — Critic Agent (L4-L5 only).

    Examines the Investigator's reasoning structure for internal contradictions
    and evidential gaps. Does not re-investigate. Feeds back into Memory
    (cross-level learning), the verdict (confidence adjustment), and the
    Orchestrator (disagreement monitoring).

    Usage:
        critic = CriticAgent(llm_client=my_client)  # LLM optional
        result = critic.verify(verdict, context)
        memory.update(tx, verdict=result["final_verdict"],
                      confidence=result["adjusted_confidence"])

    The three rule-based checks run always (free). The LLM check runs only
    when rule-based checks are inconclusive and use_llm=True.
    """

    def __init__(self, llm_client: Any = None, use_llm: bool = False):
        self._llm = llm_client
        self._use_llm = use_llm and llm_client is not None
        self.feedback = CriticFeedback()

    def verify(self, verdict: dict, context: dict) -> dict:
        """
        Verify an Investigator verdict.

        Returns:
            agree: bool
            adjusted_confidence: float
            final_verdict: str ('fraud' or 'legit')
            contradiction_found: dict | None
            evidence_gap: dict | None
            insufficiency: dict | None
            reason: str
        """
        # Skip fallback verdicts — rule-based verdicts don't have
        # hypothesis structure to examine
        if verdict.get("fallback", False):
            result = {
                "agree": True,
                "adjusted_confidence": verdict.get("confidence", 0.5),
                "final_verdict": "fraud" if verdict.get("fraud") else "legit",
                "contradiction_found": None,
                "evidence_gap": None,
                "insufficiency": None,
                "reason": "Fallback verdict — no hypothesis structure to critique.",
            }
            self.feedback.record(result)
            return result

        # Run the three checks
        contradiction = check_logical_validity(verdict)
        insufficiency = check_evidential_sufficiency(verdict)
        defeater = check_undercutting_defeaters(verdict, context)

        # Determine agreement and confidence adjustment
        original_confidence = verdict.get("confidence", 0.5)
        problems = []
        confidence_penalty = 0.0

        if contradiction:
            problems.append(f"Contradiction: {contradiction['detail']}")
            confidence_penalty += 0.25

        if insufficiency:
            problems.append(f"Insufficiency: {insufficiency['detail']}")
            confidence_penalty += 0.20

        if defeater:
            problems.append(f"Evidence gap: {defeater['detail']}")
            confidence_penalty += 0.15

        # Optional LLM check for borderline cases
        llm_result = None
        if self._use_llm and not problems and original_confidence < 0.7:
            llm_result = self._llm_verify(verdict, context)
            if llm_result and not llm_result.get("agree", True):
                problems.append(f"LLM critique: {llm_result.get('contradiction') or llm_result.get('evidence_gap') or 'disagreed'}")
                confidence_penalty += 0.15

        adjusted_confidence = max(0.0, original_confidence - confidence_penalty)

        # Agreement decision
        agree = len(problems) == 0

        # If Critic disagrees on a fraud verdict, default to conservative
        # (legitimate). If Critic disagrees on a legit verdict, flag but
        # don't override — false negatives are harder to catch.
        original_is_fraud = verdict.get("fraud", False)
        if not agree and original_is_fraud and confidence_penalty >= 0.25:
            final_verdict = "legit"
        else:
            final_verdict = "fraud" if original_is_fraud else "legit"

        reason = "; ".join(problems) if problems else "Verdict reasoning is internally consistent."

        result = {
            "agree": agree,
            "adjusted_confidence": round(adjusted_confidence, 3),
            "final_verdict": final_verdict,
            "contradiction_found": contradiction,
            "evidence_gap": defeater,
            "insufficiency": insufficiency,
            "reason": reason,
        }

        self.feedback.record(result)

        if not agree:
            logger.info(
                f"Critic DISAGREES: {reason} "
                f"(confidence {original_confidence:.2f} → {adjusted_confidence:.2f}, "
                f"verdict {original_is_fraud} → {final_verdict})"
            )

        return result

    def get_investigator_guidance(self) -> list[str]:
        """
        Get accumulated guidance for the Investigator prompt.

        Memory should call this at level boundaries and inject the
        guidance into subsequent Investigator system prompts.
        """
        return self.feedback.investigator_guidance()

    def get_feedback_summary(self) -> dict:
        """
        Summary for Orchestrator monitoring and Memory storage.
        """
        return self.feedback.summary()

    # -- LLM verification (optional) ----------------------------------------

    def _llm_verify(self, verdict: dict, context: dict) -> dict | None:
        """
        Optional LLM-based verification for borderline cases.

        Only called when rule-based checks find nothing but confidence
        is below threshold. Returns parsed JSON or None on failure.
        """
        if not self._llm:
            return None

        try:
            prompt = build_critic_prompt(verdict, context)
            response = self._call_llm(prompt, max_tokens=300)
            return self._parse_llm_response(response)
        except Exception as e:
            logger.warning(f"Critic LLM verification failed: {e}")
            return None

    def _call_llm(self, prompt: str, max_tokens: int = 300) -> str:
        if hasattr(self._llm, "generate"):
            return self._llm.generate(prompt, max_tokens=max_tokens)
        if hasattr(self._llm, "complete"):
            return self._llm.complete(prompt, max_tokens=max_tokens)
        if callable(self._llm):
            return self._llm(prompt)
        raise RuntimeError("LLM client has no recognised interface")

    @staticmethod
    def _parse_llm_response(response: str) -> dict | None:
        if not response:
            return None
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_hypothesis(hypotheses: list[dict], name: str) -> dict | None:
    for h in hypotheses:
        if h.get("hypothesis") == name:
            return h
    return None


def _find_posterior(hypotheses: list[dict], name: str) -> float:
    h = _find_hypothesis(hypotheses, name)
    return h.get("posterior", 0.0) if h else 0.0
