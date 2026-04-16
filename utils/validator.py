"""
utils/validator.py — Role A (critical safeguard).

Must be called before writing any submission file.
Raises SubmissionError on invalid output — do NOT bypass.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MIN_RECALL_FLOOR = 0.15  # competition auto-reject threshold


class SubmissionError(Exception):
    """Raised when the submission fails validation."""


def validate(
    fraud_ids: list[str],
    eval_df: pd.DataFrame,
    tx_id_col: str = "transaction_id",
) -> None:
    """
    Validate a candidate submission.

    Parameters
    ----------
    fraud_ids:
        List of transaction IDs we intend to flag as fraud.
    eval_df:
        The evaluation DataFrame (all transactions for this level).
    tx_id_col:
        Column name for transaction IDs in eval_df.

    Raises
    ------
    SubmissionError
        If the submission is empty, flags everyone, or is obviously broken.
    """
    total = len(eval_df)

    if total == 0:
        raise SubmissionError("eval_df is empty — cannot validate.")

    if len(fraud_ids) == 0:
        raise SubmissionError(
            "Submission is empty — no fraud IDs flagged. "
            "This will be auto-rejected by the competition."
        )

    flagged_pct = len(fraud_ids) / total
    if flagged_pct >= 0.99:
        raise SubmissionError(
            f"Submission flags {flagged_pct:.1%} of all transactions. "
            "Flagging everything is auto-rejected."
        )

    # Check for duplicate IDs
    dupes = len(fraud_ids) - len(set(fraud_ids))
    if dupes > 0:
        raise SubmissionError(f"Submission contains {dupes} duplicate transaction IDs.")

    # Check all IDs exist in eval set
    known_ids = set(eval_df[tx_id_col].astype(str))
    unknown = [fid for fid in fraud_ids if str(fid) not in known_ids]
    if unknown:
        raise SubmissionError(
            f"Submission contains {len(unknown)} transaction IDs not found in "
            f"eval_df: {unknown[:5]} …"
        )

    logger.info(
        "validator | PASSED — flagged %d / %d (%.1f%%)",
        len(fraud_ids),
        total,
        flagged_pct * 100,
    )


def write_submission(fraud_ids: list[str], output_path: Path) -> None:
    """
    Write validated fraud IDs to the submission file.
    Always call validate() first — this function does NOT re-validate.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(str(fid) for fid in fraud_ids) + "\n")
    logger.info("validator | Wrote %d IDs to %s", len(fraud_ids), output_path)
