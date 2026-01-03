"""
src/fairness_utils.py
=====================
Shared utility functions for the TalentLens bias and governance analysis.

Functions
---------
disparate_impact(df, outcome_col, group_col, privileged, unprivileged)
    Compute the Disparate Impact ratio between two groups.

pseudonymise(value, salt)
    Return a SHA-256 hash of value+salt, truncated to 16 characters.
"""

import hashlib
import pandas as pd


def disparate_impact(
    df: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    privileged: str,
    unprivileged: str,
) -> float:
    """
    Compute the Disparate Impact (DI) ratio.

    DI = approval_rate(unprivileged) / approval_rate(privileged)

    A value below 0.8 indicates potential disparate impact under
    the four-fifths rule (EEOC Uniform Guidelines, 1978).

    Parameters
    ----------
    df : pd.DataFrame
    outcome_col : str  — binary column (True/False or 1/0)
    group_col : str    — column containing group labels
    privileged : str   — label of the privileged group (denominator)
    unprivileged : str — label of the unprivileged group (numerator)

    Returns
    -------
    float : DI ratio
    """
    rate_priv   = df[df[group_col] == privileged][outcome_col].mean()
    rate_unpriv = df[df[group_col] == unprivileged][outcome_col].mean()

    if rate_priv == 0:
        raise ValueError("Privileged group approval rate is zero — DI undefined.")

    return rate_unpriv / rate_priv


def pseudonymise(value: str, salt: str) -> str:
    """
    Pseudonymise a string value using SHA-256 hashing with a salt.

    The original value cannot be recovered without the salt.
    The data controller retains the salt to handle GDPR Art. 17
    erasure requests.

    Parameters
    ----------
    value : str — the original identifier (e.g., applicant ID)
    salt  : str — a secret string known only to the data controller

    Returns
    -------
    str : 16-character hex digest
    """
    return hashlib.sha256(f"{salt}{value}".encode()).hexdigest()[:16]
