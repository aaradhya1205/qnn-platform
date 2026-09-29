"""Medical-sector preprocessing: domain-aware cleaning applied before the
general pipeline, for clinical/health datasets specifically.

Three concerns that generic median/mode imputation glosses over in medical
data:
  1. Identifier and free-text columns (patient ID, name, MRN) leak nothing
     useful and must never reach the model.
  2. Missingness in clinical data is often informative, not random - a lab
     test that was never ordered says something about the patient, so the
     fact it's missing is preserved as its own feature before imputing.
  3. Data-entry errors and implausible readings (a negative age, a blood
     pressure of zero) should be clipped, not left to silently distort
     scaling and, downstream, the quantum encoding's rotation angles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IDENTIFIER_KEYWORDS = [
    "patient_id", "patientid", "mrn", "ssn", "social_security",
    "name", "first_name", "last_name", "address", "phone", "email",
    "dob", "date_of_birth", "record_id", "id",
]


def find_identifier_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Column names that look like patient identifiers or free-text PII,
    matched as whole words/segments so real clinical features like
    'valid' or 'avoid' aren't caught by the 'id' keyword."""
    found = []
    for col in df.columns:
        if col == target:
            continue
        normalized = col.strip().lower().replace(" ", "_").replace("-", "_")
        tokens = normalized.split("_")
        if any(kw in tokens or normalized == kw for kw in IDENTIFIER_KEYWORDS):
            found.append(col)
    return found


def add_missingness_indicators(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """For every feature column with missing values, add a binary
    '<col>_was_missing' column before imputation runs, so the model can
    learn from the fact a value was absent, not just its filled-in value."""
    df = df.copy()
    for col in df.columns:
        if col == target:
            continue
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)
    return df


def clip_physiological_outliers(df: pd.DataFrame, target: str, iqr_factor: float = 3.0) -> pd.DataFrame:
    """Clip numeric feature columns to [Q1 - factor*IQR, Q3 + factor*IQR].
    A wide factor (3x IQR, vs. the usual 1.5x for outlier *detection*) so
    only implausible data-entry errors are clipped, not real clinical
    extremes that are the whole reason a patient is being modeled."""
    df = df.copy()
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        lower, upper = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        df[col] = df[col].clip(lower, upper)
    return df


def apply_medical_preprocessing(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Run the medical-sector cleaning steps in order and return the
    cleaned frame plus the identifier columns that were dropped (for
    display in the UI)."""
    id_cols = find_identifier_columns(df, target)
    df = df.drop(columns=id_cols)
    df = add_missingness_indicators(df, target)
    df = clip_physiological_outliers(df, target)
    return df, id_cols
