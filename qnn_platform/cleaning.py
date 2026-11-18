"""Data quality profiling and cleaning.

Cleaning is split in two, on purpose:

* **Structural cleaning** (`clean_structural`) only looks at rows and columns, never at the
  labels, so it is safe to run on the whole dataset before the train/test split:
  text tidy-up, number parsing, sparse rows/columns, duplicates, constant columns, ID-like columns.
* **Fitted cleaning** (`FittedCleaner`) learns numbers from data - the median used to fill gaps,
  the outlier limits. It is fitted on the training rows only and then applied to the test rows,
  so nothing from the test set leaks into training.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "#n/a"}
ID_NAME_TOKENS = {"id", "uuid", "guid", "index", "key", "name", "email", "phone", "address", "url", "link", "mrn", "ssn"}
_NUMBER_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
_STRIP_CHARS = re.compile(r"[,%$€£*\s]")


@dataclass
class CleaningConfig:
    # structural (label-free, run before the split)
    standardize_text: bool = True
    parse_numeric_text: bool = True
    drop_sparse_cols_pct: float = 0.6     # drop columns missing more than this share
    drop_sparse_rows_pct: float = 0.6     # drop rows missing more than this share of features
    drop_duplicates: bool = True
    drop_constant: bool = True
    drop_id_like: bool = True
    # fitted on the training rows only (run after the split)
    impute_numeric: str = "median"        # median | mean | zero
    impute_categorical: str = "most_frequent"   # most_frequent | missing_label
    add_missing_flags: bool = False
    outliers: str = "none"                # none | clip | drop_train
    outlier_iqr: float = 3.0


@dataclass
class Issue:
    severity: str   # high | medium | low
    title: str
    detail: str
    action: str


@dataclass
class DataQualityReport:
    n_rows: int
    n_cols: int
    missing_cells: int
    missing_pct: float
    missing_by_col: pd.Series
    duplicate_rows: int
    constant_cols: list[str]
    id_like_cols: list[str]
    numeric_text_cols: list[str]
    outlier_cells: int
    outliers_by_col: pd.Series
    target_missing: int
    score: int
    issues: list[Issue] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _is_text(series: pd.Series) -> bool:
    return series.dtype == object or str(series.dtype) in ("string", "category")


def _parse_number(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if text.lower() in NULL_TOKENS:
        return np.nan
    cleaned = _STRIP_CHARS.sub("", text)
    if _NUMBER_RE.match(cleaned):
        return float(cleaned)
    return None   # not a number


def numeric_text_share(series: pd.Series) -> float:
    """Share of non-empty values in a text column that look like numbers (e.g. '85%', '1,200', '12*')."""
    values = series.dropna()
    if len(values) == 0:
        return 0.0
    parsed = [_parse_number(v) for v in values]
    non_null_tokens = [p for p, v in zip(parsed, values) if str(v).strip().lower() not in NULL_TOKENS]
    if not non_null_tokens:
        return 0.0
    return sum(p is not None for p in non_null_tokens) / len(non_null_tokens)


def _name_tokens(name: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", str(name).lower().strip()))


def _iqr_bounds(series: pd.Series, factor: float):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return None
    return q1 - factor * iqr, q3 + factor * iqr


def _is_id_like(series: pd.Series, name: str, n_rows: int) -> bool:
    n_unique = series.nunique(dropna=True)
    if n_unique == 0 or n_rows == 0:
        return False
    ratio = n_unique / n_rows
    if _is_text(series):
        return (n_unique >= 30 and ratio > 0.5) or (bool(_name_tokens(name) & ID_NAME_TOKENS) and ratio > 0.8)
    if series.dtype.kind in "iu":
        return bool(_name_tokens(name) & ID_NAME_TOKENS) and ratio > 0.9
    return False


# --------------------------------------------------------------------------- #
# profiling
# --------------------------------------------------------------------------- #
def profile_dataframe(df: pd.DataFrame, target: str | None = None, outlier_factor: float = 3.0) -> DataQualityReport:
    n_rows, n_cols = df.shape
    features = df.drop(columns=[target]) if target in df.columns else df
    missing_by_col = df.isna().sum()
    missing_cells = int(missing_by_col.sum())
    n_cells = max(n_rows * n_cols, 1)

    duplicate_rows = int(df.duplicated().sum())
    constant_cols = [c for c in features.columns if features[c].nunique(dropna=True) <= 1]
    numeric_text_cols = [c for c in features.columns if _is_text(features[c]) and numeric_text_share(features[c]) >= 0.8]
    id_like_cols = [c for c in features.columns
                    if c not in numeric_text_cols and c not in constant_cols and _is_id_like(features[c], c, n_rows)]

    outliers = {}
    for c in features.columns:
        s = features[c]
        if s.dtype.kind in "iuf" and c not in id_like_cols and c not in constant_cols:
            bounds = _iqr_bounds(s.dropna(), outlier_factor)
            if bounds is not None:
                count = int(((s < bounds[0]) | (s > bounds[1])).sum())
                if count:
                    outliers[c] = count
    outliers_by_col = pd.Series(outliers, dtype=int).sort_values(ascending=False)
    outlier_cells = int(outliers_by_col.sum()) if len(outliers_by_col) else 0
    target_missing = int(df[target].isna().sum()) if target in df.columns else 0

    # ---- score: transparent, simple deductions out of 100
    dup_pct = duplicate_rows / max(n_rows, 1)
    out_pct = outlier_cells / max(n_rows * max(features.shape[1], 1), 1)
    score = 100.0
    score -= min(30, missing_cells / n_cells * 100 * 1.5)
    score -= min(15, dup_pct * 100 * 3)
    score -= min(10, 5 * len(constant_cols))
    score -= min(10, 5 * len(id_like_cols))
    score -= min(12, 4 * len(numeric_text_cols))
    score -= min(15, out_pct * 100 * 3)
    score -= 10 if target_missing else 0
    score = int(round(max(0, score)))

    issues: list[Issue] = []
    if target_missing:
        issues.append(Issue("high", f"{target_missing} rows have no target value",
                            "Rows without the answer cannot be used for learning.", "Drop those rows (always done)."))
    if missing_cells:
        worst = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
        top = ", ".join(f"{c} ({int(n)})" for c, n in worst.head(3).items())
        sev = "high" if missing_cells / n_cells > 0.1 else "medium"
        issues.append(Issue(sev, f"{missing_cells:,} missing cells ({missing_cells / n_cells:.1%})",
                            f"Most affected: {top}.", "Fill gaps with the median/most frequent value, learned from training rows only."))
    if duplicate_rows:
        issues.append(Issue("medium", f"{duplicate_rows} duplicate rows",
                            "Identical rows can end up in both training and test sets and inflate scores.", "Remove duplicates."))
    if numeric_text_cols:
        issues.append(Issue("medium", f"{len(numeric_text_cols)} number column(s) stored as text",
                            ", ".join(numeric_text_cols) + " (values like '85%', '1,200', '12*').", "Convert to numbers."))
    if id_like_cols:
        issues.append(Issue("medium", f"{len(id_like_cols)} ID-like column(s)",
                            ", ".join(id_like_cols) + ": almost every value is different, so they carry no pattern and can leak identity.",
                            "Drop them."))
    if constant_cols:
        issues.append(Issue("low", f"{len(constant_cols)} constant column(s)", ", ".join(constant_cols) + ": the same value in every row.", "Drop them."))
    if outlier_cells:
        top = ", ".join(f"{c} ({n})" for c, n in outliers_by_col.head(3).items())
        issues.append(Issue("medium", f"{outlier_cells} extreme values",
                            f"Far outside the usual range ({outlier_factor:g}×IQR). Most affected: {top}.",
                            "Clip them to the limit, using limits learned from training rows."))
    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: order[i.severity])

    return DataQualityReport(n_rows, n_cols, missing_cells, missing_cells / n_cells, missing_by_col, duplicate_rows,
                             constant_cols, id_like_cols, numeric_text_cols, outlier_cells, outliers_by_col,
                             target_missing, score, issues)


def recommended_config(report: DataQualityReport) -> CleaningConfig:
    """Sensible defaults based on what the profile found."""
    return CleaningConfig(
        outliers="clip" if report.outlier_cells else "none",
        add_missing_flags=False,
    )


# --------------------------------------------------------------------------- #
# structural cleaning
# --------------------------------------------------------------------------- #
@dataclass
class CleaningLogEntry:
    step: str
    detail: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    cells_changed: int = 0


def clean_structural(df: pd.DataFrame, target: str, cfg: CleaningConfig) -> tuple[pd.DataFrame, list[CleaningLogEntry]]:
    """Label-free cleaning. Returns the cleaned frame and a step-by-step log."""
    df = df.copy()
    log: list[CleaningLogEntry] = []

    def record(step, detail, before, cells=0):
        log.append(CleaningLogEntry(step, detail, before[0], len(df), before[1], df.shape[1], cells))

    # 0. rows with no target
    before = df.shape
    df = df.dropna(subset=[target])
    if len(df) != before[0]:
        record("Drop rows without a target", f"{before[0] - len(df)} rows had no answer to learn from", before)

    feature_cols = [c for c in df.columns if c != target]

    # 1. tidy text
    if cfg.standardize_text:
        before, changed = df.shape, 0
        for c in feature_cols:
            if not _is_text(df[c]):
                continue
            original = df[c].copy()
            s = df[c].astype("object").where(df[c].notna(), None)
            s = s.map(lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v)
            s = s.map(lambda v: None if isinstance(v, str) and v.lower() in {"", "nan", "null", "none", "n/a", "na", "#n/a"} else v)
            if numeric_text_share(pd.Series(s)) < 0.8:   # real categories: merge 'North' / 'north '
                lowered = s.map(lambda v: v.lower() if isinstance(v, str) else v)
                if lowered.nunique(dropna=True) < pd.Series(s).nunique(dropna=True):
                    s = lowered
            df[c] = s
            changed += int((original.fillna("§") != df[c].fillna("§")).sum())
        if changed:
            record("Tidy text", "trimmed spaces, merged upper/lower-case variants, turned placeholders into gaps", before, changed)

    # 2. numbers stored as text
    if cfg.parse_numeric_text:
        before, converted, changed = df.shape, [], 0
        for c in feature_cols:
            if _is_text(df[c]) and numeric_text_share(df[c]) >= 0.8:
                parsed = df[c].map(_parse_number)
                parsed = parsed.map(lambda v: np.nan if v is None else v)
                changed += int(df[c].notna().sum())
                df[c] = pd.to_numeric(parsed, errors="coerce")
                converted.append(c)
        if converted:
            record("Convert text to numbers", ", ".join(converted), before, changed)

    # 3. sparse columns / rows
    if cfg.drop_sparse_cols_pct < 1.0:
        before = df.shape
        sparse = [c for c in df.columns if c != target and df[c].isna().mean() > cfg.drop_sparse_cols_pct]
        if sparse:
            df = df.drop(columns=sparse)
            record("Drop mostly-empty columns", ", ".join(sparse), before)
    if cfg.drop_sparse_rows_pct < 1.0:
        before = df.shape
        feats = [c for c in df.columns if c != target]
        if feats:
            keep = df[feats].isna().mean(axis=1) <= cfg.drop_sparse_rows_pct
            if (~keep).any():
                df = df[keep]
                record("Drop mostly-empty rows", f"{int((~keep).sum())} rows", before)

    # 4. duplicates
    if cfg.drop_duplicates:
        before = df.shape
        df = df.drop_duplicates()
        if len(df) != before[0]:
            record("Remove duplicate rows", f"{before[0] - len(df)} identical rows", before)

    # 5. constant columns
    if cfg.drop_constant:
        before = df.shape
        const = [c for c in df.columns if c != target and df[c].nunique(dropna=True) <= 1]
        if const:
            df = df.drop(columns=const)
            record("Drop constant columns", ", ".join(const), before)

    # 6. ID-like columns
    if cfg.drop_id_like:
        before = df.shape
        ids = [c for c in df.columns if c != target and _is_id_like(df[c], c, len(df))]
        if ids:
            df = df.drop(columns=ids)
            record("Drop ID-like columns", ", ".join(ids), before)

    return df.reset_index(drop=True), log


# --------------------------------------------------------------------------- #
# fitted cleaning (training rows only)
# --------------------------------------------------------------------------- #
class FittedCleaner:
    """Learns fill values and outlier limits from the training rows, then applies them anywhere."""

    def __init__(self, cfg: CleaningConfig):
        self.cfg = cfg
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.fill_values: dict[str, object] = {}
        self.bounds: dict[str, tuple[float, float]] = {}
        self.flag_cols: list[str] = []

    def fit(self, X: pd.DataFrame) -> "FittedCleaner":
        self.numeric_cols = [c for c in X.columns if X[c].dtype.kind in "iuf"]
        self.categorical_cols = [c for c in X.columns if c not in self.numeric_cols]
        for c in self.numeric_cols:
            s = X[c]
            if self.cfg.impute_numeric == "mean":
                value = s.mean()
            elif self.cfg.impute_numeric == "zero":
                value = 0.0
            else:
                value = s.median()
            self.fill_values[c] = 0.0 if not np.isfinite(value) else float(value)
            if self.cfg.outliers != "none":
                b = _iqr_bounds(s.dropna(), self.cfg.outlier_iqr)
                if b is not None:
                    self.bounds[c] = b
        for c in self.categorical_cols:
            if self.cfg.impute_categorical == "missing_label":
                self.fill_values[c] = "missing"
            else:
                mode = X[c].mode(dropna=True)
                self.fill_values[c] = mode.iloc[0] if len(mode) else "missing"
        if self.cfg.add_missing_flags:
            self.flag_cols = [c for c in X.columns if X[c].isna().any()]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.flag_cols:
            X[f"{c}_was_missing"] = X[c].isna().astype(int)
        for c, value in self.fill_values.items():
            if c in X.columns:
                X[c] = X[c].fillna(value)
        if self.cfg.outliers == "clip":
            for c, (lo, hi) in self.bounds.items():
                if c in X.columns:
                    X[c] = X[c].clip(lo, hi)
        return X

    def outlier_row_mask(self, X: pd.DataFrame) -> np.ndarray:
        """Rows with no value outside the learned limits (used by outliers='drop_train')."""
        keep = np.ones(len(X), dtype=bool)
        for c, (lo, hi) in self.bounds.items():
            if c in X.columns:
                v = X[c]
                keep &= ~(((v < lo) | (v > hi)).fillna(False).to_numpy())
        return keep
