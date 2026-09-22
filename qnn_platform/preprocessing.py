"""Dataset loading, feature typing, cleaning, scaling and dimensionality reduction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler


@dataclass
class PreparedData:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    task: str  # "classification" | "regression"
    class_names: list[str] | None
    n_qubits: int


def infer_column_types(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric and categorical, excluding the target."""
    feature_df = df.drop(columns=[target])
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def infer_task(df: pd.DataFrame, target: str) -> str:
    """Guess classification vs regression from the target column's dtype/cardinality."""
    y = df[target]
    is_numeric = y.dtype.kind in "iuf"
    if not is_numeric:
        return "classification"
    n_unique = y.nunique()
    is_integer = y.dtype.kind in "iu"
    if is_integer and n_unique <= max(20, int(0.05 * len(y))):
        return "classification"
    if n_unique <= 15:
        return "classification"
    return "regression"


def build_feature_pipeline(numeric_cols: list[str], categorical_cols: list[str],
                            scaler: str = "minmax") -> ColumnTransformer:
    """A ColumnTransformer that imputes, encodes categoricals, and scales numerics
    into [0, 1] (minmax, the default) or zero-mean/unit-variance (standard) so
    that features are ready for angle/amplitude quantum encoding."""
    scaler_obj = MinMaxScaler(feature_range=(0, np.pi)) if scaler == "minmax" else StandardScaler()

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", scaler_obj),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers)


def reduce_dimensionality(X: np.ndarray, y: np.ndarray, task: str, n_components: int,
                           method: str = "pca") -> tuple[np.ndarray, object | None]:
    """Reduce X to n_components features so it fits within available qubits.
    Returns the transformed X and the fitted reducer (or None if no reduction needed)."""
    if X.shape[1] <= n_components:
        return X, None

    if method == "pca":
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
    elif method == "kbest":
        score_func = f_classif if task == "classification" else f_regression
        reducer = SelectKBest(score_func=score_func, k=n_components)
        X_reduced = reducer.fit_transform(X, y)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    return X_reduced, reducer


def prepare_dataset(df: pd.DataFrame, target: str, task: str | None, n_qubits: int,
                     reduction_method: str = "pca", scaler: str = "minmax") -> PreparedData:
    """Full pipeline: clean -> encode categoricals -> scale -> reduce to n_qubits features."""
    df = df.dropna(subset=[target]).reset_index(drop=True)
    numeric_cols, categorical_cols = infer_column_types(df, target)
    task = task or infer_task(df, target)

    class_names = None
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(df[target].astype(str))
        class_names = list(le.classes_)
    else:
        y = df[target].astype(float).to_numpy()

    pipeline = build_feature_pipeline(numeric_cols, categorical_cols, scaler=scaler)
    X = pipeline.fit_transform(df.drop(columns=[target]))
    X = np.asarray(X, dtype=float)

    feature_names = numeric_cols + categorical_cols
    X_reduced, reducer = reduce_dimensionality(X, y, task, n_qubits, method=reduction_method)

    if reducer is not None:
        feature_names = [f"component_{i}" for i in range(X_reduced.shape[1])]
        # Re-scale post-reduction (PCA output is not bounded) for angle encoding.
        rescaler = MinMaxScaler(feature_range=(0, np.pi))
        X_reduced = rescaler.fit_transform(X_reduced)

    return PreparedData(
        X=X_reduced, y=y, feature_names=feature_names, task=task,
        class_names=class_names, n_qubits=X_reduced.shape[1],
    )
