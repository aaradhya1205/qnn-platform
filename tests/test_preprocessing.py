import numpy as np
import pandas as pd
import pytest

from qnn_platform.preprocessing import infer_task, prepare_dataset


def _iris_df():
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    df = data.frame.rename(columns={"target": "species"})
    df["species"] = df["species"].map(dict(enumerate(data.target_names)))
    return df


def test_infer_task_classification_for_string_target():
    df = _iris_df()
    assert infer_task(df, "species") == "classification"


def test_infer_task_regression_for_continuous_target():
    df = pd.DataFrame({
        "x": np.arange(200, dtype=float),
        "y": np.linspace(0, 100, 200) + np.random.RandomState(0).randn(200),
    })
    assert infer_task(df, "y") == "regression"


def test_prepare_dataset_reduces_to_requested_qubits():
    df = _iris_df()
    prepared = prepare_dataset(df, "species", "classification", n_qubits=2)
    assert prepared.X.shape == (150, 2)
    assert prepared.n_qubits == 2
    assert prepared.class_names == ["setosa", "versicolor", "virginica"]


def test_prepare_dataset_keeps_all_features_when_under_qubit_budget():
    df = _iris_df()
    prepared = prepare_dataset(df, "species", "classification", n_qubits=10)
    assert prepared.X.shape[1] == 4  # iris has 4 features, no reduction needed


def test_prepare_dataset_scales_features_into_bounded_range():
    df = _iris_df()
    prepared = prepare_dataset(df, "species", "classification", n_qubits=4)
    assert prepared.X.min() >= -1e-6
    assert prepared.X.max() <= np.pi + 1e-6


def test_prepare_dataset_drops_rows_with_missing_target():
    df = _iris_df()
    df.loc[0, "species"] = None
    prepared = prepare_dataset(df, "species", "classification", n_qubits=4)
    assert prepared.X.shape[0] == 149


def test_high_cardinality_text_columns_are_dropped_automatically():
    n = 120
    df = pd.DataFrame({
        "player_name": [f"Player {i}" for i in range(n)],     # all different -> useless when one-hot encoded
        "country": ["A", "B", "C"] * (n // 3),
        "runs": np.arange(n, dtype=float),
        "label": ["x", "y"] * (n // 2),
    })
    prepared = prepare_dataset(df, "label", "classification", n_qubits=3)
    assert prepared.dropped_high_cardinality == ["player_name"]
    assert prepared.X.shape[0] == n
