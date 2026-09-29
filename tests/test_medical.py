import numpy as np
import pandas as pd

from qnn_platform.medical import (
    add_missingness_indicators,
    apply_medical_preprocessing,
    clip_physiological_outliers,
    find_identifier_columns,
)
from qnn_platform.preprocessing import prepare_dataset


def _toy_clinical_df():
    rng = np.random.RandomState(0)
    n = 60
    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "name": [f"Patient {i}" for i in range(n)],
        "age": rng.randint(20, 90, n).astype(float),
        "blood_pressure": rng.normal(120, 15, n),
        "diagnosis": rng.choice(["healthy", "at_risk"], n),
    })


def test_find_identifier_columns_matches_known_pii():
    df = _toy_clinical_df()
    found = find_identifier_columns(df, target="diagnosis")
    assert "patient_id" in found
    assert "name" in found
    assert "age" not in found
    assert "blood_pressure" not in found
    assert "diagnosis" not in found  # never flags the target itself


def test_find_identifier_columns_does_not_false_positive_on_substrings():
    df = pd.DataFrame({"valid": [1, 2], "avoid": [3, 4], "target": [0, 1]})
    found = find_identifier_columns(df, target="target")
    assert found == []  # "id" inside "valid"/"avoid" must not match


def test_add_missingness_indicators_only_for_columns_with_gaps():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1, 2, 3], "target": [0, 1, 0]})
    out = add_missingness_indicators(df, target="target")
    assert "a_was_missing" in out.columns
    assert "b_was_missing" not in out.columns
    assert list(out["a_was_missing"]) == [0, 1, 0]


def test_clip_physiological_outliers_bounds_extreme_values():
    df = pd.DataFrame({"age": [30, 31, 29, 32, 30, 10000], "target": [0, 1, 0, 1, 0, 1]})
    out = clip_physiological_outliers(df, target="target")
    assert out["age"].max() < 10000


def test_apply_medical_preprocessing_drops_identifiers_and_reports_them():
    df = _toy_clinical_df()
    cleaned, dropped = apply_medical_preprocessing(df, target="diagnosis")
    assert "patient_id" not in cleaned.columns
    assert "name" not in cleaned.columns
    assert set(dropped) == {"patient_id", "name"}
    assert "diagnosis" in cleaned.columns


def test_prepare_dataset_medical_sector_end_to_end():
    df = _toy_clinical_df()
    prepared = prepare_dataset(df, "diagnosis", "classification", n_qubits=3, sector="medical")
    assert prepared.dropped_identifier_columns is not None
    assert set(prepared.dropped_identifier_columns) == {"patient_id", "name"}
    assert prepared.X.shape[0] == 60


def test_prepare_dataset_general_sector_keeps_all_columns():
    df = _toy_clinical_df()
    prepared = prepare_dataset(df, "diagnosis", "classification", n_qubits=3, sector="general")
    assert prepared.dropped_identifier_columns is None
