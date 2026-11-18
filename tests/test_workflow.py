import numpy as np
import pandas as pd
import pytest

from qnn_platform.cleaning import CleaningConfig
from qnn_platform.demo_data import load_demo
from qnn_platform.workflow import (ModelConfig, PrepConfig, QNN_NAME, TrainConfig, prepare_split, run_study,
                                   run_sweep, verdict)


def _skewed_df():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({"a": rng.rand(n), "b": rng.rand(n) * 10, "c": rng.rand(n), "label": rng.choice(["x", "y"], n)})
    df.loc[rng.choice(n, 20, replace=False), "a"] = np.nan
    return df


def test_split_is_stratified_and_disjoint():
    s = prepare_split(_skewed_df(), PrepConfig(target="label", n_qubits=3, test_size=0.25))
    assert set(s.train_index).isdisjoint(s.val_index)
    assert abs(len(s.val_index) - 50) <= 1
    assert abs(s.y_train.mean() - s.y_val.mean()) < 0.08


def test_gap_filling_uses_training_rows_only():
    df = _skewed_df()
    s = prepare_split(df, PrepConfig(target="label", n_qubits=3, cleaning=CleaningConfig(impute_numeric="median")))
    train_median = df.loc[s.train_index, "a"].median()
    assert s.cleaner.fill_values["a"] == pytest.approx(train_median)
    assert s.cleaner.fill_values["a"] != pytest.approx(df["a"].median())


def test_scaling_is_fitted_on_training_rows_only():
    df = pd.DataFrame({"x": np.r_[np.linspace(0, 1, 150), np.full(50, 1000.0)], "y": np.r_[np.zeros(100), np.ones(100)],
                       "z": np.random.RandomState(1).rand(200), "label": np.r_[["a"] * 100, ["b"] * 100]})
    s = prepare_split(df, PrepConfig(target="label", n_qubits=3, test_size=0.3))
    assert s.X_train.min() >= -1e-6 and s.X_train.max() <= np.pi + 1e-6      # fitted range = training range
    assert s.X_val.min() >= -1e-6 and s.X_val.max() <= np.pi + 1e-6          # held-out values are clipped into it


def test_pca_is_reduced_to_the_qubit_budget_and_bounded():
    demo = load_demo("breast_cancer")
    s = prepare_split(demo.df, PrepConfig(target=demo.target, n_qubits=4))
    assert s.X_train.shape[1] == 4 and s.n_qubits == 4 and s.n_encoded == 30
    assert s.explained_variance is not None and len(s.explained_variance) == 4
    assert s.X_val.min() >= -1e-6 and s.X_val.max() <= np.pi + 1e-6


def test_kbest_and_standard_scaling_run():
    demo = load_demo("wine")
    s = prepare_split(demo.df, PrepConfig(target=demo.target, n_qubits=3, reduction="kbest", scaler="standard"))
    assert s.X_train.shape[1] == 3


def test_medical_domain_drops_identifiers_and_adds_flags():
    demo = load_demo("messy_clinical")
    from qnn_platform.cleaning import clean_structural
    clean, _ = clean_structural(demo.df.assign(patient_name="n"), demo.target, CleaningConfig(drop_id_like=False, drop_constant=False))
    s = prepare_split(clean, PrepConfig(target=demo.target, n_qubits=4, domain="medical"))
    assert "patient_id" in s.dropped_domain_columns
    assert any("was missing" in d for _, d in s.steps) or s.cleaner.flag_cols


def test_regression_split_has_float_target():
    demo = load_demo("diabetes")
    s = prepare_split(demo.df, PrepConfig(target=demo.target, task="regression", n_qubits=4))
    assert s.class_names is None and s.y_train.dtype.kind == "f"


def test_study_runs_with_seeds_baselines_and_verdict():
    demo = load_demo("iris")
    res = run_study(demo.df, PrepConfig(target=demo.target, n_qubits=4), ModelConfig(n_layers=1),
                    TrainConfig(epochs=4, batch_size=32), n_seeds=2, baselines=["logistic_linear", "svm"])
    assert res.n_seeds == 2 and len(res.per_seed) == 2 * 3
    assert set(res.summary["model"]) == {QNN_NAME, "Logistic regression", "SVM (RBF kernel)"}
    assert res.summary.iloc[0]["model"] == QNN_NAME
    label, text = verdict(res)
    assert label in ("ahead", "comparable", "behind") and "quantum" in text


def test_regression_study_reports_r2_rmse_mae():
    demo = load_demo("diabetes")
    res = run_study(demo.df, PrepConfig(target=demo.target, task="regression", n_qubits=4), ModelConfig(n_layers=1),
                    TrainConfig(epochs=4, batch_size=32, lr=0.05), n_seeds=1, baselines=["logistic_linear"])
    assert {"r2_mean", "rmse_mean", "mae_mean"} <= set(res.summary.columns)
    qnn = res.summary.iloc[0]
    assert qnn["rmse_mean"] > 0 and qnn["rmse_mean"] < 1000        # in original units of the target


def test_sweep_returns_mean_and_spread():
    demo = load_demo("iris")
    out = run_sweep(demo.df, PrepConfig(target=demo.target, n_qubits=3), ModelConfig(n_layers=1), TrainConfig(epochs=3, batch_size=32),
                    "n_layers", [1, 2], n_seeds=2)
    assert list(out["value"]) == [1, 2] and {"qnn_mean", "qnn_std", "classical_mean"} <= set(out.columns)
    frac = run_sweep(demo.df, PrepConfig(target=demo.target, n_qubits=3), ModelConfig(n_layers=1), TrainConfig(epochs=3, batch_size=32),
                     "train_fraction", [0.3, 1.0], n_seeds=1)
    assert frac["n_train"].iloc[0] < frac["n_train"].iloc[1]


def test_paired_difference_separates_signal_from_noise():
    from qnn_platform.workflow import paired_difference
    consistent = paired_difference([0.95, 0.96, 0.955, 0.952, 0.958], [0.90, 0.91, 0.905, 0.902, 0.908])
    assert consistent["significant"] and consistent["mean"] > 0.04
    noisy = paired_difference([0.95, 0.91, 0.94], [0.92, 0.95, 0.90])
    assert not noisy["significant"]
    single = paired_difference([0.95], [0.94], n_val=60, classification=True)
    assert not single["significant"] and single["half_width"] > 0.03     # one run on 60 rows says very little


def test_verdict_is_comparable_when_gap_is_inside_noise():
    import pandas as pd
    from qnn_platform.workflow import StudyResult, verdict
    per = pd.DataFrame({"seed": [0, 1, 2] * 2, "model": [QNN_NAME] * 3 + ["Random forest"] * 3,
                        "accuracy": [0.93, 0.95, 0.91, 0.94, 0.92, 0.95]})
    summ = pd.DataFrame({"model": [QNN_NAME, "Random forest"], "accuracy_mean": [0.93, 0.937], "accuracy_std": [0.02, 0.015]})
    res = StudyResult("classification", "accuracy", per, summ, {}, 3, 100, ["a", "b"], {})
    label, text = verdict(res)
    assert label == "comparable" and "noise" in text
