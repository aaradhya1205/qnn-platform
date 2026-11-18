import numpy as np
import pytest

from qnn_platform.image_data import load_digits_subset
from qnn_platform.image_experiments import scaling_table, sweep_noise, sweep_shots
from qnn_platform.image_pipeline import (
    ImageConfig, heldout_samples, predict_images, run_experiment, split_indices, wilson_interval,
)


@pytest.fixture(scope="module")
def ds():
    return load_digits_subset([3, 8])


@pytest.fixture(scope="module")
def result(ds):
    cfg = ImageConfig(encoding="frqi", size=2, n_layers=1, epochs=5, per_class=40)
    return run_experiment(ds, cfg, with_baselines=True)


def test_run_experiment_learns_something(result):
    assert result.test_metrics.accuracy > 0.6
    assert result.history.train_loss[-1] < result.history.train_loss[0]
    assert result.model.n_parameters()["circuit"] == 6


def test_baselines_use_the_same_test_set(result):
    n = len(result.y_test)
    assert set(result.baselines) == {"Logistic regression", "SVM (RBF)", "MLP (small DNN)", "Random forest"}
    for entry in result.baselines.values():
        assert entry["metrics"].confusion.sum() == n


def test_predict_images_matches_stored_predictions(result):
    pred, proba = predict_images(result, result.X_test_img)
    np.testing.assert_array_equal(pred, result.test_pred)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_heldout_samples_are_disjoint_from_training(ds):
    cfg = ImageConfig(per_class=40)
    sub = ds.subsample_per_class(cfg.per_class, cfg.seed)
    tr, te = split_indices(sub, cfg)
    assert set(tr).isdisjoint(te)
    images, labels = heldout_samples(ds, cfg)
    assert len(labels) == len(te)


def test_train_fraction_reduces_training_set(ds):
    full = split_indices(ds.subsample_per_class(40, 0), ImageConfig(per_class=40))[0]
    small = split_indices(ds.subsample_per_class(40, 0), ImageConfig(per_class=40, train_fraction=0.25))[0]
    assert len(small) < len(full)


def test_shots_sweep_converges_to_exact(result):
    df = sweep_shots(result, shots_list=(5, 5000), repeats=3)
    exact = df.iloc[0]["Test accuracy"]
    assert abs(df.iloc[-1]["Test accuracy"] - exact) <= 0.15


def test_noise_sweep_returns_one_row_per_level(result):
    df = sweep_noise(result, noise_levels=(0.0, 0.2), n_samples=10)
    assert len(df) == 2


def test_wilson_interval_contains_point_estimate_and_narrows():
    lo, hi = wilson_interval(45, 50)
    assert lo < 0.9 < hi
    lo2, hi2 = wilson_interval(450, 500)
    assert (hi2 - lo2) < (hi - lo)


def test_scaling_table_shows_growth():
    df = scaling_table(sizes=(2, 4))
    frqi = df[df["Encoding"] == "FRQI"].sort_values("Pixels")
    assert frqi["Native gates"].iloc[1] > frqi["Native gates"].iloc[0]
    assert frqi["Qubits"].tolist() == [3, 5]


def test_frozen_circuit_ablation_leaves_weights_unchanged(ds):
    import torch
    from dataclasses import replace
    cfg = ImageConfig(encoding="frqi", size=2, n_layers=1, epochs=3, per_class=30, freeze_circuit=True)
    from qnn_platform.quantum_image import QuantumImageClassifier, CircuitSpec
    reference = QuantumImageClassifier(cfg.spec, 2, seed=cfg.seed).weights.detach().clone()
    result = run_experiment(ds, cfg)
    assert torch.allclose(result.model.weights.detach(), reference)
    trained = run_experiment(ds, replace(cfg, freeze_circuit=False))
    assert not torch.allclose(trained.model.weights.detach(), reference)
