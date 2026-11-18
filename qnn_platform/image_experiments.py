"""Sweeps that turn single runs into evidence: encoding, depth, image size,
training-set size, measurement shots and hardware noise. Every sweep returns a
tidy DataFrame (mean and std over seeds where relevant)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from qnn_platform.image_data import ImageDataset
from qnn_platform.image_pipeline import ExperimentResult, ImageConfig, predict_images, run_experiment
from qnn_platform.quantum_image import (
    CircuitSpec, circuit_resources, predict_with_noise, predict_with_shots, time_forward_pass,
)
from sklearn.metrics import accuracy_score


def _run_seeds(ds: ImageDataset, cfg: ImageConfig, n_seeds: int) -> dict:
    accs, f1s, secs = [], [], []
    for s in range(n_seeds):
        r = run_experiment(ds, replace(cfg, seed=cfg.seed + s))
        accs.append(r.test_metrics.accuracy)
        f1s.append(r.test_metrics.f1)
        secs.append(r.train_seconds)
    return {
        "Test accuracy": float(np.mean(accs)), "Accuracy std": float(np.std(accs)),
        "Test F1": float(np.mean(f1s)), "Train time (s)": float(np.mean(secs)),
    }


def _sweep(ds, base: ImageConfig, field: str, values, n_seeds: int, progress_cb=None,
           extra=None) -> pd.DataFrame:
    rows = []
    for i, v in enumerate(values):
        cfg = replace(base, **{field: v})
        row = {field: v, "Qubits": cfg.spec.n_qubits, "Circuit params": cfg.spec.n_circuit_params}
        if extra:
            row.update(extra(cfg))
        row.update(_run_seeds(ds, cfg, n_seeds))
        rows.append(row)
        if progress_cb:
            progress_cb(i + 1, len(values))
    return pd.DataFrame(rows)


def sweep_encodings(ds, base, n_seeds=1, progress_cb=None) -> pd.DataFrame:
    def gates(cfg):
        res = circuit_resources(cfg.spec)
        return {"Native gates": res.get("native_gates"), "Native CNOTs": res.get("native_cnots")}
    return _sweep(ds, base, "encoding", ["frqi", "qpie", "angle_pca"], n_seeds, progress_cb, gates)


def sweep_depth(ds, base, layers=(1, 2, 3, 4, 6), n_seeds=1, progress_cb=None) -> pd.DataFrame:
    return _sweep(ds, base, "n_layers", list(layers), n_seeds, progress_cb)


def sweep_image_size(ds, base, sizes=(2, 4), n_seeds=1, progress_cb=None) -> pd.DataFrame:
    return _sweep(ds, base, "size", list(sizes), n_seeds, progress_cb)


def sweep_train_fraction(ds, base, fractions=(0.1, 0.25, 0.5, 1.0), n_seeds=1,
                         progress_cb=None) -> pd.DataFrame:
    return _sweep(ds, base, "train_fraction", list(fractions), n_seeds, progress_cb)


def sweep_shots(result: ExperimentResult, shots_list=(10, 50, 200, 1000, 5000), repeats=5,
                progress_cb=None) -> pd.DataFrame:
    images = result.X_test_img.reshape(len(result.X_test_img), -1)
    feats = result.featurizer.transform(images)
    exact_pred, _ = predict_images(result, result.X_test_img)
    exact_acc = accuracy_score(result.y_test, exact_pred)
    rows = [{"Shots": "exact (∞)", "Test accuracy": exact_acc, "Accuracy std": 0.0}]
    for i, shots in enumerate(shots_list):
        accs = []
        for r in range(repeats):
            pred, _, _ = predict_with_shots(result.model, feats, shots, seed=100 + r)
            accs.append(accuracy_score(result.y_test, pred))
        rows.append({"Shots": str(shots), "Test accuracy": float(np.mean(accs)),
                     "Accuracy std": float(np.std(accs))})
        if progress_cb:
            progress_cb(i + 1, len(shots_list))
    return pd.DataFrame(rows)


def sweep_noise(result: ExperimentResult, noise_levels=(0.0, 0.002, 0.01, 0.03, 0.08),
                n_samples=40, progress_cb=None) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    idx = rng.choice(len(result.y_test), size=min(n_samples, len(result.y_test)), replace=False)
    images = result.X_test_img[idx].reshape(len(idx), -1)
    feats = result.featurizer.transform(images)
    y = result.y_test[idx]
    rows = []
    for i, p in enumerate(noise_levels):
        pred, _, _ = predict_with_noise(result.model, feats, p)
        rows.append({"Depolarising probability p": p, "Test accuracy": float(accuracy_score(y, pred))})
        if progress_cb:
            progress_cb(i + 1, len(noise_levels))
    return pd.DataFrame(rows)


def scaling_table(sizes=(2, 4, 8), n_layers=2, ansatz="hardware_efficient") -> pd.DataFrame:
    """How qubits, compiled gate count and simulation time grow with image size."""
    rows = []
    for size in sizes:
        for enc in ("frqi", "qpie"):
            spec = CircuitSpec(enc, size, n_layers, ansatz)
            res = circuit_resources(spec)
            rows.append({
                "Image": f"{size}x{size}", "Pixels": size * size, "Encoding": enc.upper(),
                "Qubits": spec.n_qubits,
                "Native gates": res.get("native_gates"), "Native CNOTs": res.get("native_cnots"),
                "Native depth": res.get("native_depth"),
                "Forward pass, batch of 8 (s)": time_forward_pass(spec),
            })
    return pd.DataFrame(rows)


def ablation_circuit(ds, base, n_seeds=1, progress_cb=None) -> pd.DataFrame:
    """Does training the circuit matter, or does the classical head do all the work?"""
    rows = []
    variants = [("Trained circuit + trained head", False), ("Random frozen circuit + trained head", True)]
    for i, (name, frozen) in enumerate(variants):
        row = {"Variant": name}
        row.update(_run_seeds(ds, replace(base, freeze_circuit=frozen), n_seeds))
        rows.append(row)
        if progress_cb:
            progress_cb(i + 1, len(variants))
    return pd.DataFrame(rows)
