"""Automated sweeps over qubit count, circuit depth, and training-set size, so
the effect of each on QNN performance can be measured directly instead of by
re-running the app manually for every combination."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from qnn_platform.preprocessing import prepare_dataset
from qnn_platform.quantum_model import QNNModel
from qnn_platform.train import train_model


@dataclass
class SweepPoint:
    varied_param: str
    value: float
    val_metric: float
    train_seconds: float


def _train_once(df, target, task, n_qubits, n_layers, encoding, ansatz,
                 epochs, batch_size, optimizer_name, lr, test_size,
                 train_fraction: float = 1.0, reduction_method: str = "pca",
                 scaler: str = "minmax", sector: str = "general") -> tuple[float, float]:
    import time

    prepared = prepare_dataset(df, target, task, n_qubits,
                                reduction_method=reduction_method, scaler=scaler,
                                sector=sector)
    X_train, X_val, y_train, y_val = train_test_split(
        prepared.X, prepared.y, test_size=test_size, random_state=0,
        stratify=prepared.y if prepared.task == "classification" else None,
    )
    if train_fraction < 1.0:
        n_keep = max(8, int(len(X_train) * train_fraction))
        rng = np.random.RandomState(0)
        idx = rng.choice(len(X_train), size=n_keep, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    n_classes = len(prepared.class_names) if prepared.class_names else 2
    model = QNNModel(n_features=prepared.X.shape[1], n_qubits=prepared.n_qubits,
                      n_layers=n_layers, task=prepared.task, n_outputs=n_classes,
                      encoding=encoding, ansatz=ansatz)

    start = time.time()
    history = train_model(model, X_train, y_train, X_val, y_val, task=prepared.task,
                           n_classes=n_classes, epochs=epochs, batch_size=batch_size,
                           optimizer_name=optimizer_name, lr=lr)
    elapsed = time.time() - start
    return history.val_metric[-1], elapsed


def sweep_qubits(df, target, task, qubit_values, n_layers, encoding, ansatz,
                  epochs, batch_size, optimizer_name, lr, test_size,
                  progress_cb=None, sector: str = "general") -> list[SweepPoint]:
    results = []
    for i, nq in enumerate(qubit_values):
        metric, secs = _train_once(df, target, task, nq, n_layers, encoding, ansatz,
                                    epochs, batch_size, optimizer_name, lr, test_size,
                                    sector=sector)
        results.append(SweepPoint("n_qubits", nq, metric, secs))
        if progress_cb:
            progress_cb(i + 1, len(qubit_values))
    return results


def sweep_depth(df, target, task, n_qubits, layer_values, encoding, ansatz,
                 epochs, batch_size, optimizer_name, lr, test_size,
                 progress_cb=None, sector: str = "general") -> list[SweepPoint]:
    results = []
    for i, nl in enumerate(layer_values):
        metric, secs = _train_once(df, target, task, n_qubits, nl, encoding, ansatz,
                                    epochs, batch_size, optimizer_name, lr, test_size,
                                    sector=sector)
        results.append(SweepPoint("n_layers", nl, metric, secs))
        if progress_cb:
            progress_cb(i + 1, len(layer_values))
    return results


def sweep_dataset_size(df, target, task, n_qubits, n_layers, encoding, ansatz,
                        epochs, batch_size, optimizer_name, lr, test_size,
                        fractions, progress_cb=None, sector: str = "general") -> list[SweepPoint]:
    results = []
    for i, frac in enumerate(fractions):
        metric, secs = _train_once(df, target, task, n_qubits, n_layers, encoding, ansatz,
                                    epochs, batch_size, optimizer_name, lr, test_size,
                                    train_fraction=frac, sector=sector)
        results.append(SweepPoint("train_fraction", frac, metric, secs))
        if progress_cb:
            progress_cb(i + 1, len(fractions))
    return results
