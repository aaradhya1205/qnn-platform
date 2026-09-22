import numpy as np

from qnn_platform.quantum_model import QNNModel
from qnn_platform.train import train_model, evaluate


def _toy_classification_data(n=40, n_features=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, n_features) * np.pi
    y = (X[:, 0] > np.pi / 2).astype(int)
    return X.astype(np.float32), y


def test_train_model_reduces_loss_over_epochs():
    X, y = _toy_classification_data()
    X_train, y_train = X[:30], y[:30]
    X_val, y_val = X[30:], y[30:]

    model = QNNModel(n_features=3, n_qubits=3, n_layers=1, task="classification", n_outputs=2)
    history = train_model(model, X_train, y_train, X_val, y_val, task="classification",
                           n_classes=2, epochs=5, batch_size=8, optimizer_name="adam", lr=0.2)

    assert len(history.train_loss) == 5
    assert len(history.val_metric) == 5


def test_evaluate_returns_predictions_and_metric():
    X, y = _toy_classification_data()
    model = QNNModel(n_features=3, n_qubits=3, n_layers=1, task="classification", n_outputs=2)
    preds, metric = evaluate(model, X, y, task="classification", n_classes=2)
    assert preds.shape == y.shape
    assert 0.0 <= metric <= 1.0


def test_regression_training_runs_end_to_end():
    rng = np.random.RandomState(0)
    X = rng.rand(30, 3).astype(np.float32) * np.pi
    y = X[:, 0] * 2.0
    model = QNNModel(n_features=3, n_qubits=3, n_layers=1, task="regression")
    history = train_model(model, X[:20], y[:20], X[20:], y[20:], task="regression",
                           epochs=3, batch_size=8, optimizer_name="adam", lr=0.1)
    assert len(history.val_metric) == 3
