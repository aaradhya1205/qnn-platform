"""Classical ML/DNN baselines used as a point of comparison against the QNN."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor

BASELINES = ["logistic_linear", "mlp", "random_forest"]


def build_baseline(name: str, task: str, hidden_layer_sizes=(32, 16), random_state: int = 0):
    if task == "classification":
        if name == "logistic_linear":
            return LogisticRegression(max_iter=2000)
        if name == "mlp":
            return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=500,
                                  random_state=random_state)
        if name == "random_forest":
            return RandomForestClassifier(n_estimators=200, random_state=random_state)
    else:
        if name == "logistic_linear":
            return Ridge()
        if name == "mlp":
            return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=500,
                                 random_state=random_state)
        if name == "random_forest":
            return RandomForestRegressor(n_estimators=200, random_state=random_state)
    raise ValueError(f"Unknown baseline: {name}")
