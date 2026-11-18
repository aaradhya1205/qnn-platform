"""Train/evaluate quantum image classifiers and classical baselines on the same split."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from qnn_platform.image_data import ImageDataset
from qnn_platform.quantum_image import (
    CircuitSpec, QuantumImageClassifier, logits_to_predictions, logits_to_probabilities,
)
from qnn_platform.train import train_model

BASELINE_NAMES = ["Logistic regression", "SVM (RBF)", "MLP (small DNN)", "Random forest"]


@dataclass
class ImageConfig:
    encoding: str = "frqi"
    size: int = 4
    n_layers: int = 2
    ansatz: str = "hardware_efficient"
    n_pca: int = 4
    epochs: int = 12
    batch_size: int = 16
    optimizer: str = "adam"
    lr: float = 0.08
    test_size: float = 0.25
    seed: int = 0
    per_class: int = 120
    train_fraction: float = 1.0
    freeze_circuit: bool = False

    @property
    def spec(self) -> CircuitSpec:
        return CircuitSpec(self.encoding, self.size, self.n_layers, self.ansatz, self.n_pca)


class ImageFeaturizer:
    """Pixels -> circuit inputs. FRQI/QPIE feed raw normalised pixels; angle encoding
    first compresses to k PCA features rescaled to [0, pi]."""

    def __init__(self, encoding: str, n_pca: int = 4):
        self.encoding = encoding
        self.n_pca = n_pca
        self.pca: PCA | None = None
        self.scaler: MinMaxScaler | None = None

    def fit(self, X_flat: np.ndarray) -> "ImageFeaturizer":
        if self.encoding == "angle_pca":
            k = min(self.n_pca, X_flat.shape[1], X_flat.shape[0])
            self.pca = PCA(n_components=k).fit(X_flat)
            self.scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(self.pca.transform(X_flat))
        return self

    def transform(self, X_flat: np.ndarray) -> np.ndarray:
        X_flat = np.atleast_2d(X_flat)
        if self.encoding != "angle_pca":
            return X_flat.astype(np.float32)
        z = self.scaler.transform(self.pca.transform(X_flat))
        return np.clip(z, 0, np.pi).astype(np.float32)


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray

    def as_dict(self) -> dict:
        return {"Accuracy": self.accuracy, "Precision": self.precision,
                "Recall": self.recall, "F1": self.f1}


def compute_metrics(y_true, y_pred, n_classes: int) -> Metrics:
    avg = "binary" if n_classes <= 2 else "macro"
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        confusion=confusion_matrix(y_true, y_pred, labels=list(range(n_classes))),
    )


@dataclass
class ExperimentResult:
    config: ImageConfig
    model: QuantumImageClassifier
    featurizer: ImageFeaturizer
    n_classes: int
    class_names: list[str]
    X_train_img: np.ndarray  # (n, size, size)
    X_test_img: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    history: object
    train_seconds: float
    test_pred: np.ndarray
    test_proba: np.ndarray
    train_metrics: Metrics
    test_metrics: Metrics
    baselines: dict = field(default_factory=dict)


def split_indices(ds: ImageDataset, cfg: ImageConfig):
    """Train/test indices into `ds` (already subsampled per class)."""
    idx = np.arange(len(ds.y))
    tr, te = train_test_split(idx, test_size=cfg.test_size, random_state=cfg.seed, stratify=ds.y)
    if cfg.train_fraction < 1.0:
        rng = np.random.RandomState(cfg.seed)
        keep = max(2 * len(ds.class_names), int(len(tr) * cfg.train_fraction))
        tr = rng.choice(tr, size=keep, replace=False)
    return tr, te


def make_split(ds: ImageDataset, cfg: ImageConfig):
    images = ds.at_size(cfg.size)
    tr, te = split_indices(ds, cfg)
    return images[tr], images[te], ds.y[tr], ds.y[te]


def heldout_samples(ds: ImageDataset, cfg: ImageConfig):
    """Full-resolution (8x8) images and labels of the held-out test split, so the
    dashboard can show a genuinely unseen example."""
    sub = ds.subsample_per_class(cfg.per_class, cfg.seed)
    _, te = split_indices(sub, cfg)
    return sub.X[te], sub.y[te]


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for an accuracy measured on n test images."""
    if n == 0:
        return 0.0, 1.0
    p = correct / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def predict_images(result: ExperimentResult, images: np.ndarray):
    """Run (n, size, size) images through the trained model (exact simulation)."""
    feats = result.featurizer.transform(images.reshape(len(images), -1))
    with torch.no_grad():
        logits = result.model(torch.tensor(feats))
    n = result.n_classes
    return logits_to_predictions(logits, n), logits_to_probabilities(logits, n)


def run_experiment(ds: ImageDataset, cfg: ImageConfig, progress_cb=None,
                   with_baselines: bool = False) -> ExperimentResult:
    ds = ds.subsample_per_class(cfg.per_class, cfg.seed)
    Xtr_img, Xte_img, ytr, yte = make_split(ds, cfg)
    n_classes = len(ds.class_names)

    featurizer = ImageFeaturizer(cfg.encoding, cfg.n_pca).fit(Xtr_img.reshape(len(Xtr_img), -1))
    Ftr = featurizer.transform(Xtr_img.reshape(len(Xtr_img), -1))
    Fte = featurizer.transform(Xte_img.reshape(len(Xte_img), -1))

    model = QuantumImageClassifier(cfg.spec, n_classes, seed=cfg.seed)
    if cfg.freeze_circuit:  # ablation: keep the random circuit, train only the classical head
        model.weights.requires_grad_(False)
    t0 = time.perf_counter()
    history = train_model(model, Ftr, ytr, Fte, yte, task="classification", n_classes=n_classes,
                          epochs=cfg.epochs, batch_size=cfg.batch_size,
                          optimizer_name=cfg.optimizer, lr=cfg.lr, progress_cb=progress_cb)
    train_seconds = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        tr_logits = model(torch.tensor(Ftr))
        te_logits = model(torch.tensor(Fte))
    train_pred = logits_to_predictions(tr_logits, n_classes)
    test_pred = logits_to_predictions(te_logits, n_classes)

    result = ExperimentResult(
        config=cfg, model=model, featurizer=featurizer, n_classes=n_classes,
        class_names=ds.class_names, X_train_img=Xtr_img, X_test_img=Xte_img,
        y_train=ytr, y_test=yte, history=history, train_seconds=train_seconds,
        test_pred=test_pred, test_proba=logits_to_probabilities(te_logits, n_classes),
        train_metrics=compute_metrics(ytr, train_pred, n_classes),
        test_metrics=compute_metrics(yte, test_pred, n_classes),
    )
    if with_baselines:
        result.baselines = fit_baselines(Xtr_img, ytr, Xte_img, yte, n_classes, cfg.seed)
    return result


def fit_baselines(Xtr_img, ytr, Xte_img, yte, n_classes: int, seed: int = 0) -> dict:
    """Classical models trained on the same pixels and the same split."""
    Xtr = Xtr_img.reshape(len(Xtr_img), -1)
    Xte = Xte_img.reshape(len(Xte_img), -1)
    models = {
        "Logistic regression": LogisticRegression(max_iter=2000),
        "SVM (RBF)": SVC(kernel="rbf", random_state=seed),
        "MLP (small DNN)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=600, random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=150, random_state=seed),
    }
    out = {}
    for name, clf in models.items():
        t0 = time.perf_counter()
        clf.fit(Xtr, ytr)
        seconds = time.perf_counter() - t0
        pred = clf.predict(Xte)
        if name == "Logistic regression":
            n_params = int(clf.coef_.size + clf.intercept_.size)
        elif name == "MLP (small DNN)":
            n_params = int(sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_))
        else:
            n_params = None
        out[name] = {"metrics": compute_metrics(yte, pred, n_classes), "seconds": seconds,
                     "n_params": n_params, "pred": pred}
    return out


def misclassified_indices(result: ExperimentResult, limit: int = 12) -> list[int]:
    wrong = np.where(result.test_pred != result.y_test)[0]
    return wrong[:limit].tolist()
