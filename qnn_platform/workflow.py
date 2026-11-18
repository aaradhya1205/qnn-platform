"""Leakage-free study workflow: prepare -> train QNN + classical baselines -> compare -> sweep.

Everything that *learns from data* (gap filling, outlier limits, one-hot encoding, scaling, PCA/feature
selection) is fitted on the training rows only and then applied to the held-out rows. Results are
repeated over several random splits so they come with a spread, not a single lucky number.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler

from qnn_platform.classical_model import BASELINES, baseline_label, build_baseline
from qnn_platform.cleaning import CleaningConfig, FittedCleaner
from qnn_platform.medical import find_identifier_columns
from qnn_platform.quantum_model import QNNModel
from qnn_platform.train import train_model


@dataclass
class PrepConfig:
    target: str
    task: str = "classification"          # classification | regression
    n_qubits: int = 4
    reduction: str = "pca"                # pca | kbest (used when there are more columns than qubits)
    scaler: str = "minmax"                # minmax (0..pi) | standard
    test_size: float = 0.25
    seed: int = 0
    domain: str = "general"               # general | medical
    train_fraction: float = 1.0
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)


@dataclass
class ModelConfig:
    encoding: str = "angle"
    ansatz: str = "strongly_entangling"
    n_layers: int = 2


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 16
    optimizer: str = "adam"
    lr: float = 0.1


@dataclass
class PreparedSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    task: str
    class_names: list[str] | None
    n_qubits: int
    n_raw_columns: int
    n_encoded: int
    n_after_reduction: int
    train_index: np.ndarray
    val_index: np.ndarray
    cleaner: FittedCleaner
    explained_variance: np.ndarray | None = None
    dropped_domain_columns: list[str] = field(default_factory=list)
    steps: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.class_names) if self.class_names else 1


# --------------------------------------------------------------------------- #
# preparation
# --------------------------------------------------------------------------- #
def prepare_split(df: pd.DataFrame, cfg: PrepConfig) -> PreparedSplit:
    df = df.copy()
    steps: list[tuple[str, str]] = []
    dropped_domain: list[str] = []
    if cfg.domain == "medical":
        dropped_domain = find_identifier_columns(df, cfg.target)
        df = df.drop(columns=dropped_domain)
        if dropped_domain:
            steps.append(("Medical profile", "dropped identifier columns: " + ", ".join(dropped_domain)))
    df = df.dropna(subset=[cfg.target]).reset_index(drop=True)
    X_all = df.drop(columns=[cfg.target])

    class_names = None
    if cfg.task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(df[cfg.target].astype(str))
        class_names = list(le.classes_)
    else:
        y = df[cfg.target].astype(float).to_numpy()

    stratify = None
    if cfg.task == "classification" and pd.Series(y).value_counts().min() >= 2:
        stratify = y
    idx_all = np.arange(len(y))
    train_idx, val_idx = train_test_split(idx_all, test_size=cfg.test_size, random_state=cfg.seed, stratify=stratify)
    if cfg.train_fraction < 1.0:
        rng = np.random.RandomState(cfg.seed)
        keep = max(int(len(train_idx) * cfg.train_fraction), 8)
        train_idx = rng.choice(train_idx, size=min(keep, len(train_idx)), replace=False)
    steps.append(("Split", f"{len(train_idx):,} training rows / {len(val_idx):,} held-out rows"
                  + (" (stratified)" if stratify is not None else "")))

    X_train, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    ccfg = cfg.cleaning
    if cfg.domain == "medical":
        ccfg = replace(ccfg, add_missing_flags=True, outliers=ccfg.outliers if ccfg.outliers != "none" else "clip")
    cleaner = FittedCleaner(ccfg).fit(X_train)
    if ccfg.outliers == "drop_train":
        mask = cleaner.outlier_row_mask(X_train)
        if mask.sum() >= max(10, 0.5 * len(mask)):
            X_train, y_train = X_train[mask], y_train[mask]
            train_idx = train_idx[mask]
    X_train, X_val = cleaner.transform(X_train), cleaner.transform(X_val)
    fills = []
    if ccfg.impute_numeric:
        fills.append(f"gaps filled with the training {ccfg.impute_numeric}")
    if ccfg.outliers == "clip":
        fills.append(f"extremes clipped at {ccfg.outlier_iqr:g}×IQR")
    elif ccfg.outliers == "drop_train":
        fills.append("extreme training rows removed")
    if ccfg.add_missing_flags:
        fills.append("'was missing' flags added")
    steps.append(("Fit on training rows only", "; ".join(fills) if fills else "no fitted cleaning"))

    num_cols = [c for c in X_train.columns if X_train[c].dtype.kind in "iuf"]
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    scaler = MinMaxScaler(feature_range=(0, np.pi)) if cfg.scaler == "minmax" else StandardScaler()
    parts = []
    if num_cols:
        parts.append(("num", scaler, num_cols))
    if cat_cols:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    ct = ColumnTransformer(parts)
    A = np.asarray(ct.fit_transform(X_train), dtype=float)
    B = np.asarray(ct.transform(X_val), dtype=float)
    n_encoded = A.shape[1]
    steps.append(("Encode and scale", f"{len(num_cols)} numeric + {len(cat_cols)} text columns → {n_encoded} numbers "
                  f"({'0 to π' if cfg.scaler == 'minmax' else 'mean 0, spread 1'}), fitted on training rows"))

    explained = None
    if n_encoded > cfg.n_qubits:
        if cfg.reduction == "pca":
            reducer = PCA(n_components=cfg.n_qubits, random_state=cfg.seed).fit(A)
            explained = reducer.explained_variance_ratio_
        else:
            score = f_classif if cfg.task == "classification" else f_regression
            reducer = SelectKBest(score_func=score, k=cfg.n_qubits).fit(A, y_train)
        A, B = reducer.transform(A), reducer.transform(B)
        rescaler = MinMaxScaler(feature_range=(0, np.pi)).fit(A)
        A, B = rescaler.transform(A), np.clip(rescaler.transform(B), 0, np.pi)
        steps.append(("Shrink to fit the qubits", f"{n_encoded} → {A.shape[1]} features with "
                      f"{'PCA' if cfg.reduction == 'pca' else 'SelectKBest'}, fitted on training rows"))
    n_after = A.shape[1]
    return PreparedSplit(
        X_train=A.astype(np.float32), X_val=B.astype(np.float32), y_train=y_train, y_val=y_val, task=cfg.task,
        class_names=class_names, n_qubits=max(2, n_after), n_raw_columns=X_all.shape[1], n_encoded=n_encoded,
        n_after_reduction=n_after, train_index=np.asarray(train_idx), val_index=np.asarray(val_idx), cleaner=cleaner,
        explained_variance=explained, dropped_domain_columns=dropped_domain, steps=steps)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def primary_metric(task: str) -> str:
    return "accuracy" if task == "classification" else "r2"


def metric_label(name: str) -> str:
    return {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro F1",
            "r2": "R²", "rmse": "RMSE", "mae": "MAE"}[name]


def compute_metrics(task: str, y_true, y_pred) -> dict[str, float]:
    if task == "classification":
        return {"accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}
    return {"r2": float(r2_score(y_true, y_pred)), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred))}


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = correct / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def paired_difference(a, b, n_val: int | None = None, classification: bool = True) -> dict:
    """Is model/setting A really different from B? Paired by run, 95% confidence interval on the mean gap.

    With 2+ runs we use a paired t-interval over the per-run gaps. With a single run we can only fall back on the
    sampling noise of a small held-out set (Wilson interval), which is wide."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    k = len(d)
    mean = float(d.mean())
    if k >= 2:
        from scipy import stats
        half = float(stats.t.ppf(0.975, k - 1) * d.std(ddof=1) / np.sqrt(k))
    elif classification and n_val:
        lo, hi = wilson_interval(int(round(max(a[0], b[0]) * n_val)), n_val)
        half = (hi - lo) / 2
    else:
        half = float("inf")
    return {"mean": mean, "half_width": half, "significant": bool(abs(mean) > half), "runs": k}


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def predict_qnn(model: QNNModel, X: np.ndarray, task: str, n_classes: int) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32))
    if task == "classification":
        if n_classes <= 2:
            return (torch.sigmoid(out.squeeze(-1)) > 0.5).long().numpy()
        return out.argmax(dim=1).numpy()
    z = out.numpy().ravel()
    return z * getattr(model, "target_std", 1.0) + getattr(model, "target_mean", 0.0)


def n_parameters(model: QNNModel) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def train_qnn(split: PreparedSplit, mcfg: ModelConfig, tcfg: TrainConfig, seed: int, progress_cb=None) -> dict:
    torch.manual_seed(seed)
    model = QNNModel(n_features=split.X_train.shape[1], n_qubits=split.n_qubits, n_layers=mcfg.n_layers,
                     task=split.task, n_outputs=split.n_classes, encoding=mcfg.encoding, ansatz=mcfg.ansatz)
    t0 = time.perf_counter()
    history = train_model(model, split.X_train, split.y_train, split.X_val, split.y_val, task=split.task,
                          n_classes=split.n_classes, epochs=tcfg.epochs, batch_size=tcfg.batch_size,
                          optimizer_name=tcfg.optimizer, lr=tcfg.lr, progress_cb=progress_cb)
    seconds = time.perf_counter() - t0
    pred = predict_qnn(model, split.X_val, split.task, split.n_classes)
    return {"model": model, "history": history, "pred": pred, "seconds": seconds, "n_params": n_parameters(model)}


def train_baseline(name: str, split: PreparedSplit, seed: int) -> dict:
    clf = build_baseline(name, split.task, random_state=seed)
    t0 = time.perf_counter()
    clf.fit(split.X_train, split.y_train)
    seconds = time.perf_counter() - t0
    pred = clf.predict(split.X_val)
    n_params = None
    if name == "logistic_linear":
        n_params = int(np.size(clf.coef_) + np.size(clf.intercept_))
    elif name == "mlp":
        n_params = int(sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_))
    return {"pred": pred, "seconds": seconds, "n_params": n_params}


QNN_NAME = "Quantum neural network"


@dataclass
class StudyResult:
    task: str
    primary: str
    per_seed: pd.DataFrame
    summary: pd.DataFrame
    artifacts: dict
    n_seeds: int
    n_val: int
    class_names: list[str] | None
    config: dict


def run_study(df: pd.DataFrame, prep: PrepConfig, mcfg: ModelConfig, tcfg: TrainConfig, n_seeds: int = 3,
              baselines: list[str] | None = None, progress_cb=None) -> StudyResult:
    baselines = BASELINES if baselines is None else baselines
    records: list[dict] = []
    artifacts: dict = {}
    total = n_seeds * (1 + len(baselines))
    done = 0
    split0 = None
    for k in range(n_seeds):
        seed = prep.seed + k
        split = prepare_split(df, replace(prep, seed=seed))
        if k == 0:
            split0 = split
            artifacts["split"] = split

        def on_epoch(epoch, n_epochs, hist, _k=k):
            if progress_cb:
                progress_cb(done / total, f"Run {_k + 1}/{n_seeds} · quantum model epoch {epoch}/{n_epochs} · "
                                          f"held-out {hist.val_metric[-1]:.3f}")

        q = train_qnn(split, mcfg, tcfg, seed, on_epoch)
        records.append({"seed": seed, "model": QNN_NAME, "n_params": q["n_params"], "seconds": q["seconds"],
                        **compute_metrics(split.task, split.y_val, q["pred"])})
        done += 1
        if k == 0:
            artifacts["qnn"] = q
            artifacts["preds"] = {QNN_NAME: q["pred"]}
            artifacts["y_val"] = split.y_val
        for name in baselines:
            if progress_cb:
                progress_cb(done / total, f"Run {k + 1}/{n_seeds} · {baseline_label(name, split.task)}")
            b = train_baseline(name, split, seed)
            records.append({"seed": seed, "model": baseline_label(name, split.task), "n_params": b["n_params"],
                            "seconds": b["seconds"], **compute_metrics(split.task, split.y_val, b["pred"])})
            if k == 0:
                artifacts["preds"][baseline_label(name, split.task)] = b["pred"]
            done += 1
    per_seed = pd.DataFrame(records)
    metrics = [c for c in per_seed.columns if c not in ("seed", "model", "n_params", "seconds")]
    agg = per_seed.groupby("model", sort=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics}, **{f"{m}_std": (m, "std") for m in metrics},
        seconds=("seconds", "mean"), n_params=("n_params", "first")).reset_index()
    agg[[c for c in agg.columns if c.endswith("_std")]] = agg[[c for c in agg.columns if c.endswith("_std")]].fillna(0.0)
    primary = primary_metric(prep.task)
    agg = pd.concat([agg[agg["model"] == QNN_NAME],
                     agg[agg["model"] != QNN_NAME].sort_values(f"{primary}_mean", ascending=False)], ignore_index=True)
    if progress_cb:
        progress_cb(1.0, "Done")
    config = {"prep": {k: v for k, v in prep.__dict__.items() if k != "cleaning"}, "cleaning": prep.cleaning.__dict__,
              "model": mcfg.__dict__, "train": tcfg.__dict__, "seeds": n_seeds}
    return StudyResult(prep.task, primary, per_seed, agg, artifacts, n_seeds, len(split0.y_val), split0.class_names, config)


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def verdict(result: StudyResult) -> tuple[str, str]:
    """('ahead' | 'comparable' | 'behind', plain-English sentence) comparing the QNN with the best classical model.

    The call is made from a paired 95% confidence interval over the runs, so it is 'comparable' unless the gap is
    larger than the run-to-run noise."""
    s, p = result.summary, result.primary
    q = s[s["model"] == QNN_NAME].iloc[0]
    others = s[s["model"] != QNN_NAME]
    if others.empty:
        return "comparable", "No classical baseline was run."
    best = others.loc[others[f"{p}_mean"].idxmax()]
    ps = result.per_seed
    qa = ps[ps["model"] == QNN_NAME].sort_values("seed")[p].to_numpy()
    ca = ps[ps["model"] == best["model"]].sort_values("seed")[p].to_numpy()
    diff = paired_difference(qa, ca, result.n_val, result.task == "classification")
    name = metric_label(p)
    msg = f"{name}: quantum {q[f'{p}_mean']:.3f} vs best classical ({best['model']}) {best[f'{p}_mean']:.3f}. "
    hw = diff["half_width"]
    ci = f"gap {diff['mean']:+.3f}, 95% interval ±{hw:.3f}" if np.isfinite(hw) else f"gap {diff['mean']:+.3f}"
    if not diff["significant"]:
        return "comparable", msg + f"The gap ({ci}) is within the run-to-run noise, so the two cannot be told apart."
    if diff["mean"] > 0:
        return "ahead", msg + f"The quantum model is ahead ({ci}). Confirm with more runs and other datasets before claiming it."
    return "behind", msg + f"The quantum model is behind ({ci})."


# --------------------------------------------------------------------------- #
# sweeps
# --------------------------------------------------------------------------- #
SWEEP_PARAMS = {
    "n_qubits": "Number of qubits",
    "n_layers": "Circuit depth (layers)",
    "train_fraction": "Share of training data used",
    "encoding": "Data encoding",
    "ansatz": "Circuit type (ansatz)",
}


def run_sweep(df: pd.DataFrame, prep: PrepConfig, mcfg: ModelConfig, tcfg: TrainConfig, param: str, values: list,
              n_seeds: int = 3, with_reference: bool = True, progress_cb=None) -> pd.DataFrame:
    rows = []
    total = len(values) * n_seeds
    done = 0
    pm = primary_metric(prep.task)
    for v in values:
        p2, m2 = prep, mcfg
        if param == "n_qubits":
            p2 = replace(prep, n_qubits=int(v))
        elif param == "train_fraction":
            p2 = replace(prep, train_fraction=float(v))
        elif param == "n_layers":
            m2 = replace(mcfg, n_layers=int(v))
        elif param in ("encoding", "ansatz"):
            m2 = replace(mcfg, **{param: v})
        for k in range(n_seeds):
            seed = prep.seed + k
            split = prepare_split(df, replace(p2, seed=seed))
            q = train_qnn(split, m2, tcfg, seed)
            row = {"value": v, "seed": seed, "qnn": compute_metrics(split.task, split.y_val, q["pred"])[pm],
                   "seconds": q["seconds"], "qubits": split.n_qubits, "params": q["n_params"], "n_train": len(split.y_train)}
            if with_reference:
                b = train_baseline("logistic_linear", split, seed)
                row["classical"] = compute_metrics(split.task, split.y_val, b["pred"])[pm]
            rows.append(row)
            done += 1
            if progress_cb:
                progress_cb(done / total, f"{SWEEP_PARAMS[param]} = {v} · run {k + 1}/{n_seeds}")
    raw = pd.DataFrame(rows)
    agg = raw.groupby("value", sort=False).agg(
        qnn_mean=("qnn", "mean"), qnn_std=("qnn", "std"), seconds=("seconds", "mean"), qubits=("qubits", "first"),
        params=("params", "first"), n_train=("n_train", "mean")).reset_index()
    agg["qnn_std"] = agg["qnn_std"].fillna(0.0)
    if with_reference:
        ref = raw.groupby("value", sort=False).agg(classical_mean=("classical", "mean"), classical_std=("classical", "std")).reset_index()
        agg = agg.merge(ref, on="value")
        agg["classical_std"] = agg["classical_std"].fillna(0.0)
    agg.attrs["raw"] = raw
    return agg
