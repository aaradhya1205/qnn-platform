"""Built-in demo datasets (all public or synthetic), including deliberately messy ones for the cleaning lessons."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine, make_classification


@dataclass
class Demo:
    key: str
    title: str
    df: pd.DataFrame
    target: str
    task: str
    description: str
    source: str
    tag: str


def _named(loader, target_name: str) -> pd.DataFrame:
    data = loader(as_frame=True)
    df = data.frame.rename(columns={"target": target_name})
    df[target_name] = df[target_name].map(dict(enumerate(data.target_names)))
    return df


def messy_clinical(seed: int = 7) -> pd.DataFrame:
    """Ten real breast-cancer cell measurements wrapped in realistic mess: an ID column, a constant
    column, text-formatted numbers, inconsistent capitalisation, gaps, typo-sized outliers and duplicates."""
    data = load_breast_cancer(as_frame=True)
    rename = {"mean radius": "radius_mean", "mean texture": "texture_mean", "mean perimeter": "perimeter_mean",
              "mean area": "area_mean", "mean smoothness": "smoothness_mean", "mean compactness": "compactness_mean",
              "mean concavity": "concavity_mean", "mean concave points": "concave_points_mean",
              "mean symmetry": "symmetry_mean", "mean fractal dimension": "fractal_dimension_mean"}
    df = data.data[list(rename)].rename(columns=rename).copy()
    rng = np.random.RandomState(seed)
    n = len(df)
    df.insert(0, "patient_id", [f"P-{i:05d}" for i in range(n)])
    df["clinic_site"] = rng.choice(["North", "north ", "NORTH", "South", "south", "East"], n, p=[.25, .1, .05, .3, .1, .2])
    df["study_phase"] = "Phase II"
    df["lab_ratio"] = [f"{v:.1f}%" if rng.rand() > 0.08 else "-" for v in rng.uniform(40, 99, n)]
    df["diagnosis"] = data.target.map(dict(enumerate(data.target_names))).to_numpy()
    numeric = list(rename.values())
    for c in numeric:                                    # ~6% missing
        df.loc[rng.rand(n) < 0.06, c] = np.nan
    for idx in rng.choice(n, 6, replace=False):         # typo-sized outliers (value x100)
        col = rng.choice(["area_mean", "perimeter_mean"])
        if pd.notna(df.loc[idx, col]):
            df.loc[idx, col] = df.loc[idx, col] * 100
    dups = df.sample(12, random_state=seed)
    df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def lesson_messy_table(seed: int = 3) -> pd.DataFrame:
    """A tiny, friendly table used in the Learning Lab cleaning lesson."""
    rng = np.random.RandomState(seed)
    n = 70
    hours = np.clip(rng.normal(5, 2, n), 0.5, 10).round(1)
    sleep = np.clip(rng.normal(7, 1, n), 4, 10).round(1)
    attend = np.clip(rng.normal(82, 10, n), 40, 100).round(0)
    prev = np.clip(rng.normal(65, 12, n), 20, 100).round(0)
    score = 0.45 * hours + 0.25 * sleep + 0.05 * attend + 0.04 * prev + rng.normal(0, 0.6, n)
    passed = np.where(score > np.median(score), "yes", "no")
    df = pd.DataFrame({"student_id": [f"S{i:03d}" for i in range(n)], "hours_studied": hours, "sleep_hours": sleep,
                       "attendance": [f"{int(a)}%" for a in attend], "previous_score": prev,
                       "school_year": "2025/26", "passed": passed})
    df.loc[rng.choice(n, 6, replace=False), "sleep_hours"] = np.nan
    df.loc[rng.choice(n, 4, replace=False), "hours_studied"] = np.nan
    df.loc[rng.choice(n, 3, replace=False), "attendance"] = "-"
    df.loc[[5, 41], "hours_studied"] = [240.0, 190.0]
    df = pd.concat([df, df.iloc[[2, 9, 17]]], ignore_index=True)
    return df


def large_synthetic() -> pd.DataFrame:
    X, y = make_classification(n_samples=5000, n_features=100, n_informative=5, n_redundant=30, class_sep=1.5,
                               n_classes=2, random_state=0)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(100)])
    df["label"] = np.where(y == 1, "class_b", "class_a")
    return df


DEMO_KEYS = ["messy_clinical", "breast_cancer", "iris", "wine", "diabetes", "large"]

_META = {
    "messy_clinical": ("Messy clinical table", "diagnosis", "classification",
                       "Real breast-cancer cell measurements with realistic mess added: IDs, gaps, duplicates, text numbers, outliers. Built to show the cleaning step.",
                       "Wisconsin Diagnostic data (UCI) plus synthetic defects", "Cleaning showcase"),
    "breast_cancer": ("Breast cancer (clinical)", "diagnosis", "classification",
                      "569 patients, 30 cell measurements, benign or malignant. Clean and well known.", "UCI Breast Cancer Wisconsin (Diagnostic)", "Clinical"),
    "iris": ("Iris flowers", "species", "classification", "150 flowers, 4 measurements, 3 species. The classic first dataset.", "R. A. Fisher, 1936 (via scikit-learn)", "Beginner"),
    "wine": ("Wine chemistry", "cultivar", "classification", "178 wines, 13 chemical measurements, 3 grape types.", "UCI Wine (via scikit-learn)", "Beginner"),
    "diabetes": ("Diabetes progression", "target", "regression", "442 patients, 10 measurements, predict a disease-progress score (a number).",
                 "Efron et al. 2004 (via scikit-learn)", "Regression"),
    "large": ("Large synthetic (5,000 × 100)", "label", "classification", "5,000 rows and 100 columns: shows how big data is shrunk and batched to fit a few qubits.",
              "Generated with scikit-learn", "Big data"),
}


def load_demo(key: str) -> Demo:
    title, target, task, desc, source, tag = _META[key]
    if key == "messy_clinical":
        df = messy_clinical()
    elif key == "breast_cancer":
        df = _named(load_breast_cancer, "diagnosis")
    elif key == "iris":
        df = _named(load_iris, "species")
    elif key == "wine":
        df = _named(load_wine, "cultivar")
    elif key == "diabetes":
        df = load_diabetes(as_frame=True).frame
    elif key == "large":
        df = large_synthetic()
    else:
        raise KeyError(key)
    return Demo(key, title, df, target, task, desc, source, tag)


def demo_catalog() -> list[dict]:
    return [{"key": k, "title": _META[k][0], "task": _META[k][2], "description": _META[k][3], "tag": _META[k][5]} for k in DEMO_KEYS]
