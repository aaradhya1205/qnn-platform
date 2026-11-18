"""Ablation runs for the paper: circuit design and data-cleaning choices.

    python scripts/run_ablation.py       # a few minutes

Writes data/ablation_results.json (5 random splits per setting, seeds 0-4, same as the main benchmark).
"""
import json, sys, time, warnings
from dataclasses import replace
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from qnn_platform.cleaning import CleaningConfig  # noqa: E402
from qnn_platform.demo_data import load_demo  # noqa: E402
from qnn_platform.workflow import (ModelConfig, PrepConfig, TrainConfig, compute_metrics, prepare_split,  # noqa: E402
                                   primary_metric, train_baseline, train_qnn)

SEEDS = 5


def run(df, prep, mcfg, tcfg, baseline=True):
    pm = primary_metric(prep.task)
    q, b = [], []
    for s in range(SEEDS):
        sp = prepare_split(df, replace(prep, seed=s))
        r = train_qnn(sp, mcfg, tcfg, s)
        q.append(compute_metrics(sp.task, sp.y_val, r["pred"])[pm])
        if baseline:
            bb = train_baseline("logistic_linear", sp, s)
            b.append(compute_metrics(sp.task, sp.y_val, bb["pred"])[pm])
    return q, b


def summ(x):
    return {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)), "runs": [float(v) for v in x]}


out = {"seeds": SEEDS, "circuit": {}, "cleaning": {}}
t0 = time.time()
bc = load_demo("breast_cancer")
prep = PrepConfig(target=bc.target, n_qubits=4)
tcfg = TrainConfig(epochs=25, batch_size=16, lr=0.1)
variants = {
    "Reference (angle, strongly entangling, 2 layers)": ModelConfig(),
    "Basic entangler ansatz": ModelConfig(ansatz="basic_entangler"),
    "Hardware-efficient ansatz": ModelConfig(ansatz="hardware_efficient"),
    "Amplitude encoding": ModelConfig(encoding="amplitude"),
    "IQP encoding": ModelConfig(encoding="iqp"),
    "1 layer": ModelConfig(n_layers=1),
    "4 layers": ModelConfig(n_layers=4),
}
for name, m in variants.items():
    try:
        q, b = run(bc.df, prep, m, tcfg, baseline=(name.startswith("Reference")))
    except Exception as e:  # report, don't hide
        print("FAILED", name, repr(e)); continue
    out["circuit"][name] = summ(q)
    if b:
        out["circuit"]["Logistic regression (reference)"] = summ(b)
    print(f"{name:55s} {np.mean(q):.3f} ± {np.std(q, ddof=1):.3f}  ({time.time() - t0:.0f}s)", flush=True)

mc = load_demo("messy_clinical")
base = PrepConfig(target=mc.target, task=mc.task, n_qubits=4)
cleans = {
    "Default (median fill, no outlier limit)": CleaningConfig(),
    "Zero fill": CleaningConfig(impute_numeric="zero"),
    "Median fill + missing flags": CleaningConfig(add_missing_flags=True),
    "Median fill + outlier clipping (3 IQR)": CleaningConfig(outliers="clip"),
}
for name, c in cleans.items():
    q, b = run(mc.df, replace(base, cleaning=c), ModelConfig(), tcfg)
    out["cleaning"][name] = {"qnn": summ(q), "logistic": summ(b)}
    print(f"{name:45s} qnn {np.mean(q):.3f}  logit {np.mean(b):.3f}  ({time.time() - t0:.0f}s)", flush=True)
(ROOT / "data" / "ablation_results.json").write_text(json.dumps(out, indent=1))
print("done")
