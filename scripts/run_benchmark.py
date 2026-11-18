"""Reproduce the benchmark shown on the Home page.

    python scripts/run_benchmark.py            # 5 random splits per dataset (about 8 minutes)

Writes data/benchmark_results.json. Everything uses the leakage-free workflow in qnn_platform/workflow.py:
cleaning, scaling and PCA are fitted on the training rows only.
"""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from qnn_platform.demo_data import load_demo  # noqa: E402
from qnn_platform.workflow import (ModelConfig, PrepConfig, TrainConfig, run_study, run_sweep)  # noqa: E402

SEEDS = 5
DATASETS = [("iris", "Iris", 16), ("wine", "Wine", 16), ("breast_cancer", "Breast cancer", 16),
            ("large", "Large 5,000 × 100", 32), ("diabetes", "Diabetes (R²)", 16)]


def main():
    out = {"seeds": SEEDS, "settings": "4 qubits, angle encoding, strongly entangling circuit, 2 layers, Adam", "datasets": {}, "sweeps": {}}
    t0 = time.time()
    for key, label, batch in DATASETS:
        demo = load_demo(key)
        reg = demo.task == "regression"
        res = run_study(demo.df, PrepConfig(target=demo.target, task=demo.task, n_qubits=4), ModelConfig(),
                        TrainConfig(epochs=30 if reg else 25, batch_size=batch, lr=0.05 if reg else 0.1), n_seeds=SEEDS)
        p = res.primary
        out["datasets"][label] = {
            "task": demo.task, "metric": p, "rows": len(demo.df), "columns": demo.df.shape[1] - 1, "n_val": res.n_val,
            "models": {r["model"]: {"mean": float(r[f"{p}_mean"]), "std": float(r[f"{p}_std"]), "seconds": float(r["seconds"]),
                                    "n_params": None if r["n_params"] != r["n_params"] else int(r["n_params"])}
                       for _, r in res.summary.iterrows()},
            "per_seed": {m: g.sort_values("seed")[p].tolist() for m, g in res.per_seed.groupby("model")}}
        print(f"{label:20s} done  ({time.time() - t0:.0f}s)", flush=True)
    bc = load_demo("breast_cancer")
    prep = PrepConfig(target=bc.target, n_qubits=4)
    tcfg = TrainConfig(epochs=25, batch_size=16, lr=0.1)
    q = run_sweep(bc.df, prep, ModelConfig(), tcfg, "n_qubits", [2, 3, 4, 6, 8], n_seeds=SEEDS, with_reference=True)
    f = run_sweep(bc.df, prep, ModelConfig(), tcfg, "train_fraction", [0.1, 0.25, 0.5, 1.0], n_seeds=SEEDS, with_reference=True)
    out["sweeps"]["breast_cancer_qubits"] = q.to_dict(orient="list")
    out["sweeps"]["breast_cancer_train_fraction"] = f.to_dict(orient="list")
    out["torch"] = torch.__version__
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "benchmark_results.json").write_text(json.dumps(out, indent=1))
    print("wrote data/benchmark_results.json", f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
