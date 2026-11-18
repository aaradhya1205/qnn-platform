"""Builds every figure in the paper as vector PDF from data/*.json (no screenshots)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
bench = json.load(open(ROOT / "data" / "benchmark_results.json"))
abl = json.load(open(ROOT / "data" / "ablation_results.json"))
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": .25})
C = {"qnn": "#5b4bd6", "Logistic regression": "#2a9d8f", "SVM (RBF kernel)": "#e9a23b", "MLP (small neural net)": "#d1495b",
     "Random forest": "#6b7280", "Ridge regression": "#2a9d8f"}


def box(ax, x, y, w, h, text, fc="#eef0ff", ec="#5b4bd6", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=1, color="#333"))


# Fig 1: architecture of the hybrid model
fig, ax = plt.subplots(figsize=(7.2, 2.0)); ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 3.4); ax.grid(False)
items = [("Prepared\nfeatures\n$q$ values\nin [0, π]", "#f3f4f6", "#6b7280"),
         ("Encoding\n$R_Y(x_i)$\non each\nqubit", "#eef0ff", "#5b4bd6"),
         ("Variational\nlayers\n$L$ × (rotations\n+ CNOT ring)", "#eef0ff", "#5b4bd6"),
         ("Readout\n⟨$Z_i$⟩ per\nqubit", "#eef0ff", "#5b4bd6"),
         ("Linear\nhead", "#e6f6f3", "#2a9d8f"), ("Class scores\nor value", "#f3f4f6", "#6b7280")]
xs = np.linspace(0.1, 12.3, len(items)); bw = 2.3
for (t, fc, ec), x in zip(items, xs):
    box(ax, x, 0.4, bw, 2.0, t, fc, ec, fs=7.5)
for i in range(len(items) - 1):
    arrow(ax, (xs[i] + bw + 0.02, 1.4), (xs[i + 1] - 0.02, 1.4))
ax.plot([xs[1], xs[3] + bw], [2.85, 2.85], color="#5b4bd6", lw=1.2)
ax.text((xs[1] + xs[3] + bw) / 2, 3.05, "simulated quantum circuit (PennyLane)", ha="center", fontsize=8, color="#5b4bd6")
fig.savefig(OUT / "architecture.pdf", bbox_inches="tight"); plt.close(fig)

# Fig 2: workflow (two rows)
fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 6.4); ax.grid(False)
bw, bh = 3.1, 1.7
row1 = [("1 Data\nupload or demo", "#eef0ff", "#5b4bd6"), ("2 Structural cleaning\n(label-free)", "#eef0ff", "#5b4bd6"),
        ("3 Split\ntrain / held-out", "#eef0ff", "#5b4bd6"), ("4 Fitted preparation\n(training rows only)", "#e6f6f3", "#2a9d8f")]
row2 = [("5 Train QNN\nover $S$ seeds", "#eef0ff", "#5b4bd6"), ("6 Compare with\n4 baselines", "#eef0ff", "#5b4bd6"),
        ("Paired 95% interval\n→ verdict", "#fff4e0", "#e9a23b"), ("7 Report\nMarkdown, CSV", "#f3f4f6", "#6b7280")]
xs = [0.1 + i * 3.7 for i in range(4)]
for (t, fc, ec), x in zip(row1, xs):
    box(ax, x, 4.2, bw, bh, t, fc, ec, fs=7.5)
for (t, fc, ec), x in zip(row2, xs):
    box(ax, x, 0.7, bw, bh, t, fc, ec, fs=7.5)
for i in range(3):
    arrow(ax, (xs[i] + bw + 0.02, 5.05), (xs[i + 1] - 0.02, 5.05))
    arrow(ax, (xs[i] + bw + 0.02, 1.55), (xs[i + 1] - 0.02, 1.55))
ax.plot([xs[3] + bw / 2, xs[3] + bw / 2], [4.18, 3.3], color="#333", lw=1)
ax.plot([xs[3] + bw / 2, xs[0] + bw / 2], [3.3, 3.3], color="#333", lw=1)
arrow(ax, (xs[0] + bw / 2, 3.3), (xs[0] + bw / 2, 2.42))
ax.text(7.6, 3.5, "held-out rows pass through the training-fitted transformations only", ha="center", fontsize=7.5, style="italic")
fig.savefig(OUT / "workflow.pdf", bbox_inches="tight"); plt.close(fig)

# Fig 3: results across datasets
names = [k for k in bench["datasets"]]
models = ["Quantum neural network", "Logistic regression", "SVM (RBF kernel)", "MLP (small neural net)", "Random forest"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [4, 1.3]})
cls = [n for n in names if bench["datasets"][n]["task"] == "classification"]
w = 0.16
for j, m in enumerate(models):
    mu, sd = [], []
    for n in cls:
        md = bench["datasets"][n]["models"]
        key = m if m in md else None
        mu.append(md[key]["mean"] if key else np.nan); sd.append(md[key]["std"] if key else 0)
    axs[0].bar(np.arange(len(cls)) + (j - 2) * w, mu, w, yerr=sd, capsize=1.5, label=m.replace(" (small neural net)", "").replace(" (RBF kernel)", ""),
               color=C.get(m, C["qnn"]) if m != "Quantum neural network" else C["qnn"], error_kw={"lw": .7})
axs[0].set_xticks(range(len(cls))); axs[0].set_xticklabels([c.replace(" 5,000 × 100", "\n(5,000×100)") for c in cls])
axs[0].set_ylim(0.85, 1.03); axs[0].set_yticks([0.85,0.90,0.95,1.0]); axs[0].set_ylabel("Held-out accuracy (mean ± std, 5 splits)")
axs[0].legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
dn = bench["datasets"][[n for n in names if bench["datasets"][n]["task"] == "regression"][0]]["models"]
order = ["Quantum neural network", "Ridge regression", "Random forest", "MLP (small neural net)", "SVM (RBF kernel)"]
axs[1].bar(range(5), [dn[m]["mean"] for m in order], yerr=[dn[m]["std"] for m in order], capsize=1.5,
           color=[C.get(m, C["qnn"]) if m != "Quantum neural network" else C["qnn"] for m in order], error_kw={"lw": .7})
axs[1].set_xticks(range(5)); axs[1].set_xticklabels(["QNN", "Ridge", "RF", "MLP", "SVM"], fontsize=6, rotation=45)
axs[1].set_ylabel("Diabetes R² (mean ± std)")
fig.tight_layout(); fig.savefig(OUT / "results_datasets.pdf", bbox_inches="tight"); plt.close(fig)

# Fig 4: sweeps
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
for ax, key, xl in [(axs[0], "breast_cancer_qubits", "Number of qubits"), (axs[1], "breast_cancer_train_fraction", "Share of training rows used")]:
    s = bench["sweeps"][key]; x = s["value"]
    ax.errorbar(x, s["qnn_mean"], s["qnn_std"], marker="o", ms=4, capsize=2, color=C["qnn"], label="QNN")
    ax.errorbar(x, s["classical_mean"], s["classical_std"], marker="s", ms=4, capsize=2, color=C["Logistic regression"], label="Logistic regression")
    ax.set_xlabel(xl); ax.set_ylabel("Held-out accuracy"); ax.legend(fontsize=7, frameon=False)
axs[1].set_xscale("log"); axs[1].set_xticks(bench["sweeps"]["breast_cancer_train_fraction"]["value"])
axs[1].set_xticklabels(["10%", "25%", "50%", "100%"])
fig.tight_layout(); fig.savefig(OUT / "sweeps.pdf", bbox_inches="tight"); plt.close(fig)

# Fig 5: circuit ablation
names5 = [k for k in abl["circuit"] if not k.startswith("Logistic")]
lab = {"Reference (angle, strongly entangling, 2 layers)": "Reference", "Basic entangler ansatz": "Basic\nentangler",
       "Hardware-efficient ansatz": "Hardware-\nefficient", "Amplitude encoding": "Amplitude\nencoding", "IQP encoding": "IQP\nencoding",
       "1 layer": "1 layer", "4 layers": "4 layers"}
fig, ax = plt.subplots(figsize=(7, 2.5))
mu = [abl["circuit"][k]["mean"] for k in names5]; sd = [abl["circuit"][k]["std"] for k in names5]
cols = [C["qnn"] if k.startswith("Ref") else "#a9a3e8" for k in names5]
ax.bar(range(len(names5)), mu, yerr=sd, capsize=2, color=cols, error_kw={"lw": .8})
for i, k in enumerate(names5):
    ax.scatter([i] * len(abl["circuit"][k]["runs"]), abl["circuit"][k]["runs"], s=7, color="k", zorder=3, alpha=.6)
ax.axhline(abl["circuit"]["Logistic regression (reference)"]["mean"], color=C["Logistic regression"], ls="--", lw=1, label="Logistic regression")
ax.set_xticks(range(len(names5))); ax.set_xticklabels([lab[k] for k in names5], fontsize=8)
ax.set_ylim(0.6, 1.0); ax.set_ylabel("Held-out accuracy"); ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "ablation_circuit.pdf", bbox_inches="tight"); plt.close(fig)
print("figures written")
