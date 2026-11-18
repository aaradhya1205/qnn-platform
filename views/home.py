"""Home: the use case, the evidence behind it, and two doors in (Research Workbench, Learning Lab)."""
from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from qnn_platform import ui
from qnn_platform.workflow import paired_difference

BENCH = json.loads((Path(__file__).resolve().parent.parent / "data" / "benchmark_results.json").read_text())
QNN = "Quantum neural network"
MODEL_COLOURS = {QNN: ui.VIOLET, "Logistic regression": ui.TEAL, "SVM (RBF kernel)": ui.AMBER,
                 "MLP (small neural net)": ui.ROSE, "Random forest": "#7dd3fc"}


def link(text: str, url: str) -> str:
    return f'<a href="{url}" target="_blank" rel="noopener" style="color:#9af2e4;text-decoration:none;border-bottom:1px solid rgba(154,242,228,.4)">{text}</a>'


def verdict_for(ds: dict) -> tuple[str, float, str]:
    """ahead / comparable / behind from a paired 95% interval over the runs (same rule as the Research Workbench)."""
    models = ds["models"]
    others = {k: v for k, v in models.items() if k != QNN}
    best_name = max(others, key=lambda k: others[k]["mean"])
    d = paired_difference(ds["per_seed"][QNN], ds["per_seed"][best_name], ds["n_val"], ds["task"] == "classification")
    label = "comparable" if not d["significant"] else ("ahead" if d["mean"] > 0 else "behind")
    return label, models[QNN]["mean"] - models[best_name]["mean"], best_name


# --------------------------------------------------------------------------- #
# hero + two doors
# --------------------------------------------------------------------------- #
ui.hero("QNN Studio · quantum machine learning workspace",
        "Learn and benchmark quantum neural networks, without writing quantum code",
        "Bring a small table of data and see exactly how a quantum model is built, cleaned, trained and tested, then compare it "
        "fairly with classical machine learning. Built for students, instructors and early researchers.",
        [ui.chip("No quantum code", "t"), ui.chip("Leakage-free evaluation", ""), ui.chip("Honest, repeatable comparisons", ""),
         ui.chip("Runs on your laptop", "a")])

left, right = st.columns(2, gap="large")
with left:
    ui.md(ui.card("Research Workbench",
                  "<p>Run a complete, leakage-free study on your own CSV: <b>clean</b> it, prepare it for qubits, train the quantum model over several "
                  "random splits, and compare it with four classical models. Run sweeps, then export a report.</p>"
                  "<ul><li>Data quality score and cleaning audit trail</li><li>Mean ± spread over runs, confidence intervals</li>"
                  "<li>Sweeps over qubits, depth, data size, encoding</li></ul>",
                  ic="flask", badge='<span class="q-badge research">For researchers</span>',
                  foot="Seven steps · about 10 minutes for a first result"))
    st.page_link("views/research.py", label="Open the Research Workbench", icon=":material/science:", width="stretch")
with right:
    ui.md(ui.card("Learning Lab",
                  "<p>Seven short, hands-on lessons that teach how a quantum neural network works, in plain English. Each lesson has a small "
                  "experiment and a quick check.</p>"
                  "<ul><li>Measure a qubit, clean a messy table, encode numbers</li><li>Read a circuit and watch a model train</li>"
                  "<li>Compare models honestly and spot over-claims</li></ul>",
                  ic="grad", tone="t", badge='<span class="q-badge learn">For students and instructors</span>',
                  foot="Seven lessons · about 70 minutes · glossary with full forms"))
    st.page_link("views/learning.py", label="Start the Learning Lab", icon=":material/school:", width="stretch")

# --------------------------------------------------------------------------- #
# the use case
# --------------------------------------------------------------------------- #
ui.section("The use case", "A no-code sandbox to learn and benchmark quantum machine learning",
           "Two problems make quantum neural networks hard to try: they need specialist skills, and real datasets are bigger than quantum machines. "
           "QNN Studio hides the first and shrinks the second, so the only thing left is the question that matters: is it any good on my data?")
bc = BENCH["datasets"]["Breast cancer"]["models"]
ui.kpis([("Datasets benchmarked", str(len(BENCH["datasets"])), f"{BENCH['seeds']} random splits each"),
         ("Quantum parameters", f"{bc[QNN]['n_params']}", f"vs {bc['MLP (small neural net)']['n_params']} for a small neural net"),
         ("Time to a first result", f"{bc[QNN]['seconds']:.1f}s", "569 patients · 4 qubits"),
         ("Quantum advantage claimed", "None", "competitive, not better", "warn")])

ui.grid([
    ui.card("A real skills barrier",
            "<p>Building a QNN means circuits, encodings, optimizers and measurement. McKinsey (2022) calls the quantum talent shortage a bottleneck, "
            "and platforms such as Terra Quantum's TQ42 Studio (2025) and Bloq Quantum (2026) now target people who lack quantum programming skills.</p>",
            ic="users", foot=f"{link('TQ42 Studio', 'https://thequantuminsider.com/2025/04/14/terra-quantum-launches-tq42-studio-supports-no-code-quantum-ai-development-in-closed-beta/')} · "
                             f"{link('Bloq Quantum', 'https://quantumcomputingreport.com/bloq-quantum-launches-no-code-platform-to-accelerate-algorithm-development/')}"),
    ui.card("Fair comparison is the hard part",
            "<p>A 2024 benchmark of 12 quantum models on 160 datasets found out-of-the-box classical models generally beat the quantum ones, and a 2026 "
            "cancer-data benchmark found no evidence of quantum advantage. Researchers need quick, repeatable, honest comparisons.</p>",
            ic="bars", tone="t", foot=f"{link('Bowles et al. 2024', 'https://arxiv.org/abs/2403.07059')} · {link('Leither et al. 2026', 'https://arxiv.org/abs/2608.11373')}"),
    ui.card("Built for today's small machines",
            "<p>NISQ (Noisy Intermediate-Scale Quantum) devices hold tens to a few hundred qubits and are error-prone. Small tables, few features and short "
            "circuits are exactly where a simulator-based sandbox is useful.</p>",
            ic="cpu", tone="a", foot=link("Preskill 2018", "https://arxiv.org/abs/1801.00862")),
], cols=3)

# --------------------------------------------------------------------------- #
# how it works
# --------------------------------------------------------------------------- #
ui.section("How it works", "From a raw table to an honest answer", "Every step is visible, adjustable and explained.")
ui.grid([
    ui.card("1 · Upload", "<p>A CSV file or a curated dataset. Column types and the task (classification or regression) are detected for you.</p>", ic="upload"),
    ui.card("2 · Clean", "<p>A data-quality score, a list of problems, one-click fixes and a log of every change. Duplicates, gaps, text numbers, IDs, extremes.</p>", ic="filter", tone="t"),
    ui.card("3 · Prepare", "<p>Split first. Gap filling, scaling and PCA are learned from the training rows only, so test rows never leak in.</p>", ic="shield"),
    ui.card("4 · Encode and train", "<p>Numbers become qubit rotations. A circuit with adjustable angles is trained in mini-batches on a simulator.</p>", ic="atom", tone="a"),
    ui.card("5 · Compare", "<p>Four classical models on the same split. Results come as mean ± spread over several runs, with a plain-English verdict.</p>", ic="bars"),
    ui.card("6 · Experiment and report", "<p>Sweep qubits, depth, data size or encoding. Export the report, per-run results and settings.</p>", ic="file", tone="t"),
], cols=3)

# --------------------------------------------------------------------------- #
# evidence
# --------------------------------------------------------------------------- #
ui.section("Evidence", "What a user learns in minutes", f"Our own benchmark on five datasets, {BENCH['seeds']} random splits each, using the leakage-free pipeline. {BENCH['settings']}.")
cls = {k: v for k, v in BENCH["datasets"].items() if v["task"] == "classification"}
c1, c2 = st.columns([1.7, 1], gap="large")
with c1:
    fig = go.Figure()
    order = [QNN, "Logistic regression", "SVM (RBF kernel)", "MLP (small neural net)", "Random forest"]
    for m in order:
        fig.add_trace(go.Bar(name=m, x=list(cls), y=[cls[d]["models"][m]["mean"] * 100 for d in cls], marker_color=MODEL_COLOURS[m],
                             error_y=dict(type="data", array=[cls[d]["models"][m]["std"] * 100 for d in cls], color="rgba(236,235,248,.55)", thickness=1.2)))
    fig.update_layout(barmode="group", bargap=0.22)
    fig.update_yaxes(range=[85, 100], title="Test accuracy (%)")
    st.plotly_chart(ui.style_fig(fig, 400, "Test accuracy, mean ± 1 spread (axis starts at 85%)"), width="stretch")
with c2:
    rows = []
    for name, ds in cls.items():
        label, diff, best = verdict_for(ds)
        tone = {"comparable": "g", "behind": "a", "ahead": "t"}[label]
        rows.append(f'<div class="q-issue"><div><b>{name}</b> {ui.chip(label, tone)}<small>quantum {BENCH["datasets"][name]["models"][QNN]["mean"] * 100:.1f}% · '
                    f'best classical ({best}) {ds["models"][best]["mean"] * 100:.1f}% · gap {diff * 100:+.1f} pts</small></div></div>')
    reg = BENCH["datasets"]["Diabetes (R²)"]
    label, diff, best = verdict_for(reg)
    rows.append(f'<div class="q-issue"><div><b>Diabetes (predict a number, R²)</b> {ui.chip(label, {"comparable": "g", "behind": "a", "ahead": "t"}[label])}'
                f'<small>quantum {reg["models"][QNN]["mean"]:.2f} · best classical ({best}) {reg["models"][best]["mean"]:.2f}</small></div></div>')
    ui.md("".join(rows))
    ui.callout("", "<b>How to read it.</b> “Comparable” means the gap is inside the run-to-run noise (a paired 95% interval over the runs includes zero). The quantum model is competitive on these small "
                   "tables but not better, which is exactly what published benchmarks report.")

s1, s2 = st.columns(2, gap="large")
q_sw, f_sw = BENCH["sweeps"]["breast_cancer_qubits"], BENCH["sweeps"]["breast_cancer_train_fraction"]
with s1:
    fig = go.Figure(go.Scatter(x=[str(v) for v in q_sw["value"]], y=[m * 100 for m in q_sw["qnn_mean"]], mode="lines+markers",
                               line=dict(color=ui.VIOLET, width=3), marker=dict(size=10),
                               error_y=dict(type="data", array=[s * 100 for s in q_sw["qnn_std"]], color=ui.MUTED)))
    fig.update_xaxes(title="Number of qubits", type="category")
    fig.update_yaxes(title="Accuracy (%)", range=[88, 98])
    st.plotly_chart(ui.style_fig(fig, 320, "More qubits did not clearly help (breast cancer)", legend=False), width="stretch")
with s2:
    x = [f"{int(n)} patients" for n in f_sw["n_train"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[m * 100 for m in f_sw["qnn_mean"]], name="Quantum", mode="lines+markers", line=dict(color=ui.VIOLET, width=3),
                             error_y=dict(type="data", array=[s * 100 for s in f_sw["qnn_std"]], color=ui.MUTED)))
    fig.add_trace(go.Scatter(x=x, y=[m * 100 for m in f_sw["classical_mean"]], name="Logistic regression", mode="lines+markers",
                             line=dict(color=ui.TEAL, width=3, dash="dot"), error_y=dict(type="data", array=[s * 100 for s in f_sw["classical_std"]], color=ui.MUTED)))
    fig.update_xaxes(type="category", title="Training rows used")
    fig.update_yaxes(title="Accuracy (%)", range=[85, 98])
    st.plotly_chart(ui.style_fig(fig, 320, "Less data: the gap stays inside the noise"), width="stretch")

# --------------------------------------------------------------------------- #
# safeguards, limits, sources
# --------------------------------------------------------------------------- #
ui.section("Trust", "Built-in safeguards", "The workspace is designed so that the easy path is also the honest path.")
ui.grid([
    ui.card("No leakage", "<p>Everything that learns from data is fitted on training rows only and then applied to the held-out rows.</p>", ic="shield"),
    ui.card("Spread, not luck", "<p>Results are averaged over several random splits and shown with a spread and a confidence interval.</p>", ic="trend", tone="t"),
    ui.card("Honest verdicts", "<p>The verdict says ahead, comparable or behind based on the spread. It never claims a quantum advantage on its own.</p>", ic="check", tone="a"),
    ui.card("Audit trail", "<p>Every cleaning step is logged with rows and columns before and after. Reports include seeds and settings.</p>", ic="file"),
], cols=4)

ui.callout("warn", "<b>Limits and responsible use.</b> Quantum circuits run on a classical <i>simulator</i>, not on quantum hardware. The datasets are small and public. "
                   "QNN Studio is for learning and research; do not use its output for clinical, financial or safety decisions. "
                   "No user study has been run yet, so the learning benefit is a plan, not a measurement.")

ui.section("Sources", "Where the evidence comes from", "Each source below was opened and read; the two marked † were confirmed through search summaries only.")
ui.grid([
    ui.card("Benchmarking", "<ul><li>" + link("Bowles, Ahmed, Schuld (2024). Better than classical?", "https://arxiv.org/abs/2403.07059") + "</li><li>"
            + link("Kölle et al. (2023). Disentangling quantum and classical contributions", "https://arxiv.org/abs/2311.05559") + "</li><li>"
            + link("Leither et al. (2026). Benchmarking on oncological data", "https://arxiv.org/abs/2608.11373") + "</li><li>"
            + link("Schuld, Killoran (2022). Is quantum advantage the right goal?", "https://arxiv.org/abs/2203.01340") + "</li></ul>"),
    ui.card("Tools and skills", "<ul><li>" + link("Terra Quantum, TQ42 Studio (April 2025)", "https://thequantuminsider.com/2025/04/14/terra-quantum-launches-tq42-studio-supports-no-code-quantum-ai-development-in-closed-beta/") + "</li><li>"
            + link("Bloq Quantum no-code platform (reported May 2026)", "https://quantumcomputingreport.com/bloq-quantum-launches-no-code-platform-to-accelerate-algorithm-development/") + "</li><li>"
            + "QuantuMoonLight, SoftwareX 22 (2023) 101399 †</li><li>McKinsey (2022), Five lessons from AI on closing quantum's talent gap †</li></ul>", tone="t"),
    ui.card("Hardware and data", "<ul><li>" + link("Preskill (2018). Quantum computing in the NISQ era and beyond", "https://arxiv.org/abs/1801.00862") + "</li><li>"
            + link("UCI Breast Cancer Wisconsin (Diagnostic): 569 patients, 30 measurements", "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic") + "</li><li>"
            "Iris, Wine and Diabetes via scikit-learn</li></ul>", tone="a"),
], cols=3)
ui.footer()
