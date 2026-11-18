"""Research Workbench: a leakage-free, multi-run study of a quantum neural network on your own data."""
from __future__ import annotations

import io
import json
import platform
from dataclasses import replace

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from qnn_platform import ui  # noqa: E402
from qnn_platform.classical_model import BASELINES, baseline_label  # noqa: E402
from qnn_platform.cleaning import CleaningConfig, clean_structural, profile_dataframe, recommended_config  # noqa: E402
from qnn_platform.demo_data import demo_catalog, load_demo
from qnn_platform.figures import circuit_png, n_circuit_params  # noqa: E402
from qnn_platform.preprocessing import infer_task  # noqa: E402
from qnn_platform.quantum_model import ANSATZE, ENCODINGS  # noqa: E402
from qnn_platform.workflow import (QNN_NAME, SWEEP_PARAMS, ModelConfig, PrepConfig, TrainConfig, confusion,  # noqa: E402
                                   metric_label, paired_difference, prepare_split, primary_metric, run_study, run_sweep, verdict, wilson_interval)

ENC_LABELS = {"angle": "Angle: each number rotates one qubit",
              "amplitude": "Amplitude: numbers become the strengths of the state",
              "iqp": "IQP: rotate and link the qubits (Instantaneous Quantum Polynomial)"}
ANS_LABELS = {"basic_entangler": "Basic entangler: one rotation per qubit, then a ring of links",
              "strongly_entangling": "Strongly entangling: fuller rotations and richer links (default)",
              "hardware_efficient": "Hardware efficient: two rotations per qubit and a chain of links"}
OUTLIER_LABELS = {"none": "Leave as they are", "clip": "Clip to the limit (learned from training rows)",
                  "drop_train": "Remove from the training rows only"}


# --------------------------------------------------------------------------- #
# state and cached helpers
# --------------------------------------------------------------------------- #
def rs() -> dict:
    return st.session_state.setdefault("rs", {})


def read_csv(file) -> pd.DataFrame:
    raw = file.getvalue()
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
        except Exception:
            continue
    raise ValueError("Could not read this file as a CSV.")


@st.cache_data(show_spinner=False)
def cached_profile(df: pd.DataFrame, target: str):
    return profile_dataframe(df, target)


@st.cache_data(show_spinner=False)
def cached_split(df: pd.DataFrame, cfg: PrepConfig):
    return prepare_split(df, cfg)


def set_dataset(df: pd.DataFrame, name: str, source: str, target: str | None = None, task: str | None = None, blurb: str = ""):
    state = rs()
    state.clear()
    target = target or df.columns[-1]
    state.update(raw=df, name=name, source=source, target=target, task=task or infer_task(df, target), blurb=blurb)


def invalidate(*keys):
    for k in keys:
        rs().pop(k, None)


def step_states() -> list[tuple[str, str]]:
    S = rs()
    flags = [("Data", "raw" in S), ("Clean", "clean" in S), ("Prepare", "clean" in S), ("Model", "clean" in S),
             ("Train & compare", "study" in S), ("Experiments", bool(S.get("sweeps"))), ("Report", "study" in S)]
    out, found_active = [], False
    for label, done in flags:
        if done:
            out.append((label, "done"))
        elif not found_active:
            out.append((label, "active"))
            found_active = True
        else:
            out.append((label, "todo"))
    return out


def df_to_md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def fmt(mean: float, std: float, pct: bool = False) -> str:
    if pct:
        return f"{mean * 100:.1f}% ± {std * 100:.1f}"
    return f"{mean:.3f} ± {std:.3f}"


# --------------------------------------------------------------------------- #
# page header and sidebar inspector
# --------------------------------------------------------------------------- #
S = rs()
ui.hero("Research workbench",
        "Benchmark a quantum neural network on your own data",
        "A leakage-free study in seven steps: bring a table, clean it, prepare it, build the circuit, train it over several random "
        "splits, and compare it fairly with classical machine learning.",
        [ui.chip("Leakage-free evaluation", "t"), ui.chip("Mean ± spread over runs", ""), ui.chip("4 classical baselines", ""),
         ui.chip("Simulator, not hardware", "a")])
ui.stepper(step_states())

with st.sidebar:
    st.markdown("### Study inspector")
    if "raw" not in S:
        st.caption("No dataset yet. Start with step 1.")
    else:
        df = S["raw"]
        st.markdown(f"**{S['name']}**")
        st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
        st.caption(f"Target: `{S['target']}` · {S['task']}")
        if "clean" in S:
            b, a = S["clean"]["before"].score, S["clean"]["after"].score
            st.caption(f"Data quality score: {b} → **{a}**")
        else:
            st.caption("Cleaning: not applied yet")
        if "prep_summary" in S:
            st.caption(S["prep_summary"])
        if "study" in S:
            v, _ = verdict(S["study"])
            st.caption(f"Last result: quantum is **{v}** with classical models")
        st.divider()
        if st.button("Start a new study", width="stretch"):
            st.session_state["rs"] = {}
            st.rerun()

tabs = st.tabs(["1  Data", "2  Clean", "3  Prepare", "4  Model", "5  Train & compare", "6  Experiments", "7  Report"])

# =========================================================================== #
# 1 DATA
# =========================================================================== #
with tabs[0]:
    ui.section("Step 1", "Bring your data", "Upload a CSV with one row per example, or start from a curated dataset.")
    up_col, tip_col = st.columns([1.3, 1], gap="large")
    with up_col:
        st.markdown("#### Upload your own CSV")
        up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed", key="rs_upload")
        if up is not None and S.get("source_file") != (up.name, up.size):
            try:
                set_dataset(read_csv(up), up.name, "upload")
                rs()["source_file"] = (up.name, up.size)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with tip_col:
        st.write("")
        st.write("")
        ui.callout("", "<b>Tip.</b> One row per example, one column per measurement, and one column holding the answer you want to predict. "
                       "Messy data is fine: the next step cleans it.")
    st.markdown("#### Or start from a curated dataset")
    catalog = demo_catalog()
    for row in range(0, len(catalog), 3):
        cols = st.columns(3, gap="medium")
        for col, item in zip(cols, catalog[row:row + 3]):
            with col:
                tone = "a" if item["tag"] == "Cleaning showcase" else ""
                ui.md(ui.card(item["title"], f"<p>{item['description']}</p>", tone=tone, min_h=290,
                              badge=f'<span class="q-badge learn">{item["tag"]}</span>', foot=item["task"].capitalize()))
                if st.button("Load this dataset", key=f"load_{item['key']}", width="stretch"):
                    d = load_demo(item["key"])
                    set_dataset(d.df, d.title, "demo", d.target, d.task, d.description)
                    st.rerun()

    if "raw" in S:
        df = S["raw"]
        ui.section("Your dataset", S["name"], S.get("blurb", ""))
        n_num = int(sum(df[c].dtype.kind in "iuf" for c in df.columns))
        ui.kpis([("Rows", f"{len(df):,}", ""), ("Columns", f"{df.shape[1]}", f"{n_num} numeric · {df.shape[1] - n_num} text"),
                 ("Missing cells", f"{int(df.isna().sum().sum()):,}", f"{df.isna().mean().mean():.1%} of all cells", "warn" if df.isna().any().any() else ""),
                 ("Duplicate rows", f"{int(df.duplicated().sum())}", "", "warn" if df.duplicated().any() else "")])
        st.markdown("#### What do you want to predict?")
        c1, c2 = st.columns([1.4, 1])
        cols_list = list(df.columns)
        new_target = c1.selectbox("Target column (the answer)", cols_list, index=cols_list.index(S["target"]), key="rs_target_sel")
        auto = infer_task(df, new_target)
        new_task = c2.radio("Task", ["classification", "regression"], index=0 if auto == "classification" else 1, horizontal=True,
                            help="Classification: pick a category. Regression: predict a number. Detected automatically; change it if needed.",
                            key=f"rs_task_{new_target}")
        if new_target != S["target"] or new_task != S["task"]:
            S["target"], S["task"] = new_target, new_task
            invalidate("clean", "study", "sweeps", "prep_summary")
            st.rerun()
        st.dataframe(df.head(15), width="stretch", height=360)

# =========================================================================== #
# 2 CLEAN
# =========================================================================== #
with tabs[1]:
    ui.section("Step 2", "Clean the data", "Find what is wrong, fix it, and keep an audit trail of every change.")
    if "raw" not in S:
        ui.callout("warn", "Load a dataset in step 1 first.")
    else:
        raw, target = S["raw"], S["target"]
        rep = cached_profile(raw, target)
        c1, c2 = st.columns([1, 2.4], gap="large")
        with c1:
            ui.md(ui.card("Data quality (raw data)", ui.score_ring(rep.score), tone="t",
                          foot=f"{rep.n_rows:,} rows · {rep.n_cols} columns"))
        with c2:
            ui.kpis([("Missing cells", f"{rep.missing_cells:,}", f"{rep.missing_pct:.1%} of cells", "warn" if rep.missing_cells else ""),
                     ("Duplicate rows", f"{rep.duplicate_rows}", "", "warn" if rep.duplicate_rows else ""),
                     ("Extreme values", f"{rep.outlier_cells}", "beyond 3×IQR", "warn" if rep.outlier_cells else ""),
                     ("Problem columns", f"{len(rep.constant_cols) + len(rep.id_like_cols) + len(rep.numeric_text_cols)}",
                      "constant · ID · text numbers", "warn" if (rep.constant_cols or rep.id_like_cols or rep.numeric_text_cols) else "")])
            if rep.issues:
                issues_html = "".join(
                    f'<div class="q-issue"><div><span class="q-sev {i.severity}">{i.severity}</span><b>{ui.esc(i.title)}</b>'
                    f'<small>{ui.esc(i.detail)} <i>Fix: {ui.esc(i.action)}</i></small></div></div>' for i in rep.issues)
                ui.md(issues_html)
            else:
                ui.callout("good", "<b>No problems found.</b> The data is already clean.")

        with st.expander("See where the problems are", expanded=False):
            g1, g2 = st.columns(2)
            miss = rep.missing_by_col[rep.missing_by_col > 0].sort_values(ascending=False).head(12)
            if len(miss):
                fig = go.Figure(go.Bar(x=miss.values, y=miss.index, orientation="h", marker_color=ui.AMBER))
                fig.update_yaxes(autorange="reversed")
                g1.plotly_chart(ui.style_fig(fig, 320, "Missing values per column", legend=False), width="stretch")
            else:
                g1.caption("No missing values.")
            if len(rep.outliers_by_col):
                o = rep.outliers_by_col.head(12)
                fig = go.Figure(go.Bar(x=o.values, y=o.index, orientation="h", marker_color=ui.ROSE))
                fig.update_yaxes(autorange="reversed")
                g2.plotly_chart(ui.style_fig(fig, 320, "Extreme values per column", legend=False), width="stretch")
            else:
                g2.caption("No extreme values.")

        st.markdown("### Cleaning plan")
        st.caption("Structural steps look only at rows and columns. Fitted steps learn numbers (a median, a limit) and run later, "
                   "on the training rows only, so the test rows never influence training.")
        p1, p2 = st.columns(2, gap="large")
        with p1:
            st.markdown("**Structural steps** (whole dataset, label-free)")
            c_std = st.checkbox("Tidy text: trim spaces, merge 'North' and 'north'", True, key="cl_std")
            c_num = st.checkbox("Convert numbers stored as text ('85%', '1,200', '12*')", True, key="cl_num")
            c_dup = st.checkbox("Remove duplicate rows", True, key="cl_dup")
            c_const = st.checkbox("Drop constant columns", True, key="cl_const")
            c_id = st.checkbox("Drop ID-like columns (names, IDs, URLs)", True, key="cl_id")
            c_scol = st.slider("Drop columns that are empty more than", 0.3, 1.0, 0.6, 0.05, key="cl_scol",
                               help="1.0 means never drop columns for being empty.")
            c_srow = st.slider("Drop rows that are empty more than", 0.3, 1.0, 0.6, 0.05, key="cl_srow",
                               help="1.0 means never drop rows for being empty.")
        with p2:
            st.markdown("**Fitted steps** (learned from training rows only)")
            f_num = st.selectbox("Fill missing numbers with", ["median", "mean", "zero"], key="cl_fnum")
            f_cat = st.selectbox("Fill missing categories with", ["most_frequent", "missing_label"],
                                 format_func=lambda v: "the most common value" if v == "most_frequent" else "a 'missing' label", key="cl_fcat")
            f_out = st.selectbox("Extreme values", list(OUTLIER_LABELS), index=1 if rep.outlier_cells else 0,
                                 format_func=OUTLIER_LABELS.get, key="cl_fout")
            f_iqr = st.slider("Extreme-value limit (× IQR)", 1.5, 5.0, 3.0, 0.5, key="cl_fiqr",
                              help="IQR = interquartile range. A larger number only catches more extreme values.")
            f_flag = st.checkbox("Add a 'was missing' flag column for every column with gaps", False, key="cl_flag")
        b1, b2, _ = st.columns([1.1, 1.1, 2])
        run_rec = b1.button("Apply recommended", type="primary", width="stretch")
        run_plan = b2.button("Apply my plan", width="stretch")
        if run_rec or run_plan:
            if run_rec:
                cfg = replace(recommended_config(rep))
            else:
                cfg = CleaningConfig(standardize_text=c_std, parse_numeric_text=c_num, drop_sparse_cols_pct=c_scol,
                                     drop_sparse_rows_pct=c_srow, drop_duplicates=c_dup, drop_constant=c_const, drop_id_like=c_id,
                                     impute_numeric=f_num, impute_categorical=f_cat, add_missing_flags=f_flag,
                                     outliers=f_out, outlier_iqr=f_iqr)
            cleaned, log = clean_structural(raw, target, cfg)
            S["clean"] = {"cfg": cfg, "df": cleaned, "log": log, "before": rep, "after": profile_dataframe(cleaned, target)}
            invalidate("study", "sweeps", "prep_summary")
            st.rerun()

        if "clean" in S:
            C = S["clean"]
            ui.section("Result", "Before and after")
            ui.kpis([("Quality score", f"{C['after'].score}", f"was {C['before'].score}", "" if C["after"].score >= C["before"].score else "bad"),
                     ("Rows", f"{C['after'].n_rows:,}", f"{C['after'].n_rows - C['before'].n_rows:+,} vs before"),
                     ("Columns", f"{C['after'].n_cols}", f"{C['after'].n_cols - C['before'].n_cols:+} vs before"),
                     ("Duplicates", f"{C['after'].duplicate_rows}", f"was {C['before'].duplicate_rows}")])
            if C["log"]:
                steps_html = "".join(f'<div class="it"><b>{ui.esc(e.step)}</b>{ui.esc(e.detail)}'
                                     f' · rows {e.rows_before:,} → {e.rows_after:,} · columns {e.cols_before} → {e.cols_after}</div>' for e in C["log"])
                ui.md(f'<div class="q-tl">{steps_html}</div>')
            else:
                ui.callout("good", "Nothing needed changing in the structural step.")
            fit = C["cfg"]
            ui.callout("", "<b>Runs after the split, on training rows only:</b> missing numbers → "
                           f"{fit.impute_numeric}; missing categories → {fit.impute_categorical.replace('_', ' ')}; extremes → "
                           f"{OUTLIER_LABELS[fit.outliers].lower()}" + ("; 'was missing' flags added" if fit.add_missing_flags else "") + ".")
            num_cols = [c for c in C["df"].columns if C["df"][c].dtype.kind in "iuf" and c != target]
            if num_cols:
                col_pick = st.selectbox("Compare a column before and after", num_cols, key="cl_cmp")
                fig = go.Figure()
                if col_pick in raw.columns and raw[col_pick].dtype.kind in "iuf":
                    fig.add_trace(go.Histogram(x=raw[col_pick].dropna(), name="before", opacity=0.55, marker_color=ui.AMBER, nbinsx=40))
                fig.add_trace(go.Histogram(x=C["df"][col_pick].dropna(), name="after", opacity=0.65, marker_color=ui.TEAL, nbinsx=40))
                fig.update_layout(barmode="overlay")
                st.plotly_chart(ui.style_fig(fig, 300, f"Distribution of {col_pick}"), width="stretch")
            with st.expander("Cleaned data preview"):
                st.dataframe(C["df"].head(20), width="stretch")
            st.download_button("Download cleaned CSV", C["df"].to_csv(index=False).encode(), "cleaned_data.csv", "text/csv")

# =========================================================================== #
# 3 PREPARE
# =========================================================================== #
with tabs[2]:
    ui.section("Step 3", "Prepare the data for qubits", "Split first, then learn every transformation from the training rows only.")
    if "clean" not in S:
        ui.callout("warn", "Run the cleaning step (step 2) first. It takes one click with 'Apply recommended'.")
    else:
        c1, c2, c3 = st.columns(3)
        domain = c1.selectbox("Domain profile", ["general", "medical"], format_func=str.capitalize,
                              help="Medical also drops patient-identifier columns, adds 'was missing' flags and clips extreme readings.", key="pp_dom")
        scaler = c2.selectbox("Scaling", ["minmax", "standard"], key="pp_scaler",
                              format_func=lambda v: "Min-max: squeeze into 0 to π" if v == "minmax" else "Standard: centre on 0, spread 1",
                              help="Puts all numbers on one scale. Min-max maps each column to 0..π, the range of a rotation angle.")
        reduction = c3.selectbox("Shrinking method", ["pca", "kbest"], key="pp_red",
                                 format_func=lambda v: "PCA: mix columns into a few summaries" if v == "pca" else "SelectKBest: keep the best columns",
                                 help="Used only when there are more columns than qubits.")
        d1, d2, d3, d4 = st.columns(4)
        n_qubits = d1.slider("Number of qubits", 2, 10, 4, help="Qubit = quantum bit. More qubits can hold more columns but simulate slower.", key="pp_q")
        test_size = d2.slider("Held-out share", 0.1, 0.5, 0.25, 0.05, help="Rows hidden during training and used only for testing.", key="pp_test")
        seed = d3.number_input("Base random seed", 0, 9999, 0, key="pp_seed")
        n_seeds = d4.select_slider("Number of runs", [1, 3, 5, 10], value=3,
                                   help="Each run uses a different split and starting angles. More runs = a trustworthy spread.", key="pp_runs")
        prep = PrepConfig(target=S["target"], task=S["task"], n_qubits=n_qubits, reduction=reduction, scaler=scaler,
                          test_size=test_size, seed=int(seed), domain=domain, cleaning=S["clean"]["cfg"])
        S["prep"], S["n_seeds"] = prep, n_seeds
        try:
            split = cached_split(S["clean"]["df"], prep)
        except Exception as exc:  # pragma: no cover - surfaced to the user
            st.error(f"Could not prepare the data: {exc}")
            split = None
        if split is not None:
            batches = int(np.ceil(len(split.y_train) / 16))
            S["prep_summary"] = (f"{len(split.y_train):,} train / {len(split.y_val):,} test · {split.n_raw_columns} columns → "
                                 f"{split.n_after_reduction} features · {split.n_qubits} qubits")
            ui.kpis([("Training rows", f"{len(split.y_train):,}", f"{batches} batches per epoch (size 16)"),
                     ("Held-out rows", f"{len(split.y_val):,}", "never used for fitting"),
                     ("Columns → numbers", f"{split.n_raw_columns} → {split.n_encoded}", "after one-hot encoding"),
                     ("Features → qubits", f"{split.n_after_reduction} → {split.n_qubits}", "what the circuit sees")])
            ui.md(f'<div class="q-tl">' + "".join(f'<div class="it"><b>{ui.esc(a)}</b>{ui.esc(b)}</div>' for a, b in split.steps) + "</div>")
            ui.callout("good", "<b>No leakage.</b> Gap filling, extreme-value limits, scaling and PCA are fitted on the training rows only, "
                               "then applied unchanged to the held-out rows. Scores therefore reflect new, unseen data.")
            if split.explained_variance is not None:
                ev = split.explained_variance
                fig = go.Figure()
                fig.add_trace(go.Bar(x=[f"PC{i + 1}" for i in range(len(ev))], y=ev, name="Each component", marker_color=ui.VIOLET))
                fig.add_trace(go.Scatter(x=[f"PC{i + 1}" for i in range(len(ev))], y=np.cumsum(ev), name="Cumulative", mode="lines+markers",
                                         line=dict(color=ui.TEAL, width=3)))
                fig.update_yaxes(tickformat=".0%", title="share of variation kept")
                st.plotly_chart(ui.style_fig(fig, 300, f"How much information the {len(ev)} PCA summaries keep"), width="stretch")
            with st.expander("Preview the numbers the circuit receives"):
                st.dataframe(pd.DataFrame(split.X_train[:12], columns=[f"qubit {i}" for i in range(split.X_train.shape[1])]).round(3), width="stretch")

# =========================================================================== #
# 4 MODEL
# =========================================================================== #
with tabs[3]:
    ui.section("Step 4", "Design the quantum circuit", "Choose how data enters the qubits and how the circuit learns.")
    if "clean" not in S or "prep" not in S:
        ui.callout("warn", "Finish steps 2 and 3 first.")
    else:
        split = cached_split(S["clean"]["df"], S["prep"])
        m1, m2, m3 = st.columns([1.3, 1.5, 1])
        enc = m1.selectbox("Data encoding", ENCODINGS, index=0, format_func=ENC_LABELS.get, key="md_enc",
                           help="How the numbers are written into qubits.")
        ans = m2.selectbox("Circuit type (variational ansatz)", ANSATZE, index=1, format_func=ANS_LABELS.get, key="md_ans",
                           help="Variational = the gates have adjustable angles. Ansatz = the chosen gate pattern.")
        layers = m3.slider("Layers (depth)", 1, 6, 2, help="How many times the learning pattern repeats.", key="md_layers")
        S["mcfg"] = ModelConfig(encoding=enc, ansatz=ans, n_layers=layers)
        nq = split.n_qubits
        n_head = nq * (split.n_classes if split.n_classes > 2 else 1) + (split.n_classes if split.n_classes > 2 else 1)
        cp = n_circuit_params(nq, layers, ans)
        left, right = st.columns([1.5, 1], gap="large")
        with left:
            st.image(circuit_png(nq, layers, enc, ans), caption="The circuit that will be trained (encoder, learning layers, measurement)", width="stretch")
        with right:
            ui.kpis([("Qubits", str(nq), "register width"), ("Circuit angles", str(cp), f"{layers} layers × {nq} qubits"),
                     ("Head weights", str(n_head), "turn measurements into the answer")])
            ui.callout("", "<b>Reading the picture.</b> Each line is a qubit. The first block <i>encodes</i> your numbers. The next block is the "
                           "<i>learning layers</i>: rotations with adjustable angles, plus links between qubits. The meters at the end "
                           "<i>measure</i> every qubit, and a small linear layer turns those numbers into the prediction.")
            if enc == "amplitude":
                ui.callout("warn", "Amplitude encoding pads the numbers to a power of two and normalises them, so absolute size is lost.")

# =========================================================================== #
# 5 TRAIN & COMPARE
# =========================================================================== #
with tabs[4]:
    ui.section("Step 5", "Train and compare", "The quantum model and four classical models, on identical data, over several runs.")
    if "clean" not in S or "prep" not in S or "mcfg" not in S:
        ui.callout("warn", "Finish steps 2 to 4 first.")
    else:
        reg = S["task"] == "regression"
        t1, t2, t3, t4 = st.columns(4)
        epochs = t1.slider("Epochs", 3, 80, 30 if reg else 25, help="Passes over the training data.", key="tr_ep")
        batch = t2.select_slider("Batch size", [8, 16, 32, 64], value=16, help="Rows learned from at once. Small batches let big data be handled in pieces.", key="tr_bs")
        opt = t3.selectbox("Optimizer", ["adam", "sgd", "rmsprop"], help="Adam = Adaptive Moment Estimation; SGD = Stochastic Gradient Descent; RMSprop = Root Mean Square Propagation.", key="tr_opt")
        lr = t4.number_input("Learning rate", 0.001, 1.0, 0.05 if reg else 0.1, 0.01, format="%.3f", help="Size of each learning step.", key="tr_lr")
        baselines = st.multiselect("Classical models to compare with", BASELINES, default=BASELINES,
                                   format_func=lambda n: baseline_label(n, S["task"]), key="tr_base")
        n_runs = S.get("n_seeds", 3)
        go_btn = st.button(f"Run study ({n_runs} runs)", type="primary", key="tr_go")
        if go_btn:
            bar = st.progress(0.0, text="Preparing…")
            tcfg = TrainConfig(epochs=epochs, batch_size=batch, optimizer=opt, lr=float(lr))
            res = run_study(S["clean"]["df"], S["prep"], S["mcfg"], tcfg, n_seeds=n_runs, baselines=baselines,
                            progress_cb=lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t))
            S["study"], S["tcfg"] = res, tcfg
            invalidate("sweeps")
            st.rerun()

        if "study" in S:
            R = S["study"]
            p = R.primary
            v, vtext = verdict(R)
            summ = R.summary
            q = summ[summ["model"] == QNN_NAME].iloc[0]
            others = summ[summ["model"] != QNN_NAME]
            best = others.loc[others[f"{p}_mean"].idxmax()] if len(others) else None
            kind = {"ahead": "warn", "comparable": "good", "behind": "warn"}[v]
            ui.callout(kind, f"<b>Verdict: {v}.</b> {vtext}")
            pct = p == "accuracy"
            items = [(f"Quantum {metric_label(p)}", fmt(q[f"{p}_mean"], q[f"{p}_std"], pct), f"mean ± spread over {R.n_seeds} runs")]
            if best is not None:
                items.append((f"Best classical", fmt(best[f"{p}_mean"], best[f"{p}_std"], pct), best["model"]))
                gap = q[f"{p}_mean"] - best[f"{p}_mean"]
                items.append(("Gap", f"{gap * 100:+.1f} pts" if pct else f"{gap:+.3f}", "quantum minus best classical"))
            items.append(("Quantum training time", f"{q['seconds']:.1f}s", "per run, on a simulator"))
            ui.kpis(items)
            if R.task == "classification":
                lo, hi = wilson_interval(int(round(q["accuracy_mean"] * R.n_val)), R.n_val)
                st.caption(f"The held-out set has {R.n_val} rows, so a single accuracy has a 95% confidence interval of about "
                           f"{lo * 100:.1f}% to {hi * 100:.1f}%. Differences smaller than this are within the noise.")

            st.markdown("#### Leaderboard")
            vmax = max(summ[f"{p}_mean"].max(), 1e-9)
            rows_html = []
            for _, r in summ.iterrows():
                me = r["model"] == QNN_NAME
                width = max(3.0, 100 * max(r[f"{p}_mean"], 0) / vmax)
                params = "n/a" if pd.isna(r["n_params"]) else f"{int(r['n_params']):,}"
                other_m = [m for m in ("macro_f1", "balanced_accuracy", "rmse") if f"{m}_mean" in r.index][:1]
                extra = f"{metric_label(other_m[0])} {r[f'{other_m[0]}_mean']:.3f}" if other_m else ""
                rows_html.append(f'<tr class="{"me" if me else ""}"><td>{ui.esc(r["model"])}</td>'
                                 f'<td>{fmt(r[f"{p}_mean"], r[f"{p}_std"], pct)}</td>'
                                 f'<td><div class="q-bar {"" if me else "alt"}"><i style="width:{width:.0f}%"></i></div></td>'
                                 f'<td>{extra}</td><td>{r["seconds"]:.2f}s</td><td>{params}</td></tr>')
            ui.md('<table class="q-table"><thead><tr><th>Model</th><th>' + metric_label(p) + ' (mean ± spread)</th><th></th>'
                  '<th>Also</th><th>Train time</th><th>Parameters</th></tr></thead><tbody>' + "".join(rows_html) + "</tbody></table>")

            g1, g2 = st.columns(2)
            fig = go.Figure()
            order_df = summ.iloc[::-1]
            fig.add_trace(go.Bar(y=order_df["model"], x=order_df[f"{p}_mean"], orientation="h", showlegend=False,
                                 marker_color=[ui.VIOLET if m == QNN_NAME else "#64748b" for m in order_df["model"]],
                                 error_x=dict(type="data", array=order_df[f"{p}_std"], color=ui.MUTED),
                                 text=[f"{v * 100:.1f}%" if pct else f"{v:.2f}" for v in order_df[f"{p}_mean"]], textposition="inside",
                                 insidetextanchor="start", textfont=dict(color="#fff")))
            lo_ax = max(0.0, float(summ[f"{p}_mean"].min()) - 0.15) if pct else None
            fig.update_xaxes(title=metric_label(p), range=[lo_ax, 1.0] if pct else None)
            g1.plotly_chart(ui.style_fig(fig, 340, f"{metric_label(p)} by model (error bar = 1 spread)", legend=False), width="stretch")
            h = R.artifacts["qnn"]["history"]
            ep = list(range(1, len(h.train_loss) + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ep, y=h.train_loss, name="training loss", line=dict(color=ui.VIOLET, width=3)))
            fig.add_trace(go.Scatter(x=ep, y=h.val_loss, name="held-out loss", line=dict(color=ui.TEAL, width=3)))
            fig.update_xaxes(title="epoch")
            fig.update_yaxes(title="loss")
            g2.plotly_chart(ui.style_fig(fig, 340, "Quantum model: learning curve (first run)"), width="stretch")

            y_val = R.artifacts["y_val"]
            if R.task == "classification":
                names = R.class_names
                cm_models = [m for m in R.artifacts["preds"]]
                pick = st.selectbox("Compare the quantum confusion matrix with", [m for m in cm_models if m != QNN_NAME], key="tr_cm")
                a, b = st.columns(2)
                for col, model_name in ((a, QNN_NAME), (b, pick)):
                    cm = confusion(y_val, R.artifacts["preds"][model_name], len(names))
                    fig = go.Figure(go.Heatmap(z=cm, x=[f"pred {n}" for n in names], y=[f"true {n}" for n in names],
                                               colorscale=[[0, "#17142b"], [1, ui.VIOLET if model_name == QNN_NAME else "#94a3b8"]],
                                               text=cm, texttemplate="%{text}", textfont=dict(size=16), showscale=False))
                    fig.update_yaxes(autorange="reversed")
                    col.plotly_chart(ui.style_fig(fig, 320, model_name, legend=False), width="stretch")
            else:
                preds = R.artifacts["preds"][QNN_NAME]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_val, y=preds, mode="markers", marker=dict(color=ui.VIOLET, size=8, opacity=0.75), name="held-out rows"))
                lo_, hi_ = float(min(y_val.min(), preds.min())), float(max(y_val.max(), preds.max()))
                fig.add_trace(go.Scatter(x=[lo_, hi_], y=[lo_, hi_], mode="lines", line=dict(color=ui.TEAL, dash="dash"), name="perfect prediction"))
                fig.update_xaxes(title="actual")
                fig.update_yaxes(title="predicted by the quantum model")
                st.plotly_chart(ui.style_fig(fig, 380, "Predicted vs actual (first run)"), width="stretch")
            with st.expander("Every run, one row each"):
                st.dataframe(R.per_seed.round(4), width="stretch")

# =========================================================================== #
# 6 EXPERIMENTS
# =========================================================================== #
with tabs[5]:
    ui.section("Step 6", "Experiments", "Change one setting at a time and see whether it really matters, with a spread for every point.")
    if "study" not in S:
        ui.callout("warn", "Run the study in step 5 first. Sweeps start from those settings.")
    else:
        e1, e2 = st.columns([1.2, 2])
        param = e1.selectbox("Setting to vary", list(SWEEP_PARAMS), format_func=SWEEP_PARAMS.get, key="ex_param")
        defaults = {"n_qubits": [2, 3, 4, 6], "n_layers": [1, 2, 3, 4], "train_fraction": [0.1, 0.25, 0.5, 1.0],
                    "encoding": list(ENCODINGS), "ansatz": list(ANSATZE)}
        options = {"n_qubits": [2, 3, 4, 5, 6, 8], "n_layers": [1, 2, 3, 4, 5, 6], "train_fraction": [0.1, 0.25, 0.5, 0.75, 1.0],
                   "encoding": list(ENCODINGS), "ansatz": list(ANSATZE)}
        values = e2.multiselect("Values to try", options[param], default=defaults[param], key=f"ex_vals_{param}")
        runs = st.select_slider("Runs per value", [1, 3, 5], value=3, key="ex_runs")
        if st.button("Run experiment", type="primary", key="ex_go") and values:
            bar = st.progress(0.0, text="Starting…")
            order = options[param]
            vals = sorted(values, key=order.index)
            out = run_sweep(S["clean"]["df"], S["prep"], S["mcfg"], S["tcfg"], param, vals, n_seeds=runs,
                            progress_cb=lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t))
            S.setdefault("sweeps", {})[param] = {"table": out, "runs": runs, "metric": primary_metric(S["task"])}
            st.rerun()
        for pkey, sw in (S.get("sweeps") or {}).items():
            tbl, pm = sw["table"], sw["metric"]
            ui.section("Result", SWEEP_PARAMS[pkey])
            x = tbl["value"].astype(str)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=tbl["qnn_mean"], mode="lines+markers", name="Quantum model", line=dict(color=ui.VIOLET, width=3),
                                     marker=dict(size=10), error_y=dict(type="data", array=tbl["qnn_std"], color=ui.MUTED)))
            if "classical_mean" in tbl:
                fig.add_trace(go.Scatter(x=x, y=tbl["classical_mean"], mode="lines+markers", name="Logistic / Ridge reference",
                                         line=dict(color=ui.TEAL, width=3, dash="dot"), marker=dict(size=8),
                                         error_y=dict(type="data", array=tbl["classical_std"], color=ui.MUTED)))
            fig.update_xaxes(title=SWEEP_PARAMS[pkey], type="category")
            fig.update_yaxes(title=metric_label(pm))
            a, b = st.columns([1.6, 1])
            a.plotly_chart(ui.style_fig(fig, 360, f"{metric_label(pm)} vs {SWEEP_PARAMS[pkey].lower()} ({sw['runs']} runs per point)"), width="stretch")
            with b:
                raw = tbl.attrs.get("raw")
                best_row, worst_row = tbl.loc[tbl["qnn_mean"].idxmax()], tbl.loc[tbl["qnn_mean"].idxmin()]
                if sw["runs"] == 1 or raw is None:
                    ui.callout("warn", "<b>One run per point.</b> Differences cannot be separated from luck. Repeat with 3 or 5 runs.")
                elif best_row["value"] == worst_row["value"]:
                    ui.callout("", "Only one setting was tried.")
                else:
                    a_ = raw[raw["value"] == best_row["value"]].sort_values("seed")["qnn"].to_numpy()
                    b_ = raw[raw["value"] == worst_row["value"]].sort_values("seed")["qnn"].to_numpy()
                    pdiff = paired_difference(a_, b_, S["study"].n_val, S["task"] == "classification")
                    ci = f"{pdiff['mean']:+.3f} ± {pdiff['half_width']:.3f}"
                    if pdiff["significant"]:
                        ui.callout("", f"<b>A measurable difference.</b> Best at <b>{best_row['value']}</b> ({best_row['qnn_mean']:.3f}) vs worst at "
                                       f"<b>{worst_row['value']}</b> ({worst_row['qnn_mean']:.3f}); paired gap {ci} (95% interval) excludes zero.")
                    else:
                        ui.callout("good", f"<b>Within the noise.</b> Best ({best_row['value']}: {best_row['qnn_mean']:.3f}) vs worst "
                                           f"({worst_row['value']}: {worst_row['qnn_mean']:.3f}); paired gap {ci} (95% interval) includes zero. "
                                           "This setting does not clearly matter here.")
                st.dataframe(tbl.drop(columns=["n_train"], errors="ignore").round(4), width="stretch", height=250)

# =========================================================================== #
# 7 REPORT
# =========================================================================== #
def build_report() -> str:
    R, C, P, M, T = S["study"], S["clean"], S["prep"], S["mcfg"], S["tcfg"]
    p = R.primary
    v, vtext = verdict(R)
    lines = [f"# QNN Studio study report", "", f"*Dataset:* {S['name']}  ", f"*Task:* {R.task}, target `{S['target']}`  ",
             f"*Runs:* {R.n_seeds} random splits, held-out rows per run: {R.n_val}", "", "## 1. Data cleaning",
             f"Quality score {C['before'].score} → {C['after'].score}. Rows {C['before'].n_rows:,} → {C['after'].n_rows:,}.", ""]
    lines += [f"- {e.step}: {e.detail} (rows {e.rows_before:,} → {e.rows_after:,}, columns {e.cols_before} → {e.cols_after})" for e in C["log"]] or ["- No structural changes were needed."]
    lines += ["", f"Fitted on training rows only: fill numbers with {C['cfg'].impute_numeric}, categories with {C['cfg'].impute_categorical}, extremes: {C['cfg'].outliers}.", "",
              "## 2. Preparation", f"Qubits {P.n_qubits}, shrinking {P.reduction}, scaling {P.scaler}, held-out share {P.test_size}, domain {P.domain}, base seed {P.seed}.", "",
              "## 3. Model", f"Encoding {M.encoding}, circuit {M.ansatz}, {M.n_layers} layers. Training: {T.epochs} epochs, batch {T.batch_size}, {T.optimizer}, learning rate {T.lr}.", "",
              "## 4. Results", f"**Verdict: {v}.** {vtext}", "", f"| Model | {metric_label(p)} (mean ± spread) | Train time (s) |", "|---|---|---|"]
    for _, r in R.summary.iterrows():
        lines.append(f"| {r['model']} | {fmt(r[f'{p}_mean'], r[f'{p}_std'], p == 'accuracy')} | {r['seconds']:.2f} |")
    for pkey, sw in (S.get("sweeps") or {}).items():
        lines += ["", f"### Experiment: {SWEEP_PARAMS[pkey]} ({sw['runs']} runs per point)", "", df_to_md(sw["table"].round(4))]
    lines += ["", "## 5. Limits", "- Quantum circuits run on a classical simulator, not real quantum hardware.",
              "- Results come from the held-out rows of this dataset only; they may not carry over to other data.",
              "- A comparable result is not evidence of quantum advantage.", "",
              f"*Environment:* Python {platform.python_version()}, torch {torch.__version__}."]
    return "\n".join(lines)


with tabs[6]:
    ui.section("Step 7", "Report and reproducibility", "Everything needed to explain, repeat and share this study.")
    if "study" not in S:
        ui.callout("warn", "Run the study in step 5 to generate a report.")
    else:
        report = build_report()
        a, b, c = st.columns(3)
        a.download_button("Download report (Markdown)", report.encode(), "qnn_study_report.md", "text/markdown", type="primary", width="stretch")
        b.download_button("Download per-run results (CSV)", S["study"].per_seed.to_csv(index=False).encode(), "per_run_results.csv", "text/csv", width="stretch")
        c.download_button("Download settings (JSON)", json.dumps(S["study"].config, indent=2, default=str).encode(), "study_settings.json", "application/json", width="stretch")
        ui.callout("", "<b>Reproducibility.</b> The seeds, every setting and the cleaning log are saved in these files. "
                       "Re-running with the same seed and data gives the same result.")
        st.markdown(report)

ui.footer()
