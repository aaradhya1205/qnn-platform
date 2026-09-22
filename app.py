"""Interactive Quantum Neural Network platform.

Upload a dataset, configure preprocessing + a variational QNN, train it with
mini-batches, and compare it against classical ML/DNN baselines - all through
a browser UI, no quantum-circuit code required.
"""
import io
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from qnn_platform.classical_model import BASELINES, build_baseline
from qnn_platform.preprocessing import infer_task, prepare_dataset
from qnn_platform.quantum_model import ANSATZE, ENCODINGS, QNNModel
from qnn_platform.train import train_model

st.set_page_config(page_title="QNN Platform", layout="wide")
st.title("🧪 Interactive Quantum Neural Network Platform")
st.caption(
    "Upload a dataset, configure a quantum neural network, train it, and compare "
    "it against classical baselines - no quantum programming required."
)

# --------------------------------------------------------------------------- #
# 1. Data upload
# --------------------------------------------------------------------------- #
st.header("1. Upload dataset")
uploaded = st.file_uploader("CSV file", type=["csv"])
use_demo = st.checkbox("Use a built-in demo dataset instead", value=uploaded is None)

df = None
if uploaded is not None and not use_demo:
    df = pd.read_csv(uploaded)
elif use_demo:
    demo_choice = st.selectbox("Demo dataset", ["Iris (classification)", "Diabetes (regression)"])
    from sklearn.datasets import load_iris, load_diabetes
    if demo_choice.startswith("Iris"):
        data = load_iris(as_frame=True)
        df = data.frame.rename(columns={"target": "species"})
        df["species"] = df["species"].map(dict(enumerate(data.target_names)))
    else:
        data = load_diabetes(as_frame=True)
        df = data.frame

if df is None:
    st.info("Upload a CSV or select a demo dataset to continue.")
    st.stop()

st.dataframe(df.head(10), width="stretch")
st.caption(f"{df.shape[0]} rows x {df.shape[1]} columns")

# --------------------------------------------------------------------------- #
# 2. Task configuration
# --------------------------------------------------------------------------- #
st.header("2. Target & task")
col1, col2 = st.columns(2)
with col1:
    target = st.selectbox("Target column", df.columns.tolist(), index=len(df.columns) - 1)
with col2:
    auto_task = infer_task(df, target)
    task = st.selectbox("Task type", ["classification", "regression"],
                         index=0 if auto_task == "classification" else 1,
                         help=f"Auto-detected: {auto_task}")

# --------------------------------------------------------------------------- #
# 3. Preprocessing & dimensionality reduction
# --------------------------------------------------------------------------- #
st.header("3. Preprocessing")
c1, c2, c3 = st.columns(3)
with c1:
    n_qubits = st.slider("Number of qubits (features fed to circuit)", 2, 10, 4)
with c2:
    reduction_method = st.selectbox("Dimensionality reduction", ["pca", "kbest"],
                                     help="Applied only if the dataset has more features than qubits.")
with c3:
    scaler = st.selectbox("Feature scaling", ["minmax", "standard"])

test_size = st.slider("Validation split", 0.1, 0.5, 0.25)

# --------------------------------------------------------------------------- #
# 4. QNN architecture
# --------------------------------------------------------------------------- #
st.header("4. QNN architecture")
c1, c2, c3 = st.columns(3)
with c1:
    encoding = st.selectbox("Data encoding", ENCODINGS,
                             help="angle: 1 feature/qubit via rotations. amplitude: encodes "
                                  "into state amplitudes (needs 2^n_qubits features padded/truncated). "
                                  "iqp: repeated-feature entangling encoding.")
with c2:
    ansatz = st.selectbox("Variational ansatz", ANSATZE)
with c3:
    n_layers = st.slider("Circuit depth (layers)", 1, 8, 2)

# --------------------------------------------------------------------------- #
# 5. Training configuration
# --------------------------------------------------------------------------- #
st.header("5. Training")
c1, c2, c3, c4 = st.columns(4)
with c1:
    epochs = st.slider("Epochs", 1, 100, 15)
with c2:
    batch_size = st.selectbox("Batch size", [4, 8, 16, 32, 64], index=2)
with c3:
    optimizer_name = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
with c4:
    lr = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.05, step=0.01)

st.header("6. Classical baseline")
baseline_name = st.selectbox("Baseline model", BASELINES)

run = st.button("🚀 Prepare data & train", type="primary")

if run:
    with st.spinner("Preprocessing..."):
        prepared = prepare_dataset(df, target, task, n_qubits,
                                    reduction_method=reduction_method, scaler=scaler)
    st.success(f"Prepared {prepared.X.shape[0]} samples x {prepared.X.shape[1]} features "
               f"({prepared.task}).")

    X_train, X_val, y_train, y_val = train_test_split(
        prepared.X, prepared.y, test_size=test_size, random_state=0,
        stratify=prepared.y if prepared.task == "classification" else None,
    )

    n_classes = len(prepared.class_names) if prepared.class_names else 2

    st.header("7. Training progress")
    qnn_col, chart_col = st.columns([1, 2])
    progress_bar = st.progress(0.0)
    status = st.empty()
    chart_ph = st.empty()

    model = QNNModel(n_features=prepared.X.shape[1], n_qubits=prepared.n_qubits,
                      n_layers=n_layers, task=prepared.task, n_outputs=n_classes,
                      encoding=encoding, ansatz=ansatz)

    with st.expander("Quantum circuit diagram"):
        try:
            st.text(model.qlayer.draw())
        except Exception as e:
            st.warning(f"Could not render circuit diagram: {e}")

    start = time.time()

    def progress_cb(epoch, total, history):
        progress_bar.progress(epoch / total)
        status.text(f"Epoch {epoch}/{total} | train_loss={history.train_loss[-1]:.4f} "
                     f"val_loss={history.val_loss[-1]:.4f} | "
                     f"train_metric={history.train_metric[-1]:.4f} "
                     f"val_metric={history.val_metric[-1]:.4f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history.train_loss, name="train loss"))
        fig.add_trace(go.Scatter(y=history.val_loss, name="val loss"))
        fig.update_layout(title="Loss", xaxis_title="epoch", height=300,
                           margin=dict(t=40, b=20))
        chart_ph.plotly_chart(fig, width="stretch")

    history = train_model(model, X_train, y_train, X_val, y_val, task=prepared.task,
                           n_classes=n_classes, epochs=epochs, batch_size=batch_size,
                           optimizer_name=optimizer_name, lr=lr, progress_cb=progress_cb)
    qnn_time = time.time() - start

    st.header("8. Results")
    metric_label = "Accuracy" if prepared.task == "classification" else "R²"

    with st.spinner("Training classical baseline..."):
        baseline = build_baseline(baseline_name, prepared.task)
        t0 = time.time()
        baseline.fit(X_train, y_train)
        baseline_time = time.time() - t0
        base_pred = baseline.predict(X_val)
        if prepared.task == "classification":
            base_metric = accuracy_score(y_val, base_pred)
        else:
            base_metric = r2_score(y_val, base_pred)

    c1, c2 = st.columns(2)
    with c1:
        st.metric(f"QNN val {metric_label}", f"{history.val_metric[-1]:.4f}")
        st.caption(f"Training time: {qnn_time:.1f}s | qubits={prepared.n_qubits} | "
                   f"layers={n_layers} | params={sum(p.numel() for p in model.parameters())}")
    with c2:
        st.metric(f"{baseline_name} val {metric_label}", f"{base_metric:.4f}")
        st.caption(f"Training time: {baseline_time:.1f}s")

    fig_cmp = go.Figure(data=[
        go.Bar(name="QNN", x=[metric_label], y=[history.val_metric[-1]]),
        go.Bar(name=baseline_name, x=[metric_label], y=[base_metric]),
    ])
    fig_cmp.update_layout(title="QNN vs classical baseline", height=350)
    st.plotly_chart(fig_cmp, width="stretch")

    fig_metric = go.Figure()
    fig_metric.add_trace(go.Scatter(y=history.train_metric, name=f"train {metric_label}"))
    fig_metric.add_trace(go.Scatter(y=history.val_metric, name=f"val {metric_label}"))
    fig_metric.update_layout(title=f"{metric_label} over training", xaxis_title="epoch", height=350)
    st.plotly_chart(fig_metric, width="stretch")

    st.subheader("Effect of qubits / depth / dataset size")
    st.caption(
        "Re-run with different values in section 4-5 above to explore this tradeoff "
        "directly; each run's final metrics are shown above for comparison."
    )

    csv_buf = io.StringIO()
    pd.DataFrame({
        "epoch": range(1, epochs + 1),
        "train_loss": history.train_loss, "val_loss": history.val_loss,
        "train_metric": history.train_metric, "val_metric": history.val_metric,
    }).to_csv(csv_buf, index=False)
    st.download_button("Download training history (CSV)", csv_buf.getvalue(),
                        file_name="qnn_training_history.csv")
