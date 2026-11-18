"""Quantum Image Lab: upload an image and watch every stage of a quantum image
classifier - preprocessing, quantum encoding, circuit, training, measurement,
evaluation against classical baselines, experiments, and limitations."""
from __future__ import annotations

import io
import re
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

from qnn_platform import dashboard_plots as dp
from qnn_platform.image_data import (
    BASE_SIZE, ImageDataset, array_to_pil, load_digits_subset, load_zip_dataset, preprocess_image,
)
from qnn_platform.image_experiments import (
    ablation_circuit, scaling_table, sweep_depth, sweep_encodings, sweep_image_size, sweep_noise, sweep_shots,
    sweep_train_fraction,
)
from qnn_platform.image_pipeline import (
    BASELINE_NAMES, ImageConfig, ImageFeaturizer, heldout_samples, misclassified_indices,
    predict_images, run_experiment, wilson_interval,
)
from qnn_platform.quantum_image import (
    ANSATZE, ENCODINGS, CircuitSpec, circuit_resources, draw_circuit, draw_circuit_text,
    encoding_state, frqi_state_formula, predict_with_shots, qhed_circuit_text, qhed_edges,
    qpie_state_formula, reconstruct_frqi_from_shots,
)

DEMO_DIR = Path(__file__).resolve().parent / "demo_images"
IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "webp"]


# --------------------------------------------------------------------------- #
# Cached helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cached_digits(classes: tuple[int, ...]) -> ImageDataset:
    return load_digits_subset(list(classes))


@st.cache_data(show_spinner=False)
def cached_zip(data: bytes, invert: bool, per_class: int) -> ImageDataset:
    return load_zip_dataset(data, invert, per_class)


@st.cache_data(show_spinner=False)
def cached_resources(spec: CircuitSpec) -> dict:
    return circuit_resources(spec)


@st.cache_data(show_spinner=False)
def cached_reconstruction(image: np.ndarray, shots: int) -> np.ndarray:
    return reconstruct_frqi_from_shots(image, shots)


@st.cache_data(show_spinner=False)
def cached_qhed(image: np.ndarray, shots: int | None) -> dict:
    return qhed_edges(image, shots=shots)


def fig_to_png(fig, dpi: int = 160) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def small_image(arr01: np.ndarray, px: int = 200) -> Image.Image:
    arr = np.clip(np.asarray(arr01, dtype=float), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L").convert("RGB").resize((px, px), Image.NEAREST)


def kpi_row(items: list[tuple[str, str, str]]):
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(dp.kpi(label, value, sub), unsafe_allow_html=True)


def step_header(number: str, title: str, blurb: str):
    st.markdown(f'<div class="step">{number}</div>', unsafe_allow_html=True)
    st.subheader(title)
    st.caption(blurb)


# --------------------------------------------------------------------------- #
# Sidebar: every parameter of the pipeline
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## ⚛️ Quantum Image Lab")
    st.caption("Set up the experiment here, press **Run**, then walk the tabs left to right.")

    st.markdown("### 1 · Dataset")
    source = st.radio("Source", ["Handwritten digits (built-in)", "My own images (ZIP)"],
                      label_visibility="collapsed")
    ds: ImageDataset | None = None
    if source.startswith("Handwritten"):
        classes = st.multiselect("Digit classes to separate", list(range(10)), default=[3, 8],
                                 help="3 vs 8 is a good demo: visually similar, so not trivial.")
        if len(classes) >= 2:
            ds = cached_digits(tuple(sorted(classes)))
        else:
            st.warning("Pick at least two classes.")
    else:
        zip_file = st.file_uploader("ZIP with one folder per class", type=["zip"],
                                    help="e.g. cats/1.png cats/2.png dogs/1.png ...")
        zip_invert = st.checkbox("Dark objects on light background (invert)", value=True)
        if zip_file is not None:
            try:
                ds = cached_zip(zip_file.getvalue(), zip_invert, 200)
            except ValueError as exc:
                st.error(str(exc))
    per_class = st.slider("Max images per class", 20, 200, 120, step=10,
                          help="Fewer images = faster training and noisier accuracy.")

    st.markdown("### 2 · Quantum image encoding")
    encoding = st.selectbox("Representation", list(ENCODINGS), format_func=lambda k: ENCODINGS[k])
    size = st.select_slider("Image size (pixels per side)", options=[2, 4, 8], value=4,
                            help="FRQI needs 2·log2(size)+1 qubits; gates grow with the pixel count.")
    n_pca = 4
    if encoding == "angle_pca":
        n_pca = st.slider("PCA features = qubits", 2, 8, 4)

    st.markdown("### 3 · Variational circuit")
    ansatz = st.selectbox("Ansatz", list(ANSATZE), format_func=lambda k: ANSATZE[k])
    n_layers = st.slider("Layers (circuit depth)", 1, 6, 2)

    st.markdown("### 4 · Training")
    c1, c2 = st.columns(2)
    epochs = c1.slider("Epochs", 2, 40, 12)
    batch_size = c2.selectbox("Batch", [8, 16, 32, 64], index=1)
    optimizer = c1.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
    lr = c2.number_input("Learning rate", 0.005, 0.5, 0.08, step=0.01, format="%.3f")
    test_size = c1.slider("Test split", 0.1, 0.5, 0.25, step=0.05)
    seed = c2.number_input("Random seed", 0, 999, 0, step=1)

    cfg = ImageConfig(encoding=encoding, size=size, n_layers=n_layers, ansatz=ansatz, n_pca=n_pca,
                      epochs=epochs, batch_size=batch_size, optimizer=optimizer, lr=float(lr),
                      test_size=float(test_size), seed=int(seed), per_class=per_class)
    spec = cfg.spec
    st.markdown(
        f'<span class="pill">{spec.n_qubits} qubits</span>'
        f'<span class="pill">{spec.n_circuit_params} circuit params</span>'
        f'<span class="pill">{size}×{size} image</span>',
        unsafe_allow_html=True,
    )
    if spec.n_qubits >= 7:
        st.warning("7+ qubits: simulation is ~5-10× slower. Fine for a demo, plan extra time.")
    run_clicked = st.button("🚀 Run pipeline", type="primary", width="stretch", disabled=ds is None)

ds_key = (ds.source, tuple(ds.class_names), len(ds.y)) if ds is not None else None

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
if run_clicked and ds is not None:
    st.session_state.pop("exp", None)
    bar = st.progress(0.0, text="Preparing…")

    def on_epoch(epoch, total, hist):
        bar.progress(epoch / total, text=(f"Training epoch {epoch}/{total} · loss "
                                          f"{hist.train_loss[-1]:.3f} · held-out accuracy "
                                          f"{hist.val_metric[-1]:.1%}"))

    with st.spinner("Training the quantum circuit on a simulator…"):
        st.session_state["run"] = run_experiment(ds, cfg, progress_cb=on_epoch, with_baselines=True)
    st.session_state["run_ds_key"] = ds_key
    bar.empty()

run = st.session_state.get("run")
stale = run is not None and (run.config != cfg or st.session_state.get("run_ds_key") != ds_key)

st.markdown(
    '<div class="hero"><h1>⚛️ Quantum Image Lab</h1>'
    '<p>Upload an image, see exactly how it becomes a quantum state, passes through a '
    'variational circuit, gets measured, and is classified — then judge it against classical '
    'baselines, with every limitation on the table.</p></div>',
    unsafe_allow_html=True,
)
if run is not None and stale:
    st.markdown('<div class="callout">Sidebar settings differ from the last trained model. '
                'Tabs 5–9 show the <b>last trained</b> model; press <b>Run pipeline</b> to retrain.</div>',
                unsafe_allow_html=True)

tabs = st.tabs(["🏠 Overview", "1 Input", "2 Preprocess", "3 Encode", "4 Circuit", "5 Train",
                "6 Measure", "7 Evaluate", "8 Experiments", "9 Advantage & limits", "⭐ Bonus: edges"])


def need_run() -> bool:
    if run is None:
        st.info("Press **🚀 Run pipeline** in the sidebar to train the quantum model first.")
        return True
    return False


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### The problem")
        st.markdown(
            "**Image classification on a quantum computer.** Images are large (thousands of pixels) "
            "while today's quantum processors and simulators offer only a handful of usable qubits. "
            "The question is *how to represent an image as a quantum state*, *how to process it "
            "with a trainable circuit*, and *whether that beats a classical model*."
        )
        st.markdown("#### Why a quantum approach?")
        st.markdown(
            "- **Compact representation:** an image with *N* pixels fits in **log₂N + O(1) qubits** "
            "(a state over 2ᵠ amplitudes).\n"
            "- **Native linear algebra:** circuits apply unitary transforms to all amplitudes at once, "
            "so some image operations (edge detection, transforms) are constant-depth circuits.\n"
            "- **Open research question:** whether such models can beat classical ones — this lab measures "
            "that honestly instead of assuming it."
        )
    with right:
        st.markdown("#### Where each evaluation criterion lives")
        st.dataframe(pd.DataFrame({
            "Criterion": ["Problem & objective", "Quantum formulation", "Methodology", "Experiments",
                          "Results & comparison", "Advantage & analysis", "Limitations & failures"],
            "Tab": ["Overview", "3 Encode · 4 Circuit · ⭐ Bonus", "2 Preprocess · 5 Train",
                    "8 Experiments", "7 Evaluate", "9 Advantage & limits", "6 Measure · 8 · 9"],
        }), hide_index=True, width="stretch")
    st.markdown("#### Pipeline")
    st.graphviz_chart(dp.pipeline_dot(ENCODINGS[encoding].split(" (")[0], spec.n_qubits, n_layers),
                      width="stretch")
    st.divider()
    if run is None:
        st.markdown('<div class="callout">Configure the sidebar and press <b>🚀 Run pipeline</b> '
                    '(about 10 s for the default 4×4 FRQI setup).</div>', unsafe_allow_html=True)
    else:
        best_name, best = max(run.baselines.items(), key=lambda kv: kv[1]["metrics"].accuracy)
        n_test = len(run.y_test)
        kpi_row([
            ("Qubits", str(run.config.spec.n_qubits), f"for {run.config.size**2} pixels"),
            ("Circuit parameters", str(run.model.n_parameters()["circuit"]),
             f"+ {run.model.n_parameters()['head']} in the linear head"),
            ("Quantum test accuracy", f"{run.test_metrics.accuracy:.1%}", f"{n_test} held-out images"),
            ("Best classical", f"{best['metrics'].accuracy:.1%}", best_name),
            ("Quantum training time", f"{run.train_seconds:.1f}s", "simulator, not hardware"),
        ])


# --------------------------------------------------------------------------- #
# 1 Input
# --------------------------------------------------------------------------- #
with tabs[1]:
    step_header("Step 1", "Choose the image to push through the pipeline",
                "Upload your own file, use the camera, take a guaranteed-unseen test image, or pick a demo.")
    mode = st.radio("Image source", ["Upload a file", "Take a photo", "Held-out test sample", "Demo image"],
                    horizontal=True)
    invert = st.checkbox("Dark ink on light paper → invert so ink becomes bright", value=True,
                         help="The classifier treats bright pixels as 'ink'. Untick for white-on-black images.")
    img, label, true_label = None, "", None

    if mode == "Upload a file":
        up = st.file_uploader("PNG / JPG / BMP / WEBP", type=IMAGE_TYPES)
        if up is not None:
            img = Image.open(io.BytesIO(up.getvalue()))
            img.load()
            label = up.name
            m = re.search(r"digit_(\d)", up.name)
            true_label = m.group(1) if m else None
    elif mode == "Take a photo":
        shot = st.camera_input("Photograph a handwritten digit (dark pen, white paper)")
        if shot is not None:
            img = Image.open(io.BytesIO(shot.getvalue()))
            img.load()
            label = "camera photo"
    elif mode == "Held-out test sample":
        if ds is None:
            st.warning("Select a dataset in the sidebar first.")
        else:
            hx, hy = heldout_samples(ds, run.config if run is not None else cfg)
            k = st.slider("Held-out sample #", 0, len(hy) - 1, 0)
            img = array_to_pil(hx[k], 224)
            true_label = ds.class_names[int(hy[k])]
            label = f"held-out sample {k}"
    else:
        demo_files = sorted(DEMO_DIR.glob("*.png"))
        if demo_files:
            choice = st.selectbox("Demo image", [p.name for p in demo_files],
                                  index=[p.name for p in demo_files].index("digit_3.png")
                                  if "digit_3.png" in [p.name for p in demo_files] else 0)
            img = Image.open(DEMO_DIR / choice)
            img.load()
            label = choice
            m = re.search(r"digit_(\d)", choice)
            true_label = m.group(1) if m else None
        else:
            st.warning("No demo images found. Run `python scripts/make_demo_images.py`.")

    if img is None:
        fallback = DEMO_DIR / "digit_3.png"
        if fallback.exists():
            img = Image.open(fallback)
            img.load()
            label, true_label = "digit_3.png (default demo)", "3"
            st.info("No image chosen yet — using a default demo image so the other tabs have something to show.")
    st.session_state["input"] = {"img": img, "label": label, "true_label": true_label, "invert": invert}

    if img is not None:
        c1, c2 = st.columns([1, 1])
        c1.image(img, caption=f"{label} · {img.size[0]}×{img.size[1]} px · mode {img.mode}", width=320)
        with c2:
            st.markdown(dp.card("What happens next", (
                "<ol><li><b>Preprocess</b> – grayscale, crop, resize, normalise</li>"
                "<li><b>Encode</b> – write the pixels into a quantum state</li>"
                "<li><b>Circuit</b> – trainable gates mix the qubits</li>"
                "<li><b>Measure</b> – read out ⟨Z⟩ on every qubit</li>"
                "<li><b>Classify</b> – a small linear layer picks the class</li></ol>")),
                unsafe_allow_html=True)
            if true_label is not None:
                known = ds is not None and true_label in ds.class_names
                st.markdown(f"**Known label:** `{true_label}`" +
                            ("" if known else " — ⚠️ not one of the trained classes, so the model "
                                              "cannot get this one right (a deliberate failure case)."))

inp = st.session_state["input"]


def current_steps():
    if inp["img"] is None:
        return None
    return preprocess_image(inp["img"], cfg.size, inp["invert"])


steps = current_steps()


# --------------------------------------------------------------------------- #
# 2 Preprocess
# --------------------------------------------------------------------------- #
with tabs[2]:
    step_header("Step 2", "Preprocessing: from a photo to a tiny normalised grid",
                f"Every image is reduced to {cfg.size}×{cfg.size} = {cfg.size**2} numbers in [0, 1].")
    if steps is None:
        st.info("Choose an image in tab 1.")
    else:
        stages = [
            ("Original", steps.original, f"{steps.original.shape[1]}×{steps.original.shape[0]} RGB"),
            ("Grayscale", steps.gray / 255, "1 intensity channel"),
            ("Square crop", steps.cropped / 255, f"{steps.cropped.shape[0]}×{steps.cropped.shape[1]}"),
            ("Ink bright" if steps.inverted else "Orientation kept", steps.oriented / 255,
             "inverted" if steps.inverted else "not inverted"),
            (f"Resize {cfg.size}×{cfg.size}", steps.resized / 255, "area averaging"),
            ("Normalise [0,1]", steps.normalized, "min-max stretch"),
        ]
        cols = st.columns(len(stages))
        for col, (title, arr, note) in zip(cols, stages):
            with col:
                if title == "Original":
                    col.image(arr, caption=f"{title}\n{note}", width="stretch")
                elif arr.shape[0] <= 8:
                    col.image(small_image(arr, 200), caption=f"{title}\n{note}", width="stretch")
                else:
                    col.image(np.clip(arr, 0, 1), caption=f"{title}\n{note}", width="stretch", clamp=True)
        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(dp.pixel_heatmap(steps.normalized, "What the quantum encoder receives",
                                             annotate=True), width="stretch")
        with right:
            st.markdown("**Pixel matrix** (row-major order = the pixel index used in the next step)")
            st.dataframe(pd.DataFrame(steps.normalized).style.format("{:.2f}")
                         .background_gradient(cmap="Purples", axis=None), width="stretch")
            kpi_row([("Pixels", str(cfg.size ** 2), f"{cfg.size}×{cfg.size}"),
                     ("Min / max", f"{steps.normalized.min():.2f} / {steps.normalized.max():.2f}", "after stretch"),
                     ("Mean ink", f"{steps.normalized.mean():.2f}", "fraction of bright")])
        with st.expander("Why each step exists"):
            st.markdown(
                "- **Grayscale** – FRQI stores *one* intensity per pixel, so colour is collapsed first.\n"
                "- **Square crop** – keeps the aspect ratio; the circuit needs a 2ⁿ×2ⁿ grid.\n"
                "- **Invert** – the classifier treats bright = ink, like the training digits.\n"
                "- **Resize** – qubits grow like log₂(pixels), but *gates* grow like the pixel count, "
                "so images must stay tiny on near-term machines.\n"
                "- **Normalise** – rotation angles must live in a fixed range (θ ∈ [0, π/2] for FRQI)."
            )


# --------------------------------------------------------------------------- #
# 3 Encode
# --------------------------------------------------------------------------- #
def get_featurizer() -> ImageFeaturizer | None:
    if run is not None and not stale:
        return run.featurizer
    if ds is None:
        return None
    flat = ds.subsample_per_class(per_class, cfg.seed).at_size(cfg.size).reshape(-1, cfg.size ** 2)
    return ImageFeaturizer(cfg.encoding, cfg.n_pca).fit(flat)


with tabs[3]:
    step_header("Step 3", "Quantum encoding: writing the image into a quantum state",
                ENCODINGS[encoding])
    if steps is None:
        st.info("Choose an image in tab 1.")
    else:
        image = steps.normalized
        flat = image.reshape(-1)
        n_side = int(np.log2(cfg.size))
        n_pix = flat.size
        enc_spec = CircuitSpec(encoding, cfg.size, 1, ansatz, n_pca)

        if encoding == "frqi":
            st.latex(r"|I\rangle=\frac{1}{2^{n}}\sum_{i=0}^{2^{2n}-1}"
                     r"\big(\cos\theta_i|0\rangle+\sin\theta_i|1\rangle\big)\otimes|i\rangle,"
                     r"\qquad \theta_i=\tfrac{\pi}{2}\,p_i")
            st.markdown(
                f"Image is {cfg.size}×{cfg.size} → n = {n_side}. Register = **{2 * n_side} position qubits** "
                f"(which pixel, binary row·column) + **1 colour qubit** (its intensity as a rotation angle) "
                f"= **{enc_spec.n_qubits} qubits** for **{n_pix} pixels**.")
            theta = flat * np.pi / 2
            table = pd.DataFrame({
                "pixel i": range(n_pix), "row": np.arange(n_pix) // cfg.size, "col": np.arange(n_pix) % cfg.size,
                "position |i⟩": [format(i, f"0{2 * n_side}b") for i in range(n_pix)],
                "intensity p": flat, "θ = p·π/2": theta, "cos θ": np.cos(theta), "sin θ": np.sin(theta),
                "amp ⟨i,0|I⟩": np.cos(theta) / 2 ** n_side, "amp ⟨i,1|I⟩": np.sin(theta) / 2 ** n_side,
            })
            st.dataframe(table.style.format({c: "{:.4f}" for c in table.columns[4:]}), height=260,
                         width="stretch")
            state = encoding_state(enc_spec, flat)
            reference = frqi_state_formula(image)
            err = float(np.abs(state - reference).max())
            probs = np.abs(state) ** 2
            labels = [f"{format(i >> 1, f'0{2 * n_side}b')}|{i & 1}" for i in range(len(state))]
            colours = [dp.VIOLET if i % 2 == 0 else dp.TEAL for i in range(len(state))]
            st.plotly_chart(dp.basis_state_bars(probs, labels, colours,
                                                "Measurement probabilities of the encoded state "
                                                "(violet: colour qubit = 0, teal: = 1)"), width="stretch")
            st.markdown(f'<div class="good">✔ Circuit-prepared state equals the textbook FRQI formula: '
                        f'max |Δamplitude| = <b>{err:.1e}</b> (norm = {np.linalg.norm(state):.6f})</div>',
                        unsafe_allow_html=True)
            with st.expander("🔬 Read the image back by measurement (why readout is expensive)"):
                shots = st.select_slider("Measurement shots", options=[100, 300, 1000, 3000, 10000, 100000],
                                         value=10000)
                rec = cached_reconstruction(image, shots)
                mse = float(np.mean((rec - image) ** 2))
                c1, c2 = st.columns(2)
                c1.plotly_chart(dp.pixel_heatmap(image, "Original", annotate=False, height=280), width="stretch")
                c2.plotly_chart(dp.pixel_heatmap(rec, f"Reconstructed from {shots:,} shots (MSE {mse:.4f})",
                                                 annotate=False, height=280), width="stretch")
                st.caption("Each pixel's intensity is estimated from P(colour=1 | position). Statistical error "
                           "shrinks only like 1/√shots, and every pixel needs its own samples — so reading "
                           "out a full image costs far more than loading it.")
        elif encoding == "qpie":
            st.latex(r"|I\rangle=\sum_{i=0}^{N-1}\frac{p_i}{\lVert p\rVert}\,|i\rangle,\qquad N=2^{2n}")
            st.markdown(f"The {n_pix} pixel values become the **amplitudes** of a {enc_spec.n_qubits}-qubit state "
                        f"(2^{enc_spec.n_qubits} = {n_pix}). Intensities are encoded in amplitudes, so one "
                        f"fewer qubit than FRQI — but the global brightness is lost to normalisation.")
            norm = np.linalg.norm(flat + 1e-8)
            amp = (flat + 1e-8) / norm
            table = pd.DataFrame({"pixel i": range(n_pix), "position |i⟩": [format(i, f"0{enc_spec.n_qubits}b") for i in range(n_pix)],
                                  "intensity p": flat, "amplitude p/‖p‖": amp, "probability": amp ** 2})
            st.dataframe(table.style.format({c: "{:.4f}" for c in table.columns[2:]}), height=260, width="stretch")
            state = encoding_state(enc_spec, flat)
            err = float(np.abs(state - qpie_state_formula(image)).max())
            st.plotly_chart(dp.basis_state_bars(np.abs(state) ** 2, [format(i, f"0{enc_spec.n_qubits}b") for i in range(len(state))],
                                                [dp.VIOLET] * len(state),
                                                "Measurement probabilities of the encoded state"), width="stretch")
            st.markdown(f'<div class="good">✔ Circuit-prepared state equals the amplitude-encoding formula: '
                        f'max |Δamplitude| = <b>{err:.1e}</b></div>', unsafe_allow_html=True)
        else:
            feat = get_featurizer()
            if feat is None or feat.pca is None:
                st.info("Select a dataset so PCA can be fitted.")
            else:
                st.latex(r"|x\rangle=\bigotimes_{j=1}^{k}R_Y(\phi_j)|0\rangle,\qquad "
                         r"\phi=\mathrm{minmax}_{[0,\pi]}\big(\mathrm{PCA}_k(p)\big)")
                st.markdown(f"The {n_pix} pixels are compressed by PCA to **{n_pca} features**, each rotated "
                            f"onto its own qubit: **{n_pca} qubits**. Compact and hardware-friendly, but the "
                            f"compression is lossy and learned from the training set.")
                angles = feat.transform(flat[None, :])[0]
                recon = feat.pca.inverse_transform(feat.pca.transform(flat[None, :]))[0].reshape(cfg.size, cfg.size)
                c1, c2 = st.columns(2)
                c1.plotly_chart(dp.basis_state_bars(angles, [f"qubit {j}" for j in range(len(angles))],
                                                    [dp.VIOLET] * len(angles), "Rotation angle φ per qubit (rad)"),
                                width="stretch")
                c2.plotly_chart(dp.pixel_heatmap(np.clip(recon, 0, 1), f"What survives compression to {n_pca} "
                                                 f"features (MSE {np.mean((recon - image) ** 2):.4f})",
                                                 annotate=False, height=320), width="stretch")
                st.caption("Explained variance of the kept components: " +
                           ", ".join(f"{v:.1%}" for v in feat.pca.explained_variance_ratio_))
        st.markdown(dp.card("Qubits vs pixels", (
            f"<p>{n_pix} pixels → <b>{enc_spec.n_qubits} qubits</b>. The state has 2^{enc_spec.n_qubits} = "
            f"{2 ** enc_spec.n_qubits} amplitudes. This is the sense in which quantum representations are "
            f"compact. The catch (tab 9): preparing the state costs gates roughly proportional to the pixel count.</p>")),
            unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# 4 Circuit
# --------------------------------------------------------------------------- #
with tabs[4]:
    step_header("Step 4", "The quantum circuit", "Encode → trainable layers → measure.")
    st.graphviz_chart(dp.circuit_blocks_dot(ENCODINGS[encoding].split(" (")[0], spec.n_qubits, n_layers,
                                            len(ds.class_names) if ds is not None else 2), width="stretch")
    trained_here = run is not None and run.config.spec == spec
    weights = run.model.weights.detach().numpy() if trained_here else np.zeros(spec.weight_shape)
    x_for_draw = None
    if steps is not None:
        feat = get_featurizer()
        if feat is not None:
            x_for_draw = feat.transform(steps.normalized.reshape(1, -1))[0]
    expanded = False
    if encoding == "frqi" and size <= 4:
        expanded = st.toggle("Expand the FRQI block into individual gates", value=False)
    try:
        png = fig_to_png(draw_circuit(spec, x=x_for_draw, weights=weights, expanded=expanded))
        st.image(png, caption=("Trained parameter values shown" if trained_here
                               else "Untrained (all θ = 0) — train to see learned values"), width="stretch")
    except Exception as exc:
        st.warning(f"Could not draw the circuit figure: {exc}")
        st.code(draw_circuit_text(spec, x=x_for_draw, weights=weights, expanded=expanded))
    with st.expander("Text version"):
        st.code(draw_circuit_text(spec, x=x_for_draw, weights=weights, expanded=expanded))

    res = cached_resources(spec)
    kpi_row([("Qubits", str(spec.n_qubits), "register width"),
             ("Trainable circuit params", str(spec.n_circuit_params), f"weights shape {spec.weight_shape}"),
             ("Gates as written", str(res.get("logical_gates", "?")), f"depth {res.get('logical_depth', '?')}"),
             ("Gates on hardware basis", f"{res.get('native_gates', '?'):,}" if isinstance(res.get("native_gates"), int) else "?",
              f"{res.get('native_cnots', '?')} CNOTs · depth {res.get('native_depth', '?')}")])
    if "native_gate_types" in res:
        types = res["native_gate_types"]
        fig = go.Figure(go.Bar(x=list(types), y=list(types.values()), marker_color=dp.VIOLET))
        st.plotly_chart(dp.style_fig(fig, 280, "Gate mix after compiling to {CNOT, RX, RY, RZ, H, X}"),
                        width="stretch")
    if trained_here:
        w = weights.reshape(spec.n_layers, -1)
        st.plotly_chart(dp.style_fig(go.Figure(go.Heatmap(
            z=w, colorscale="RdBu", zmid=0, colorbar=dict(title="rad"),
            y=[f"layer {i + 1}" for i in range(w.shape[0])])), 260, "Learned rotation angles (rows = layers)"),
            width="stretch")
    with st.expander("How to read this circuit (explain it in 30 seconds)"):
        st.markdown(
            f"- Each horizontal line is a **qubit**; time runs left → right. There are **{spec.n_qubits}**.\n"
            f"- The first box is the **encoder** ({ENCODINGS[encoding].split(' (')[0]}). It has no trainable "
            f"parameters — it just loads the image.\n"
            f"- Then **{spec.n_layers} layers**: single-qubit rotations (the learnable angles θ) followed by "
            f"**CNOT** gates (⊕ on the target, ● on the control) that entangle neighbouring qubits so "
            f"information about distant pixels can interact.\n"
            f"- At the end every qubit is **measured** in the Z basis, giving ⟨Z⟩ ∈ [−1, 1]. A small classical "
            f"linear layer turns those {spec.n_qubits} numbers into class scores.\n"
            f"- Training adjusts the {spec.n_circuit_params} angles (and the head) to minimise the loss."
        )


# --------------------------------------------------------------------------- #
# 5 Train
# --------------------------------------------------------------------------- #
with tabs[5]:
    step_header("Step 5", "Training", "Mini-batch gradient descent through the simulated circuit.")
    if not need_run():
        loss_fig, acc_fig = dp.history_figures(run.history)
        c1, c2 = st.columns(2)
        c1.plotly_chart(loss_fig, width="stretch")
        c2.plotly_chart(acc_fig, width="stretch")
        p = run.model.n_parameters()
        rc = run.config
        kpi_row([("Final train accuracy", f"{run.train_metrics.accuracy:.1%}", f"{len(run.y_train)} images"),
                 ("Final test accuracy", f"{run.test_metrics.accuracy:.1%}", f"{len(run.y_test)} images"),
                 ("Parameters", str(p["total"]), f"{p['circuit']} circuit + {p['head']} head"),
                 ("Training time", f"{run.train_seconds:.1f}s", f"{rc.epochs} epochs · batch {rc.batch_size}")])
        st.caption(f"Optimiser **{rc.optimizer}**, learning rate **{rc.lr}**, seed **{rc.seed}**. Gradients flow "
                   f"through the exact state-vector simulation (backpropagation); on real hardware this would "
                   f"use the parameter-shift rule, costing 2 circuit runs per parameter per sample.")
        gap = run.train_metrics.accuracy - run.test_metrics.accuracy
        if gap > 0.08:
            st.markdown('<div class="callout">Train accuracy is noticeably above test accuracy — a sign of '
                        'overfitting on this small dataset.</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# 6 Measure
# --------------------------------------------------------------------------- #
with tabs[6]:
    step_header("Step 6", "Measurement and prediction for your image",
                "The circuit's output is read by measuring each qubit; a linear layer picks the class.")
    if not need_run() and steps is not None:
        rcfg = run.config
        x_img = preprocess_image(inp["img"], rcfg.size, inp["invert"]).normalized
        feats_in = run.featurizer.transform(x_img.reshape(1, -1))
        with torch.no_grad():
            q_feats = run.model.quantum_features(torch.tensor(feats_in))
            logits = run.model.head(q_feats)
        from qnn_platform.quantum_image import logits_to_probabilities
        proba = logits_to_probabilities(logits, run.n_classes)[0]
        pred = int(np.argmax(proba))
        c1, c2, c3 = st.columns([1, 1.4, 1.2])
        c1.image(small_image(x_img, 220), caption=f"Preprocessed {rcfg.size}×{rcfg.size} input", width=220)
        c2.plotly_chart(dp.basis_state_bars(q_feats[0].numpy(), [f"⟨Z{j}⟩" for j in range(q_feats.shape[1])],
                                            [dp.VIOLET] * q_feats.shape[1], "Measured expectation per qubit",
                                            height=290), width="stretch")
        fig = go.Figure(go.Bar(x=[f"class {c}" for c in run.class_names], y=proba,
                               marker_color=[dp.TEAL if i == pred else dp.VIOLET for i in range(len(proba))]))
        fig.update_yaxes(range=[0, 1])
        c3.plotly_chart(dp.style_fig(fig, 290, "Class probabilities"), width="stretch")
        verdict = f"Predicted class <b>{run.class_names[pred]}</b> with confidence <b>{proba[pred]:.1%}</b>"
        if inp["true_label"] is not None:
            ok = run.class_names[pred] == inp["true_label"]
            verdict += f" · known label <code>{inp['true_label']}</code> → " + ("✅ correct" if ok else "❌ wrong")
        st.markdown(f'<div class="good">{verdict}</div>', unsafe_allow_html=True)

        st.markdown("#### Finite measurements (what hardware really gives you)")
        st.caption("A simulator can compute ⟨Z⟩ exactly. A real device estimates it from a limited number of "
                   "shots, so predictions become random near the decision boundary.")
        shots = st.select_slider("Shots per circuit", options=[10, 30, 100, 300, 1000, 5000], value=100)
        reps = 40
        agree = 0
        sampled_runs = []
        for r in range(reps):
            sp, sf, _ = predict_with_shots(run.model, feats_in, shots, seed=1000 + r)
            sampled_runs.append(sf[0])
            agree += int(sp[0] == pred)
        sampled = np.array(sampled_runs)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"⟨Z{j}⟩" for j in range(sampled.shape[1])], y=q_feats[0].numpy(),
                             name="exact", marker_color=dp.VIOLET))
        fig.add_trace(go.Bar(x=[f"⟨Z{j}⟩" for j in range(sampled.shape[1])], y=sampled.mean(axis=0),
                             error_y=dict(type="data", array=sampled.std(axis=0), color=dp.MUTED),
                             name=f"{shots} shots (mean ± std over {reps} runs)", marker_color=dp.AMBER))
        fig.update_layout(barmode="group")
        c1, c2 = st.columns([2, 1])
        c1.plotly_chart(dp.style_fig(fig, 300, "Exact vs sampled expectation values"), width="stretch")
        c2.markdown(dp.kpi("Same prediction as exact", f"{agree / reps:.0%}", f"over {reps} repeated runs"),
                    unsafe_allow_html=True)
    elif steps is None:
        st.info("Choose an image in tab 1.")


# --------------------------------------------------------------------------- #
# 7 Evaluate
# --------------------------------------------------------------------------- #
with tabs[7]:
    step_header("Step 7", "Evaluation against classical baselines",
                "Same images, same split, same pixels — only the model differs.")
    if not need_run():
        n_test = len(run.y_test)
        rows = [{"Model": "⚛️ Quantum (this lab)", **run.test_metrics.as_dict(),
                 "Parameters": run.model.n_parameters()["total"], "Train time (s)": run.train_seconds}]
        for name in BASELINE_NAMES:
            b = run.baselines[name]
            rows.append({"Model": name, **b["metrics"].as_dict(), "Parameters": b["n_params"],
                         "Train time (s)": b["seconds"]})
        table = pd.DataFrame(rows)
        st.dataframe(table.style.format({"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
                                         "F1": "{:.3f}", "Parameters": "{:.0f}", "Train time (s)": "{:.2f}"},
                                        na_rep="n/a"),
                     hide_index=True, width="stretch")
        correct = int(round(run.test_metrics.accuracy * n_test))
        lo, hi = wilson_interval(correct, n_test)
        st.markdown(f'<div class="callout">Test set has only <b>{n_test}</b> images, so the quantum accuracy of '
                    f'<b>{run.test_metrics.accuracy:.1%}</b> has a 95% confidence interval of '
                    f'<b>{lo:.1%} – {hi:.1%}</b>. Differences of a few points between models are within noise — '
                    f'use the multi-seed experiments (tab 8) before claiming one is better.</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        for metric, colour in (("Accuracy", dp.VIOLET), ("F1", dp.TEAL)):
            fig.add_trace(go.Bar(x=table["Model"], y=table[metric], name=metric, marker_color=colour))
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[0, 1.02])
        st.plotly_chart(dp.style_fig(fig, 340, "Held-out accuracy and F1"), width="stretch")

        c1, c2 = st.columns(2)
        c1.plotly_chart(dp.confusion_heatmap(run.test_metrics.confusion, run.class_names, "Quantum model"),
                        width="stretch")
        other = c2.selectbox("Compare confusion matrix with", BASELINE_NAMES, index=2)
        c2.plotly_chart(dp.confusion_heatmap(run.baselines[other]["metrics"].confusion, run.class_names, other),
                        width="stretch")

        st.markdown("#### Failure cases: images the quantum model got wrong")
        wrong = misclassified_indices(run, limit=8)
        if not wrong:
            st.markdown('<div class="good">No misclassified test images for this run.</div>', unsafe_allow_html=True)
        else:
            cols = st.columns(len(wrong))
            for col, i in zip(cols, wrong):
                col.image(small_image(run.X_test_img[i], 140),
                          caption=f"true {run.class_names[run.y_test[i]]} → pred {run.class_names[run.test_pred[i]]}",
                          width="stretch")
            st.caption("Typically ambiguous strokes at this very low resolution — the 4×4 grid has lost the detail "
                       "that separates the classes.")


# --------------------------------------------------------------------------- #
# 8 Experiments
# --------------------------------------------------------------------------- #
def progress_runner(label: str, fn):
    bar = st.progress(0.0, text=label)

    def cb(i, total):
        bar.progress(i / total, text=f"{label} ({i}/{total})")

    out = fn(cb)
    bar.empty()
    return out


with tabs[8]:
    step_header("Step 8", "Experiments: turning one run into evidence",
                "Each sweep changes one factor and holds the rest at the sidebar values.")
    exp = st.session_state.setdefault("exp", {})
    c1, c2, c3 = st.columns(3)
    exp_epochs = c1.slider("Epochs per run", 3, 25, 8, key="exp_epochs")
    rigorous = c2.checkbox("Rigorous: 3 seeds, mean ± std", value=False,
                           help="Triples the time but separates real effects from luck.")
    c3.caption("Tip: run these *before* the presentation and keep the tab open.")
    n_seeds = 3 if rigorous else 1
    base = replace(cfg, epochs=exp_epochs)

    def show(key: str, build_fig, note=None):
        if key in exp:
            fig, df = build_fig(exp[key])
            st.plotly_chart(fig, width="stretch")
            st.dataframe(df, hide_index=True, width="stretch")
            if note:
                st.caption(note(exp[key]))

    # ---- encodings
    with st.expander("A · Which quantum image representation is best?", expanded=True):
        if ds is None:
            st.info("Select a dataset first.")
        elif st.button("Run encoding comparison", key="run_enc"):
            exp["enc"] = progress_runner("Encodings", lambda cb: sweep_encodings(ds, base, n_seeds, cb))

        def enc_fig(df):
            fig = go.Figure(go.Bar(x=df["encoding"].str.upper(), y=df["Test accuracy"], marker_color=dp.VIOLET,
                                   error_y=dict(type="data", array=df["Accuracy std"], color=dp.MUTED)))
            fig.update_yaxes(range=[0, 1.02], title="test accuracy")
            return dp.style_fig(fig, 320, "Test accuracy by encoding"), df.round(3)

        show("enc", enc_fig, lambda df: "FRQI needs the most gates by far (see 'Native gates'); QPIE and "
                                         "angle+PCA reach similar accuracy with a fraction of the circuit.")

    # ---- depth
    with st.expander("B · Does a deeper circuit help?"):
        if ds is not None and st.button("Run depth sweep", key="run_depth"):
            exp["depth"] = progress_runner("Depth", lambda cb: sweep_depth(ds, base, (1, 2, 3, 4, 6), n_seeds, cb))
        show("depth", lambda df: (dp.sweep_line(df, "n_layers", "Test accuracy", "Accuracy vs circuit depth",
                                                err="Accuracy std", xlabel="layers"), df.round(3)),
             lambda df: "More layers add parameters and gates; beyond a point accuracy plateaus while "
                        "training time keeps growing (and gradients can vanish — barren plateaus).")

    # ---- size
    with st.expander("C · Image size, qubits and accuracy"):
        sizes = st.multiselect("Image sizes", [2, 4, 8], default=[2, 4],
                               help="8×8 FRQI uses 7 qubits and is ~10× slower.")
        if ds is not None and sizes and st.button("Run image-size sweep", key="run_size"):
            exp["size"] = progress_runner("Image size", lambda cb: sweep_image_size(ds, base, tuple(sorted(sizes)),
                                                                                    n_seeds, cb))
        show("size", lambda df: (dp.sweep_line(df, "size", "Test accuracy", "Accuracy vs image size",
                                               err="Accuracy std", xlabel="pixels per side"), df.round(3)),
             lambda df: "Bigger images keep more detail (higher accuracy) but need exponentially more "
                        "simulation effort and, for FRQI, proportionally more gates.")

    # ---- data size
    with st.expander("D · How much training data is needed?"):
        if ds is not None and st.button("Run dataset-size sweep", key="run_frac"):
            exp["frac"] = progress_runner("Training size", lambda cb: sweep_train_fraction(
                ds, base, (0.1, 0.25, 0.5, 1.0), n_seeds, cb))
        show("frac", lambda df: (dp.sweep_line(df, "train_fraction", "Test accuracy", "Accuracy vs training-set size",
                                               err="Accuracy std", xlabel="fraction of training data"), df.round(3)))

    # ---- shots
    with st.expander("E · Finite measurement shots (needs a trained model)"):
        if run is None:
            st.info("Run the pipeline first.")
        elif st.button("Run shots sweep", key="run_shots"):
            exp["shots"] = progress_runner("Shots", lambda cb: sweep_shots(run, (10, 30, 100, 300, 1000, 5000), 5, cb))

        def shots_fig(df):
            fig = go.Figure(go.Scatter(x=df["Shots"].astype(str), y=df["Test accuracy"], mode="lines+markers",
                                       line=dict(color=dp.VIOLET, width=3), marker=dict(size=10, color=dp.TEAL),
                                       error_y=dict(type="data", array=df["Accuracy std"], color=dp.MUTED)))
            fig.update_yaxes(range=[0, 1.02], title="test accuracy")
            fig.update_xaxes(title="shots per expectation value", type="category")
            return dp.style_fig(fig, 320, "Accuracy vs number of measurement shots"), df.round(3)

        show("shots", shots_fig, lambda df: "With few shots the estimated ⟨Z⟩ are noisy and accuracy falls and "
                                             "fluctuates; thousands of shots per circuit are needed for stable results.")

    # ---- noise
    with st.expander("F · Hardware noise (needs a trained model)"):
        n_noise = st.slider("Test images used (density-matrix simulation is slow)", 10, 80, 30, key="noise_n")
        if run is None:
            st.info("Run the pipeline first.")
        elif st.button("Run noise sweep", key="run_noise"):
            exp["noise"] = progress_runner("Noise", lambda cb: sweep_noise(
                run, (0.0, 0.01, 0.05, 0.15, 0.3), n_noise, cb))
        show("noise", lambda df: (dp.sweep_line(df, "Depolarising probability p", "Test accuracy",
                                                "Accuracy under depolarising noise", xlabel="error probability p per qubit per layer"),
                                  df.round(3)),
             lambda df: "Depolarising noise after the encoder and every layer shrinks the measured values toward "
                        "zero; the trained model tolerates mild noise but degrades as it grows.")

    # ---- ablation
    with st.expander("H · Ablation: does training the quantum circuit matter?"):
        if ds is not None and st.button("Run ablation", key="run_ablation"):
            exp["ablation"] = progress_runner("Ablation", lambda cb: ablation_circuit(ds, base, n_seeds, cb))

        def ablation_fig(df):
            fig = go.Figure(go.Bar(x=df["Variant"], y=df["Test accuracy"], marker_color=[dp.VIOLET, dp.AMBER],
                                   error_y=dict(type="data", array=df["Accuracy std"], color=dp.MUTED)))
            fig.update_yaxes(range=[0, 1.02], title="test accuracy")
            return dp.style_fig(fig, 320, "Trained vs random frozen circuit"), df.round(3)

        show("ablation", ablation_fig, lambda df: "If the frozen random circuit scores close to the trained one, the "
                                                   "circuit is acting as a fixed feature map and the classical head is "
                                                   "doing most of the learning — an honest check on how much the "
                                                   "variational training contributes.")

    # ---- scaling
    with st.expander("G · Resource scaling: qubits, gates and time vs image size"):
        if st.button("Compute scaling table", key="run_scaling"):
            with st.spinner("Compiling circuits…"):
                exp["scaling"] = scaling_table((2, 4, 8), n_layers, ansatz)

        def scaling_fig(df):
            fig = go.Figure()
            for enc, colour in (("FRQI", dp.VIOLET), ("QPIE", dp.TEAL)):
                sub = df[df["Encoding"] == enc]
                fig.add_trace(go.Scatter(x=sub["Pixels"], y=sub["Native gates"], mode="lines+markers", name=enc,
                                         line=dict(color=colour, width=3)))
            fig.update_xaxes(type="log", title="pixels")
            fig.update_yaxes(type="log", title="compiled gates")
            return dp.style_fig(fig, 320, "Compiled gate count vs image size (log–log)"), df.round(4)

        show("scaling", scaling_fig, lambda df: "Qubit count grows logarithmically with the pixel count, but the "
                                                 "gates needed to *load* the image grow roughly with the pixel count.")


# --------------------------------------------------------------------------- #
# 9 Advantage & limits
# --------------------------------------------------------------------------- #
with tabs[9]:
    step_header("Step 9", "Where quantum helps, where it doesn't, and what can go wrong",
                "A frank assessment backed by numbers from this lab.")
    if run is not None:
        best_name, best = max(run.baselines.items(), key=lambda kv: kv[1]["metrics"].accuracy)
        n_test = len(run.y_test)
        lo, hi = wilson_interval(int(round(run.test_metrics.accuracy * n_test)), n_test)
        diff = run.test_metrics.accuracy - best["metrics"].accuracy
        verdict = ("is within noise of" if abs(diff) < (hi - lo) / 2 else ("beats" if diff > 0 else "trails"))
        st.markdown(f'<div class="callout"><b>Headline result for the last run:</b> the quantum classifier '
                    f'({run.test_metrics.accuracy:.1%}, CI {lo:.1%}–{hi:.1%}) {verdict} the best classical '
                    f'baseline ({best_name}, {best["metrics"].accuracy:.1%}) on {n_test} held-out images. '
                    f'<b>No quantum advantage is claimed or observed.</b></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(dp.card("Where the quantum approach has merit", (
            "<ul><li><b>Compact state space.</b> N pixels live in ~log₂N qubits; the model manipulates 2ⁿ "
            "amplitudes with n qubits.</li>"
            "<li><b>Constant-depth image operations.</b> Edge detection (⭐ tab) is one Hadamard layer on an "
            "amplitude-encoded image, versus O(N) classical operations.</li>"
            "<li><b>Few trainable parameters.</b> A handful of angles reach accuracy similar to models with "
            "many more weights.</li>"
            "<li><b>Research value.</b> A controllable testbed for encodings, expressivity and noise.</li></ul>")),
            unsafe_allow_html=True)
    with c2:
        st.markdown(dp.card("Where it does not (today)", (
            "<ul><li><b>No accuracy advantage.</b> Classical logistic regression / SVM / MLP match or beat it "
            "on this task, in milliseconds.</li>"
            "<li><b>State preparation dominates.</b> Loading an image costs ~O(N) gates; FRQI 8×8 compiles to "
            "thousands of gates — the 'exponential speed-up' disappears once you pay to load the data.</li>"
            "<li><b>Readout is expensive.</b> Estimating outputs needs many shots (tab 6, 8-E).</li>"
            "<li><b>Simulation cost is exponential</b> in qubits; this lab is a classical simulation.</li></ul>")),
            unsafe_allow_html=True)
    st.markdown("#### Limitations and failure cases")
    st.markdown(
        "- **Tiny images.** 4×4 / 8×8 digits are far below real image-processing scale; accuracy here does not extrapolate.\n"
        "- **Easy, small dataset.** scikit-learn digits has ~180 images per class; test accuracy has a wide confidence interval.\n"
        "- **Simulator, not hardware.** Real devices add gate errors, decoherence and connectivity limits; "
        "we model only depolarising noise (tab 8-F) and finite shots (8-E).\n"
        "- **Domain shift.** Photos taken by a camera differ from the dataset's scanning style; uploaded images "
        "may be misclassified for reasons unrelated to quantum (lighting, thickness, centring).\n"
        "- **Out-of-class inputs.** The model only knows its trained classes; any other digit is forced into one of them.\n"
        "- **Grayscale only.** FRQI/QPIE here store one intensity channel; colour needs extra qubits (e.g. NEQR/MCQI).\n"
        "- **Trainability.** Deeper / wider circuits can hit barren plateaus; our depth sweep shows diminishing returns.\n"
        "- **Resizing loses information**, which is a classical bottleneck shared by every model compared here.\n"
        "- **Fairness of comparison.** Baselines use default hyperparameters; neither side is heavily tuned."
    )
    st.markdown("#### Sources of error")
    st.markdown(
        "- *Statistical:* small test set, random split and seed (use multi-seed in tab 8).\n"
        "- *Optimisation:* local minima, learning-rate sensitivity.\n"
        "- *Modelling:* lossy resize/PCA; angle normalisation per image removes absolute brightness.\n"
        "- *Hardware (not simulated here):* readout error, crosstalk, calibration drift."
    )
    if "exp" in st.session_state and "scaling" in st.session_state["exp"]:
        st.markdown("#### Resource scaling (from tab 8-G)")
        st.dataframe(st.session_state["exp"]["scaling"].round(4), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# Bonus: Quantum Hadamard edge detection
# --------------------------------------------------------------------------- #
with tabs[10]:
    step_header("Bonus", "Quantum edge detection (QHED) on your image",
                "A second image-processing task solved by a tiny, constant-depth circuit — no training involved.")
    if inp["img"] is None:
        st.info("Choose an image in tab 1.")
    else:
        c1, c2 = st.columns([1, 1])
        q_size = c1.select_slider("Edge-detection image size", options=[8, 16, 32], value=16,
                                  help="8→6 qubits, 16→8 qubits, 32→10 qubits")
        shots_choice = c2.select_slider("Measurement shots", options=["exact", 1000, 10000, 100000, 1000000],
                                        value="exact")
        shots_val = None if shots_choice == "exact" else int(shots_choice)
        edge_steps = preprocess_image(inp["img"], q_size, inp["invert"])
        out = cached_qhed(edge_steps.normalized, shots_val)
        norm_ref = max(out["classical_combined"].max(), 1e-9)
        cols = st.columns(4)
        cols[0].plotly_chart(dp.pixel_heatmap(edge_steps.normalized, "Input", annotate=False, height=260), width="stretch")
        cols[1].plotly_chart(dp.pixel_heatmap(out["combined"] / norm_ref, "Quantum edges", annotate=False, height=260,
                                              zmax=1.0), width="stretch")
        cols[2].plotly_chart(dp.pixel_heatmap(out["classical_combined"] / norm_ref, "Classical differences",
                                              annotate=False, height=260, zmax=1.0), width="stretch")
        cols[3].plotly_chart(dp.pixel_heatmap(np.abs(out["combined"] - out["classical_combined"]) / norm_ref,
                                              "|Quantum − classical|", annotate=False, height=260, zmax=1.0),
                             width="stretch")
        kpi_row([("Qubits", str(out["n_qubits"]), f"for {q_size ** 2} pixels"),
                 ("Circuits run", str(out["n_circuits"]), "2 directions × (plain + shifted)"),
                 ("Max error vs classical", f"{out['max_error']:.1e}",
                  "exact simulation" if shots_val is None else f"{shots_val:,} shots"),
                 ("Classical ops", str(2 * q_size * (q_size - 1)), "pixel differences")])
        st.latex(r"|I\rangle=\sum_i a_i|i\rangle\ \xrightarrow{\ H\ \text{on the last qubit}\ }\ "
                 r"\sum_{k}\Big[\tfrac{a_{2k}+a_{2k+1}}{\sqrt2}|2k\rangle+\tfrac{a_{2k}-a_{2k+1}}{\sqrt2}|2k{+}1\rangle\Big]")
        st.markdown(
            "Amplitude-encode the image. One Hadamard on the least-significant qubit turns every adjacent pixel pair "
            "(aₖ, aₖ₊₁) into their **sum** (even index) and **difference** (odd index). The probability of the odd basis "
            "states is the squared difference — an edge map. A second run with the pixel index shifted by one covers the "
            "remaining pairs; doing it along rows and along columns gives horizontal and vertical edges.")
        with st.expander("Circuit (one of the four runs)"):
            st.code(qhed_circuit_text(min(q_size, 8)), language="text")
        st.markdown('<div class="callout"><b>Honest caveat:</b> the circuit itself is constant depth, but loading the '
                    'image (amplitude encoding) costs O(N) gates and reading the edge map needs many shots, so there '
                    'is no practical speed-up over a classical O(N) pass — the value here is demonstrating a genuine '
                    'quantum image algorithm and checking it against the classical answer.</div>',
                    unsafe_allow_html=True)
