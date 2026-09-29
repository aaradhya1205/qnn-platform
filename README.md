# Interactive QNN Platform

A Streamlit app for applying Quantum Neural Networks to classification and
regression problems without writing quantum circuit code.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## What it does

1. **Upload** a CSV, or use one of the built-in demos: Iris (3-class),
   Wine (3-class), Breast cancer (binary), Diabetes (regression).
2. **Pick a target column** — task type (classification/regression) is
   auto-detected, and overridable.
3. **Preprocess**: numeric/categorical columns are auto-identified, imputed,
   one-hot encoded, and scaled; if the dataset has more features than the
   chosen qubit count, PCA or SelectKBest reduces it down to fit. A
   **Medical sector profile** runs domain-aware cleaning first: patient
   identifier/PII columns (patient ID, name, MRN, DOB, contact info) are
   dropped, missingness on any feature is kept as its own `*_was_missing`
   flag before imputation (clinically, a missing lab value is often
   informative rather than random), and numeric outliers are clipped to a
   wide IQR band to catch data-entry errors without touching real clinical
   extremes.
4. **Configure the QNN**: number of qubits, encoding (angle / amplitude /
   IQP), variational ansatz (basic entangler / strongly entangling /
   hardware-efficient), and circuit depth.
5. **Train** with configurable epochs, mini-batch size, optimizer
   (Adam/SGD/RMSprop), and learning rate, with live loss/metric charts.
6. **Compare** against a classical baseline (logistic/ridge, MLP, or random
   forest) trained on the same data split.
7. **Inspect** the rendered circuit diagram (visual + text) and download the
   training history as CSV.
8. **Track run history**: every configuration trained in the session is kept
   in a comparison table, so different settings can be judged side by side.
9. **Run automated sweeps**: vary number of qubits, circuit depth, or
   training-set size while holding everything else fixed, and see accuracy
   (or R²) and training time plotted against each factor — directly answers
   "what does more qubits / depth / data actually buy me?" without manually
   re-running the app for every combination.

## Structure

- `qnn_platform/preprocessing.py` — dataset loading, feature typing, scaling,
  dimensionality reduction.
- `qnn_platform/medical.py` — medical-sector preprocessing: identifier/PII
  dropping, missingness indicators, physiological outlier clipping.
- `qnn_platform/quantum_model.py` — PennyLane encodings/ansätze wrapped as a
  differentiable PyTorch `QNNModel`, plus text/matplotlib circuit diagrams.
- `qnn_platform/classical_model.py` — scikit-learn baselines.
- `qnn_platform/train.py` — mini-batch training loop shared by all tasks.
- `qnn_platform/experiments.py` — qubit/depth/dataset-size sweep utilities.
- `app.py` — the Streamlit UI.
- `tests/` — pytest suite covering preprocessing, the QNN model, training,
  and classical baselines.

## Notes

- Requires `pandas<3.0` — pandas 3.0's new default string dtype currently
  crashes on `numpy>=2.4` when checking column types via `np.issubdtype`.
- `default.qubit` (PennyLane's simulator) is used; qubit counts above ~12-14
  will get slow since it's a full state-vector simulation.
- Sweeps train with a reduced epoch count for speed — treat the resulting
  trend as indicative, not a fully-converged comparison.
