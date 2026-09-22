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

## What it does

1. **Upload** a CSV (or use the built-in Iris / Diabetes demos).
2. **Pick a target column** — task type (classification/regression) is
   auto-detected, and overridable.
3. **Preprocess**: numeric/categorical columns are auto-identified, imputed,
   one-hot encoded, and scaled; if the dataset has more features than the
   chosen qubit count, PCA or SelectKBest reduces it down to fit.
4. **Configure the QNN**: number of qubits, encoding (angle / amplitude /
   IQP), variational ansatz (basic entangler / strongly entangling /
   hardware-efficient), and circuit depth.
5. **Train** with configurable epochs, mini-batch size, optimizer
   (Adam/SGD/RMSprop), and learning rate, with live loss/metric charts.
6. **Compare** against a classical baseline (logistic/ridge, MLP, or random
   forest) trained on the same data split.
7. **Inspect** the rendered circuit diagram and download the training
   history as CSV.

## Structure

- `qnn_platform/preprocessing.py` — dataset loading, feature typing, scaling,
  dimensionality reduction.
- `qnn_platform/quantum_model.py` — PennyLane encodings/ansätze wrapped as a
  differentiable PyTorch `QNNModel`.
- `qnn_platform/classical_model.py` — scikit-learn baselines.
- `qnn_platform/train.py` — mini-batch training loop shared by all tasks.
- `app.py` — the Streamlit UI.

## Notes

- Requires `pandas<3.0` — pandas 3.0's new default string dtype currently
  crashes on `numpy>=2.4` when checking column types via `np.issubdtype`.
- `default.qubit` (PennyLane's simulator) is used; qubit counts above ~12-14
  will get slow since it's a full state-vector simulation.
