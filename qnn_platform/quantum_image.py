"""Quantum image representations, variational classifiers, and quantum edge detection.

Encodings
---------
FRQI  (Flexible Representation of Quantum Images, Le et al. 2011)
    A 2^n x 2^n image uses 2n position qubits + 1 colour qubit:
        |I> = 1/2^n * sum_i ( cos(theta_i)|0> + sin(theta_i)|1> ) (x) |i>,
        theta_i = pixel_i * pi/2 in [0, pi/2].
    Prepared with Hadamards on the position register followed by one
    multi-controlled RY(2 theta_i) per pixel.

QPIE  (Quantum Probability Image Encoding)
    The normalised pixel vector becomes the amplitude vector of a 2n-qubit
    state: |I> = sum_i p_i |i> / ||p||.

Angle + PCA
    The image is compressed to k PCA features, each written to one qubit as a
    Y rotation. The compact, near-term-friendly baseline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pennylane as qml
import torch
from torch import nn

ENCODINGS = {
    "frqi": "FRQI (2n+1 qubits, flexible representation)",
    "qpie": "QPIE amplitude (2n qubits)",
    "angle_pca": "Angle encoding of PCA features (k qubits)",
}
ANSATZE = {
    "hardware_efficient": "Hardware-efficient (RY, RZ + CNOT chain)",
    "strongly_entangling": "Strongly entangling (Rot + CNOT ring)",
    "basic_entangler": "Basic entangler (RX + CNOT ring)",
}
NATIVE_GATES = {"CNOT", "RY", "RZ", "RX", "Hadamard", "PauliX"}


def log2_int(n: int) -> int:
    k = int(round(np.log2(n)))
    if 2 ** k != n:
        raise ValueError(f"{n} is not a power of two")
    return k


@dataclass(frozen=True)
class CircuitSpec:
    encoding: str = "frqi"
    size: int = 4
    n_layers: int = 2
    ansatz: str = "hardware_efficient"
    n_pca: int = 4

    @property
    def n_qubits(self) -> int:
        n = log2_int(self.size)
        if self.encoding == "frqi":
            return 2 * n + 1
        if self.encoding == "qpie":
            return 2 * n
        return self.n_pca

    @property
    def n_features(self) -> int:
        return self.n_pca if self.encoding == "angle_pca" else self.size * self.size

    @property
    def weight_shape(self) -> tuple[int, ...]:
        n, L = self.n_qubits, self.n_layers
        return {"hardware_efficient": (L, n, 2),
                "strongly_entangling": (L, n, 3),
                "basic_entangler": (L, n)}[self.ansatz]

    @property
    def n_circuit_params(self) -> int:
        return int(np.prod(self.weight_shape))


# --------------------------------------------------------------------------- #
# Encodings
# --------------------------------------------------------------------------- #
class FRQIEmbedding(qml.operation.Operation):
    """FRQI state preparation as a single drawable block that decomposes into
    Hadamards plus one multi-controlled RY per pixel."""

    grad_method = None

    def __init__(self, pixels, wires, id=None):
        super().__init__(pixels, wires=wires, id=id)

    @property
    def ndim_params(self):
        return (1,)

    @staticmethod
    def compute_decomposition(pixels, wires):
        wires = list(wires)
        pos, colour = wires[:-1], wires[-1]
        ops = [qml.Hadamard(w) for w in pos]
        for i in range(2 ** len(pos)):
            bits = [int(b) for b in format(i, f"0{len(pos)}b")]
            theta = pixels[..., i] * (np.pi / 2)
            ops.append(qml.ctrl(qml.RY(2 * theta, wires=colour), control=pos, control_values=bits))
        return ops


def apply_encoding(x, wires, encoding: str):
    if encoding == "frqi":
        FRQIEmbedding(x, wires=wires)
    elif encoding == "qpie":
        qml.AmplitudeEmbedding(x, wires=wires, normalize=True)
    elif encoding == "angle_pca":
        qml.AngleEmbedding(x, wires=wires, rotation="Y")
    else:
        raise ValueError(f"Unknown encoding {encoding}")


def prepare_inputs(x, encoding: str):
    """Small numerical guard so an all-black image never gives a zero-norm state."""
    return x + 1e-8 if encoding == "qpie" else x


def frqi_state_formula(image: np.ndarray) -> np.ndarray:
    """The textbook FRQI state vector. Index = 2*position + colour."""
    p = np.asarray(image, dtype=float).reshape(-1)
    theta = p * np.pi / 2
    state = np.zeros(2 * p.size)
    state[0::2] = np.cos(theta)
    state[1::2] = np.sin(theta)
    return state / np.sqrt(p.size)


def qpie_state_formula(image: np.ndarray) -> np.ndarray:
    v = np.asarray(image, dtype=float).reshape(-1) + 1e-8
    return v / np.linalg.norm(v)


def encoding_state(spec: CircuitSpec, x: np.ndarray) -> np.ndarray:
    """State produced by the encoding circuit alone (no variational layers)."""
    wires = list(range(spec.n_qubits))
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circuit(inp):
        apply_encoding(inp, wires, spec.encoding)
        return qml.state()

    return np.asarray(circuit(prepare_inputs(np.asarray(x, dtype=float), spec.encoding)))


# --------------------------------------------------------------------------- #
# Variational circuit
# --------------------------------------------------------------------------- #
def _ansatz_layer(w, wires, ansatz: str):
    n = len(wires)
    if ansatz == "hardware_efficient":
        for i, q in enumerate(wires):
            qml.RY(w[i, 0], wires=q)
            qml.RZ(w[i, 1], wires=q)
        for i in range(n - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
    elif ansatz == "strongly_entangling":
        for i, q in enumerate(wires):
            qml.Rot(w[i, 0], w[i, 1], w[i, 2], wires=q)
        for i in range(n):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])
    elif ansatz == "basic_entangler":
        for i, q in enumerate(wires):
            qml.RX(w[i], wires=q)
        for i in range(n):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])
    else:
        raise ValueError(f"Unknown ansatz {ansatz}")


def build_qnode(spec: CircuitSpec, device: str = "default.qubit", noise_p: float = 0.0,
                mode: str = "expval", interface: str = "torch", diff_method="backprop",
                seed: int | None = None):
    """The full circuit: encode -> L variational layers -> measure.

    mode: "expval" (PauliZ on every qubit), "probs" (full register), "state".
    noise_p > 0 inserts a depolarising channel on every qubit after the
    encoding and after every layer (needs device="default.mixed").
    """
    wires = list(range(spec.n_qubits))
    kwargs = {"seed": seed} if (seed is not None and device == "default.qubit") else {}
    dev = qml.device(device, wires=wires, **kwargs)

    def noise():
        if noise_p > 0:
            for q in wires:
                qml.DepolarizingChannel(noise_p, wires=q)

    @qml.qnode(dev, interface=interface, diff_method=diff_method)
    def circuit(x, weights):
        apply_encoding(x, wires, spec.encoding)
        noise()
        for layer in range(spec.n_layers):
            _ansatz_layer(weights[layer], wires, spec.ansatz)
            noise()
        if mode == "expval":
            return [qml.expval(qml.PauliZ(q)) for q in wires]
        if mode == "probs":
            return qml.probs(wires=wires)
        return qml.state()

    return circuit


def _stack_expvals(outputs) -> torch.Tensor:
    """List of per-qubit expectation values -> (batch, n_qubits), also for a batch of one."""
    return torch.stack([torch.as_tensor(o).reshape(-1) for o in outputs], dim=1).float()


class QuantumImageClassifier(nn.Module):
    """Encoded image -> variational circuit -> <Z> on every qubit -> linear head.

    The circuit is the only place the image is processed; the head just maps the
    n_qubits expectation values to class logits (1 logit when binary)."""

    def __init__(self, spec: CircuitSpec, n_classes: int, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.spec = spec
        self.n_classes = n_classes
        self.weights = nn.Parameter(0.3 * torch.randn(spec.weight_shape))
        self.head = nn.Linear(spec.n_qubits, 1 if n_classes <= 2 else n_classes)
        self._qnode = build_qnode(spec)

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self._qnode(prepare_inputs(x, self.spec.encoding), self.weights)
        return _stack_expvals(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.quantum_features(x))

    def n_parameters(self) -> dict:
        circuit = self.weights.numel()
        head = sum(p.numel() for p in self.head.parameters())
        return {"circuit": circuit, "head": head, "total": circuit + head}


def logits_to_predictions(logits: torch.Tensor, n_classes: int) -> np.ndarray:
    if n_classes <= 2:
        return (torch.sigmoid(logits.squeeze(-1)) > 0.5).long().numpy()
    return logits.argmax(dim=1).numpy()


def logits_to_probabilities(logits: torch.Tensor, n_classes: int) -> np.ndarray:
    if n_classes <= 2:
        p1 = torch.sigmoid(logits.squeeze(-1)).numpy()
        return np.stack([1 - p1, p1], axis=1)
    return torch.softmax(logits, dim=1).numpy()


# --------------------------------------------------------------------------- #
# Non-ideal execution: finite shots and noise
# --------------------------------------------------------------------------- #
def predict_with_shots(model: QuantumImageClassifier, X: np.ndarray, shots: int, seed: int = 0):
    """Replace exact expectation values with estimates from `shots` measurements."""
    qn = build_qnode(model.spec, diff_method=None, seed=seed)
    qn = qml.set_shots(qn, shots=shots)
    with torch.no_grad():
        # float64 so amplitude-encoded states normalise tightly enough for the shot sampler
        x = prepare_inputs(torch.tensor(X, dtype=torch.float64), model.spec.encoding)
        feats = _stack_expvals(qn(x, model.weights.double()))
        logits = model.head(feats)
    return logits_to_predictions(logits, model.n_classes), feats.numpy(), logits


def predict_with_noise(model: QuantumImageClassifier, X: np.ndarray, noise_p: float):
    """Run the trained circuit on a depolarising-noise density-matrix simulator."""
    qn = build_qnode(model.spec, device="default.mixed", noise_p=noise_p, diff_method=None)
    feats = []
    with torch.no_grad():
        for row in X:
            x = prepare_inputs(torch.tensor(row, dtype=torch.float32), model.spec.encoding)
            feats.append(_stack_expvals(qn(x, model.weights))[0])
        feats = torch.stack(feats)
        logits = model.head(feats)
    return logits_to_predictions(logits, model.n_classes), feats.numpy(), logits


# --------------------------------------------------------------------------- #
# Circuit inspection
# --------------------------------------------------------------------------- #
def _dummy_inputs(spec: CircuitSpec):
    x = np.full(spec.n_features, 0.5)
    w = np.zeros(spec.weight_shape)
    return x, w


def draw_circuit(spec: CircuitSpec, x=None, weights=None, expanded: bool = False):
    """matplotlib figure of the circuit. expanded=True shows every FRQI gate."""
    dx, dw = _dummy_inputs(spec)
    x = dx if x is None else np.asarray(x, dtype=float)
    weights = dw if weights is None else np.asarray(weights, dtype=float)
    qn = build_qnode(spec, interface="numpy", diff_method=None)
    kwargs = {"level": "device"} if expanded else {}
    fig, _ = qml.draw_mpl(qn, decimals=2, style="pennylane", **kwargs)(x, weights)
    return fig


def draw_circuit_text(spec: CircuitSpec, x=None, weights=None, expanded: bool = False) -> str:
    dx, dw = _dummy_inputs(spec)
    x = dx if x is None else np.asarray(x, dtype=float)
    weights = dw if weights is None else np.asarray(weights, dtype=float)
    qn = build_qnode(spec, interface="numpy", diff_method=None)
    kwargs = {"level": "device"} if expanded else {}
    return qml.draw(qn, decimals=2, **kwargs)(x, weights)


def circuit_resources(spec: CircuitSpec) -> dict:
    """Gate counts: as written (logical) and after compiling to a hardware-style
    basis of {CNOT, RX, RY, RZ, H, X}."""
    x, w = _dummy_inputs(spec)
    qn = build_qnode(spec, interface="numpy", diff_method=None, mode="expval")
    out = {"n_qubits": spec.n_qubits}
    try:
        logical = qml.specs(qn, level="device")(x, w).resources
        out["logical_gates"] = int(logical.num_gates)
        out["logical_depth"] = int(logical.depth)
    except Exception as exc:  # pragma: no cover - defensive, shown in UI
        out["logical_error"] = str(exc)
    try:
        compiled = qml.transforms.decompose(qn, gate_set=NATIVE_GATES)
        res = qml.specs(compiled)(x, w).resources
        types = {k: int(v) for k, v in dict(res.gate_types).items()}
        out["native_gates"] = int(res.num_gates)
        out["native_depth"] = int(res.depth)
        out["native_cnots"] = types.get("CNOT", 0)
        out["native_gate_types"] = types
    except Exception as exc:  # pragma: no cover
        out["native_error"] = str(exc)
    return out


def time_forward_pass(spec: CircuitSpec, batch: int = 8, repeats: int = 2) -> float:
    """Wall-clock seconds for one batched forward pass of the encode+ansatz circuit."""
    model = QuantumImageClassifier(spec, n_classes=2)
    x = torch.rand(batch, spec.n_features)
    with torch.no_grad():
        model.quantum_features(x)
        t = time.perf_counter()
        for _ in range(repeats):
            model.quantum_features(x)
    return (time.perf_counter() - t) / repeats


# --------------------------------------------------------------------------- #
# Measurement of the encoded image (reconstruction from shots)
# --------------------------------------------------------------------------- #
def reconstruct_frqi_from_shots(image: np.ndarray, shots: int, seed: int = 0) -> np.ndarray:
    """Prepare the FRQI state of `image`, measure every qubit `shots` times, and
    estimate each pixel's intensity from P(colour=1 | position)."""
    size = image.shape[0]
    spec = CircuitSpec("frqi", size)
    wires = list(range(spec.n_qubits))
    dev = qml.device("default.qubit", wires=wires, seed=seed)

    @qml.qnode(dev, shots=shots)
    def circuit(inp):
        apply_encoding(inp, wires, "frqi")
        return qml.counts(wires=wires)

    counts = circuit(np.asarray(image, dtype=float).reshape(-1))
    n_pix = size * size
    ones = np.zeros(n_pix)
    totals = np.zeros(n_pix)
    for bitstring, c in counts.items():
        idx = int(bitstring, 2)
        pos, colour = idx >> 1, idx & 1
        totals[pos] += c
        ones[pos] += c * colour
    p1 = np.divide(ones, totals, out=np.full(n_pix, np.nan), where=totals > 0)
    theta = np.arcsin(np.sqrt(np.clip(np.nan_to_num(p1, nan=0.0), 0, 1)))
    return (theta / (np.pi / 2)).reshape(size, size)


# --------------------------------------------------------------------------- #
# Quantum Hadamard Edge Detection (QHED)
# --------------------------------------------------------------------------- #
def _shift_matrix(n_states: int, shift: int) -> np.ndarray:
    return np.roll(np.eye(n_states), shift, axis=0)


def _qhed_probabilities(image: np.ndarray, hadamard_wire: int, shift: int, shots: int | None,
                         seed: int = 0) -> np.ndarray:
    size = image.shape[0]
    n_wires = 2 * log2_int(size)
    wires = list(range(n_wires))
    dev = qml.device("default.qubit", wires=wires, seed=seed)

    @qml.qnode(dev, shots=shots)
    def circuit(inp):
        qml.AmplitudeEmbedding(inp, wires=wires, normalize=True)
        if shift:
            qml.QubitUnitary(_shift_matrix(2 ** n_wires, shift), wires=wires)
        qml.Hadamard(wires=hadamard_wire)
        return qml.probs(wires=wires)

    return np.asarray(circuit(image.reshape(-1) + 1e-8))


def qhed_edges(image: np.ndarray, shots: int | None = None, seed: int = 0) -> dict:
    """Edge map of a 2^n x 2^n image computed by a quantum circuit.

    Amplitude-encode the image, optionally cyclically shift the pixel index, then
    apply one Hadamard to the least-significant qubit. In every adjacent pixel pair
    (a, b) the |...1> amplitude becomes (a - b)/sqrt(2): the intensity difference.
    Two circuits (unshifted + shifted by one) cover every neighbouring pair.
    Done along rows (horizontal neighbours) and along columns (vertical)."""
    image = np.asarray(image, dtype=float)
    size = image.shape[0]
    n = log2_int(size)
    norm = np.linalg.norm(image.reshape(-1) + 1e-8)

    def directional(col_wire: int, step: int) -> np.ndarray:
        probs_a = _qhed_probabilities(image, col_wire, 0, shots, seed)
        probs_b = _qhed_probabilities(image, col_wire, step, shots, seed + 1)
        # amplitude (a-b)/sqrt2  ->  |a-b| = sqrt(2 * prob) * ||image||
        diff_a = np.sqrt(2 * probs_a) * norm
        diff_b = np.sqrt(2 * probs_b) * norm
        edge = np.zeros(size * size)
        for j in range(step, size * size):
            # boundary between pixel (j - step) and pixel j along the chosen direction
            index_in_pair = (j // step) % 2
            if index_in_pair == 1:
                edge[j] = diff_a[j]
            else:
                edge[j] = diff_b[j + step]
        return edge.reshape(size, size)

    horiz = directional(2 * n - 1, 1)   # LSB of column index
    vert = directional(n - 1, size)     # LSB of row index
    horiz[:, 0] = 0.0                   # row wrap-around is not a real edge
    vert[0, :] = 0.0

    classical_h = np.zeros_like(image)
    classical_h[:, 1:] = np.abs(np.diff(image, axis=1))
    classical_v = np.zeros_like(image)
    classical_v[1:, :] = np.abs(np.diff(image, axis=0))

    return {
        "horizontal": horiz, "vertical": vert,
        "combined": np.sqrt(horiz ** 2 + vert ** 2),
        "classical_combined": np.sqrt(classical_h ** 2 + classical_v ** 2),
        "classical_horizontal": classical_h, "classical_vertical": classical_v,
        "max_error": float(max(np.abs(horiz - classical_h).max(), np.abs(vert - classical_v).max())),
        "n_qubits": 2 * n, "n_circuits": 4,
    }


def qhed_circuit_text(size: int, shifted: bool = False, direction: str = "horizontal") -> str:
    """Text diagram of one QHED circuit (the building block of qhed_edges)."""
    n = log2_int(size)
    n_wires = 2 * n
    wires = list(range(n_wires))
    hadamard_wire = n_wires - 1 if direction == "horizontal" else n - 1
    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circuit(inp):
        qml.AmplitudeEmbedding(inp, wires=wires, normalize=True)
        if shifted:
            qml.QubitUnitary(_shift_matrix(2 ** n_wires, 1), wires=wires)
        qml.Hadamard(wires=hadamard_wire)
        return qml.probs(wires=wires)

    return qml.draw(circuit, decimals=None)(np.ones(size * size))
