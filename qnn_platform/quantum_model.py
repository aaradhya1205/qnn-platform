"""Configurable quantum encodings, variational circuits, and QNN architectures
built on PennyLane, exposed as differentiable PyTorch modules."""
from __future__ import annotations

import pennylane as qml
import torch
from torch import nn

ENCODINGS = ["angle", "amplitude", "iqp"]
ANSATZE = ["basic_entangler", "strongly_entangling", "hardware_efficient"]


def apply_encoding(x, wires, encoding: str):
    if encoding == "angle":
        qml.AngleEmbedding(x, wires=wires, rotation="Y")
    elif encoding == "amplitude":
        qml.AmplitudeEmbedding(x, wires=wires, normalize=True, pad_with=0.0)
    elif encoding == "iqp":
        qml.IQPEmbedding(x, wires=wires, n_repeats=2)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def apply_ansatz(weights, wires, ansatz: str):
    if ansatz == "basic_entangler":
        qml.BasicEntanglerLayers(weights, wires=wires)
    elif ansatz == "strongly_entangling":
        qml.StronglyEntanglingLayers(weights, wires=wires)
    elif ansatz == "hardware_efficient":
        n_layers = weights.shape[0]
        for layer in range(n_layers):
            for i, w in enumerate(wires):
                qml.RY(weights[layer, i, 0], wires=w)
                qml.RZ(weights[layer, i, 1], wires=w)
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
    else:
        raise ValueError(f"Unknown ansatz: {ansatz}")


def weight_shape(ansatz: str, n_qubits: int, n_layers: int):
    if ansatz == "basic_entangler":
        return (n_layers, n_qubits)
    if ansatz == "strongly_entangling":
        return (n_layers, n_qubits, 3)
    if ansatz == "hardware_efficient":
        return (n_layers, n_qubits, 2)
    raise ValueError(f"Unknown ansatz: {ansatz}")


class QNNLayer(nn.Module):
    """A single variational quantum circuit wrapped as a differentiable torch layer.

    Input features are encoded onto `n_qubits` qubits, passed through `n_layers`
    of a configurable variational ansatz, and measured (PauliZ expectation on
    every wire) to produce `n_qubits` outputs. Uses the parameter-shift rule via
    PennyLane's torch interface so it trains alongside ordinary torch layers.
    """

    def __init__(self, n_qubits: int, n_layers: int, encoding: str = "angle",
                 ansatz: str = "strongly_entangling", diff_method: str = "best"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.ansatz = ansatz
        wires = list(range(n_qubits))

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            apply_encoding(inputs, wires, encoding)
            apply_ansatz(weights, wires, ansatz)
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        self.circuit = circuit
        shape = weight_shape(ansatz, n_qubits, n_layers)
        weight_shapes = {"weights": shape}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)

    def draw(self, sample_input=None):
        """Return a text diagram of the circuit for a representative input."""
        if sample_input is None:
            sample_input = torch.zeros(self.n_qubits)
        weights = torch.zeros(weight_shape(self.ansatz, self.n_qubits, self.n_layers))
        return qml.draw(self.circuit)(sample_input, weights)

    def draw_mpl(self, sample_input=None, **style_kwargs):
        """Return a (fig, ax) matplotlib rendering of the circuit for a representative input."""
        if sample_input is None:
            sample_input = torch.ones(self.n_qubits)
        weights = torch.zeros(weight_shape(self.ansatz, self.n_qubits, self.n_layers))
        return qml.draw_mpl(self.circuit, **style_kwargs)(sample_input, weights)


class QNNModel(nn.Module):
    """Full QNN: optional classical pre-layer -> quantum layer -> classical head.

    task="classification" ends in n_classes logits (or 1 logit for binary);
    task="regression" ends in a single continuous output.
    """

    def __init__(self, n_features: int, n_qubits: int, n_layers: int, task: str,
                 n_outputs: int = 1, encoding: str = "angle",
                 ansatz: str = "strongly_entangling"):
        super().__init__()
        self.task = task
        self.n_qubits = n_qubits
        # Classical layer maps raw features to n_qubits values in [0, pi] for encoding.
        self.pre = nn.Linear(n_features, n_qubits) if n_features != n_qubits else nn.Identity()
        self.qlayer = QNNLayer(n_qubits, n_layers, encoding=encoding, ansatz=ansatz)
        out_dim = n_outputs if task == "classification" and n_outputs > 2 else 1
        self.post = nn.Linear(n_qubits, out_dim)

    def forward(self, x):
        x = self.pre(x)
        if not isinstance(self.pre, nn.Identity):
            x = torch.pi * torch.sigmoid(x)  # keep inputs bounded for angle encoding
        x = self.qlayer(x)
        return self.post(x)
