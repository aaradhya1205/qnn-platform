"""Small shared figure helpers."""
from __future__ import annotations

import io
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qnn_platform.quantum_model import QNNLayer  # noqa: E402


@lru_cache(maxsize=64)
def circuit_png(n_qubits: int, n_layers: int, encoding: str, ansatz: str) -> bytes:
    """Dark-theme picture of the circuit: encoder, learning layers and measurement."""
    fig, _ = QNNLayer(n_qubits, n_layers, encoding, ansatz).draw_mpl(style="black_white_dark")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def n_circuit_params(n_qubits: int, n_layers: int, ansatz: str) -> int:
    per = {"basic_entangler": 1, "strongly_entangling": 3, "hardware_efficient": 2}[ansatz]
    return n_layers * n_qubits * per
