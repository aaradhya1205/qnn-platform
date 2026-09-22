import torch

from qnn_platform.quantum_model import QNNModel, QNNLayer, weight_shape


def test_weight_shape_matches_ansatz_conventions():
    assert weight_shape("basic_entangler", n_qubits=4, n_layers=2) == (2, 4)
    assert weight_shape("strongly_entangling", n_qubits=4, n_layers=2) == (2, 4, 3)
    assert weight_shape("hardware_efficient", n_qubits=4, n_layers=2) == (2, 4, 2)


def test_qnn_layer_forward_shape():
    layer = QNNLayer(n_qubits=3, n_layers=1, encoding="angle", ansatz="basic_entangler")
    x = torch.rand(5, 3) * torch.pi
    out = layer(x)
    assert out.shape == (5, 3)


def test_qnn_model_classification_output_shape_multiclass():
    model = QNNModel(n_features=4, n_qubits=3, n_layers=1, task="classification", n_outputs=3)
    x = torch.rand(6, 4)
    out = model(x)
    assert out.shape == (6, 3)


def test_qnn_model_regression_output_shape():
    model = QNNModel(n_features=4, n_qubits=3, n_layers=1, task="regression")
    x = torch.rand(6, 4)
    out = model(x)
    assert out.shape == (6, 1)


def test_qnn_model_gradients_flow():
    model = QNNModel(n_features=4, n_qubits=3, n_layers=1, task="regression")
    x = torch.rand(4, 4)
    out = model(x)
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
