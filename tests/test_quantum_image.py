import numpy as np
import pytest
import torch

from qnn_platform.image_data import (
    array_to_pil, load_digits_subset, load_zip_dataset, normalize01, preprocess_image, resize_array,
)
from qnn_platform.quantum_image import (
    CircuitSpec, QuantumImageClassifier, circuit_resources, encoding_state, frqi_state_formula,
    predict_with_noise, predict_with_shots, qhed_edges, qpie_state_formula, reconstruct_frqi_from_shots,
)


@pytest.fixture(scope="module")
def image4():
    rng = np.random.RandomState(1)
    return rng.rand(4, 4)


# ---------------------------------------------------------------- representations
@pytest.mark.parametrize("size", [2, 4])
def test_frqi_circuit_matches_textbook_state(size):
    img = np.random.RandomState(size).rand(size, size)
    spec = CircuitSpec("frqi", size)
    state = encoding_state(spec, img.reshape(-1))
    np.testing.assert_allclose(state, frqi_state_formula(img), atol=1e-10)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_frqi_qubit_count_is_2n_plus_1():
    assert CircuitSpec("frqi", 2).n_qubits == 3
    assert CircuitSpec("frqi", 4).n_qubits == 5
    assert CircuitSpec("frqi", 8).n_qubits == 7


def test_qpie_circuit_matches_amplitude_formula(image4):
    spec = CircuitSpec("qpie", 4)
    assert spec.n_qubits == 4
    state = encoding_state(spec, image4.reshape(-1))
    np.testing.assert_allclose(np.abs(state), np.abs(qpie_state_formula(image4)), atol=1e-6)


def test_frqi_shot_reconstruction_converges(image4):
    img = normalize01(image4)
    coarse = np.mean((reconstruct_frqi_from_shots(img, 200) - img) ** 2)
    fine = np.mean((reconstruct_frqi_from_shots(img, 200000) - img) ** 2)
    assert fine < coarse
    assert fine < 5e-3


# ---------------------------------------------------------------- classifier
@pytest.mark.parametrize("encoding", ["frqi", "qpie", "angle_pca"])
@pytest.mark.parametrize("ansatz", ["hardware_efficient", "strongly_entangling", "basic_entangler"])
def test_classifier_forward_shape_and_gradients(encoding, ansatz):
    spec = CircuitSpec(encoding, 2, n_layers=2, ansatz=ansatz, n_pca=3)
    model = QuantumImageClassifier(spec, n_classes=3)
    x = torch.rand(5, spec.n_features)
    out = model(x)
    assert out.shape == (5, 3)
    out.sum().backward()
    assert model.weights.grad is not None and model.weights.grad.abs().sum() > 0


def test_binary_classifier_has_single_logit():
    model = QuantumImageClassifier(CircuitSpec("frqi", 2), n_classes=2)
    assert model(torch.rand(4, 4)).shape == (4, 1)


def test_single_image_batch_and_shots_and_noise_paths_run():
    spec = CircuitSpec("frqi", 2, n_layers=1)
    model = QuantumImageClassifier(spec, n_classes=2)
    X = np.random.RandomState(0).rand(3, 4).astype(np.float32)
    pred, feats, _ = predict_with_shots(model, X[:1], shots=50, seed=0)
    assert pred.shape == (1,) and feats.shape == (1, spec.n_qubits)
    pred, feats, _ = predict_with_noise(model, X, noise_p=0.05)
    assert pred.shape == (3,)


def test_noise_shrinks_expectation_values():
    spec = CircuitSpec("frqi", 2, n_layers=1)
    model = QuantumImageClassifier(spec, n_classes=2)
    X = np.random.RandomState(0).rand(4, 4).astype(np.float32)
    _, clean, _ = predict_with_noise(model, X, noise_p=0.0)
    _, noisy, _ = predict_with_noise(model, X, noise_p=0.3)
    assert np.abs(noisy).mean() < np.abs(clean).mean()


def test_resources_show_frqi_costs_more_than_qpie():
    frqi = circuit_resources(CircuitSpec("frqi", 4))
    qpie = circuit_resources(CircuitSpec("qpie", 4))
    assert frqi["native_gates"] > 10 * qpie["native_gates"]
    assert frqi["n_qubits"] == qpie["n_qubits"] + 1


# ---------------------------------------------------------------- QHED
@pytest.mark.parametrize("size", [4, 8])
def test_qhed_matches_classical_neighbour_differences(size):
    img = np.random.RandomState(size).rand(size, size)
    out = qhed_edges(img)
    assert out["max_error"] < 1e-9
    np.testing.assert_allclose(out["combined"], out["classical_combined"], atol=1e-9)


def test_qhed_detects_a_vertical_step_edge():
    img = np.zeros((8, 8))
    img[:, 4:] = 1.0
    out = qhed_edges(img)
    assert out["horizontal"][:, 4].min() > 0.5
    assert out["horizontal"][:, 5:].max() < 1e-6
    assert out["vertical"].max() < 1e-6


# ---------------------------------------------------------------- data layer
def test_digits_subset_shapes_and_labels():
    ds = load_digits_subset([3, 8])
    assert ds.X.shape[1:] == (8, 8)
    assert set(ds.y) == {0, 1}
    assert ds.class_names == ["3", "8"]
    assert ds.X.min() >= 0 and ds.X.max() <= 1
    assert ds.at_size(4).shape[1:] == (4, 4)


def test_digits_requires_two_classes():
    with pytest.raises(ValueError):
        load_digits_subset([3])


def test_preprocess_steps_for_rgb_upload():
    from PIL import Image
    rng = np.random.RandomState(0)
    img = Image.fromarray((rng.rand(60, 90, 3) * 255).astype(np.uint8))
    steps = preprocess_image(img, 4, invert=True)
    assert steps.cropped.shape == (60, 60)
    assert steps.resized.shape == (4, 4)
    assert steps.normalized.min() == 0 and np.isclose(steps.normalized.max(), 1.0)
    np.testing.assert_allclose(steps.oriented, 255 - steps.cropped)


def test_preprocess_rejects_non_power_of_two():
    from PIL import Image
    with pytest.raises(ValueError):
        preprocess_image(Image.new("RGB", (10, 10)), 5, invert=False)


def test_resize_array_block_average():
    a = np.arange(16, dtype=float).reshape(4, 4)
    np.testing.assert_allclose(resize_array(a, 2), [[2.5, 4.5], [10.5, 12.5]])


def test_zip_dataset_roundtrip(tmp_path):
    import io, zipfile
    from PIL import Image
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for cls, level in (("dark", 40), ("light", 220)):
            for i in range(3):
                img = Image.new("L", (16, 16), level + i)
                b = io.BytesIO()
                img.save(b, format="PNG")
                zf.writestr(f"data/{cls}/{i}.png", b.getvalue())
    ds = load_zip_dataset(buf.getvalue(), invert=False)
    assert ds.class_names == ["dark", "light"]
    assert ds.X.shape == (6, 8, 8)


def test_zip_with_single_class_is_rejected():
    import io, zipfile
    from PIL import Image
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        b = io.BytesIO()
        Image.new("L", (8, 8), 5).save(b, format="PNG")
        zf.writestr("only/1.png", b.getvalue())
    with pytest.raises(ValueError):
        load_zip_dataset(buf.getvalue(), invert=False)


def test_array_to_pil_returns_rgb_image():
    assert array_to_pil(np.zeros((4, 4)), 64).size == (64, 64)
