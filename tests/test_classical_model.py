import pytest

from qnn_platform.classical_model import BASELINES, build_baseline


@pytest.mark.parametrize("name", BASELINES)
def test_build_baseline_classification(name):
    model = build_baseline(name, "classification")
    assert hasattr(model, "fit") and hasattr(model, "predict")


@pytest.mark.parametrize("name", BASELINES)
def test_build_baseline_regression(name):
    model = build_baseline(name, "regression")
    assert hasattr(model, "fit") and hasattr(model, "predict")


def test_build_baseline_unknown_name_raises():
    with pytest.raises(ValueError):
        build_baseline("not_a_model", "classification")
