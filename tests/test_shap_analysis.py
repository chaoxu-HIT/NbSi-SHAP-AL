"""
Tests for the SHAPAnalyser (src/shap_analysis.py).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from src.model import NbSiModel
from src.shap_analysis import SHAPAnalyser, _minmax_norm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fitted_model():
    X_arr, y_arr = make_regression(n_samples=60, n_features=7, noise=0.5, random_state=1)
    cols = ["Si", "Mo", "Hf", "Ti", "Cr", "Al", "Nb"]
    X = pd.DataFrame(X_arr, columns=cols)
    y = pd.Series(y_arr, name="KIC")
    model = NbSiModel(n_estimators=50, random_state=1)
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_shap_compute_shape(fitted_model):
    model, X, _ = fitted_model
    analyser = SHAPAnalyser(model)
    shap_vals = analyser.compute(X)
    assert shap_vals.shape == (len(X), len(X.columns)), "SHAP shape mismatch."


def test_mean_abs_shap_positive(fitted_model):
    model, X, _ = fitted_model
    analyser = SHAPAnalyser(model)
    analyser.compute(X)
    mean_abs = analyser.mean_abs_shap()
    assert (mean_abs >= 0).all(), "Mean |SHAP| values must be non-negative."
    assert len(mean_abs) == len(X.columns)


def test_composite_score_shape(fitted_model):
    model, X, _ = fitted_model
    analyser = SHAPAnalyser(model)
    analyser.compute(X)
    score = analyser.composite_score()
    assert score.shape == (len(X),), "Composite score shape mismatch."


def test_shap_dataframe(fitted_model):
    model, X, _ = fitted_model
    analyser = SHAPAnalyser(model)
    analyser.compute(X)
    df = analyser.shap_dataframe(X)
    assert df.shape == (len(X), len(X.columns))
    assert list(df.columns) == list(X.columns)


def test_compute_before_mean_abs_raises(fitted_model):
    model, _, _ = fitted_model
    analyser = SHAPAnalyser(model)
    with pytest.raises(RuntimeError, match="computed"):
        analyser.mean_abs_shap()


def test_unfitted_model_raises():
    model = NbSiModel()
    with pytest.raises(ValueError, match="fitted"):
        SHAPAnalyser(model)


def test_minmax_norm_range():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normed = _minmax_norm(arr)
    assert abs(normed.min()) < 1e-9
    assert abs(normed.max() - 1.0) < 1e-9


def test_minmax_norm_constant():
    arr = np.array([3.0, 3.0, 3.0])
    normed = _minmax_norm(arr)
    assert (normed == 0).all()
