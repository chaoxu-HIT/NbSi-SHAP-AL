"""
Tests for the NbSiModel (src/model.py).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from src.model import NbSiModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def toy_data():
    """Simple synthetic regression dataset (50 samples, 7 features)."""
    X_arr, y_arr = make_regression(n_samples=50, n_features=7, noise=0.5, random_state=0)
    cols = ["Si", "Mo", "Hf", "Ti", "Cr", "Al", "Nb"]
    X = pd.DataFrame(X_arr, columns=cols)
    y = pd.Series(y_arr, name="KIC")
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fit_predict(toy_data):
    X, y = toy_data
    model = NbSiModel(n_estimators=50, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),), "Prediction shape mismatch."
    assert np.isfinite(preds).all(), "Predictions contain non-finite values."


def test_predict_with_uncertainty(toy_data):
    X, y = toy_data
    model = NbSiModel(n_estimators=50, random_state=0)
    model.fit(X, y)
    mean, std = model.predict_with_uncertainty(X)
    assert mean.shape == (len(X),)
    assert std.shape == (len(X),)
    assert (std >= 0).all(), "Standard deviation must be non-negative."


def test_cross_validate_keys(toy_data):
    X, y = toy_data
    model = NbSiModel(n_estimators=20, random_state=0)
    metrics = model.cross_validate(X, y, cv=3)
    for key in ("r2_mean", "r2_std", "rmse_mean", "rmse_std"):
        assert key in metrics, f"Missing key: {key}"
    assert metrics["rmse_mean"] >= 0, "RMSE must be non-negative."


def test_feature_importances(toy_data):
    X, y = toy_data
    model = NbSiModel(n_estimators=50, random_state=0)
    model.fit(X, y)
    importances = model.get_feature_importances()
    assert len(importances) == len(X.columns)
    assert abs(importances.sum() - 1.0) < 1e-6, "Importances must sum to 1."


def test_predict_before_fit_raises():
    model = NbSiModel()
    X = pd.DataFrame({"a": [1.0]})
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(X)


def test_is_fitted_flag(toy_data):
    X, y = toy_data
    model = NbSiModel(n_estimators=10, random_state=0)
    assert not model.is_fitted
    model.fit(X, y)
    assert model.is_fitted
