"""
Tests for the ActiveLearner (src/active_learning.py).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from src.active_learning import ActiveLearner, ALResult, _minmax_norm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def nbsi_like_data():
    """70-sample, 7-feature dataset mimicking Nb-Si composition space."""
    X_arr, y_arr = make_regression(n_samples=70, n_features=7, noise=1.0, random_state=2)
    cols = ["Si", "Mo", "Hf", "Ti", "Cr", "Al", "Nb"]
    X = pd.DataFrame(X_arr, columns=cols)
    y = pd.Series(y_arr + 16.0, name="KIC")  # shift to positive range
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_run(nbsi_like_data):
    """Run 5 AL iterations and check that ALResult is well-formed."""
    X, y = nbsi_like_data
    initial = list(range(10))
    learner = ActiveLearner(alpha=0.5, model_kwargs={"n_estimators": 30, "random_state": 0})
    result = learner.run(X, y, initial_indices=initial, n_iterations=5, verbose=False)

    assert isinstance(result, ALResult)
    assert len(result.history) == 5
    assert len(result.labelled_indices) == 10 + 5


def test_labelled_pool_grows(nbsi_like_data):
    X, y = nbsi_like_data
    learner = ActiveLearner(alpha=0.5, model_kwargs={"n_estimators": 30, "random_state": 0})
    result = learner.run(X, y, initial_indices=list(range(8)), n_iterations=4, verbose=False)
    n_labelled = [rec.n_labelled for rec in result.history]
    assert n_labelled == sorted(n_labelled), "Labelled pool must grow monotonically."


def test_no_duplicate_queries(nbsi_like_data):
    X, y = nbsi_like_data
    learner = ActiveLearner(alpha=0.5, model_kwargs={"n_estimators": 30, "random_state": 0})
    result = learner.run(X, y, initial_indices=list(range(10)), n_iterations=10, verbose=False)
    assert len(result.labelled_indices) == len(set(result.labelled_indices)), \
        "Duplicate indices in labelled pool."


def test_to_dataframe_columns(nbsi_like_data):
    X, y = nbsi_like_data
    learner = ActiveLearner(alpha=0.5, model_kwargs={"n_estimators": 20, "random_state": 0})
    result = learner.run(X, y, initial_indices=list(range(8)), n_iterations=3, verbose=False)
    df = result.to_dataframe()
    expected_cols = {"iteration", "n_labelled", "rmse", "r2", "best_kic", "selected_kic"}
    assert expected_cols.issubset(set(df.columns))


def test_shap_importance_over_time(nbsi_like_data):
    X, y = nbsi_like_data
    learner = ActiveLearner(alpha=0.5, model_kwargs={"n_estimators": 20, "random_state": 0})
    result = learner.run(X, y, initial_indices=list(range(8)), n_iterations=3, verbose=False)
    df = result.shap_importance_over_time()
    assert df.shape == (3, len(X.columns))


def test_alpha_pure_uncertainty(nbsi_like_data):
    """alpha=1 (pure uncertainty) should still run without errors."""
    X, y = nbsi_like_data
    learner = ActiveLearner(alpha=1.0, model_kwargs={"n_estimators": 20, "random_state": 0})
    result = learner.run(X, y, initial_indices=list(range(8)), n_iterations=3, verbose=False)
    assert len(result.history) == 3


def test_alpha_pure_shap(nbsi_like_data):
    """alpha=0 (pure SHAP exploration) should still run without errors."""
    X, y = nbsi_like_data
    learner = ActiveLearner(alpha=0.0, model_kwargs={"n_estimators": 20, "random_state": 0})
    result = learner.run(X, y, initial_indices=list(range(8)), n_iterations=3, verbose=False)
    assert len(result.history) == 3


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        ActiveLearner(alpha=1.5)


def test_custom_oracle(nbsi_like_data):
    """Custom oracle should be called with the selected index."""
    X, y = nbsi_like_data
    queried = []

    def my_oracle(idx):
        queried.append(idx)
        return float(y.iloc[idx])

    learner = ActiveLearner(alpha=0.5, model_kwargs={"n_estimators": 20, "random_state": 0})
    result = learner.run(
        X, y,
        initial_indices=list(range(8)),
        n_iterations=3,
        oracle=my_oracle,
        verbose=False,
    )
    assert len(queried) == 3, "Oracle must be called once per iteration."


def test_minmax_norm(nbsi_like_data):
    arr = np.array([0.0, 2.0, 4.0, 6.0])
    normed = _minmax_norm(arr)
    assert abs(normed[0]) < 1e-9
    assert abs(normed[-1] - 1.0) < 1e-9
