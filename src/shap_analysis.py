"""
Dynamic SHAP analysis for SHAP-guided active learning.

SHAP (SHapley Additive exPlanations) values are recomputed at every active-
learning iteration to provide a current, model-aware picture of how each
composition feature drives the KIC prediction for every unlabelled candidate.

Key design choices
------------------
* TreeExplainer is used (exact, fast for Random Forest).
* The signed SHAP values capture direction of influence; the absolute values
  are used to quantify *importance* per feature.
* A normalised composite SHAP score is returned so it can be directly added
  to the uncertainty score in the acquisition function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.model import NbSiModel


class SHAPAnalyser:
    """Compute and analyse SHAP values for the NbSiModel.

    Parameters
    ----------
    model:
        A fitted NbSiModel instance.
    """

    def __init__(self, model: NbSiModel) -> None:
        if not model.is_fitted:
            raise ValueError(
                "SHAPAnalyser requires a fitted NbSiModel. Call model.fit() first."
            )
        self.model = model
        self._explainer: shap.TreeExplainer | None = None
        self.shap_values: np.ndarray | None = None
        self.feature_names: list[str] = model.feature_names

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for all rows in *X*.

        Parameters
        ----------
        X:
            Feature matrix (n_samples × n_features).  Must have the same
            columns that the model was trained on.

        Returns
        -------
        shap_values : np.ndarray, shape (n_samples, n_features)
            Signed SHAP values.  Positive value → feature pushes KIC higher.
        """
        # Re-create the explainer from the current (freshly trained) forest
        self._explainer = shap.TreeExplainer(self.model._rf)
        X_scaled = self.model._scaler.transform(X.values.astype(float))
        raw = self._explainer.shap_values(X_scaled)
        # TreeExplainer for regression returns a single 2-D array
        if isinstance(raw, list):
            raw = raw[0]
        self.shap_values = raw
        return self.shap_values

    def mean_abs_shap(self) -> pd.Series:
        """Return mean |SHAP| per feature, sorted descending.

        Requires a prior call to :meth:`compute`.

        Returns
        -------
        pd.Series indexed by feature name.
        """
        self._check_computed()
        importances = np.abs(self.shap_values).mean(axis=0)
        return (
            pd.Series(importances, index=self.feature_names)
            .sort_values(ascending=False)
        )

    def composite_score(self) -> np.ndarray:
        """Return a per-sample SHAP diversity/exploration score.

        The score captures how *uncertain* the model is about *why* each
        candidate would achieve a high KIC.  It combines:

        1. **SHAP magnitude** – candidates whose prediction relies on a large
           absolute SHAP signal are already well-explained.
        2. **SHAP entropy** – candidates where the SHAP weight is spread
           across many features are in a more uncertain region of composition
           space and are therefore preferred for exploration.

        Score = SHAP_entropy − λ · SHAP_magnitude (both normalised to [0,1])

        where λ = 0.3 balances exploitation of high-magnitude regions against
        exploration of high-entropy regions.

        Requires a prior call to :meth:`compute`.

        Returns
        -------
        score : np.ndarray, shape (n_samples,)
            Higher → candidate is more informative from a SHAP perspective.
        """
        self._check_computed()
        abs_sv = np.abs(self.shap_values)  # (n_samples, n_features)

        # ---- SHAP entropy: spread of |SHAP| across features ----
        row_sum = abs_sv.sum(axis=1, keepdims=True) + 1e-12
        p = abs_sv / row_sum
        entropy = -(p * np.log(p + 1e-12)).sum(axis=1)  # (n_samples,)

        # ---- SHAP magnitude: total absolute SHAP value ----
        magnitude = abs_sv.sum(axis=1)  # (n_samples,)

        entropy_norm = _minmax_norm(entropy)
        magnitude_norm = _minmax_norm(magnitude)

        return entropy_norm - 0.3 * magnitude_norm

    def shap_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of SHAP values with the same index as *X*.

        Parameters
        ----------
        X:
            The same DataFrame passed to :meth:`compute`.

        Returns
        -------
        pd.DataFrame, shape (n_samples, n_features).
        """
        self._check_computed()
        return pd.DataFrame(
            self.shap_values, index=X.index, columns=self.feature_names
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_computed(self) -> None:
        if self.shap_values is None:
            raise RuntimeError(
                "SHAP values have not been computed yet.  Call compute() first."
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Scale *arr* to [0, 1]; returns zeros if all values are equal."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)
