"""
Machine learning model for Nb-Si alloy fracture toughness prediction.

Uses a Random Forest regressor whose per-tree predictions provide a natural
uncertainty estimate (standard deviation across trees) that is exploited by the
active-learning acquisition function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class NbSiModel:
    """Random Forest model with built-in uncertainty quantification.

    Parameters
    ----------
    n_estimators:
        Number of trees in the forest.
    max_features:
        Maximum number of features considered for each split.
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_features: float = 0.6,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state

        self._rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        self._scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NbSiModel":
        """Fit the model on labelled training data.

        Parameters
        ----------
        X:
            Feature matrix (n_samples × n_features).
        y:
            Target vector (fracture toughness, KIC).
        """
        self.feature_names = list(X.columns)
        X_arr = self._scaler.fit_transform(X.values.astype(float))
        self._rf.fit(X_arr, y.values.astype(float))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return mean prediction across all trees.

        Parameters
        ----------
        X:
            Feature matrix with the same columns used during training.
        """
        self._check_fitted()
        X_arr = self._scaler.transform(X.values.astype(float))
        return self._rf.predict(X_arr)

    def predict_with_uncertainty(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean prediction, prediction std) across trees.

        The standard deviation of per-tree predictions serves as a
        model-based uncertainty estimate suitable for active-learning
        acquisition.

        Parameters
        ----------
        X:
            Feature matrix with the same columns used during training.

        Returns
        -------
        mean : np.ndarray, shape (n_samples,)
        std  : np.ndarray, shape (n_samples,)
        """
        self._check_fitted()
        X_arr = self._scaler.transform(X.values.astype(float))
        # Collect predictions from each tree
        tree_preds = np.array(
            [tree.predict(X_arr) for tree in self._rf.estimators_]
        )  # shape: (n_estimators, n_samples)
        mean = tree_preds.mean(axis=0)
        std = tree_preds.std(axis=0)
        return mean, std

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> dict[str, float]:
        """Run k-fold cross-validation and return performance metrics.

        Parameters
        ----------
        X:
            Full feature matrix.
        y:
            Full target vector.
        cv:
            Number of folds.

        Returns
        -------
        Dictionary with 'r2_mean', 'r2_std', 'rmse_mean', 'rmse_std'.
        """
        X_arr = self._scaler.fit_transform(X.values.astype(float))

        r2_scores = cross_val_score(
            self._rf, X_arr, y.values.astype(float), cv=cv, scoring="r2"
        )
        neg_mse_scores = cross_val_score(
            self._rf,
            X_arr,
            y.values.astype(float),
            cv=cv,
            scoring="neg_mean_squared_error",
        )
        rmse_scores = np.sqrt(-neg_mse_scores)

        return {
            "r2_mean": float(r2_scores.mean()),
            "r2_std": float(r2_scores.std()),
            "rmse_mean": float(rmse_scores.mean()),
            "rmse_std": float(rmse_scores.std()),
        }

    def get_feature_importances(self) -> pd.Series:
        """Return impurity-based feature importances from the forest.

        Returns
        -------
        pd.Series indexed by feature name, sorted descending.
        """
        self._check_fitted()
        importances = self._rf.feature_importances_
        return (
            pd.Series(importances, index=self.feature_names)
            .sort_values(ascending=False)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() first."
            )
