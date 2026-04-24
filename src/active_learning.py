"""
Active learning loop with dynamic SHAP-guided acquisition.

Algorithm overview
------------------
1. Split data into initial labelled pool and unlabelled candidate pool.
2. For each iteration *t*:
   a. Fit NbSiModel on current labelled set.
   b. Run SHAPAnalyser to compute SHAP values on all *unlabelled* candidates.
   c. Compute per-candidate acquisition score:

          score = α · uncertainty_score + (1 − α) · shap_composite_score

      where uncertainty_score is the per-tree prediction std (normalised) and
      shap_composite_score is the SHAP entropy/magnitude blend (normalised).

   d. Select the candidate with the highest acquisition score and move it to
      the labelled pool (the oracle reveals its true KIC from the data).
   e. Record metrics (RMSE, R², best KIC discovered so far).
3. Return full history for analysis and plotting.

The "dynamic" nature of the method lies in step 2b: SHAP values are
recomputed from the freshly trained model at every iteration, so the
acquisition strategy adapts as the model improves.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from src.model import NbSiModel
from src.shap_analysis import SHAPAnalyser


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class IterationRecord:
    """Stores diagnostics produced at one AL iteration."""

    iteration: int
    n_labelled: int
    rmse: float
    r2: float
    best_kic: float
    selected_index: int
    selected_kic: float
    acquisition_scores: np.ndarray
    mean_abs_shap: pd.Series


@dataclasses.dataclass
class ALResult:
    """Aggregated result of a full active-learning run."""

    history: list[IterationRecord]
    labelled_indices: list[int]
    feature_names: list[str]

    # ------------------------------------------------------------------ #
    # Convenience accessors                                                #
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        """Summarise the history as a tidy DataFrame."""
        rows = []
        for rec in self.history:
            rows.append(
                {
                    "iteration": rec.iteration,
                    "n_labelled": rec.n_labelled,
                    "rmse": rec.rmse,
                    "r2": rec.r2,
                    "best_kic": rec.best_kic,
                    "selected_kic": rec.selected_kic,
                }
            )
        return pd.DataFrame(rows)

    def shap_importance_over_time(self) -> pd.DataFrame:
        """Return a DataFrame of mean |SHAP| per feature over iterations."""
        rows = []
        for rec in self.history:
            row = {"iteration": rec.iteration}
            row.update(rec.mean_abs_shap.to_dict())
            rows.append(row)
        return pd.DataFrame(rows).set_index("iteration")


# ---------------------------------------------------------------------------
# Core active-learning class
# ---------------------------------------------------------------------------

class ActiveLearner:
    """SHAP-guided active learning for Nb-Si fracture toughness.

    Parameters
    ----------
    alpha:
        Weight for uncertainty vs. SHAP composite score.
        alpha=1 → pure uncertainty sampling.
        alpha=0 → pure SHAP-guided exploration.
    model_kwargs:
        Keyword arguments forwarded to :class:`NbSiModel`.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        model_kwargs: dict | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = alpha
        self.model_kwargs = model_kwargs or {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        initial_indices: list[int],
        n_iterations: int,
        *,
        oracle: Callable[[int], float] | None = None,
        verbose: bool = True,
    ) -> ALResult:
        """Execute the active-learning loop.

        Parameters
        ----------
        X:
            Full feature matrix (all samples).
        y:
            Full target vector (true KIC for all samples).
        initial_indices:
            Indices (into X/y) of the initially labelled samples.
        n_iterations:
            Number of AL query iterations to run.
        oracle:
            Optional callable that returns the true KIC for a given index.
            If *None*, the true value is taken directly from *y* (simulated
            oracle).
        verbose:
            Whether to print per-iteration progress.

        Returns
        -------
        ALResult
        """
        if oracle is None:
            oracle = lambda idx: float(y.iloc[idx])

        labelled = list(initial_indices)
        history: list[IterationRecord] = []

        for t in range(n_iterations):
            unlabelled = [i for i in range(len(X)) if i not in set(labelled)]
            if not unlabelled:
                warnings.warn("No unlabelled candidates remain; stopping early.")
                break

            X_train = X.iloc[labelled]
            y_train = y.iloc[labelled]
            X_cand = X.iloc[unlabelled]

            # ---- (a) Train model ----
            model = NbSiModel(**self.model_kwargs)
            model.fit(X_train, y_train)

            # ---- (b) Compute SHAP values on candidates ----
            analyser = SHAPAnalyser(model)
            analyser.compute(X_cand)
            shap_score = analyser.composite_score()  # (n_unlabelled,)
            mean_abs = analyser.mean_abs_shap()

            # ---- (c) Compute acquisition score ----
            _, std = model.predict_with_uncertainty(X_cand)
            unc_score = _minmax_norm(std)
            shap_score_norm = _minmax_norm(shap_score)
            acq = self.alpha * unc_score + (1.0 - self.alpha) * shap_score_norm

            # ---- (d) Select best candidate ----
            best_local = int(np.argmax(acq))
            selected_global = unlabelled[best_local]
            true_kic = oracle(selected_global)
            labelled.append(selected_global)

            # ---- (e) Record metrics on test (validation) set ----
            # Use the remaining unlabelled (after removal) as proxy test set,
            # or the full dataset minus current training set.
            test_indices = [i for i in range(len(X)) if i not in set(labelled)]
            if test_indices:
                X_test = X.iloc[test_indices]
                y_test = y.iloc[test_indices]
                y_pred = model.predict(X_test)
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                r2 = float(r2_score(y_test, y_pred))
            else:
                rmse, r2 = 0.0, 1.0

            best_kic = float(y.iloc[labelled].max())

            record = IterationRecord(
                iteration=t + 1,
                n_labelled=len(labelled),
                rmse=rmse,
                r2=r2,
                best_kic=best_kic,
                selected_index=selected_global,
                selected_kic=true_kic,
                acquisition_scores=acq,
                mean_abs_shap=mean_abs,
            )
            history.append(record)

            if verbose:
                print(
                    f"  Iter {t+1:3d} | labelled={len(labelled):3d} | "
                    f"RMSE={rmse:.3f} | R²={r2:.3f} | "
                    f"best KIC={best_kic:.2f} | "
                    f"selected KIC={true_kic:.2f}"
                )

        return ALResult(
            history=history,
            labelled_indices=labelled,
            feature_names=list(X.columns),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Scale *arr* to [0, 1]; return zeros if all values are equal."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)
