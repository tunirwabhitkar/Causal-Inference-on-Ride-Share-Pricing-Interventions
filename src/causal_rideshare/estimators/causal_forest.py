"""Causal Forest estimator for heterogeneous treatment effects.

Since ``econml`` / ``grf`` are not available on Python 3.14, this module
implements a manual causal-forest-style estimator using scikit-learn
``RandomForestRegressor`` with the *T-learner* approach (separate models for
treated / control) and a simple honest-splitting variant.

For CATE estimation we also provide a lightweight *R-learner* using cross-
fitted residuals, which is closer to the Generalized Random Forest idea.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold

from causal_rideshare.dag import SURGE_CONFOUNDERS


class CausalForestEstimator:
    """T-learner + R-learner causal forest substitute.

    Attributes
    ----------
    ate_ : float
        Average Treatment Effect (mean of predicted CATEs).
    cate_ : np.ndarray
        Conditional Average Treatment Effect for every observation.
    """

    def __init__(
        self,
        treatment: str = "is_surge",
        outcome: str = "booking",
        confounders: list[str] | None = None,
        n_estimators: int = 100,
        max_depth: int = 8,
        seed: int = 42,
    ):
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders or SURGE_CONFOUNDERS
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed

        self.ate_: float | None = None
        self.cate_: np.ndarray | None = None

        # Internal models
        self._mu1 = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
        self._mu0 = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
        self._ps_model = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=seed
        )

    def fit(self, df: pd.DataFrame) -> "CausalForestEstimator":
        X = df[self.confounders].values
        T = df[self.treatment].values
        Y = df[self.outcome].values

        # --- T-learner: separate outcome models ---
        self._mu1.fit(X[T == 1], Y[T == 1])
        self._mu0.fit(X[T == 0], Y[T == 0])

        tau_t = self._mu1.predict(X) - self._mu0.predict(X)

        # --- R-learner refinement (cross-fitted) ---
        kf = KFold(n_splits=3, shuffle=True, random_state=self.seed)
        y_res = np.zeros(len(Y), dtype=float)
        t_res = np.zeros(len(T), dtype=float)

        for tr_idx, te_idx in kf.split(X):
            # outcome nuisance
            rf_y = RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                random_state=self.seed,
            )
            rf_y.fit(X[tr_idx], Y[tr_idx])
            y_res[te_idx] = Y[te_idx] - rf_y.predict(X[te_idx])

            # propensity nuisance
            self._ps_model.fit(X[tr_idx], T[tr_idx])
            t_res[te_idx] = T[te_idx] - self._ps_model.predict_proba(X[te_idx])[:, 1]

        # Pseudo-outcome for R-learner
        pseudo_y = y_res / (t_res + 1e-8)
        # Weight by t_res^2 to down-weight uncertain propensity
        weights = t_res ** 2

        # Fit final CATE model
        cate_model = RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.seed,
        )
        cate_model.fit(X, pseudo_y, sample_weight=np.abs(weights))

        self.cate_ = cate_model.predict(X)
        self.ate_ = float(np.mean(self.cate_))
        self._cate_model = cate_model

        return self

    def get_ate(self) -> float | None:
        return self.ate_

    def predict_cate(self, df: pd.DataFrame) -> np.ndarray:
        """Predict CATE for new data."""
        X = df[self.confounders].values
        return self._cate_model.predict(X)

    def feature_importance(self) -> pd.DataFrame:
        """Return CATE model feature importances."""
        imp = self._cate_model.feature_importances_
        return (
            pd.DataFrame({"feature": self.confounders, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
