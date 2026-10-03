"""Double / Debiased Machine Learning (DML) estimator.

Implements Chernozhukov et al. (2018) with cross-fitting using
``RandomForest`` for nuisance estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold

from causal_rideshare.dag import SURGE_CONFOUNDERS


class DMLEstimator:
    """Double Machine Learning (Partially Linear Model).

    Attributes
    ----------
    ate_ : float
    se_ : float
    """

    def __init__(
        self,
        treatment: str = "is_surge",
        outcome: str = "booking",
        confounders: list[str] | None = None,
        n_splits: int = 3,
        seed: int = 42,
    ):
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders or SURGE_CONFOUNDERS
        self.n_splits = n_splits
        self.seed = seed
        self.ate_: float | None = None
        self.se_: float | None = None

    def fit(self, df: pd.DataFrame) -> "DMLEstimator":
        X = df[self.confounders].values
        T = df[self.treatment].values.astype(float)
        Y = df[self.outcome].values.astype(float)
        n = len(Y)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        t_res = np.zeros(n)
        y_res = np.zeros(n)

        for train_idx, test_idx in kf.split(X):
            # Treatment nuisance
            ps = RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=self.seed
            )
            ps.fit(X[train_idx], T[train_idx].astype(int))
            t_res[test_idx] = T[test_idx] - ps.predict_proba(X[test_idx])[:, 1]

            # Outcome nuisance
            om = RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=self.seed
            )
            om.fit(X[train_idx], Y[train_idx])
            y_res[test_idx] = Y[test_idx] - om.predict(X[test_idx])

        # Final regression: Y_res = theta * T_res + eps
        theta = float(np.sum(t_res * y_res) / np.sum(t_res ** 2))
        eps = y_res - theta * t_res
        var_theta = float(np.mean(eps ** 2 * t_res ** 2) / (np.mean(t_res ** 2) ** 2))
        se = float(np.sqrt(var_theta / n))

        self.ate_ = theta
        self.se_ = se
        return self

    def get_ate(self) -> float | None:
        return self.ate_

    def get_se(self) -> float | None:
        return self.se_
