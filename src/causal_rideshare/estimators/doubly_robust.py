"""Doubly Robust / Augmented IPW estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

from causal_rideshare.dag import SURGE_CONFOUNDERS


class DoublyRobustEstimator:
    """Augmented Inverse Probability Weighting (AIPW).

    Consistent if *either* the propensity model or the outcome model is
    correctly specified — hence "doubly robust".

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
    ):
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders or SURGE_CONFOUNDERS
        self.ps_model = LogisticRegression(max_iter=2000, C=1.0)
        self.outcome_model = LinearRegression()
        self.ate_: float | None = None
        self.se_: float | None = None

    def fit(self, df: pd.DataFrame) -> "DoublyRobustEstimator":
        X = df[self.confounders].values
        T = df[self.treatment].values
        Y = df[self.outcome].values
        n = len(Y)

        # 1. Propensity model
        self.ps_model.fit(X, T)
        e = self.ps_model.predict_proba(X)[:, 1]
        e = np.clip(e, 1e-4, 1 - 1e-4)

        # 2. Outcome model  E[Y | X, T]
        XT = np.column_stack([X, T])
        self.outcome_model.fit(XT, Y)

        # Counterfactual predictions
        mu1 = self.outcome_model.predict(np.column_stack([X, np.ones(n)]))
        mu0 = self.outcome_model.predict(np.column_stack([X, np.zeros(n)]))

        # 3. AIPW score
        dr1 = mu1 + T * (Y - mu1) / e
        dr0 = mu0 + (1 - T) * (Y - mu0) / (1 - e)
        psi = dr1 - dr0

        self.ate_ = float(np.mean(psi))
        self.se_ = float(np.sqrt(np.var(psi) / n))
        return self

    def get_ate(self) -> float | None:
        return self.ate_

    def get_se(self) -> float | None:
        return self.se_
