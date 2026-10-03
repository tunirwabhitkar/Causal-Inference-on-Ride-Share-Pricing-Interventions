"""Inverse Probability Weighting (IPW) estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from causal_rideshare.dag import SURGE_CONFOUNDERS


class IPWEstimator:
    """Horwitz–Thompson / Hájek IPW estimator.

    Attributes
    ----------
    ate_ : float
        Estimated ATE.
    se_ : float
        Standard error (sandwich estimator).
    propensity_scores_ : np.ndarray
        Estimated propensity scores.
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
        self.ate_: float | None = None
        self.se_: float | None = None
        self.propensity_scores_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "IPWEstimator":
        X = df[self.confounders].values
        T = df[self.treatment].values
        Y = df[self.outcome].values
        n = len(Y)

        self.ps_model.fit(X, T)
        e = self.ps_model.predict_proba(X)[:, 1]
        e = np.clip(e, 1e-4, 1 - 1e-4)
        self.propensity_scores_ = e

        # Hájek estimator (normalised weights)
        w1 = T / e
        w0 = (1 - T) / (1 - e)
        mu1 = np.sum(Y * w1) / np.sum(w1)
        mu0 = np.sum(Y * w0) / np.sum(w0)
        self.ate_ = float(mu1 - mu0)

        # Influence-function based SE
        psi = T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e) + mu1 - mu0
        self.se_ = float(np.sqrt(np.var(psi) / n))

        return self

    def get_ate(self) -> float | None:
        return self.ate_

    def get_se(self) -> float | None:
        return self.se_
