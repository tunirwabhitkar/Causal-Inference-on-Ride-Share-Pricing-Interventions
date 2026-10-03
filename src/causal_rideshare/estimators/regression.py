"""Regression adjustment estimator (Linear Probability Model)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causal_rideshare.dag import SURGE_CONFOUNDERS


class RegressionEstimator:
    """OLS regression controlling for observed confounders.

    Uses a Linear Probability Model so the coefficient on treatment
    directly estimates the ATE under the assumption of linearity.
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
        self.model_ = None
        self.ate_: float | None = None
        self.se_: float | None = None

    def fit(self, df: pd.DataFrame) -> "RegressionEstimator":
        X = df[[self.treatment] + self.confounders].copy()
        X = sm.add_constant(X, has_constant="add")
        y = df[self.outcome]

        self.model_ = sm.OLS(y.values, X.values).fit(cov_type="HC1")
        # Treatment coefficient is the second column (after constant)
        treat_idx = 1  # constant is 0, treatment is 1
        self.ate_ = float(self.model_.params[treat_idx])
        self.se_ = float(self.model_.bse[treat_idx])
        return self

    def get_ate(self) -> float | None:
        return self.ate_

    def get_se(self) -> float | None:
        return self.se_
