"""Naive difference-in-means estimator (biased baseline)."""

from __future__ import annotations

import numpy as np
import pandas as pd


class NaiveEstimator:
    """Compute a simple difference in means — no confounder adjustment.

    This estimator is intentionally biased in the presence of confounding
    and serves as the baseline that causal methods improve upon.
    """

    def __init__(self, treatment: str = "is_surge", outcome: str = "booking"):
        self.treatment = treatment
        self.outcome = outcome
        self.ate_: float | None = None
        self.se_: float | None = None

    def fit(self, df: pd.DataFrame) -> "NaiveEstimator":
        t = df[self.treatment].values
        y = df[self.outcome].values
        y1 = y[t == 1]
        y0 = y[t == 0]
        self.ate_ = float(y1.mean() - y0.mean())
        self.se_ = float(np.sqrt(y1.var() / len(y1) + y0.var() / len(y0)))
        return self

    def get_ate(self) -> float | None:
        return self.ate_

    def get_se(self) -> float | None:
        return self.se_
