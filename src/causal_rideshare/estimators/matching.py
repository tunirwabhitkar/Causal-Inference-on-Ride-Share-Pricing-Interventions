"""Propensity Score Matching estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from causal_rideshare.dag import SURGE_CONFOUNDERS


class MatchingEstimator:
    """Nearest-neighbour propensity-score matching.

    Attributes
    ----------
    ate_ : float
        Average Treatment Effect after matching.
    att_ : float
        Average Treatment Effect on the Treated.
    matched_df_ : pd.DataFrame
        Matched dataset with ``_match_id`` column.
    balance_ : pd.DataFrame
        Standardised mean differences before / after matching.
    """

    def __init__(
        self,
        treatment: str = "is_surge",
        outcome: str = "booking",
        confounders: list[str] | None = None,
        n_neighbors: int = 1,
        caliper: float | None = 0.05,
    ):
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders or SURGE_CONFOUNDERS
        self.n_neighbors = n_neighbors
        self.caliper = caliper

        self.ps_model = LogisticRegression(max_iter=2000, C=1.0)
        self.ate_: float | None = None
        self.att_: float | None = None
        self.matched_df_: pd.DataFrame | None = None
        self.balance_: pd.DataFrame | None = None
        self.propensity_scores_: np.ndarray | None = None

    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame) -> "MatchingEstimator":
        X = df[self.confounders].values
        T = df[self.treatment].values
        Y = df[self.outcome].values

        # Propensity model
        self.ps_model.fit(X, T)
        ps = self.ps_model.predict_proba(X)[:, 1]
        ps = np.clip(ps, 1e-4, 1 - 1e-4)
        self.propensity_scores_ = ps

        # Nearest-neighbour matching on logit(ps)
        logit_ps = np.log(ps / (1 - ps)).reshape(-1, 1)

        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]

        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")
        nn.fit(logit_ps[control_idx])
        distances, indices = nn.kneighbors(logit_ps[treated_idx])

        # Apply caliper
        mask = np.ones(len(treated_idx), dtype=bool)
        if self.caliper is not None:
            mask = distances[:, 0] <= self.caliper

        matched_treated = treated_idx[mask]
        matched_control = control_idx[indices[mask, 0]]

        # ATT
        y_t = Y[matched_treated]
        y_c = Y[matched_control]
        self.att_ = float(np.mean(y_t - y_c))
        self.ate_ = self.att_  # With 1:1 matching on treated → ATT ≈ ATE

        # Balance diagnostics
        self.balance_ = self._balance_table(df, matched_treated, matched_control)

        # Matched dataframe
        df_t = df.iloc[matched_treated].copy()
        df_c = df.iloc[matched_control].copy()
        df_t["_match_id"] = range(len(df_t))
        df_c["_match_id"] = range(len(df_c))
        df_t["_group"] = "treated"
        df_c["_group"] = "control"
        self.matched_df_ = pd.concat([df_t, df_c], ignore_index=True)

        return self

    def get_ate(self) -> float | None:
        return self.ate_

    # ------------------------------------------------------------------ #
    # Balance diagnostics
    # ------------------------------------------------------------------ #

    def _balance_table(
        self,
        df: pd.DataFrame,
        t_idx: np.ndarray,
        c_idx: np.ndarray,
    ) -> pd.DataFrame:
        rows = []
        for col in self.confounders:
            vals = df[col].values
            # Before matching
            mean_t_bef = vals[df[self.treatment].values == 1].mean()
            mean_c_bef = vals[df[self.treatment].values == 0].mean()
            std_bef = vals.std()
            smd_bef = (mean_t_bef - mean_c_bef) / (std_bef + 1e-8)

            # After matching
            mean_t_aft = vals[t_idx].mean()
            mean_c_aft = vals[c_idx].mean()
            smd_aft = (mean_t_aft - mean_c_aft) / (std_bef + 1e-8)

            rows.append(
                {
                    "covariate": col,
                    "smd_before": round(smd_bef, 4),
                    "smd_after": round(smd_aft, 4),
                }
            )
        return pd.DataFrame(rows)
