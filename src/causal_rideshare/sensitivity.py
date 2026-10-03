"""Sensitivity analysis for unobserved confounding.

Implements:
  1. A simplified Rosenbaum-bounds–style analysis.
  2. A parametric bias analysis (Oster-style partial R²).
  3. Bootstrap confidence intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from causal_rideshare.estimators.doubly_robust import DoublyRobustEstimator
from causal_rideshare.dag import SURGE_CONFOUNDERS


def rosenbaum_bounds(
    df: pd.DataFrame,
    treatment: str = "is_surge",
    outcome: str = "booking",
    gamma_values: list[float] | None = None,
) -> pd.DataFrame:
    """Simplified Rosenbaum sensitivity analysis.

    For each Γ (odds-ratio of differential treatment assignment due to
    an unobserved confounder), compute the upper and lower bounds on
    the p-value for the treatment effect.

    Parameters
    ----------
    gamma_values : list of float
        Values of Γ ≥ 1.0 to evaluate.

    Returns
    -------
    pd.DataFrame
        Columns: gamma, p_lower, p_upper, conclusion
    """
    if gamma_values is None:
        gamma_values = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0]

    T = df[treatment].values
    Y = df[outcome].values
    n1 = T.sum()
    n0 = len(T) - n1
    diff = Y[T == 1].mean() - Y[T == 0].mean()
    se = np.sqrt(Y[T == 1].var() / n1 + Y[T == 0].var() / n0)

    rows = []
    for g in gamma_values:
        # Bounds on the treatment-assignment probability shift
        shift = np.log(g) * se
        z_lower = (diff - shift) / se
        z_upper = (diff + shift) / se
        p_lower = float(stats.norm.sf(z_upper))
        p_upper = float(stats.norm.sf(z_lower))
        sig = "significant" if p_upper < 0.05 else "not significant"
        rows.append({
            "gamma": g,
            "p_lower": round(p_lower, 4),
            "p_upper": round(p_upper, 4),
            "conclusion": sig,
        })
    return pd.DataFrame(rows)


def bootstrap_ate(
    df: pd.DataFrame,
    estimator_cls=DoublyRobustEstimator,
    n_bootstrap: int = 200,
    seed: int = 42,
    **estimator_kwargs,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for the ATE.

    Returns
    -------
    dict with keys: ate, se, ci_lower, ci_upper
    """
    rng = np.random.RandomState(seed)
    estimates = []
    n = len(df)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sample = df.iloc[idx].reset_index(drop=True)
        est = estimator_cls(**estimator_kwargs)
        est.fit(sample)
        estimates.append(est.get_ate())

    estimates = np.array(estimates)
    return {
        "ate": float(np.mean(estimates)),
        "se": float(np.std(estimates)),
        "ci_lower": float(np.percentile(estimates, 2.5)),
        "ci_upper": float(np.percentile(estimates, 97.5)),
    }


def partial_r2_sensitivity(
    df: pd.DataFrame,
    treatment: str = "is_surge",
    outcome: str = "booking",
    confounders: list[str] | None = None,
) -> dict[str, float]:
    """Oster-style sensitivity: how much partial R² from an unobserved
    confounder would be needed to drive the estimate to zero.

    Returns the critical value of partial R² (treatment → outcome | X).
    """
    import statsmodels.api as sm

    confounders = confounders or SURGE_CONFOUNDERS

    # Short regression (without confounders)
    X_short = sm.add_constant(df[[treatment]])
    r_short = sm.OLS(df[outcome], X_short).fit()
    r2_short = r_short.rsquared
    beta_short = r_short.params[treatment]

    # Long regression (with confounders)
    X_long = sm.add_constant(df[[treatment] + confounders])
    r_long = sm.OLS(df[outcome], X_long).fit()
    r2_long = r_long.rsquared
    beta_long = r_long.params[treatment]

    # Oster δ: ratio of selection on unobservables vs observables
    # that would drive β to zero, given proportional selection
    if abs(beta_short - beta_long) < 1e-12:
        delta = float("inf")
    else:
        delta = beta_long / (beta_short - beta_long)

    return {
        "beta_short": float(beta_short),
        "beta_long": float(beta_long),
        "r2_short": float(r2_short),
        "r2_long": float(r2_long),
        "oster_delta": float(delta),
        "interpretation": (
            f"An unobserved confounder would need δ = {delta:.2f} "
            f"(relative to observables) to nullify the estimate."
        ),
    }
