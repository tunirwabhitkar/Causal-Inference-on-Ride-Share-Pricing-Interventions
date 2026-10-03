"""Causal refutation tests.

Implements:
  1. Random Common Cause — add a random confounder and re-estimate.
  2. Placebo Treatment — replace treatment with random noise.
  3. Data Subset Validation — re-estimate on random subsets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_rideshare.estimators.doubly_robust import DoublyRobustEstimator


def random_common_cause(
    df: pd.DataFrame,
    estimator_cls=DoublyRobustEstimator,
    n_trials: int = 10,
    seed: int = 42,
    **est_kwargs,
) -> dict[str, float]:
    """Add a random variable to the confounder set and re-estimate.

    If the estimate changes substantially, the original estimate may
    be fragile.  If it stays stable, we gain (limited) confidence.
    """
    rng = np.random.RandomState(seed)
    baseline = estimator_cls(**est_kwargs)
    baseline.fit(df)
    base_ate = baseline.get_ate()

    perturbed = []
    for i in range(n_trials):
        df_aug = df.copy()
        df_aug["_random_cause"] = rng.normal(size=len(df))
        # Add random variable to confounders
        kw = dict(est_kwargs)
        conf = list(kw.get("confounders", baseline.confounders)) + ["_random_cause"]
        kw["confounders"] = conf
        est = estimator_cls(**kw)
        est.fit(df_aug)
        perturbed.append(est.get_ate())

    perturbed = np.array(perturbed)
    return {
        "baseline_ate": float(base_ate),
        "mean_perturbed_ate": float(np.mean(perturbed)),
        "std_perturbed_ate": float(np.std(perturbed)),
        "max_deviation": float(np.max(np.abs(perturbed - base_ate))),
        "pass": bool(np.max(np.abs(perturbed - base_ate)) < abs(base_ate) * 0.5),
    }


def placebo_treatment(
    df: pd.DataFrame,
    estimator_cls=DoublyRobustEstimator,
    n_trials: int = 10,
    seed: int = 42,
    **est_kwargs,
) -> dict[str, float]:
    """Replace the actual treatment with a random placebo.

    Expected result: estimated effect ≈ 0.
    """
    rng = np.random.RandomState(seed)
    treat_col = est_kwargs.get("treatment", "is_surge")
    treat_rate = df[treat_col].mean()

    estimates = []
    for _ in range(n_trials):
        df_pla = df.copy()
        df_pla[treat_col] = rng.binomial(1, treat_rate, size=len(df))
        est = estimator_cls(**est_kwargs)
        est.fit(df_pla)
        estimates.append(est.get_ate())

    estimates = np.array(estimates)
    return {
        "mean_placebo_ate": float(np.mean(estimates)),
        "std_placebo_ate": float(np.std(estimates)),
        "pass": bool(abs(np.mean(estimates)) < 0.02),
    }


def data_subset_validation(
    df: pd.DataFrame,
    estimator_cls=DoublyRobustEstimator,
    n_subsets: int = 5,
    frac: float = 0.7,
    seed: int = 42,
    **est_kwargs,
) -> dict[str, float]:
    """Estimate ATE on multiple random subsets.

    The estimates should be reasonably stable across subsets.
    """
    rng = np.random.RandomState(seed)
    estimates = []
    for _ in range(n_subsets):
        sub = df.sample(frac=frac, random_state=rng.randint(0, 1_000_000))
        est = estimator_cls(**est_kwargs)
        est.fit(sub)
        estimates.append(est.get_ate())

    estimates = np.array(estimates)
    return {
        "estimates": estimates.tolist(),
        "mean": float(np.mean(estimates)),
        "std": float(np.std(estimates)),
        "range": float(np.ptp(estimates)),
        "pass": bool(np.std(estimates) < abs(np.mean(estimates)) * 0.5),
    }
