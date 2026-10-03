"""Policy simulation and counterfactual analysis.

All outputs are clearly labelled as *model-based counterfactual
simulations*, not observed real-world outcomes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def simulate_policy(
    df: pd.DataFrame,
    surge_cap: float = 3.0,
    discount_pct_override: float | None = None,
    target_segment: str | None = None,
    causal_effects: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run a counterfactual policy simulation.

    Parameters
    ----------
    df : pd.DataFrame
        The original (observed) dataset.
    surge_cap : float
        Maximum allowed surge multiplier.
    discount_pct_override : float, optional
        If set, replace discount_pct for all (or targeted) trips.
    target_segment : str, optional
        Income segment to apply discount to (``"low"`` / ``"medium"`` / ``"high"``).
    causal_effects : dict, optional
        Ground-truth causal parameters. Defaults to the standard DGP values.

    Returns
    -------
    dict with scenario metrics.
    """
    fx = causal_effects or {
        "surge_on_booking": -0.12,
        "discount_on_booking": 0.15,
        "surge_on_cancellation": 0.08,
        "discount_on_cancellation": -0.05,
    }

    sim = df.copy()

    # Apply surge cap
    sim["cf_surge_mult"] = np.minimum(sim["surge_multiplier"], surge_cap)

    # Apply discount override
    sim["cf_discount_pct"] = sim["discount_pct"].copy()
    if discount_pct_override is not None:
        mask = np.ones(len(sim), dtype=bool)
        if target_segment is not None:
            mask = sim["income_segment"] == target_segment
        sim.loc[mask, "cf_discount_pct"] = discount_pct_override

    # Re-compute counterfactual fare
    sim["cf_final_fare"] = sim["base_fare"] * sim["cf_surge_mult"] * (1 - sim["cf_discount_pct"])

    # Re-compute booking probability using the *same* DGP formula
    logit_book = (
        0.5
        + 0.8 * sim["demand_intensity"]
        - 0.3 * sim["weather"]
        + fx["surge_on_booking"] * (sim["cf_surge_mult"] - 1.0) * sim["price_sensitivity"]
        + fx["discount_on_booking"] * sim["cf_discount_pct"] * sim["price_sensitivity"]
        - 0.04 * sim["distance_km"]
        + 0.3 * sim["loyalty_score"]
        - 0.1 * sim["eta"] / 10
    )
    sim["cf_booking_prob"] = _sigmoid(logit_book)

    # Cancellation probability
    logit_cancel = (
        -2.0
        + fx["surge_on_cancellation"] * (sim["cf_surge_mult"] - 1.0) * sim["price_sensitivity"]
        + fx["discount_on_cancellation"] * sim["cf_discount_pct"] * sim["price_sensitivity"]
        + 0.3 * sim["eta"] / 10
        - 0.2 * sim["loyalty_score"]
    )
    sim["cf_cancel_prob"] = _sigmoid(logit_cancel)

    # Expected outcomes (deterministic — use probabilities, not random draws)
    sim["cf_completed_prob"] = sim["cf_booking_prob"] * (1 - sim["cf_cancel_prob"])
    sim["cf_revenue"] = sim["cf_completed_prob"] * sim["cf_final_fare"] * 0.25
    sim["cf_driver_earnings"] = sim["cf_completed_prob"] * sim["cf_final_fare"] * 0.75

    # Compare to observed
    obs_booking = df["booking"].mean()
    obs_cancel = df["cancellation"].mean()
    obs_completed = df["completed"].mean()
    obs_revenue = df["platform_revenue"].mean()
    obs_driver = df["driver_earnings"].mean()

    cf_booking = float(sim["cf_booking_prob"].mean())
    cf_cancel = float(sim["cf_cancel_prob"].mean())
    cf_completed = float(sim["cf_completed_prob"].mean())
    cf_revenue = float(sim["cf_revenue"].mean())
    cf_driver = float(sim["cf_driver_earnings"].mean())

    return {
        "scenario": {
            "surge_cap": surge_cap,
            "discount_pct_override": discount_pct_override,
            "target_segment": target_segment,
        },
        "observed": {
            "booking_rate": float(obs_booking),
            "cancellation_rate": float(obs_cancel),
            "completion_rate": float(obs_completed),
            "avg_revenue": float(obs_revenue),
            "avg_driver_earnings": float(obs_driver),
        },
        "counterfactual": {
            "booking_rate": cf_booking,
            "cancellation_rate": cf_cancel,
            "completion_rate": cf_completed,
            "avg_revenue": cf_revenue,
            "avg_driver_earnings": cf_driver,
        },
        "delta": {
            "booking_rate": cf_booking - float(obs_booking),
            "cancellation_rate": cf_cancel - float(obs_cancel),
            "completion_rate": cf_completed - float(obs_completed),
            "avg_revenue": cf_revenue - float(obs_revenue),
            "avg_driver_earnings": cf_driver - float(obs_driver),
        },
        "_note": "These are MODEL-BASED counterfactual simulations, not observed outcomes.",
    }


def run_standard_scenarios(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Run the five standard policy scenarios."""
    scenarios = [
        {"surge_cap": 1.0, "discount_pct_override": None, "target_segment": None},  # No surge
        {"surge_cap": 3.0, "discount_pct_override": None, "target_segment": None},  # Status quo
        {"surge_cap": 1.5, "discount_pct_override": None, "target_segment": None},  # Reduced cap
        {"surge_cap": 3.0, "discount_pct_override": 0.15, "target_segment": None},  # Universal 15% discount
        {"surge_cap": 3.0, "discount_pct_override": 0.20, "target_segment": "low"},  # Targeted discount
    ]
    labels = [
        "A: No Surge",
        "B: Current Policy",
        "C: Surge Cap 1.5×",
        "D: Universal 15% Discount",
        "E: Targeted 20% Discount (Low Income)",
    ]
    results = []
    for label, kw in zip(labels, scenarios):
        res = simulate_policy(df, **kw)
        res["label"] = label
        results.append(res)
    return results
