"""Synthetic ride-share data generator with known causal ground truth.

The Data-Generating Process (DGP) intentionally introduces confounding so
that naive observational estimates are biased upward, while the *true*
causal effects of surge pricing and discounts are encoded as configurable
parameters.

Key confounding mechanism
-------------------------
``demand_intensity`` drives **both** the probability of surge pricing
*and* the probability of booking.  A naive comparison of booking rates
between surge / no-surge groups will therefore be positively biased
(Simpson's-paradox style).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causal_rideshare.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_data(
    config_path: str | Path | None = None,
    output_path: str | Path = "data/processed/simulated_rides.csv",
) -> pd.DataFrame:
    """Generate a synthetic ride-share dataset.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the YAML config.  Defaults to ``configs/simulation.yaml``.
    output_path : str or Path
        Where to write the CSV.

    Returns
    -------
    pd.DataFrame
    """
    config = load_config(config_path)
    sim = config["simulation"]
    fx = config["causal_effects"]

    n_trips: int = sim["n_trips"]
    n_zones: int = sim["n_zones"]
    seed: int = sim["seed"]
    rng = np.random.RandomState(seed)

    logger.info("Generating %d trips (seed=%d) …", n_trips, seed)

    # ------------------------------------------------------------------
    # 1. Context / Confounders
    # ------------------------------------------------------------------
    weather = rng.choice([0, 1, 2], size=n_trips, p=[0.65, 0.25, 0.10])
    temperature = rng.normal(loc=22 - 5 * weather, scale=4, size=n_trips)
    precipitation = (weather >= 1).astype(float) * rng.exponential(5, size=n_trips)

    hour_of_day = rng.randint(0, 24, size=n_trips)
    is_weekend = rng.binomial(1, 2 / 7, size=n_trips)
    is_holiday = rng.binomial(1, 0.03, size=n_trips)
    local_event = rng.binomial(1, 0.05, size=n_trips)

    rush = (
        ((hour_of_day >= 7) & (hour_of_day <= 9))
        | ((hour_of_day >= 17) & (hour_of_day <= 19))
    ).astype(float)

    origin_zone = rng.randint(0, n_zones, size=n_trips)
    dest_zone = rng.randint(0, n_zones, size=n_trips)
    zone_demand_factor = rng.uniform(0.5, 2.0, size=n_zones)

    # Traffic level (1-5), correlated with rush hour
    traffic_level = np.clip(
        rng.normal(loc=2 + 2 * rush + 0.5 * weather, scale=0.8), 1, 5
    ).astype(int)

    demand_intensity = np.clip(
        rng.normal(
            loc=(
                1.0
                + 1.5 * rush
                + 0.8 * weather
                + 0.5 * is_weekend
                + 1.0 * local_event
                + 0.3 * zone_demand_factor[origin_zone]
            ),
            scale=0.5,
        ),
        0.1,
        5.0,
    )

    driver_supply = np.clip(
        rng.normal(loc=50 - 10 * rush + 5 * is_weekend, scale=10), 5, 100
    )
    supply_demand_ratio = driver_supply / (demand_intensity * 30 + 1)

    # ------------------------------------------------------------------
    # 2. Rider features
    # ------------------------------------------------------------------
    rider_id = rng.randint(1, sim.get("n_riders", 100_000) + 1, size=n_trips)
    age_group = rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], size=n_trips,
                           p=[0.15, 0.35, 0.25, 0.15, 0.10])
    income_segment = rng.choice(["low", "medium", "high"], size=n_trips,
                                p=[0.30, 0.45, 0.25])
    price_sensitivity = _price_sensitivity_from_income(income_segment, rng)
    loyalty_score = rng.beta(2, 5, size=n_trips)
    hist_trip_freq = rng.poisson(8, size=n_trips)

    # ------------------------------------------------------------------
    # 3. Driver features
    # ------------------------------------------------------------------
    driver_id = rng.randint(1, sim.get("n_drivers", 20_000) + 1, size=n_trips)
    driver_experience = rng.poisson(3, size=n_trips)  # years
    driver_rating = np.clip(rng.normal(4.7, 0.3, size=n_trips), 1, 5)
    driver_acceptance_rate = np.clip(rng.beta(8, 2, size=n_trips), 0, 1)

    # ------------------------------------------------------------------
    # 4. Trip features
    # ------------------------------------------------------------------
    distance_km = np.clip(rng.lognormal(1.5, 0.8, size=n_trips), 0.5, 50.0)
    estimated_duration = distance_km * (2.5 + 0.4 * traffic_level)  # minutes
    eta = np.clip(rng.exponential(5 / supply_demand_ratio), 1, 30)  # minutes

    base_fare = 2.5 + 1.5 * distance_km

    # ------------------------------------------------------------------
    # 5. Treatment assignment — surge (confounded by demand)
    # ------------------------------------------------------------------
    surge_logit = -2.0 + 1.2 * demand_intensity - 0.5 * supply_demand_ratio
    surge_prob = _sigmoid(surge_logit)
    is_surge = rng.binomial(1, surge_prob)

    surge_multiplier = np.ones(n_trips)
    n_surged = int(is_surge.sum())
    surge_multiplier[is_surge == 1] = (
        1.0
        + 0.2 * demand_intensity[is_surge == 1]
        + rng.uniform(0.1, 0.5, size=n_surged)
    )
    surge_multiplier = np.clip(surge_multiplier, 1.0, 3.0)

    # Discount treatment — quasi-random (platform A/B test style)
    has_discount = rng.binomial(1, 0.20, size=n_trips)
    discount_pct = has_discount * rng.uniform(0.05, 0.30, size=n_trips)

    final_fare = base_fare * surge_multiplier * (1 - discount_pct)

    # Binary treatment flag for primary analysis
    treatment = is_surge.copy()

    # ------------------------------------------------------------------
    # 6. Outcomes — booking, cancellation, revenue
    # ------------------------------------------------------------------
    surge_effect: float = fx["surge_on_booking"]
    disc_effect: float = fx["discount_on_booking"]
    surge_cancel: float = fx["surge_on_cancellation"]
    disc_cancel: float = fx["discount_on_cancellation"]

    logit_book = (
        0.5
        + 0.8 * demand_intensity
        - 0.3 * weather
        + surge_effect * (surge_multiplier - 1.0) * price_sensitivity
        + disc_effect * discount_pct * price_sensitivity
        - 0.04 * distance_km
        + 0.3 * loyalty_score
        - 0.1 * eta / 10
    )
    booking_prob = _sigmoid(logit_book)
    booking = rng.binomial(1, booking_prob)

    logit_cancel = (
        -2.0
        + surge_cancel * (surge_multiplier - 1.0) * price_sensitivity
        + disc_cancel * discount_pct * price_sensitivity
        + 0.3 * eta / 10
        - 0.2 * loyalty_score
    )
    cancel_prob = _sigmoid(logit_cancel)
    cancellation = rng.binomial(1, cancel_prob) * booking  # can only cancel if booked

    completed = booking * (1 - cancellation)

    platform_revenue = completed * final_fare * 0.25
    driver_earnings_val = completed * final_fare * 0.75

    # ------------------------------------------------------------------
    # 7. Ground-truth individual treatment effects (for evaluation)
    # ------------------------------------------------------------------
    # Marginal effect of is_surge on booking probability
    # dP/d(surge_mult) * (surge_mult - 1)
    true_ite_booking = (
        surge_effect * price_sensitivity * booking_prob * (1 - booking_prob)
    )

    # ------------------------------------------------------------------
    # 8. Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "trip_id": np.arange(1, n_trips + 1),
            # Rider
            "rider_id": rider_id,
            "age_group": age_group,
            "income_segment": income_segment,
            "price_sensitivity": price_sensitivity,
            "loyalty_score": loyalty_score,
            "hist_trip_freq": hist_trip_freq,
            # Driver
            "driver_id": driver_id,
            "driver_experience": driver_experience,
            "driver_rating": driver_rating,
            "driver_acceptance_rate": driver_acceptance_rate,
            # Context
            "hour_of_day": hour_of_day,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "local_event": local_event,
            "weather": weather,
            "temperature": temperature,
            "precipitation": precipitation,
            "traffic_level": traffic_level,
            "demand_intensity": demand_intensity,
            "driver_supply": driver_supply,
            "supply_demand_ratio": supply_demand_ratio,
            # Trip
            "origin_zone": origin_zone,
            "destination_zone": dest_zone,
            "distance_km": distance_km,
            "estimated_duration": estimated_duration,
            "eta": eta,
            "base_fare": base_fare,
            # Treatment
            "is_surge": is_surge,
            "surge_multiplier": surge_multiplier,
            "has_discount": has_discount,
            "discount_pct": discount_pct,
            "final_fare": final_fare,
            "treatment": treatment,
            # Outcomes
            "booking": booking,
            "cancellation": cancellation,
            "completed": completed,
            "platform_revenue": platform_revenue,
            "driver_earnings": driver_earnings_val,
            # Ground truth
            "true_booking_prob": booking_prob,
            "true_ite_booking": true_ite_booking,
        }
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Wrote %d rows → %s", len(df), out)
    print(f"Data generated: {len(df)} rows -> {out}")
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _price_sensitivity_from_income(
    income: np.ndarray, rng: np.random.RandomState
) -> np.ndarray:
    """Higher-income riders are less price-sensitive."""
    base = np.where(
        income == "low",
        1.5,
        np.where(income == "medium", 1.0, 0.6),
    )
    return np.clip(base + rng.normal(0, 0.3, size=len(income)), 0.1, 3.0)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_data()
