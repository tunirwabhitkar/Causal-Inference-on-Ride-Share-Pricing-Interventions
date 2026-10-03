"""Heterogeneous treatment effect (CATE) analysis.

Slices estimated CATEs by rider segment, demand level, time-of-day,
zone demand, and weather to answer: *"For whom does surge pricing
have the largest negative effect?"*
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_cate_by_group(
    df: pd.DataFrame,
    cate: np.ndarray,
    group_col: str,
) -> pd.DataFrame:
    """Compute mean CATE and 95% CI within each level of *group_col*.

    Parameters
    ----------
    df : pd.DataFrame
        Must have the same length as *cate*.
    cate : np.ndarray
        Per-observation CATE predictions.
    group_col : str
        Column to group by.

    Returns
    -------
    pd.DataFrame
        Columns: group, mean_cate, se, ci_lower, ci_upper, n
    """
    tmp = df[[group_col]].copy()
    tmp["cate"] = cate
    agg = tmp.groupby(group_col)["cate"].agg(["mean", "std", "count"])
    agg.columns = ["mean_cate", "std_cate", "n"]
    agg["se"] = agg["std_cate"] / np.sqrt(agg["n"])
    agg["ci_lower"] = agg["mean_cate"] - 1.96 * agg["se"]
    agg["ci_upper"] = agg["mean_cate"] + 1.96 * agg["se"]
    return agg.reset_index().rename(columns={group_col: "group"})


def add_binned_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical bins for continuous variables used in HTE analysis."""
    out = df.copy()

    # Price sensitivity terciles
    out["ps_group"] = pd.qcut(
        out["price_sensitivity"], 3, labels=["Low", "Medium", "High"]
    )

    # Demand level
    out["demand_level"] = pd.cut(
        out["demand_intensity"],
        bins=[0, 1.5, 2.5, 5.0],
        labels=["Low", "Medium", "High"],
    )

    # Time of day
    def _time_bucket(h: int) -> str:
        if 6 <= h < 12:
            return "Morning"
        if 12 <= h < 17:
            return "Afternoon"
        if 17 <= h < 21:
            return "Evening"
        return "Night"

    out["time_of_day"] = out["hour_of_day"].apply(_time_bucket)

    # Zone demand bucket (by origin zone median demand)
    zone_med = out.groupby("origin_zone")["demand_intensity"].transform("median")
    out["zone_demand"] = pd.cut(
        zone_med, bins=[0, 1.5, 2.5, 5.0], labels=["Low", "Medium", "High"]
    )

    # Weather label
    weather_map = {0: "Clear", 1: "Rain", 2: "Extreme"}
    out["weather_label"] = out["weather"].map(weather_map)

    return out


def full_heterogeneity_report(
    df: pd.DataFrame, cate: np.ndarray
) -> dict[str, pd.DataFrame]:
    """Run CATE analysis across all standard slices.

    Returns dict mapping slice name → DataFrame.
    """
    df_b = add_binned_columns(df)
    slices = {
        "Price Sensitivity": "ps_group",
        "Demand Level": "demand_level",
        "Time of Day": "time_of_day",
        "Zone Demand": "zone_demand",
        "Weather": "weather_label",
        "Income Segment": "income_segment",
    }
    return {name: compute_cate_by_group(df_b, cate, col) for name, col in slices.items()}
