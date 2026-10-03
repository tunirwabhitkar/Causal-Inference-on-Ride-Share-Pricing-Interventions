"""Data validation and quality checks for the ride-share dataset."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_quality_checks(df: pd.DataFrame) -> dict[str, Any]:
    """Run automated data-quality checks.

    Returns a dict summarising every check (pass/fail + detail).
    """
    report: dict[str, Any] = {}

    # 1. Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].to_dict()
    report["missing_values"] = {
        "pass": len(missing_cols) == 0,
        "detail": missing_cols if missing_cols else "No missing values",
    }

    # 2. Duplicate trip IDs
    n_dup = df["trip_id"].duplicated().sum()
    report["duplicate_trip_ids"] = {
        "pass": n_dup == 0,
        "detail": f"{n_dup} duplicates found",
    }

    # 3. Invalid fares (negative or zero)
    bad_fare = (df["final_fare"] <= 0).sum()
    report["invalid_fares"] = {
        "pass": int(bad_fare) == 0,
        "detail": f"{bad_fare} rows with fare <= 0",
    }

    # 4. Negative distance
    bad_dist = (df["distance_km"] < 0).sum()
    report["negative_distance"] = {
        "pass": int(bad_dist) == 0,
        "detail": f"{bad_dist} rows with distance < 0",
    }

    # 5. Impossible surge multiplier
    bad_surge = ((df["surge_multiplier"] < 1.0) | (df["surge_multiplier"] > 5.0)).sum()
    report["impossible_surge"] = {
        "pass": int(bad_surge) == 0,
        "detail": f"{bad_surge} rows with surge outside [1, 5]",
    }

    # 6. Treatment imbalance
    treat_rate = df["is_surge"].mean()
    report["treatment_imbalance"] = {
        "pass": 0.05 < treat_rate < 0.95,
        "detail": f"Treatment rate = {treat_rate:.3f}",
    }

    # 7. Hour range
    bad_hour = ((df["hour_of_day"] < 0) | (df["hour_of_day"] > 23)).sum()
    report["invalid_hour"] = {
        "pass": int(bad_hour) == 0,
        "detail": f"{bad_hour} rows with hour outside [0, 23]",
    }

    # 8. Booking / cancellation consistency
    cancel_without_book = ((df["cancellation"] == 1) & (df["booking"] == 0)).sum()
    report["cancel_without_booking"] = {
        "pass": int(cancel_without_book) == 0,
        "detail": f"{cancel_without_book} rows cancelled without booking",
    }

    # 9. Completed consistency
    bad_completed = (
        (df["completed"] != (df["booking"] * (1 - df["cancellation"])))
    ).sum()
    report["completed_consistency"] = {
        "pass": int(bad_completed) == 0,
        "detail": f"{bad_completed} inconsistent rows",
    }

    # 10. Outliers — flag extreme propensity scores if column exists
    if "true_booking_prob" in df.columns:
        extreme = ((df["true_booking_prob"] < 0.01) | (df["true_booking_prob"] > 0.99)).sum()
        report["extreme_booking_prob"] = {
            "pass": int(extreme) < len(df) * 0.05,
            "detail": f"{extreme} rows with booking prob outside (0.01, 0.99)",
        }

    n_pass = sum(1 for v in report.values() if v["pass"])
    n_total = len(report)
    logger.info("Data quality: %d/%d checks passed", n_pass, n_total)
    report["_summary"] = {"passed": n_pass, "total": n_total}
    return report


def print_quality_report(report: dict[str, Any]) -> None:
    """Pretty-print a quality report to stdout."""
    print("\n" + "=" * 60)
    print("  DATA QUALITY REPORT")
    print("=" * 60)
    for name, info in report.items():
        if name.startswith("_"):
            continue
        status = "✅ PASS" if info["pass"] else "❌ FAIL"
        print(f"  {status}  {name}: {info['detail']}")
    s = report["_summary"]
    print("-" * 60)
    print(f"  Result: {s['passed']}/{s['total']} checks passed")
    print("=" * 60 + "\n")
