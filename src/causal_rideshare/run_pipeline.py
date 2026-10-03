"""Complete execution pipeline for the causal inference project."""

import json
from pathlib import Path

import pandas as pd

from causal_rideshare.data_generation import generate_data
from causal_rideshare.config import load_config
from causal_rideshare.estimators.naive import NaiveEstimator
from causal_rideshare.estimators.regression import RegressionEstimator
from causal_rideshare.estimators.ipw import IPWEstimator
from causal_rideshare.estimators.doubly_robust import DoublyRobustEstimator
from causal_rideshare.estimators.matching import MatchingEstimator
from causal_rideshare.estimators.dml import DMLEstimator
from causal_rideshare.estimators.causal_forest import CausalForestEstimator

from causal_rideshare.refutation import (
    random_common_cause,
    placebo_treatment,
    data_subset_validation
)


def main():
    root = Path(__file__).resolve().parent.parent.parent
    res_dir = root / "reports" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    print("1. Generating Data...")
    df = generate_data()
    
    config = load_config()
    true_fx = config["causal_effects"]
    gt = true_fx["surge_on_booking"]
    print(f"   Ground Truth ATE: {gt:.4f}")

    print("\n2. Estimating ATE across methods...")
    estimators = {
        "Naive": NaiveEstimator(),
        "Regression": RegressionEstimator(),
        "PSM": MatchingEstimator(),
        "IPW": IPWEstimator(),
        "Doubly Robust": DoublyRobustEstimator(),
        "Double ML": DMLEstimator(),
        "Causal Forest": CausalForestEstimator(),
    }

    results = {"Ground Truth": gt}
    for name, est in estimators.items():
        print(f"   Fitting {name}...")
        est.fit(df)
        val = est.get_ate()
        results[name] = val
        print(f"      Estimate: {val:.4f} (Error: {val - gt:.4f})")

    out_path = res_dir / "ate_estimates.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n   Results saved to {out_path}")
    
    print("\n3. Running Refutation Tests (on Doubly Robust)...")
    print("   Random Common Cause:")
    rcc = random_common_cause(df)
    print(f"      Passed: {rcc['pass']} (Max Deviation: {rcc['max_deviation']:.4f})")
    
    print("   Placebo Treatment:")
    pla = placebo_treatment(df)
    print(f"      Passed: {pla['pass']} (Mean ATE: {pla['mean_placebo_ate']:.4f})")
    
    print("   Data Subset Validation:")
    sub = data_subset_validation(df)
    print(f"      Passed: {sub['pass']} (Std: {sub['std']:.4f})")
    
    print("\nPipeline Complete!")


if __name__ == "__main__":
    main()
