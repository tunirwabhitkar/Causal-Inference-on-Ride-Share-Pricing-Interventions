import pytest
import pandas as pd
import numpy as np
from causal_rideshare.estimators.naive import NaiveEstimator
from causal_rideshare.estimators.regression import RegressionEstimator
from causal_rideshare.estimators.ipw import IPWEstimator

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "is_surge": np.random.binomial(1, 0.5, size=n),
        "weather": np.random.choice([0, 1], size=n),
        "hour_of_day": np.random.randint(0, 24, size=n),
        "demand_intensity": np.random.normal(1, 0.5, size=n),
        "distance_km": np.random.uniform(1, 10, size=n),
        "base_fare": np.random.uniform(5, 20, size=n)
    })
    # true ATE = -0.1
    df["booking"] = 0.5 - 0.1 * df["is_surge"] + 0.1 * df["demand_intensity"]
    return df

def test_naive(dummy_data):
    est = NaiveEstimator()
    est.fit(dummy_data)
    assert est.get_ate() is not None

def test_regression(dummy_data):
    est = RegressionEstimator()
    est.fit(dummy_data)
    assert est.get_ate() is not None

def test_ipw(dummy_data):
    est = IPWEstimator()
    est.fit(dummy_data)
    assert est.get_ate() is not None
