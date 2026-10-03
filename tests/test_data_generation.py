import pytest
import os
import pandas as pd
from causal_rideshare.data_generation import generate_data
import tempfile

def test_generate_data():
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdirname:
        output_path = os.path.join(tmpdirname, "simulated_rides.csv")
        df = generate_data(output_path=output_path)
        
        assert os.path.exists(output_path)
        assert len(df) == 100000
        assert "is_surge" in df.columns
        assert "booking" in df.columns
        assert "true_surge_effect_on_prob" in df.columns
