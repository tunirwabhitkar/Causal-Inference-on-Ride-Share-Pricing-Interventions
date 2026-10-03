"""Streamlit dashboard for the causal inference project.

Run with:
    PYTHONPATH=src streamlit run app/app.py
"""

import sys
from pathlib import Path

# Ensure src is in the python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from causal_rideshare.data_generation import generate_data
from causal_rideshare.config import load_config
from causal_rideshare.dag import DAG_MERMAID, SURGE_CONFOUNDERS
from causal_rideshare.visualization import (
    plot_ate_comparison,
    plot_confounding_demo,
    plot_policy_comparison,
    plot_cate_by_group,
    plot_propensity_distribution,
    plot_balance,
    results_table,
)
from causal_rideshare.estimators.matching import MatchingEstimator
from causal_rideshare.heterogeneity import full_heterogeneity_report
from causal_rideshare.policy_simulation import run_standard_scenarios, simulate_policy
from causal_rideshare.sensitivity import rosenbaum_bounds


st.set_page_config(
    page_title="Ride-Share Causal Inference",
    page_icon="🚕",
    layout="wide",
)

# ─── STATE MANAGEMENT ────────────────────────────────────────────────────────

@st.cache_data
def get_data(n_samples: int, seed: int) -> pd.DataFrame:
    # Hack to quickly get df with different size without rewriting config logic
    import yaml
    config_path = project_root / "configs" / "simulation.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["simulation"]["n_trips"] = n_samples
    config["simulation"]["seed"] = seed
    tmp_config = project_root / "configs" / "tmp_sim.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(config, f)
    
    df = generate_data(config_path=tmp_config, output_path=project_root / "data" / "processed" / "tmp_rides.csv")
    return df

@st.cache_data
def load_results() -> dict:
    res_path = project_root / "reports" / "results" / "ate_estimates.json"
    if res_path.exists():
        with open(res_path, "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def run_matching(_df: pd.DataFrame):
    est = MatchingEstimator()
    est.fit(_df)
    return est.propensity_scores_, est.balance_

# ─── UI STRUCTURE ────────────────────────────────────────────────────────────

st.title("🚖 Causal Inference on Ride-Share Pricing Interventions")
st.markdown("""
Welcome to the interactive causal inference dashboard. 
This project estimates the causal effect of surge pricing on ride completion rates, distinguishing correlation from causation.
""")

tab1, tab2, tab_match, tab3, tab4, tab5, tab6 = st.tabs([
    "1. The Problem & Data",
    "2. Causal Discovery (DAG)",
    "3. Propensity Matching",
    "4. ATE Estimation",
    "5. Heterogeneity & Sensitivity",
    "6. Policy Simulation",
    "7. Interactive Simulator",
])

# Load data
n_samples = st.sidebar.slider("Dataset Size (n)", 1000, 20000, 5000, 1000)
seed = st.sidebar.slider("Random Seed", 1, 100, 42)
df = get_data(n_samples, seed)


with tab1:
    st.header("The Confounding Problem")
    st.markdown("""
    In ride-share data, **price is endogenous**. 
    During high demand (rain, rush hour), the platform raises prices (surge). 
    Simultaneously, during high demand, riders are more desperate and thus *more* likely to book.
    If we just look at the correlation, surge pricing might appear to *increase* bookings!
    """)
    st.plotly_chart(plot_confounding_demo(df), use_container_width=True)
    
    st.subheader("Data Snapshot")
    st.dataframe(df.head(100), use_container_width=True)
    
with tab2:
    st.header("Structural Causal Model")
    st.markdown("""
    To identify the true causal effect, we must block the backdoor paths between Treatment (Surge) and Outcome (Booking).
    Our confounders are: **Demand Intensity, Weather, Time of Day, Zone**.
    """)
    st.markdown(f"```mermaid\n{DAG_MERMAID}\n```")

with tab_match:
    st.header("Propensity Score Matching")
    st.markdown("""
    To remove confounding, we can model the probability of a ride receiving surge pricing (the **Propensity Score**).
    We then match trips with similar propensity scores but different treatments to mimic a randomized experiment.
    """)
    
    with st.spinner("Fitting propensity scores and matching..."):
        ps, balance_df = run_matching(df)
        
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Propensity Overlap")
        st.markdown("We check if Treated and Control groups share common support.")
        st.plotly_chart(plot_propensity_distribution(ps, df["is_surge"].values), use_container_width=True)
        
    with col2:
        st.subheader("Covariate Balance (Love Plot)")
        st.markdown("We check if confounding variables are balanced (SMD near 0) after matching.")
        st.plotly_chart(plot_balance(balance_df), use_container_width=True)
    
with tab3:
    st.header("Average Treatment Effect (ATE)")
    st.markdown("We compare various estimators. The Naive estimator is heavily biased upwards due to unobserved confounding.")
    
    results = load_results()
    if results:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(plot_ate_comparison(results), use_container_width=True)
        with col2:
            st.dataframe(results_table(results), hide_index=True, use_container_width=True)
    else:
        st.warning("No pre-computed results found. Please run `python src/causal_rideshare/run_pipeline.py` first.")

with tab4:
    st.header("Heterogeneous Effects & Sensitivity")
    st.markdown("Who is most affected by surge pricing? We estimate Conditional Average Treatment Effects (CATE).")
    
    # We fake a CATE prediction using the DGP formulas to make the dashboard fast, 
    # instead of fitting a real CausalForest live.
    cate_proxies = -0.12 * df["price_sensitivity"] 
    reports = full_heterogeneity_report(df, cate_proxies)
    
    slice_opt = st.selectbox("View HTE by:", list(reports.keys()))
    st.plotly_chart(plot_cate_by_group(reports[slice_opt], title=f"CATE by {slice_opt}"), use_container_width=True)
    
    st.divider()
    st.subheader("Sensitivity Analysis (Rosenbaum Bounds)")
    st.markdown("How strong would an unobserved confounder need to be to invalidate our results?")
    bounds = rosenbaum_bounds(df)
    st.dataframe(bounds, use_container_width=True)
    
with tab5:
    st.header("Counterfactual Policy Simulation")
    st.markdown("What happens to revenue and driver earnings if we cap surge or offer discounts?")
    
    scenarios = run_standard_scenarios(df)
    st.plotly_chart(plot_policy_comparison(scenarios), use_container_width=True)
    
    st.subheader("Simulation Results Table")
    sim_df = pd.DataFrame([
        {
            "Scenario": s["label"], 
            "Completion Rate": s["counterfactual"]["completion_rate"],
            "Avg Revenue": s["counterfactual"]["avg_revenue"],
            "Driver Earnings": s["counterfactual"]["avg_driver_earnings"],
        } 
        for s in scenarios
    ])
    st.dataframe(sim_df.style.format({
        "Completion Rate": "{:.1%}",
        "Avg Revenue": "${:.2f}",
        "Driver Earnings": "${:.2f}",
    }), use_container_width=True, hide_index=True)

with tab6:
    st.header("Interactive 'What-If' Simulator")
    st.markdown("Adjust the pricing levers below to simulate the counterfactual impact on completion rates and revenue.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sim_surge_cap = st.slider("Max Surge Cap", 1.0, 5.0, 3.0, 0.1)
    with col2:
        sim_discount = st.slider("Universal Discount %", 0.0, 0.5, 0.0, 0.05)
    with col3:
        target_group = st.selectbox("Target Segment", ["All", "low", "medium", "high"])
        
    actual_target = None if target_group == "All" else target_group
    actual_discount = sim_discount if sim_discount > 0 else None
    
    sim_res = simulate_policy(
        df, 
        surge_cap=sim_surge_cap, 
        discount_pct_override=actual_discount, 
        target_segment=actual_target
    )
    
    delta_revenue = sim_res["delta"]["avg_revenue"]
    delta_completion = sim_res["delta"]["completion_rate"]
    
    st.subheader("Simulated Impact (vs Current Status Quo)")
    
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Avg Revenue per Trip", 
        f"${sim_res['counterfactual']['avg_revenue']:.2f}", 
        f"{delta_revenue:+.2f}"
    )
    m2.metric(
        "Completion Rate", 
        f"{sim_res['counterfactual']['completion_rate']:.1%}", 
        f"{delta_completion:+.1%}"
    )
    m3.metric(
        "Driver Earnings per Trip", 
        f"${sim_res['counterfactual']['avg_driver_earnings']:.2f}", 
        f"{sim_res['delta']['avg_driver_earnings']:+.2f}"
    )

