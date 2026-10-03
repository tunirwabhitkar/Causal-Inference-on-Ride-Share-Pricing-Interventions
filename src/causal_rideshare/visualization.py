"""Reusable Plotly visualisation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ── Colour palette ──────────────────────────────────────────────────────────

COLORS = {
    "primary": "#4361EE",
    "secondary": "#7209B7",
    "accent": "#F72585",
    "positive": "#06D6A0",
    "negative": "#EF476F",
    "neutral": "#FFD166",
    "bg": "#1B1B2F",
}

ESTIMATOR_COLORS = {
    "Ground Truth": "#06D6A0",
    "Naive": "#EF476F",
    "Regression": "#4361EE",
    "PSM": "#7209B7",
    "IPW": "#FFD166",
    "Doubly Robust": "#F72585",
    "Double ML": "#118AB2",
    "Causal Forest": "#073B4C",
}


# ── ATE comparison ──────────────────────────────────────────────────────────

def plot_ate_comparison(results: dict[str, float]) -> go.Figure:
    """Bar chart comparing ATE estimates with ground-truth line."""
    gt = results.get("Ground Truth", 0)
    methods = [k for k in results if k != "Ground Truth"]
    vals = [results[k] for k in methods]
    colors = [ESTIMATOR_COLORS.get(m, COLORS["primary"]) for m in methods]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=methods, y=vals, marker_color=colors,
        text=[f"{v:.4f}" for v in vals], textposition="outside",
    ))
    fig.add_hline(
        y=gt, line_dash="dash", line_color=ESTIMATOR_COLORS["Ground Truth"],
        annotation_text=f"Ground Truth = {gt:.4f}",
        annotation_position="top left",
    )
    fig.update_layout(
        title="ATE Estimates vs Ground Truth",
        yaxis_title="Estimated ATE",
        template="plotly_white",
        height=450,
    )
    return fig


# ── Propensity-score distribution ───────────────────────────────────────────

def plot_propensity_distribution(
    ps: np.ndarray, treatment: np.ndarray,
) -> go.Figure:
    """Overlapping histograms of propensity scores by treatment group."""
    df = pd.DataFrame({"ps": ps, "group": np.where(treatment == 1, "Treated", "Control")})
    fig = px.histogram(
        df, x="ps", color="group", barmode="overlay", nbins=50,
        opacity=0.6, title="Propensity Score Distribution",
        labels={"ps": "Propensity Score", "group": "Group"},
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig


# ── Covariate balance (love plot) ──────────────────────────────────────────

def plot_balance(balance_df: pd.DataFrame) -> go.Figure:
    """Love plot — SMD before and after matching."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=balance_df["smd_before"], y=balance_df["covariate"],
        mode="markers", marker=dict(size=10, color=COLORS["negative"]),
        name="Before Matching",
    ))
    fig.add_trace(go.Scatter(
        x=balance_df["smd_after"], y=balance_df["covariate"],
        mode="markers", marker=dict(size=10, color=COLORS["positive"]),
        name="After Matching",
    ))
    fig.add_vline(x=0.1, line_dash="dot", line_color="grey")
    fig.add_vline(x=-0.1, line_dash="dot", line_color="grey")
    fig.update_layout(
        title="Covariate Balance (Standardised Mean Difference)",
        xaxis_title="SMD", template="plotly_white", height=450,
    )
    return fig


# ── CATE by segment ────────────────────────────────────────────────────────

def plot_cate_by_group(cate_df: pd.DataFrame, title: str = "CATE by Group") -> go.Figure:
    """Bar chart with error bars for CATE across groups."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cate_df["group"],
        y=cate_df["mean_cate"],
        error_y=dict(type="data", array=(cate_df["ci_upper"] - cate_df["mean_cate"]).tolist()),
        marker_color=COLORS["primary"],
        text=[f"{v:.4f}" for v in cate_df["mean_cate"]],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_layout(
        title=title,
        yaxis_title="Estimated CATE",
        template="plotly_white",
        height=400,
    )
    return fig


# ── Policy simulation comparison ───────────────────────────────────────────

def plot_policy_comparison(scenarios: list[dict]) -> go.Figure:
    """Grouped bar chart comparing policy scenarios."""
    labels = [s["label"] for s in scenarios]
    metrics = ["booking_rate", "cancellation_rate", "completion_rate"]
    nice = {"booking_rate": "Booking Rate", "cancellation_rate": "Cancel Rate",
            "completion_rate": "Completion Rate"}
    colors = [COLORS["primary"], COLORS["negative"], COLORS["positive"]]

    fig = go.Figure()
    for m, c in zip(metrics, colors):
        vals = [s["counterfactual"][m] for s in scenarios]
        fig.add_trace(go.Bar(name=nice[m], x=labels, y=vals, marker_color=c))
    fig.update_layout(
        barmode="group", title="Policy Scenario Comparison",
        yaxis_title="Rate", template="plotly_white", height=450,
    )
    return fig


# ── Demand / Surge confounding ─────────────────────────────────────────────

def plot_confounding_demo(df: pd.DataFrame) -> go.Figure:
    """Scatter + marginal showing demand ↔ surge ↔ booking confounding."""
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter(
        sample,
        x="demand_intensity",
        y="surge_multiplier",
        color="booking",
        opacity=0.4,
        title="Confounding: Demand drives both Surge and Booking",
        labels={
            "demand_intensity": "Demand Intensity",
            "surge_multiplier": "Surge Multiplier",
            "booking": "Booked",
        },
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(template="plotly_white", height=450)
    return fig


# ── Ground-truth comparison table ──────────────────────────────────────────

def results_table(results: dict[str, float]) -> pd.DataFrame:
    """Create a tidy results table with error metrics."""
    gt = results.get("Ground Truth", 0)
    rows = []
    for method, est in results.items():
        rows.append({
            "Method": method,
            "Estimate": round(est, 4),
            "True Effect": round(gt, 4),
            "Error": round(est - gt, 4),
            "Abs Error": round(abs(est - gt), 4),
            "Rel Error (%)": round(abs(est - gt) / (abs(gt) + 1e-8) * 100, 1),
        })
    return pd.DataFrame(rows)
