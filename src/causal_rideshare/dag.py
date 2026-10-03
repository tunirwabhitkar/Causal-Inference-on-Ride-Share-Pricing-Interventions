"""Causal DAG definition for the ride-share pricing problem.

The DAG encodes domain knowledge about the data-generating process.  It is
used for:
  1. Documentation — making assumptions transparent.
  2. Identification — determining which confounders to adjust for.
  3. Refutation — testing causal estimates against the graph structure.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Text representation (for rendering in dashboards / docs)
# ---------------------------------------------------------------------------

DAG_DOT = """\
digraph RideshareDAG {
    rankdir=LR;
    node [shape=ellipse, fontname="Helvetica"];

    // Confounders
    Weather       [label="Weather"];
    Hour          [label="Hour of Day"];
    Weekend       [label="Weekend/Holiday"];
    LocalEvent    [label="Local Event"];
    Zone          [label="Origin Zone"];

    // Mediators / mechanisms
    Demand        [label="Demand Intensity"];
    Supply        [label="Driver Supply"];
    SDRatio       [label="Supply/Demand\\nRatio"];

    // Treatment
    Surge         [label="Surge Multiplier", style=filled, fillcolor="#FFCCCC"];
    Discount      [label="Discount", style=filled, fillcolor="#CCFFCC"];

    // Rider heterogeneity
    PriceSens     [label="Price Sensitivity"];
    Income        [label="Income Segment"];
    Loyalty       [label="Loyalty Score"];

    // Outcomes
    Booking       [label="Booking", style=filled, fillcolor="#CCE5FF"];
    Cancellation  [label="Cancellation", style=filled, fillcolor="#CCE5FF"];
    Revenue       [label="Revenue", style=filled, fillcolor="#CCE5FF"];
    DriverEarn    [label="Driver Earnings", style=filled, fillcolor="#CCE5FF"];

    // Edges — confounder → treatment & outcome
    Weather  -> Demand;
    Hour     -> Demand;
    Weekend  -> Demand;
    LocalEvent -> Demand;
    Zone     -> Demand;

    Hour     -> Supply;
    Weekend  -> Supply;

    Demand   -> SDRatio;
    Supply   -> SDRatio;

    Demand   -> Surge;
    SDRatio  -> Surge;

    Demand   -> Booking;
    Weather  -> Booking;

    // Treatment → outcomes (causal effects of interest)
    Surge    -> Booking   [color=red, penwidth=2];
    Surge    -> Cancellation [color=red, penwidth=2];
    Discount -> Booking   [color=blue, penwidth=2];
    Discount -> Cancellation [color=blue, penwidth=2];

    // Rider heterogeneity
    Income    -> PriceSens;
    PriceSens -> Booking;
    Loyalty   -> Booking;
    Loyalty   -> Cancellation;

    // Outcomes chain
    Booking       -> Revenue;
    Cancellation  -> Revenue;
    Booking       -> DriverEarn;
    Cancellation  -> DriverEarn;
    Surge         -> Revenue;
    Discount      -> Revenue;
}
"""

# ---------------------------------------------------------------------------
# Mermaid representation (for Streamlit / markdown rendering)
# ---------------------------------------------------------------------------

DAG_MERMAID = """\
graph LR
    Weather --> Demand
    Hour["Hour of Day"] --> Demand
    Weekend["Weekend/Holiday"] --> Demand
    LocalEvent["Local Event"] --> Demand
    Zone["Origin Zone"] --> Demand

    Hour --> Supply["Driver Supply"]
    Weekend --> Supply

    Demand --> SDRatio["Supply/Demand Ratio"]
    Supply --> SDRatio

    Demand --> Surge["Surge Multiplier"]
    SDRatio --> Surge

    Demand --> Booking
    Weather --> Booking

    Surge -->|causal| Booking
    Surge -->|causal| Cancellation
    Discount -->|causal| Booking
    Discount -->|causal| Cancellation

    Income["Income Segment"] --> PriceSens["Price Sensitivity"]
    PriceSens --> Booking
    Loyalty["Loyalty Score"] --> Booking
    Loyalty --> Cancellation

    Booking --> Revenue
    Cancellation --> Revenue
    Booking --> DriverEarn["Driver Earnings"]
    Cancellation --> DriverEarn
    Surge --> Revenue
    Discount --> Revenue

    style Surge fill:#FFCCCC
    style Discount fill:#CCFFCC
    style Booking fill:#CCE5FF
    style Cancellation fill:#CCE5FF
    style Revenue fill:#CCE5FF
    style DriverEarn fill:#CCE5FF
"""


# Confounders to adjust for (surge → booking analysis)
SURGE_CONFOUNDERS: list[str] = [
    "demand_intensity",
    "weather",
    "hour_of_day",
    "is_weekend",
    "distance_km",
    "supply_demand_ratio",
    "price_sensitivity",
    "loyalty_score",
    "local_event",
]

DISCOUNT_CONFOUNDERS: list[str] = [
    "demand_intensity",
    "weather",
    "hour_of_day",
    "is_weekend",
    "distance_km",
    "price_sensitivity",
    "loyalty_score",
]
