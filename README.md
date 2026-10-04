# Causal Inference on Ride-Share Pricing Interventions

## 🚀 Live Demo

👉 **[Try the Interactive Streamlit Dashboard](https://causal-inference-on-ride-share-pricing-interventions-dabxuf63j.streamlit.app/)**

## 🚖 Project Overview
This project investigates the causal impact of ride-share pricing interventions, specifically surge pricing, on rider demand and conversion. By creating a synthetic dataset with known ground-truth causal effects, we demonstrate why traditional correlation analysis fails in observational settings due to confounding variables like demand intensity, and how causal inference methods can recover the true effect.

## ⚠️ The Problem with Correlation
In ride-share data, **Demand Intensity** causes both higher Surge Pricing and higher Booking Rates. A naive analysis comparing rides with and without surge pricing will show that surge pricing is *positively* correlated with booking rates. However, the true causal effect of higher prices is a reduction in demand. This project corrects for that confounding.

## 📊 Methodology
We implemented multiple causal estimators to isolate the treatment effect of surge pricing from confounders:
- **Naive Difference-in-Means** (Baseline bias)
- **OLS Regression Adjustment**
- **Inverse Probability Weighting (IPW)**
- **Doubly Robust Estimation (AIPW)**
- **Double Machine Learning (DML)** using Random Forests

## 🚀 Results

| Method | Estimated ATE |
|--------|---------------|
| Ground Truth | -0.0220 |
| Naive | +0.0658 |
| Regression | -0.0162 |
| IPW | -0.0141 |
| Doubly Robust | -0.0140 |
| Double ML | -0.0128 |

The Naive estimator falsely concludes that surge pricing increases bookings by +6.5%. The causal estimators correctly identify the negative impact of price on conversion, recovering effects close to the -2.2% ground truth.

## 💻 How to Run

1. Clone the repo and navigate to the root directory.
2. Activate the virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Run the data generation and estimation pipeline:
   ```bash
   $env:PYTHONPATH="src"
   python src/causal_rideshare/run_pipeline.py
   ```
4. Run tests:
   ```bash
   $env:PYTHONPATH="src"
   pytest
   ```
5. Launch the Streamlit Dashboard:
   ```bash
   $env:PYTHONPATH="src"
   streamlit run app/app.py
   ```

## 📁 Repository Structure
- `configs/`: Simulation parameters
- `data/`: Raw and processed data
- `src/causal_rideshare/estimators/`: Implementations of causal estimators
- `tests/`: Unit tests (pytest)
- `app/`: Streamlit interactive dashboard
- `reports/`: JSON results and final report
