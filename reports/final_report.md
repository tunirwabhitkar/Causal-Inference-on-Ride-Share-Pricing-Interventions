# Causal Inference on Ride-Share Pricing Interventions

## Executive Summary
This project investigates the causal impact of surge pricing on ride-share booking rates and platform revenue. Because pricing interventions are endogenously assigned based on demand, simple correlation leads to biased estimates (Simpson's Paradox). By rigorously controlling for demand intensity and other confounders through a structural causal model, we uncover the true negative elasticity of surge pricing.

## 1. The Confounding Problem
In our ride-share data, **price is endogenous**. During periods of high demand, the platform initiates surge pricing. Simultaneously, riders experiencing high demand are more desperate to book a ride. 
If we ignore this confounding, a naive estimator shows a positive correlation between surge pricing and bookings, suggesting we should surge more to get more bookings. This is a classic causal fallacy.

## 2. Structural Causal Model (DAG)
To identify the true causal effect, we established the following confounders based on domain expertise:
* **Demand Intensity**: Affects both Surge and Booking.
* **Weather**: Affects Demand and Booking.
* **Time of Day**: Affects Demand.
* **Origin Zone**: Affects Demand.

By blocking the backdoor paths through these confounders, we can isolate the direct effect of Surge Pricing on Bookings.

## 3. Estimator Performance
We implemented several modern causal estimators and benchmarked them against the known ground-truth DGP (Data Generating Process) Average Treatment Effect (ATE) of exactly -0.12.

| Estimator | ATE | Error |
| :--- | :--- | :--- |
| **Ground Truth** | **-0.120** | - |
| Naive | +0.184 | 0.304 |
| Regression | -0.119 | 0.001 |
| PSM | -0.121 | 0.001 |
| IPW | -0.118 | 0.002 |
| Doubly Robust | -0.119 | 0.001 |
| Double ML | -0.123 | 0.003 |

**Key Findings:**
1. The **Naive** estimator entirely fails, predicting a positive effect (+0.18).
2. Modern estimators (**Doubly Robust**, **DML**, **Regression**) successfully recover the true negative elasticity (-0.12) by controlling for the backdoor paths.

## 4. Heterogeneous Treatment Effects (HTE)
Using our Causal Forest implementation, we identified substantial heterogeneity in price elasticity:
* **Income Segments**: Low-income riders have significantly higher price sensitivity than high-income riders.
* **Weather**: During extreme weather, price elasticity drops significantly (riders become inelastic).

## 5. Policy Counterfactuals
Using the estimated causal parameters, we simulated alternative pricing strategies:
* **A. No Surge**: Eliminating surge increases completion rates but significantly reduces platform revenue and driver earnings.
* **B. Current Policy (3x Cap)**: The baseline standard.
* **C. Reduced Cap (1.5x)**: A balanced approach that slightly decreases peak revenue but improves completion consistency.
* **D. Targeted Discount**: Offering 20% discounts to highly price-sensitive segments yields higher incremental conversion than universal discounts.

## 6. Refutation & Robustness
The estimates are robust to refutation tests:
* **Placebo Treatment**: When treatment is randomized, the estimated effect is effectively 0.0.
* **Random Common Cause**: Adding random noise to the confounder set does not significantly alter the ATE.
* **Data Subsets**: The ATE is stable across 70% random resamples of the dataset.

## Conclusion
This project successfully demonstrates the necessity of causal inference in dynamic pricing environments. By correctly adjusting for demand-induced confounding, we shift the platform's insight from "surge increases bookings" to an accurate measurement of negative price elasticity, enabling precise revenue and conversion optimizations.
