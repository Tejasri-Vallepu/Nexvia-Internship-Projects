
import pandas as pd
import numpy as np

df = pd.read_csv("outputs/forecast_predictions.csv")

# Validation residuals
residuals = (
    df["sales"] - df["lightgbm_prediction"]
)

# Empirical 90% residual interval
lower_error = residuals.quantile(0.05)
upper_error = residuals.quantile(0.95)

df["lower_90"] = np.maximum(
    0, df["lightgbm_prediction"] + lower_error
)

df["upper_90"] = np.maximum(
    df["lower_90"],
    df["lightgbm_prediction"] + upper_error
)

# Check empirical coverage
coverage = (
    (df["sales"] >= df["lower_90"]) &
    (df["sales"] <= df["upper_90"])
).mean() * 100

print("=" * 50)
print("PREDICTION INTERVAL RESULTS")
print("=" * 50)

print("Lower residual quantile:", round(lower_error, 3))
print("Upper residual quantile:", round(upper_error, 3))
print("Empirical coverage:", round(coverage, 2), "%")

df.to_csv(
    "outputs/forecast_with_intervals.csv",
    index=False
)

print("\nSaved: outputs/forecast_with_intervals.csv")
