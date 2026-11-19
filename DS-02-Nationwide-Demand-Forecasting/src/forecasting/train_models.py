
from pathlib import Path
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

DATA = Path("data/processed/forecast_features.parquet")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

print("Loading forecasting features...")
df = pd.read_parquet(DATA)

cutoff = pd.Timestamp("2016-03-27")

train = df[df["date"] <= cutoff].copy()
valid = df[df["date"] > cutoff].copy()

target = "sales"

features = [
    "item_id",
    "store_id",
    "state_id",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "month",
    "day_of_month",
    "week_of_year",
    "is_event",
    "snap",
    "sell_price",
    "price_missing"
]

# Align categorical types
for col in ["item_id", "store_id", "state_id"]:
    categories = df[col].cat.categories
    train[col] = train[col].cat.set_categories(categories)
    valid[col] = valid[col].cat.set_categories(categories)

X_train = train[features]
y_train = train[target]

X_valid = valid[features]
y_valid = valid[target]

print("\nTraining records:", len(train))
print("Validation records:", len(valid))

# 1. Naive seasonal baseline
print("\nGenerating naive predictions...")

naive_pred = valid["lag_7"].to_numpy()

# 2. LightGBM
print("\nTraining LightGBM model...")

model = LGBMRegressor(
    objective="regression",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

model.fit(
    X_train,
    y_train,
    categorical_feature=["item_id", "store_id", "state_id"]
)

lgb_pred = np.maximum(
    model.predict(X_valid),
    0
)

# Evaluation metrics
def calculate_metrics(actual, predicted, training_data):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    wape = (
        np.abs(actual - predicted).sum()
        / max(np.abs(actual).sum(), 1e-10)
    ) * 100

    errors = []
    unique_ids = training_data["id"].unique()

    for series_id in unique_ids:
        train_series = training_data.loc[
            training_data["id"] == series_id, "sales"
        ].to_numpy()

        mask = valid["id"].to_numpy() == series_id

        actual_series = actual[mask]
        pred_series = predicted[mask]

        # Seasonal scale: lag-7 differences
        scale = np.mean(
            (train_series[7:] - train_series[:-7]) ** 2
        )

        if scale > 0:
            rmsse = np.sqrt(
                np.mean((actual_series - pred_series) ** 2)
                / scale
            )
            errors.append(rmsse)

    return {
        "WAPE_percent": round(float(wape), 4),
        "RMSSE": round(float(np.mean(errors)), 4)
        if errors else None
    }


# One training history per series
history = (
    train.sort_values("date")
    .groupby("id")
    .tail(365)
)

print("\nCalculating evaluation metrics...")

naive_metrics = calculate_metrics(
    y_valid, naive_pred, history
)

lgb_metrics = calculate_metrics(
    y_valid, lgb_pred, history
)

results = {
    "Naive_Lag_7": naive_metrics,
    "LightGBM": lgb_metrics
}

print("\n" + "=" * 55)
print("FORECASTING MODEL COMPARISON")
print("=" * 55)

for name, metrics in results.items():
    print(f"\n{name}")
    print("WAPE:", metrics["WAPE_percent"], "%")
    print("RMSSE:", metrics["RMSSE"])

# Save predictions
predictions = valid[
    ["id", "item_id", "store_id", "state_id",
     "date", "sales"]
].copy()

predictions["naive_prediction"] = naive_pred
predictions["lightgbm_prediction"] = lgb_pred

predictions.to_csv(
    OUT / "forecast_predictions.csv",
    index=False
)

# Save model
model.booster_.save_model(
    "models/lightgbm_model.txt"
)

# Save metrics
with open(OUT / "model_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved:")
print("- outputs/forecast_predictions.csv")
print("- outputs/model_metrics.json")
print("- models/lightgbm_model.txt")

print("\nMODEL TRAINING COMPLETED!")
