
from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path("data/processed/sales_long.parquet")
OUTPUT = Path("data/processed")

print("Loading processed M5 data...")

df = pd.read_parquet(INPUT)

# Select 500 product-store combinations
unique_ids = df["id"].drop_duplicates().sort_values()

selected_ids = unique_ids.sample(
    n=500,
    random_state=42
)

df = df[df["id"].isin(selected_ids)].copy()

df = df.sort_values(["id", "date"]).reset_index(drop=True)

print("Selected combinations:", df["id"].nunique())

# Lag features
print("Creating lag features...")

for lag in [7, 14, 28]:
    df[f"lag_{lag}"] = (
        df.groupby("id")["sales"].shift(lag)
    )

# Rolling features using previous observations only
print("Creating rolling features...")

for window in [7, 28]:
    df[f"rolling_mean_{window}"] = (
        df.groupby("id")["sales"]
        .transform(
            lambda x: x.shift(1).rolling(
                window,
                min_periods=window
            ).mean()
        )
    )

# Calendar features
print("Creating calendar features...")

df["day_of_week"] = df["date"].dt.day_of_week
df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

# Event feature
df["is_event"] = (
    df["event_name_1"].ne("None")
).astype(int)

# SNAP feature according to state
df["snap"] = np.select(
    [
        df["state_id"].eq("CA"),
        df["state_id"].eq("TX"),
        df["state_id"].eq("WI")
    ],
    [
        df["snap_CA"],
        df["snap_TX"],
        df["snap_WI"]
    ],
    default=0
)

# Price missing indicator
df["price_missing"] = df["sell_price"].isna().astype(int)

# Training cutoff: March 27, 2016
cutoff = pd.Timestamp("2016-03-27")

train_mask = df["date"] <= cutoff

# Fill missing prices using training median
price_median = df.loc[
    train_mask, "sell_price"
].median()

df["sell_price"] = df["sell_price"].fillna(price_median)

# Remove rows without sufficient lag history
df = df.dropna(
    subset=["lag_7", "lag_14", "lag_28",
            "rolling_mean_7", "rolling_mean_28"]
)

# Categorical features
for col in ["item_id", "store_id", "state_id"]:
    df[col] = df[col].astype("category")

# Save
output_file = OUTPUT / "forecast_features.parquet"

df.to_parquet(output_file, index=False)

print("\n" + "=" * 55)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 55)

print("Rows:", len(df))
print("Products:", df["item_id"].nunique())
print("Stores:", df["store_id"].nunique())

print("\nFeatures:")
print([
    "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
    "day_of_week", "month", "day_of_month",
    "week_of_year", "is_event", "snap",
    "sell_price", "price_missing"
])

print("\nTraining rows:", (df["date"] <= cutoff).sum())
print("Validation rows:", (df["date"] > cutoff).sum())

print("\nSaved:", output_file)
