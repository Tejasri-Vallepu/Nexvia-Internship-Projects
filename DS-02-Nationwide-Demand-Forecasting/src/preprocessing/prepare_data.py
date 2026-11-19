
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
OUTPUT = Path("data/processed")
OUTPUT.mkdir(parents=True, exist_ok=True)

print("Loading M5 datasets...")

sales = pd.read_csv(RAW / "sales_train_validation.csv")
calendar = pd.read_csv(RAW / "calendar.csv")
prices = pd.read_csv(RAW / "sell_prices.csv")

# Select recent 393 days
# 365 training days + 28 validation days
start_day = 1521
end_day = 1913

day_columns = [
    f"d_{i}" for i in range(start_day, end_day + 1)
]

print("Converting sales to long format...")

id_columns = [
    "id", "item_id", "dept_id",
    "cat_id", "store_id", "state_id"
]

sales_long = sales[id_columns + day_columns].melt(
    id_vars=id_columns,
    value_vars=day_columns,
    var_name="d",
    value_name="sales"
)

print("Merging calendar information...")

calendar["date"] = pd.to_datetime(calendar["date"])

# Fill missing event labels
event_columns = [
    "event_name_1", "event_type_1",
    "event_name_2", "event_type_2"
]

calendar[event_columns] = calendar[event_columns].fillna("None")

sales_long = sales_long.merge(
    calendar,
    on="d",
    how="left"
)

# Merge selling prices
print("Merging price information...")

sales_long = sales_long.merge(
    prices,
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left"
)

# Calendar features
sales_long["day_of_week"] = sales_long["date"].dt.dayofweek
sales_long["month"] = sales_long["date"].dt.month
sales_long["year"] = sales_long["date"].dt.year
sales_long["day_of_month"] = sales_long["date"].dt.day

# Validation indicator
sales_long["is_validation"] = (
    sales_long["d"].str.replace("d_", "", regex=False).astype(int)
    > 1885
)

# Sort time series
sales_long = sales_long.sort_values(
    ["id", "date"]
).reset_index(drop=True)

# Save processed data
output_file = OUTPUT / "sales_long.parquet"

sales_long.to_parquet(
    output_file,
    index=False
)

print("\n" + "=" * 55)
print("DATA PREPARATION COMPLETED")
print("=" * 55)

print("Rows:", len(sales_long))
print("Columns:", len(sales_long.columns))
print("Products:", sales_long["item_id"].nunique())
print("Stores:", sales_long["store_id"].nunique())
print("States:", sales_long["state_id"].nunique())

print("\nDate range:")
print(sales_long["date"].min())
print(sales_long["date"].max())

print("\nMissing selling prices:",
      sales_long["sell_price"].isna().sum())

print("\nSaved:", output_file)
