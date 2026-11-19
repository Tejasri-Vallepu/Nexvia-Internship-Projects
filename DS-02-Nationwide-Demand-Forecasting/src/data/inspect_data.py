
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

files = {
    "Sales": DATA_DIR / "sales_train_validation.csv",
    "Calendar": DATA_DIR / "calendar.csv",
    "Prices": DATA_DIR / "sell_prices.csv"
}

print("=" * 60)
print("M5 DATASET INSPECTION")
print("=" * 60)

# 1. Load sales dataset
sales = pd.read_csv(files["Sales"])

print("\n1. SALES DATA")
print("Shape:", sales.shape)
print("\nFirst 5 rows:")
print(sales.head())
print("\nMissing values:", sales.isnull().sum().sum())

# 2. Load calendar dataset
calendar = pd.read_csv(files["Calendar"])

print("\n2. CALENDAR DATA")
print("Shape:", calendar.shape)
print("\nColumns:", calendar.columns.tolist())
print("\nDate range:")
print(calendar["date"].min(), "to", calendar["date"].max())
print("\nMissing values:", calendar.isnull().sum().sum())

# 3. Inspect prices dataset
prices = pd.read_csv(files["Prices"], nrows=100000)

print("\n3. SELL PRICES DATA")
print("Sample shape:", prices.shape)
print("\nColumns:", prices.columns.tolist())
print("\nFirst 5 rows:")
print(prices.head())

# 4. Hierarchy information
print("\n4. HIERARCHY INFORMATION")
print("Total products:", sales["item_id"].nunique())
print("Total stores:", sales["store_id"].nunique())
print("Total states:", sales["state_id"].nunique())

print("\nStores:")
print(sorted(sales["store_id"].unique()))

print("\nStates:")
print(sorted(sales["state_id"].unique()))

# 5. Save inspection summary
summary = {
    "sales_rows": sales.shape[0],
    "sales_columns": sales.shape[1],
    "calendar_rows": calendar.shape[0],
    "calendar_columns": calendar.shape[1],
    "products": sales["item_id"].nunique(),
    "stores": sales["store_id"].nunique(),
    "states": sales["state_id"].nunique(),
    "sales_missing_values": int(sales.isnull().sum().sum()),
    "calendar_missing_values": int(calendar.isnull().sum().sum())
}

pd.Series(summary).to_csv("outputs/dataset_summary.csv")

print("\n" + "=" * 60)
print("INSPECTION COMPLETED SUCCESSFULLY!")
print("Summary saved to outputs/dataset_summary.csv")
print("=" * 60)
