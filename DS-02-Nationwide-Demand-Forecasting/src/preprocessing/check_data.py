
import pandas as pd

df = pd.read_parquet("data/processed/sales_long.parquet")

print("=" * 55)
print("DATA QUALITY CHECK")
print("=" * 55)

print("\n1. Missing prices")
print("Missing:", df["sell_price"].isna().sum())
print("Percentage:",
      round(df["sell_price"].isna().mean() * 100, 3), "%")

print("\n2. Missing prices by state")
print(
    df.groupby("state_id")["sell_price"]
    .apply(lambda x: x.isna().mean() * 100)
)

print("\n3. Sales statistics")
print(df["sales"].describe())

print("\n4. Training and validation split")
print("Training rows:", (~df["is_validation"]).sum())
print("Validation rows:", df["is_validation"].sum())

print("\n5. Date split")
print("Training end:",
      df.loc[~df["is_validation"], "date"].max())

print("Validation start:",
      df.loc[df["is_validation"], "date"].min())

print("\n6. Zero sales percentage")
print(round((df["sales"] == 0).mean() * 100, 2), "%")

print("\nDATA QUALITY CHECK COMPLETED")
