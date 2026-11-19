
"""Train Prophet models for the 500 sampled item-store demand series."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "forecast_features.parquet"
OUTPUT_PATH = ROOT / "outputs" / "prophet_predictions.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"

def calculate_metrics(predictions, train_data):
    actual = predictions["actual"].to_numpy(dtype=float)
    forecast = predictions["Prophet_Forecast"].to_numpy(dtype=float)

    wape = (
        np.abs(actual - forecast).sum()
        / max(np.abs(actual).sum(), 1e-9)
    ) * 100

    rmsse_values = []

    for series_id, group in predictions.groupby("id"):
        train_series = train_data.loc[
            train_data["id"] == series_id, "sales"
        ].to_numpy(dtype=float)

        if len(train_series) <= 7:
            continue

        scale = np.mean((train_series[7:] - train_series[:-7]) ** 2)

        if scale <= 0:
            continue

        errors = (
            group["actual"].to_numpy(dtype=float)
            - group["Prophet_Forecast"].to_numpy(dtype=float)
        )

        rmsse_values.append(
            np.sqrt(np.mean(errors ** 2) / scale)
        )

    rmsse = float(np.mean(rmsse_values)) if rmsse_values else None

    return {
        "WAPE_percent": round(float(wape), 4),
        "RMSSE": round(rmsse, 4) if rmsse is not None else None,
    }


def main():
    print("Loading feature dataset...")
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["id", "date"])

    train = df[df["is_validation"] == False].copy()
    validation = df[df["is_validation"] == True].copy()

    predictions = []
    failed_series = []

    series_ids = sorted(validation["id"].unique())
    total = len(series_ids)

    print(f"Training Prophet for {total} item-store series...")
    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")

    for index, series_id in enumerate(series_ids, start=1):
        train_group = train[train["id"] == series_id]
        valid_group = validation[validation["id"] == series_id]

        model_data = train_group[["date", "sales"]].rename(
            columns={"date": "ds", "sales": "y"}
        )

        future = valid_group[["date"]].rename(
            columns={"date": "ds"}
        )

        try:
            model = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=False,
                daily_seasonality=False,
                uncertainty_samples=0,
                changepoint_prior_scale=0.05,
            )

            model.fit(model_data)
            forecast = model.predict(future)

            series_result = valid_group[
                ["id", "item_id", "store_id", "state_id", "date", "sales"]
            ].copy()

            series_result = series_result.rename(
                columns={"sales": "actual"}
            )

            series_result["Prophet_Forecast"] = np.maximum(
                forecast["yhat"].to_numpy(), 0
            )

            predictions.append(series_result)

        except Exception as exc:
            failed_series.append({
                "id": series_id,
                "error": str(exc),
            })

        if index % 25 == 0 or index == total:
            print(f"Completed {index}/{total}")

    if not predictions:
        raise RuntimeError("No Prophet predictions were generated.")

    result = pd.concat(predictions, ignore_index=True)
    result.to_csv(OUTPUT_PATH, index=False)

    metrics = calculate_metrics(result, train)

    existing_metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            existing_metrics = json.load(file)

    existing_metrics["Prophet"] = metrics

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(existing_metrics, file, indent=2)

    failure_path = ROOT / "outputs" / "prophet_failed_series.json"
    with open(failure_path, "w", encoding="utf-8") as file:
        json.dump(failed_series, file, indent=2)

    print("\n" + "=" * 55)
    print("PROPHET FORECASTING RESULTS")
    print("=" * 55)
    print(f"Successful series: {len(predictions)}")
    print(f"Failed series: {len(failed_series)}")
    print(f"Prediction rows: {len(result)}")
    print(f"WAPE: {metrics['WAPE_percent']}%")
    print(f"RMSSE: {metrics['RMSSE']}")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Updated: {METRICS_PATH}")


if __name__ == "__main__":
    main()
