
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def test_required_output_files_exist():
    required_files = [
        "forecast_predictions.csv",
        "forecast_with_intervals.csv",
        "inventory_recommendations.csv",
        "model_metrics.json",
    ]

    for filename in required_files:
        assert (OUTPUTS / filename).exists(), f"Missing file: {filename}"


def test_forecast_predictions_schema():
    df = pd.read_csv(OUTPUTS / "forecast_predictions.csv")

    required_columns = {
        "id", "item_id", "store_id", "state_id",
        "date", "sales", "naive_prediction",
        "lightgbm_prediction",
    }

    assert required_columns.issubset(df.columns)
    assert not df.empty


def test_forecast_values_are_valid():
    df = pd.read_csv(OUTPUTS / "forecast_with_intervals.csv")

    assert not df.empty
    assert df["sales"].notna().all()
    assert df["lightgbm_prediction"].notna().all()
    assert (df["lightgbm_prediction"] >= 0).all()
    assert (df["lower_90"] >= 0).all()
    assert (df["upper_90"] >= df["lower_90"]).all()


def test_inventory_recommendations():
    df = pd.read_csv(OUTPUTS / "inventory_recommendations.csv")

    required_columns = {
        "item_id", "store_id", "state_id",
        "forecast_7_day_demand", "order_quantity",
        "expected_shortage", "expected_excess",
    }

    assert required_columns.issubset(df.columns)
    assert not df.empty
    assert (df["order_quantity"] >= 0).all()
    assert (df["expected_shortage"] >= 0).all()
    assert (df["expected_excess"] >= 0).all()


def test_model_metrics():
    with open(OUTPUTS / "model_metrics.json", "r") as file:
        metrics = json.load(file)

    assert "Naive_Lag_7" in metrics
    assert "LightGBM" in metrics

    for model in metrics.values():
        assert "WAPE_percent" in model
        assert "RMSSE" in model
        assert model["WAPE_percent"] >= 0
        assert model["RMSSE"] >= 0


def test_lightgbm_outperforms_naive_on_wape():
    with open(OUTPUTS / "model_metrics.json", "r") as file:
        metrics = json.load(file)

    assert (
        metrics["LightGBM"]["WAPE_percent"]
        < metrics["Naive_Lag_7"]["WAPE_percent"]
    )
