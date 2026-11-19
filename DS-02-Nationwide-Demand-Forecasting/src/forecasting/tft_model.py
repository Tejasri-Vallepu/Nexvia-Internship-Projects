
"""Train a Temporal Fusion Transformer for sampled M5 demand series."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lightning.pytorch import Trainer
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "forecast_features.parquet"
PREDICTION_PATH = ROOT / "outputs" / "tft_predictions.csv"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
MODEL_PATH = ROOT / "models" / "tft_model.ckpt"

MAX_ENCODER_LENGTH = 28
MAX_PREDICTION_LENGTH = 28
SAMPLE_SERIES = 100
EPOCHS = 3
BATCH_SIZE = 64


def calculate_metrics(predictions, training_data):
    actual = predictions["actual"].to_numpy(dtype=float)
    forecast = predictions["TFT_Forecast"].to_numpy(dtype=float)

    wape = (
        np.abs(actual - forecast).sum()
        / max(np.abs(actual).sum(), 1e-9)
    ) * 100

    rmsse_values = []

    for series_id, group in predictions.groupby("id"):
        history = training_data.loc[
            training_data["id"] == series_id, "sales"
        ].to_numpy(dtype=float)

        if len(history) <= 7:
            continue

        scale = np.mean((history[7:] - history[:-7]) ** 2)

        if scale <= 0:
            continue

        errors = (
            group["actual"].to_numpy(dtype=float)
            - group["TFT_Forecast"].to_numpy(dtype=float)
        )

        rmsse_values.append(
            np.sqrt(np.mean(errors ** 2) / scale)
        )

    return {
        "WAPE_percent": round(float(wape), 4),
        "RMSSE": (
            round(float(np.mean(rmsse_values)), 4)
            if rmsse_values else None
        ),
    }


def main():
    torch.set_num_threads(4)

    print("Loading feature dataset...")
    df = pd.read_parquet(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["id", "date"]).copy()

    # Select a reproducible sample for CPU training.
    all_ids = np.array(sorted(df["id"].unique()))
    rng = np.random.RandomState(42)

    selected_ids = sorted(
        rng.choice(
            all_ids,
            size=min(SAMPLE_SERIES, len(all_ids)),
            replace=False,
        ).tolist()
    )

    df = df[df["id"].isin(selected_ids)].copy()

    # Convert dates into a continuous integer time index.
    date_values = sorted(df["date"].unique())
    date_mapping = {
        date: index for index, date in enumerate(date_values)
    }

    df["time_idx"] = df["date"].map(date_mapping).astype(int)

    train_df = df[df["is_validation"] == False].copy()
    full_df = df.copy()

    # Categorical columns must be strings.
    categorical_columns = [
        "id", "item_id", "store_id", "state_id"
    ]

    for column in categorical_columns:
        full_df[column] = full_df[column].astype(str)
        train_df[column] = train_df[column].astype(str)

    known_reals = [
        "time_idx",
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "is_event",
        "snap",
        "sell_price",
        "price_missing",
    ]

    for column in known_reals:
        full_df[column] = pd.to_numeric(
            full_df[column], errors="coerce"
        )
        train_df[column] = pd.to_numeric(
            train_df[column], errors="coerce"
        )

        median_value = train_df[column].median()
        if pd.isna(median_value):
            median_value = 0

        full_df[column] = full_df[column].fillna(median_value)
        train_df[column] = train_df[column].fillna(median_value)

    full_df["sales"] = full_df["sales"].astype(float)
    train_df["sales"] = train_df["sales"].astype(float)

    print("Selected series:", len(selected_ids))
    print("Training rows:", len(train_df))
    print("Validation rows:", len(full_df) - len(train_df))

    # Build training dataset.
    training_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="sales",
        group_ids=["id"],
        min_encoder_length=MAX_ENCODER_LENGTH,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=MAX_PREDICTION_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=[
            "item_id", "store_id", "state_id"
        ],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=["sales"],
        target_normalizer=GroupNormalizer(
            groups=["id"],
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    # Prediction dataset: 28 historical days + 28 validation days.
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        full_df,
        min_prediction_idx=int(train_df["time_idx"].max()) + 1,
        predict=True,
        stop_randomization=True,
    )

    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    validation_loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print("Training TFT model...")

    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.01,
        hidden_size=8,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=4,
        lstm_layers=1,
        loss=MAE(),
        output_size=1,
        reduce_on_plateau_patience=2,
    )

    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",
        devices=1,
        gradient_clip_val=0.1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        limit_train_batches=40,
        limit_val_batches=0,
    )

    trainer.fit(model, train_dataloaders=train_loader)

    print("Generating 28-day validation predictions...")

    prediction_result = model.predict(
        validation_loader,
        return_index=True,
        
    )

    # PyTorch Forecasting returns prediction values and their index.
    raw_predictions = prediction_result.output
    prediction_index = prediction_result.index.reset_index(drop=True)

    forecast_array = raw_predictions.detach().cpu().numpy()

    if forecast_array.ndim == 3:
        forecast_array = forecast_array.squeeze(-1)

    if forecast_array.shape[1] != MAX_PREDICTION_LENGTH:
        raise RuntimeError(
            f"Expected {MAX_PREDICTION_LENGTH} forecast days, "
            f"received {forecast_array.shape[1]}"
        )

    results = []
    validation_start = int(train_df["time_idx"].max()) + 1

    for row_number, index_row in prediction_index.iterrows():
        series_id = str(index_row["id"])

        series_actual = full_df[
            (full_df["id"] == series_id)
            & (full_df["time_idx"] >= validation_start)
        ].sort_values("time_idx")

        if len(series_actual) != MAX_PREDICTION_LENGTH:
            continue

        output = series_actual[
            [
                "id", "item_id", "store_id",
                "state_id", "date", "sales"
            ]
        ].copy()

        output.rename(columns={"sales": "actual"}, inplace=True)

        output["TFT_Forecast"] = np.maximum(
            forecast_array[row_number], 0
        )

        results.append(output)

    if not results:
        raise RuntimeError(
            "No TFT predictions matched the validation series."
        )

    final_predictions = pd.concat(results, ignore_index=True)

    if final_predictions["id"].nunique() != len(selected_ids):
        raise RuntimeError(
            "Some selected series are missing from TFT predictions."
        )

    final_predictions.to_csv(PREDICTION_PATH, index=False)

    metrics = calculate_metrics(final_predictions, train_df)

    existing_metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            existing_metrics = json.load(file)

    # Explicitly identify the sample scope.
    existing_metrics["TFT_100_series"] = metrics

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(existing_metrics, file, indent=2)

    trainer.save_checkpoint(str(MODEL_PATH))

    run_info = {
        "model": "Temporal Fusion Transformer",
        "series_trained": len(selected_ids),
        "training_epochs": EPOCHS,
        "training_batch_limit": 40,
        "encoder_length_days": MAX_ENCODER_LENGTH,
        "prediction_length_days": MAX_PREDICTION_LENGTH,
        "validation_rows": len(final_predictions),
        "evaluation_scope": "100 sampled item-store series",
        "metrics": metrics,
    }

    with open(
        ROOT / "outputs" / "tft_run_info.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_info, file, indent=2)

    print("\n" + "=" * 55)
    print("TEMPORAL FUSION TRANSFORMER RESULTS")
    print("=" * 55)
    print("Successful series:", final_predictions["id"].nunique())
    print("Prediction rows:", len(final_predictions))
    print("WAPE:", metrics["WAPE_percent"], "%")
    print("RMSSE:", metrics["RMSSE"])
    print("Saved:", PREDICTION_PATH)
    print("Model checkpoint:", MODEL_PATH)


if __name__ == "__main__":
    main()
