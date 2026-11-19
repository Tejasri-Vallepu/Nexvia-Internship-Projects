
"""
Darts Bottom-Up Hierarchical Forecast Reconciliation

Hierarchy:
Item-Store -> Store -> State -> National

Input:
    outputs/forecast_predictions.csv

Outputs:
    outputs/darts_reconciled_forecasts.csv
    outputs/darts_reconciliation_summary.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers.reconciliation import (
    BottomUpReconciliator,
)


# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "outputs" / "forecast_predictions.csv"

OUTPUT_FILE = (
    ROOT / "outputs" / "darts_reconciled_forecasts.csv"
)

SUMMARY_FILE = (
    ROOT / "outputs" / "darts_reconciliation_summary.json"
)


# ---------------------------------------------------------
# 1. LOAD FORECAST DATA
# ---------------------------------------------------------

def load_forecasts():

    print("Loading LightGBM forecasts...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "id",
        "item_id",
        "store_id",
        "state_id",
        "date",
        "lightgbm_prediction",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"])

    df["lightgbm_prediction"] = pd.to_numeric(
        df["lightgbm_prediction"],
        errors="raise",
    )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Missing values found in required columns."
        )

    if df.duplicated(["id", "date"]).any():
        raise ValueError(
            "Duplicate item-store-date records found."
        )

    # Demand forecasts cannot be negative.
    df["lightgbm_prediction"] = (
        df["lightgbm_prediction"].clip(lower=0)
    )

    print(f"Input rows: {len(df):,}")
    print(
        f"Item-store series: {df['id'].nunique():,}"
    )
    print(
        f"Forecast dates: {df['date'].nunique()}"
    )

    return df


# ---------------------------------------------------------
# 2. BUILD HIERARCHY
# ---------------------------------------------------------

def build_hierarchy(df):

    print("\nBuilding item-store hierarchy...")

    # Validate store-state mapping.
    store_state = (
        df[["store_id", "state_id"]]
        .drop_duplicates()
    )

    if store_state["store_id"].duplicated().any():
        raise ValueError(
            "A store belongs to multiple states."
        )

    dates = sorted(df["date"].unique())

    # Bottom-level item-store components.
    bottom_mapping = {}

    unique_series = (
        df[
            ["id", "item_id", "store_id", "state_id"]
        ]
        .drop_duplicates()
    )

    for row in unique_series.itertuples(index=False):

        bottom_mapping[row.id] = {
            "component": f"bottom__{row.id}",
            "item_id": row.item_id,
            "store_id": row.store_id,
            "state_id": row.state_id,
            "level": "item_store",
        }

    # Store-level components.
    store_ids = sorted(df["store_id"].unique())

    store_components = {
        store: f"store__{store}"
        for store in store_ids
    }

    # State-level components.
    state_ids = sorted(df["state_id"].unique())

    state_components = {
        state: f"state__{state}"
        for state in state_ids
    }

    # National-level component.
    national_component = "national__US"

    # -----------------------------------------------------
    # Create bottom-level forecast matrix
    # -----------------------------------------------------

    bottom_df = df.copy()

    bottom_df["component"] = bottom_df["id"].map(
        lambda x: bottom_mapping[x]["component"]
    )

    bottom_wide = bottom_df.pivot(
        index="date",
        columns="component",
        values="lightgbm_prediction",
    ).sort_index()

    if bottom_wide.isnull().any().any():
        raise ValueError(
            "Missing dates detected in bottom-level forecasts."
        )

    if len(bottom_wide.index) != len(dates):
        raise ValueError(
            "Unexpected forecast date count."
        )

    # -----------------------------------------------------
    # Create aggregate components
    # -----------------------------------------------------

    aggregate_components = (
        list(store_components.values())
        + list(state_components.values())
        + [national_component]
    )

    aggregate_wide = pd.DataFrame(
        0.0,
        index=bottom_wide.index,
        columns=aggregate_components,
    )

    all_components = pd.concat(
        [bottom_wide, aggregate_wide],
        axis=1,
    )

    all_components.index.name = "date"

    # -----------------------------------------------------
    # Build Darts hierarchy dictionary
    #
    # Child component -> Parent component
    # -----------------------------------------------------

    hierarchy = {}

    # Item-store -> Store.
    for series_id, info in bottom_mapping.items():

        hierarchy[info["component"]] = [
            store_components[info["store_id"]]
        ]

    # Store -> State.
    for row in store_state.itertuples(index=False):

        hierarchy[store_components[row.store_id]] = [
            state_components[row.state_id]
        ]

    # State -> National.
    for state in state_ids:

        hierarchy[state_components[state]] = [
            national_component
        ]

    print(
        f"Bottom-level components: {len(bottom_mapping)}"
    )

    print(
        f"Store components: {len(store_components)}"
    )

    print(
        f"State components: {len(state_components)}"
    )

    print(
        f"Hierarchy edges: {len(hierarchy)}"
    )

    return (
        all_components,
        hierarchy,
        bottom_mapping,
        store_components,
        state_components,
        national_component,
        store_state,
    )


# ---------------------------------------------------------
# 3. CONVERT TO DARTS TIMESERIES
# ---------------------------------------------------------

def convert_to_darts(
    all_components,
    hierarchy,
):

    print("\nConverting data into Darts TimeSeries...")

    frame = all_components.reset_index()

    series = TimeSeries.from_dataframe(
        frame,
        time_col="date",
        value_cols=list(all_components.columns),
        freq="D",
    )

    # Attach hierarchy metadata.
    series = series.with_hierarchy(hierarchy)

    print(
        "Darts hierarchy attached:",
        series.has_hierarchy,
    )

    print(
        "Darts components:",
        series.n_components,
    )

    print(
        "Bottom-level components:",
        len(series.bottom_level_components),
    )

    return series


# ---------------------------------------------------------
# 4. APPLY BOTTOM-UP RECONCILIATION
# ---------------------------------------------------------

def reconcile_forecasts(series):

    print("\nApplying BottomUpReconciliator...")

    reconciliator = BottomUpReconciliator()

    # Darts version uses transform(), not fit_transform().
    reconciled = reconciliator.transform(series)

    print("Darts reconciliation completed.")

    return reconciled


# ---------------------------------------------------------
# 5. VERIFY HIERARCHICAL CONSISTENCY
# ---------------------------------------------------------

def verify_hierarchy(
    wide,
    bottom_mapping,
    store_components,
    state_components,
    national_component,
    store_state,
):

    print("\nVerifying hierarchical consistency...")

    errors = []

    # -----------------------------------------------------
    # Item-store -> Store validation
    # -----------------------------------------------------

    bottom_by_store = {}

    for info in bottom_mapping.values():

        bottom_by_store.setdefault(
            info["store_id"],
            [],
        ).append(
            info["component"]
        )

    for store, store_component in store_components.items():

        expected = wide[
            bottom_by_store[store]
        ].sum(axis=1)

        difference = (
            wide[store_component] - expected
        ).abs().max()

        errors.append(float(difference))

    # -----------------------------------------------------
    # Store -> State validation
    # -----------------------------------------------------

    store_by_state = {}

    for row in store_state.itertuples(index=False):

        store_by_state.setdefault(
            row.state_id,
            [],
        ).append(
            store_components[row.store_id]
        )

    for state, state_component in state_components.items():

        expected = wide[
            store_by_state[state]
        ].sum(axis=1)

        difference = (
            wide[state_component] - expected
        ).abs().max()

        errors.append(float(difference))

    # -----------------------------------------------------
    # State -> National validation
    # -----------------------------------------------------

    expected_national = wide[
        list(state_components.values())
    ].sum(axis=1)

    national_error = (
        wide[national_component] - expected_national
    ).abs().max()

    errors.append(float(national_error))

    max_error = (
        max(errors) if errors else 0.0
    )

    passed = bool(
        np.isclose(max_error, 0.0, atol=1e-6)
    )

    print(
        f"Maximum aggregation difference: {max_error:.10f}"
    )

    print(
        "Hierarchy consistency:",
        "PASSED" if passed else "FAILED",
    )

    if not passed:

        raise AssertionError(
            "Hierarchical consistency validation failed."
        )

    return max_error


# ---------------------------------------------------------
# 6. CREATE OUTPUT METADATA
# ---------------------------------------------------------

def create_metadata(
    bottom_mapping,
    store_components,
    state_components,
    national_component,
    store_state,
):

    metadata = {}

    # Bottom-level metadata.
    for info in bottom_mapping.values():

        component = info["component"]

        metadata[component] = {
            "level": "item_store",
            "item_id": info["item_id"],
            "store_id": info["store_id"],
            "state_id": info["state_id"],
        }

    # Store-level metadata.
    for store, component in store_components.items():

        state = store_state.loc[
            store_state["store_id"] == store,
            "state_id",
        ].iloc[0]

        metadata[component] = {
            "level": "store",
            "item_id": "",
            "store_id": store,
            "state_id": state,
        }

    # State-level metadata.
    for state, component in state_components.items():

        metadata[component] = {
            "level": "state",
            "item_id": "",
            "store_id": "",
            "state_id": state,
        }

    # National-level metadata.
    metadata[national_component] = {
        "level": "national",
        "item_id": "",
        "store_id": "",
        "state_id": "ALL",
    }

    # Explicit records avoid duplicate component columns.
    metadata_df = pd.DataFrame(
        [
            {
                "component": component,
                **info,
            }
            for component, info in metadata.items()
        ]
    )

    return metadata_df


# ---------------------------------------------------------
# 7. SAVE RECONCILED FORECASTS
# ---------------------------------------------------------

def save_outputs(
    reconciled,
    bottom_mapping,
    store_components,
    state_components,
    national_component,
    store_state,
):

    print("\nPreparing reconciled forecast output...")

    # Darts 0.48.1 uses to_dataframe().
    wide = reconciled.to_dataframe()

    # Standardize index name before converting to long format.
    wide.index.name = "date"

    wide = wide.reset_index()

    long_df = wide.melt(
        id_vars="date",
        var_name="component",
        value_name="forecast",
    )

    metadata_df = create_metadata(
        bottom_mapping,
        store_components,
        state_components,
        national_component,
        store_state,
    )

    final_df = long_df.merge(
        metadata_df,
        on="component",
        how="left",
        validate="many_to_one",
    )

    final_df = final_df[
        [
            "date",
            "level",
            "component",
            "item_id",
            "store_id",
            "state_id",
            "forecast",
        ]
    ]

    final_df["date"] = pd.to_datetime(
        final_df["date"]
    )

    final_df = final_df.sort_values(
        [
            "date",
            "level",
            "state_id",
            "store_id",
            "item_id",
        ]
    )

    # Save CSV.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return final_df, wide


# ---------------------------------------------------------
# 8. SAVE SUMMARY REPORT
# ---------------------------------------------------------

def save_summary(
    final_df,
    wide,
    max_error,
    bottom_mapping,
    store_components,
    state_components,
):

    summary = {
        "method": "Darts BottomUpReconciliator",

        "hierarchy": (
            "Item-Store -> Store -> State -> National"
        ),

        "bottom_level_series": len(bottom_mapping),

        "store_nodes": len(store_components),

        "state_nodes": len(state_components),

        "national_nodes": 1,

        "forecast_dates": int(len(wide)),

        "total_hierarchy_components": int(
            len(bottom_mapping)
            + len(store_components)
            + len(state_components)
            + 1
        ),

        "output_rows": int(len(final_df)),

        "max_aggregation_difference": float(
            max_error
        ),

        "consistency_check": "PASSED",

        "output_file": str(OUTPUT_FILE),
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    return summary


# ---------------------------------------------------------
# 9. MAIN EXECUTION
# ---------------------------------------------------------

def main():

    # Step 1: Load forecasts.
    df = load_forecasts()

    # Step 2: Build hierarchy.
    (
        all_components,
        hierarchy,
        bottom_mapping,
        store_components,
        state_components,
        national_component,
        store_state,
    ) = build_hierarchy(df)

    # Step 3: Convert into Darts TimeSeries.
    series = convert_to_darts(
        all_components,
        hierarchy,
    )

    # Step 4: Reconcile forecasts.
    reconciled = reconcile_forecasts(series)

    # Step 5: Convert and verify.
    wide_for_check = reconciled.to_dataframe()

    max_error = verify_hierarchy(
        wide_for_check,
        bottom_mapping,
        store_components,
        state_components,
        national_component,
        store_state,
    )

    # Step 6: Save CSV.
    final_df, wide = save_outputs(
        reconciled,
        bottom_mapping,
        store_components,
        state_components,
        national_component,
        store_state,
    )

    # Step 7: Save JSON summary.
    summary = save_summary(
        final_df,
        wide,
        max_error,
        bottom_mapping,
        store_components,
        state_components,
    )

    # -----------------------------------------------------
    # FINAL RESULTS
    # -----------------------------------------------------

    print("\n=======================================================")
    print("DARTS HIERARCHICAL RECONCILIATION RESULTS")
    print("=======================================================")

    print(
        f"Bottom-level series: {summary['bottom_level_series']}"
    )

    print(
        f"Store nodes: {summary['store_nodes']}"
    )

    print(
        f"State nodes: {summary['state_nodes']}"
    )

    print(
        f"National nodes: {summary['national_nodes']}"
    )

    print(
        f"Total components: {summary['total_hierarchy_components']}"
    )

    print(
        f"Forecast dates: {summary['forecast_dates']}"
    )

    print(
        f"Output rows: {summary['output_rows']:,}"
    )

    print(
        "Maximum aggregation difference:",
        f"{max_error:.10f}",
    )

    print(
        "Consistency check: PASSED"
    )

    print(
        f"\nSaved forecasts: {OUTPUT_FILE}"
    )

    print(
        f"Saved summary: {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
