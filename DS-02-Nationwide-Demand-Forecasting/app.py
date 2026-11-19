
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

FORECAST_FILE = OUTPUT_DIR / "forecast_with_intervals.csv"
INVENTORY_FILE = OUTPUT_DIR / "inventory_recommendations.csv"
METRICS_FILE = OUTPUT_DIR / "model_metrics.json"
DARTS_FILE = OUTPUT_DIR / "darts_reconciled_forecasts.csv"

OPTIONAL_MODEL_FILES = {
    "Prophet": OUTPUT_DIR / "prophet_predictions.csv",
    "TFT": OUTPUT_DIR / "tft_predictions.csv",
}

for file_path in [FORECAST_FILE, INVENTORY_FILE, METRICS_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required project output missing: {file_path}"
        )

forecast_df = pd.read_csv(FORECAST_FILE)
inventory_df = pd.read_csv(INVENTORY_FILE)

with open(METRICS_FILE, "r", encoding="utf-8") as file:
    metrics = json.load(file)


# =========================================================
# 2. VALIDATE AND PREPARE BASE DATA
# =========================================================

forecast_df["date"] = pd.to_datetime(forecast_df["date"])

for column in ["id", "item_id", "store_id", "state_id"]:
    forecast_df[column] = forecast_df[column].astype(str)

for column in ["item_id", "store_id", "state_id"]:
    inventory_df[column] = inventory_df[column].astype(str)

forecast_df = forecast_df.sort_values("date")

required_forecast_columns = {
    "id", "item_id", "store_id", "state_id", "date",
    "sales", "naive_prediction", "lightgbm_prediction",
    "lower_90", "upper_90"
}

required_inventory_columns = {
    "item_id", "store_id", "state_id",
    "forecast_7_day_demand", "lower_demand", "upper_demand",
    "assumed_initial_stock", "order_quantity",
    "expected_shortage", "expected_excess"
}

if not required_forecast_columns.issubset(forecast_df.columns):
    missing = required_forecast_columns - set(forecast_df.columns)
    raise ValueError(f"Forecast CSV missing columns: {missing}")

if not required_inventory_columns.issubset(inventory_df.columns):
    missing = required_inventory_columns - set(inventory_df.columns)
    raise ValueError(f"Inventory CSV missing columns: {missing}")

if forecast_df.empty or inventory_df.empty:
    raise ValueError("Forecast or inventory output file is empty.")

if "Naive_Lag_7" not in metrics or "LightGBM" not in metrics:
    raise KeyError("Naive_Lag_7 or LightGBM metrics are missing.")


# =========================================================
# 3. LOAD OPTIONAL MODEL PREDICTIONS
# =========================================================

def load_optional_predictions(model_name, file_path, base_df):
    """
    Load Prophet/TFT predictions and merge them into the main
    forecast dataframe using the available identifying columns.
    """

    if not file_path.exists():
        print(f"{model_name}: prediction file not found. Skipping.")
        return base_df

    model_df = pd.read_csv(file_path)

    if "date" not in model_df.columns:
        print(f"{model_name}: date column missing. Skipping.")
        return base_df

    model_df["date"] = pd.to_datetime(model_df["date"])

    candidate_columns = {
        "Prophet": [
            "prophet_prediction",
            "prophet_forecast",
            "prediction",
            "yhat",
            "forecast"
        ],
        "TFT": [
            "tft_prediction",
            "tft_forecast",
            "prediction",
            "forecast",
            "yhat"
        ]
    }

    prediction_column = next(
        (
            column
            for column in candidate_columns[model_name]
            if column in model_df.columns
        ),
        None
    )

    if prediction_column is None:
        excluded = {
            "date", "sales", "actual", "lower_90", "upper_90",
            "lower", "upper", "id", "item_id", "store_id",
            "state_id", "d"
        }

        numeric_columns = [
            column
            for column in model_df.select_dtypes(include="number").columns
            if column not in excluded
        ]

        if len(numeric_columns) == 1:
            prediction_column = numeric_columns[0]

    if prediction_column is None:
        print(
            f"{model_name}: prediction column could not be identified. "
            f"Available columns: {list(model_df.columns)}"
        )
        return base_df

    possible_keys = [
        "id", "item_id", "store_id", "state_id", "date"
    ]

    if all(key in model_df.columns for key in ["id", "date"]):
        merge_keys = ["id", "date"]
    elif all(
        key in model_df.columns
        for key in ["item_id", "store_id", "state_id", "date"]
    ):
        merge_keys = [
            "item_id", "store_id", "state_id", "date"
        ]
    else:
        print(
            f"{model_name}: required merge identifiers are missing. "
            f"Available columns: {list(model_df.columns)}"
        )
        return base_df

    keep_columns = merge_keys + [prediction_column]

    model_df = model_df[keep_columns].copy()

    model_df = model_df.rename(
        columns={prediction_column: f"{model_name.lower()}_prediction"}
    )

    for column in ["item_id", "store_id", "state_id", "id"]:
        if column in model_df.columns:
            model_df[column] = model_df[column].astype(str)

    model_df = model_df.drop_duplicates(subset=merge_keys)

    prediction_name = f"{model_name.lower()}_prediction"

    if prediction_name in base_df.columns:
        base_df = base_df.drop(columns=[prediction_name])

    merged = base_df.merge(
        model_df,
        on=merge_keys,
        how="left"
    )

    available = merged[prediction_name].notna().sum()

    print(
        f"{model_name}: loaded {available:,} matching prediction rows "
        f"using column '{prediction_column}'."
    )

    return merged


for model_name, file_path in OPTIONAL_MODEL_FILES.items():
    forecast_df = load_optional_predictions(
        model_name,
        file_path,
        forecast_df
    )


# =========================================================
# 4. LOAD DARTS HIERARCHICAL FORECASTS
# =========================================================

darts_df = pd.DataFrame()

if DARTS_FILE.exists():
    darts_df = pd.read_csv(DARTS_FILE)

    required_darts_columns = {
        "date", "level", "component",
        "item_id", "store_id", "state_id", "forecast"
    }

    if required_darts_columns.issubset(darts_df.columns):
        darts_df["date"] = pd.to_datetime(darts_df["date"])

        for column in [
            "level", "item_id", "store_id", "state_id"
        ]:
            darts_df[column] = darts_df[column].astype(str)

        print(
            f"Darts hierarchy loaded: "
            f"{len(darts_df):,} rows."
        )
    else:
        print(
            "Darts file found, but required columns are missing. "
            "Hierarchy chart will be unavailable."
        )
        darts_df = pd.DataFrame()
else:
    print("Darts hierarchy file not found. Skipping hierarchy chart.")


# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================

def empty_figure(message):
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=15)
    )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=380
    )

    return fig


def style_figure(fig, title):
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=25, t=65, b=45),
        legend=dict(orientation="h", y=1.12),
        hovermode="x unified"
    )

    return fig


def card(title, value, subtitle):
    return html.Div(
        [
            html.Div(
                title,
                style={
                    "fontSize": "13px",
                    "color": "#64748b",
                    "marginBottom": "8px"
                }
            ),
            html.Div(
                value,
                style={
                    "fontSize": "27px",
                    "fontWeight": "bold",
                    "color": "#14213d"
                }
            ),
            html.Div(
                subtitle,
                style={
                    "fontSize": "11px",
                    "color": "#64748b",
                    "marginTop": "5px"
                }
            )
        ],
        style={
            "backgroundColor": "white",
            "padding": "18px",
            "borderRadius": "10px",
            "boxShadow": "0 1px 5px rgba(0,0,0,0.08)",
            "flex": "1",
            "minWidth": "180px"
        }
    )


def metric_value(model, metric_name):
    if model not in metrics:
        return None

    value = metrics[model].get(metric_name)

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# =========================================================
# 6. DASH APPLICATION
# =========================================================

app = Dash(__name__)
app.title = "Nationwide Demand Intelligence"

states = sorted(forecast_df["state_id"].unique())

initial_state = states[0]

initial_stores = sorted(
    forecast_df.loc[
        forecast_df["state_id"] == initial_state,
        "store_id"
    ].unique()
)

initial_store = initial_stores[0]

initial_items = sorted(
    forecast_df.loc[
        (forecast_df["state_id"] == initial_state) &
        (forecast_df["store_id"] == initial_store),
        "item_id"
    ].unique()
)

initial_item = initial_items[0]

lightgbm_wape = metric_value("LightGBM", "WAPE_percent")
lightgbm_rmsse = metric_value("LightGBM", "RMSSE")
naive_wape = metric_value("Naive_Lag_7", "WAPE_percent")

total_series = forecast_df["id"].nunique()
total_recommendations = len(inventory_df)

available_models = [
    model
    for model in ["Naive_Lag_7", "Prophet", "LightGBM", "TFT"]
    if model in metrics
]

app.layout = html.Div(
    style={
        "backgroundColor": "#f1f5f9",
        "minHeight": "100vh",
        "padding": "15px",
        "fontFamily": "Arial, sans-serif"
    },
    children=[

        # HEADER
        html.Div(
            [
                html.H2(
                    "NATIONWIDE DEMAND INTELLIGENCE",
                    style={"margin": "0", "fontSize": "23px"}
                ),
                html.P(
                    "Demand forecasting | Model evaluation | "
                    "Hierarchical reconciliation | Inventory planning",
                    style={"margin": "6px 0 0", "fontSize": "13px"}
                )
            ],
            style={
                "backgroundColor": "#14213d",
                "color": "white",
                "padding": "22px",
                "borderRadius": "8px",
                "marginBottom": "18px"
            }
        ),

        # FILTERS
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "State",
                            style={"fontWeight": "bold"}
                        ),
                        dcc.Dropdown(
                            id="state-filter",
                            options=[
                                {"label": s, "value": s}
                                for s in states
                            ],
                            value=initial_state,
                            clearable=False
                        )
                    ],
                    style={"flex": "1"}
                ),

                html.Div(
                    [
                        html.Label(
                            "Store",
                            style={"fontWeight": "bold"}
                        ),
                        dcc.Dropdown(
                            id="store-filter",
                            options=[
                                {"label": s, "value": s}
                                for s in initial_stores
                            ],
                            value=initial_store,
                            clearable=False
                        )
                    ],
                    style={"flex": "1"}
                ),

                html.Div(
                    [
                        html.Label(
                            "Item",
                            style={"fontWeight": "bold"}
                        ),
                        dcc.Dropdown(
                            id="item-filter",
                            options=[
                                {"label": i, "value": i}
                                for i in initial_items
                            ],
                            value=initial_item,
                            clearable=False
                        )
                    ],
                    style={"flex": "1"}
                )
            ],
            style={
                "display": "flex",
                "gap": "15px",
                "backgroundColor": "white",
                "padding": "16px",
                "borderRadius": "8px",
                "marginBottom": "18px"
            }
        ),

        # KPI CARDS
        html.Div(
            [
                card(
                    "LightGBM WAPE",
                    f"{lightgbm_wape:.2f}%" if lightgbm_wape is not None else "N/A",
                    "Lower is better"
                ),
                card(
                    "LightGBM RMSSE",
                    f"{lightgbm_rmsse:.4f}" if lightgbm_rmsse is not None else "N/A",
                    "Scaled forecast error"
                ),
                card(
                    "Forecast Series",
                    f"{total_series:,}",
                    "Sampled item-store series"
                ),
                card(
                    "Inventory Recommendations",
                    f"{total_recommendations:,}",
                    "Generated by OR-Tools"
                )
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "15px",
                "marginBottom": "18px"
            }
        ),

        # MODEL METRICS
        html.Div(
            [
                dcc.Graph(id="metrics-chart")
            ],
            style={
                "backgroundColor": "white",
                "padding": "10px",
                "borderRadius": "8px",
                "marginBottom": "18px"
            }
        ),

        # FORECAST CHARTS
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="forecast-chart"),
                    style={
                        "backgroundColor": "white",
                        "padding": "10px",
                        "borderRadius": "8px",
                        "flex": "1"
                    }
                ),
                html.Div(
                    dcc.Graph(id="model-chart"),
                    style={
                        "backgroundColor": "white",
                        "padding": "10px",
                        "borderRadius": "8px",
                        "flex": "1"
                    }
                )
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "15px",
                "marginBottom": "18px"
            }
        ),

        # HIERARCHY LEVEL SELECTOR
        html.Div(
            [
                html.H3("Darts Hierarchical Forecast Reconciliation"),
                html.P(
                    "Explore reconciled forecasts at different "
                    "aggregation levels.",
                    style={"color": "#64748b", "fontSize": "13px"}
                ),
                dcc.Dropdown(
                    id="hierarchy-level-filter",
                    options=[
                        {"label": "Item – Store", "value": "item_store"},
                        {"label": "Store", "value": "store"},
                        {"label": "State", "value": "state"},
                        {"label": "National", "value": "national"}
                    ],
                    value="store",
                    clearable=False,
                    style={"maxWidth": "350px"}
                ),
                dcc.Graph(id="hierarchy-chart"),
                html.Div(
                    id="hierarchy-summary",
                    style={
                        "fontSize": "13px",
                        "color": "#475569",
                        "padding": "8px"
                    }
                )
            ],
            style={
                "backgroundColor": "white",
                "padding": "18px",
                "borderRadius": "8px",
                "marginBottom": "18px"
            }
        ),

        # STATE AND INVENTORY CHARTS
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="state-chart"),
                    style={
                        "backgroundColor": "white",
                        "padding": "10px",
                        "borderRadius": "8px",
                        "flex": "1"
                    }
                ),
                html.Div(
                    dcc.Graph(id="inventory-chart"),
                    style={
                        "backgroundColor": "white",
                        "padding": "10px",
                        "borderRadius": "8px",
                        "flex": "1"
                    }
                )
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "15px",
                "marginBottom": "18px"
            }
        ),

        # INVENTORY TABLE
        html.Div(
            [
                html.H3("Inventory Order Recommendations"),
                html.P(
                    "Recommended order quantities for the selected store.",
                    style={"color": "#64748b", "fontSize": "13px"}
                ),
                dash_table.DataTable(
                    id="inventory-table",
                    columns=[
                        {"name": "Item ID", "id": "item_id"},
                        {"name": "Store ID", "id": "store_id"},
                        {"name": "State ID", "id": "state_id"},
                        {
                            "name": "7-Day Demand",
                            "id": "forecast_7_day_demand",
                            "type": "numeric"
                        },
                        {
                            "name": "Lower Demand",
                            "id": "lower_demand",
                            "type": "numeric"
                        },
                        {
                            "name": "Upper Demand",
                            "id": "upper_demand",
                            "type": "numeric"
                        },
                        {
                            "name": "Initial Stock",
                            "id": "assumed_initial_stock",
                            "type": "numeric"
                        },
                        {
                            "name": "Order Quantity",
                            "id": "order_quantity",
                            "type": "numeric"
                        },
                        {
                            "name": "Expected Shortage",
                            "id": "expected_shortage",
                            "type": "numeric"
                        },
                        {
                            "name": "Expected Excess",
                            "id": "expected_excess",
                            "type": "numeric"
                        }
                    ],
                    data=[],
                    page_size=10,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "padding": "10px",
                        "fontSize": "12px",
                        "minWidth": "110px"
                    },
                    style_header={
                        "backgroundColor": "#14213d",
                        "color": "white",
                        "fontWeight": "bold"
                    },
                    style_data={
                        "backgroundColor": "white",
                        "color": "#14213d"
                    }
                )
            ],
            style={
                "backgroundColor": "white",
                "padding": "18px",
                "borderRadius": "8px",
                "marginBottom": "18px"
            }
        ),

        # PROJECT SUMMARY
        html.Div(
            id="selection-summary",
            style={
                "backgroundColor": "#e2e8f0",
                "padding": "16px",
                "borderRadius": "8px",
                "fontSize": "13px",
                "lineHeight": "1.7",
                "marginBottom": "12px"
            }
        ),

        html.Footer(
            "Nationwide Demand Forecasting & Inventory Optimization | M5 Dataset",
            style={
                "textAlign": "center",
                "color": "#64748b",
                "fontSize": "11px",
                "padding": "12px"
            }
        )
    ]
)


# =========================================================
# 7. MODEL METRICS CALLBACK
# =========================================================

@app.callback(
    Output("metrics-chart", "figure"),
    Input("state-filter", "value")
)
def update_metrics_chart(selected_state):

    model_labels = {
        "Naive_Lag_7": "Naive Lag-7",
        "Prophet": "Prophet",
        "LightGBM": "LightGBM",
        "TFT": "Temporal Fusion Transformer"
    }

    rows = []

    for model in available_models:
        wape = metric_value(model, "WAPE_percent")
        rmsse = metric_value(model, "RMSSE")

        if wape is not None:
            rows.append({
                "Model": model_labels.get(model, model),
                "WAPE (%)": wape,
                "RMSSE": rmsse
            })

    if not rows:
        return empty_figure("Model metrics are unavailable.")

    metric_df = pd.DataFrame(rows)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=metric_df["Model"],
            y=metric_df["WAPE (%)"],
            name="WAPE (%)",
            text=metric_df["WAPE (%)"].round(2),
            textposition="outside"
        )
    )

    style_figure(fig, "Forecast Model Performance – WAPE")

    fig.update_yaxes(title="WAPE (%)")

    return fig


# =========================================================
# 8. UPDATE STORE OPTIONS WHEN STATE CHANGES
# =========================================================

@app.callback(
    Output("store-filter", "options"),
    Output("store-filter", "value"),
    Input("state-filter", "value")
)
def update_store_options(selected_state):

    available_stores = sorted(
        forecast_df.loc[
            forecast_df["state_id"] == selected_state,
            "store_id"
        ].unique()
    )

    options = [
        {"label": store, "value": store}
        for store in available_stores
    ]

    selected_store = (
        available_stores[0] if available_stores else None
    )

    return options, selected_store


# =========================================================
# 9. UPDATE ITEM OPTIONS WHEN STORE CHANGES
# =========================================================

@app.callback(
    Output("item-filter", "options"),
    Output("item-filter", "value"),
    Input("state-filter", "value"),
    Input("store-filter", "value")
)
def update_item_options(selected_state, selected_store):

    if not selected_state or not selected_store:
        return [], None

    available_items = sorted(
        forecast_df.loc[
            (forecast_df["state_id"] == selected_state) &
            (forecast_df["store_id"] == selected_store),
            "item_id"
        ].unique()
    )

    options = [
        {"label": item, "value": item}
        for item in available_items
    ]

    selected_item = (
        available_items[0] if available_items else None
    )

    return options, selected_item


# =========================================================
# 10. DARTS HIERARCHY CALLBACK
# =========================================================

@app.callback(
    Output("hierarchy-chart", "figure"),
    Output("hierarchy-summary", "children"),
    Input("hierarchy-level-filter", "value"),
    Input("state-filter", "value"),
    Input("store-filter", "value"),
    Input("item-filter", "value")
)
def update_hierarchy_chart(
    selected_level,
    selected_state,
    selected_store,
    selected_item
):

    if darts_df.empty:
        return (
            empty_figure("Darts hierarchy output is unavailable."),
            "Check outputs/darts_reconciled_forecasts.csv."
        )

    data = darts_df[
        darts_df["level"] == selected_level
    ].copy()

    if selected_level == "item_store":
        data = data[
            (data["state_id"] == selected_state) &
            (data["store_id"] == selected_store) &
            (data["item_id"] == selected_item)
        ]

    elif selected_level == "store":
        data = data[
            (data["state_id"] == selected_state) &
            (data["store_id"] == selected_store)
        ]

    elif selected_level == "state":
        data = data[
            data["state_id"] == selected_state
        ]

    elif selected_level == "national":
        pass

    if data.empty:
        return (
            empty_figure("No reconciled forecasts for this selection."),
            "No hierarchy records found."
        )

    trend = (
        data.groupby("date", as_index=False)
        .agg(forecast=("forecast", "sum"))
        .sort_values("date")
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trend["date"],
            y=trend["forecast"],
            mode="lines+markers",
            name="Reconciled forecast"
        )
    )

    level_labels = {
        "item_store": "Item–Store",
        "store": "Store",
        "state": "State",
        "national": "National"
    }

    style_figure(
        fig,
        f"Darts Bottom-Up Reconciled Forecast – "
        f"{level_labels.get(selected_level, selected_level)}"
    )

    fig.update_yaxes(title="Forecast demand (units)")
    fig.update_xaxes(title="Date")

    summary = (
        f"Level: {level_labels.get(selected_level, selected_level)} | "
        f"Forecast records: {len(data):,} | "
        f"Forecast dates: {data['date'].nunique()} | "
        f"Total forecast across selected level: "
        f"{data['forecast'].sum():,.2f} units."
    )

    return fig, summary


# =========================================================
# 11. MAIN DASHBOARD CALLBACK
# =========================================================

@app.callback(
    Output("forecast-chart", "figure"),
    Output("model-chart", "figure"),
    Output("state-chart", "figure"),
    Output("inventory-chart", "figure"),
    Output("inventory-table", "data"),
    Output("selection-summary", "children"),
    Input("state-filter", "value"),
    Input("store-filter", "value"),
    Input("item-filter", "value")
)
def update_dashboard(
    selected_state,
    selected_store,
    selected_item
):

    if not all([
        selected_state,
        selected_store,
        selected_item
    ]):
        message = "Select a valid state, store and item."
        empty = empty_figure(message)

        return empty, empty, empty, empty, [], message

    # -----------------------------------------------------
    # SELECTED ITEM FORECAST
    # -----------------------------------------------------

    selected_data = forecast_df[
        (forecast_df["state_id"] == selected_state) &
        (forecast_df["store_id"] == selected_store) &
        (forecast_df["item_id"] == selected_item)
    ].sort_values("date")

    if selected_data.empty:

        forecast_fig = empty_figure(
            "No forecast records for this selection."
        )

    else:

        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=selected_data["date"],
                y=selected_data["upper_90"],
                mode="lines",
                line=dict(width=0),
                name="Upper 90% bound",
                showlegend=False
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=selected_data["date"],
                y=selected_data["lower_90"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(59,130,246,0.15)",
                name="90% prediction interval"
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=selected_data["date"],
                y=selected_data["sales"],
                mode="lines+markers",
                name="Actual sales",
                line=dict(width=2)
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=selected_data["date"],
                y=selected_data["lightgbm_prediction"],
                mode="lines+markers",
                name="LightGBM forecast",
                line=dict(width=2, dash="dash")
            )
        )

        style_figure(
            forecast_fig,
            f"Actual vs LightGBM Forecast – {selected_item}"
        )

        forecast_fig.update_yaxes(title="Daily units sold")
        forecast_fig.update_xaxes(title="Date")

    # -----------------------------------------------------
    # MODEL PREDICTION COMPARISON
    # -----------------------------------------------------

    if selected_data.empty:

        model_fig = empty_figure(
            "No model comparison data."
        )

    else:

        model_fig = go.Figure()

        model_fig.add_trace(
            go.Scatter(
                x=selected_data["date"],
                y=selected_data["sales"],
                mode="lines+markers",
                name="Actual sales"
            )
        )

        prediction_columns = [
            ("naive_prediction", "Naive Lag-7"),
            ("prophet_prediction", "Prophet"),
            ("lightgbm_prediction", "LightGBM"),
            ("tft_prediction", "TFT")
        ]

        for column, label in prediction_columns:

            if column not in selected_data.columns:
                continue

            if selected_data[column].notna().sum() == 0:
                continue

            model_fig.add_trace(
                go.Scatter(
                    x=selected_data["date"],
                    y=selected_data[column],
                    mode="lines",
                    name=label
                )
            )

        style_figure(
            model_fig,
            "Actual vs Available Model Forecasts – Selected Item"
        )

        model_fig.update_yaxes(title="Daily units sold")
        model_fig.update_xaxes(title="Date")

    # -----------------------------------------------------
    # STATE-LEVEL DEMAND TREND
    # -----------------------------------------------------

    state_data = forecast_df[
        forecast_df["state_id"] == selected_state
    ]

    if state_data.empty:

        state_fig = empty_figure("No state-level data.")

    else:

        state_trend = (
            state_data.groupby("date", as_index=False)
            .agg(
                actual_demand=("sales", "sum"),
                predicted_demand=(
                    "lightgbm_prediction",
                    "sum"
                )
            )
            .sort_values("date")
        )

        state_fig = go.Figure()

        state_fig.add_trace(
            go.Scatter(
                x=state_trend["date"],
                y=state_trend["actual_demand"],
                mode="lines",
                name="Actual state demand"
            )
        )

        state_fig.add_trace(
            go.Scatter(
                x=state_trend["date"],
                y=state_trend["predicted_demand"],
                mode="lines",
                name="Predicted state demand"
            )
        )

        style_figure(
            state_fig,
            f"Sampled Demand Trend – State {selected_state}"
        )

        state_fig.update_yaxes(title="Units sold (sample)")
        state_fig.update_xaxes(title="Date")

    # -----------------------------------------------------
    # INVENTORY RECOMMENDATIONS
    # -----------------------------------------------------

    store_inventory = inventory_df[
        (inventory_df["state_id"] == selected_state) &
        (inventory_df["store_id"] == selected_store)
    ].copy()

    if store_inventory.empty:

        inventory_fig = empty_figure(
            "No inventory recommendations for this store."
        )

    else:

        top_orders = (
            store_inventory.sort_values(
                "order_quantity",
                ascending=False
            ).head(15)
        )

        inventory_fig = px.bar(
            top_orders.sort_values("order_quantity"),
            x="order_quantity",
            y="item_id",
            orientation="h",
            title=f"Top Inventory Orders – {selected_store}",
            labels={
                "order_quantity": "Recommended order quantity",
                "item_id": "Item"
            },
            hover_data=[
                "forecast_7_day_demand",
                "expected_shortage",
                "expected_excess"
            ]
        )

        style_figure(
            inventory_fig,
            f"Top 15 Recommended Orders – {selected_store}"
        )

    # -----------------------------------------------------
    # INVENTORY TABLE
    # -----------------------------------------------------

    selected_inventory = store_inventory[
        store_inventory["item_id"] == selected_item
    ]

    if selected_inventory.empty:

        table_source = store_inventory

        table_message = (
            "No exact item recommendation found. "
            "Showing other recommendations for the selected store."
        )

    else:

        table_source = selected_inventory

        table_message = (
            "Showing recommendation for the selected item."
        )

    table_columns = [
        "item_id", "store_id", "state_id",
        "forecast_7_day_demand",
        "lower_demand", "upper_demand",
        "assumed_initial_stock", "order_quantity",
        "expected_shortage", "expected_excess"
    ]

    table_data = (
        table_source[table_columns]
        .sort_values("order_quantity", ascending=False)
        .head(50)
        .round(2)
        .to_dict("records")
    )

    # -----------------------------------------------------
    # SELECTED ITEM ORDER SUMMARY
    # -----------------------------------------------------

    selected_order = (
        selected_inventory.iloc[0]
        if not selected_inventory.empty
        else None
    )

    if selected_order is not None:

        order_summary = (
            f"Selected item recommended order quantity: "
            f"{selected_order['order_quantity']:.0f} units. "
            f"Expected shortage: "
            f"{selected_order['expected_shortage']:.2f}. "
            f"Expected excess: "
            f"{selected_order['expected_excess']:.2f}."
        )

    else:

        order_summary = table_message

    # -----------------------------------------------------
    # MODEL METRICS SUMMARY
    # -----------------------------------------------------

    model_summary_parts = []

    for model in available_models:

        wape = metric_value(model, "WAPE_percent")

        if wape is not None:

            label = {
                "Naive_Lag_7": "Naive Lag-7",
                "Prophet": "Prophet",
                "LightGBM": "LightGBM",
                "TFT": "TFT"
            }.get(model, model)

            model_summary_parts.append(
                f"{label} WAPE: {wape:.2f}%"
            )

    model_summary = " | ".join(model_summary_parts)

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    summary = html.Div(
        [
            html.H4("Project Methodology and Selected Results"),

            html.P(
                f"Selected filters: State = {selected_state} | "
                f"Store = {selected_store} | "
                f"Item = {selected_item}"
            ),

            html.P(
                f"Forecast validation records for selected item: "
                f"{len(selected_data)}"
            ),

            html.P(model_summary),

            html.P(
                f"LightGBM RMSSE: "
                f"{lightgbm_rmsse:.4f}"
                if lightgbm_rmsse is not None
                else "LightGBM RMSSE unavailable."
            ),

            html.P(order_summary),

            html.P(
                "Scope: 500 sampled item-store combinations. "
                "TFT was trained on 100 sampled series, while "
                "the other model runs used 500 series. Therefore, "
                "their overall metrics are not a like-for-like "
                "comparison. Forecast evaluation uses rolling "
                "one-step-ahead validation."
            ),

            html.P(
                "Prediction intervals use global validation "
                "residual quantiles. Inventory starting stock, "
                "scenario probabilities and costs are illustrative "
                "assumptions."
            ),

            html.P(
                "Darts Bottom-Up reconciliation aggregates the "
                "sampled item-store forecasts to store, state and "
                "national levels. State-level charts represent "
                "sampled demand, not the full Walmart state demand."
            )
        ]
    )

    return (
        forecast_fig,
        model_fig,
        state_fig,
        inventory_fig,
        table_data,
        summary
    )


# =========================================================
# 12. RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(
        debug=False,
        host="127.0.0.1",
        port=8050
    )
