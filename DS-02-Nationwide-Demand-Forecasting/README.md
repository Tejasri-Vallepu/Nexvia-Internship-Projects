# Nationwide Demand Forecasting & Inventory Optimization

An end-to-end demand forecasting and inventory optimization system built using the M5 Walmart sales dataset. The project combines statistical forecasting, machine learning, deep learning, hierarchical forecast reconciliation, prediction intervals, and optimization to support data-driven inventory planning.

## 1. Problem Statement

Retail businesses need to estimate future product demand across multiple stores and locations. Poor forecasts can lead to stockouts or excess inventory.

This project develops a forecasting pipeline that predicts daily item-store demand, aggregates predictions across store, state, and national levels, and generates inventory order recommendations using scenario-based optimization.

## 2. Objectives

- Process historical retail sales data.
- Build item-store demand forecasting features.
- Compare baseline, statistical, machine learning, and deep learning models.
- Generate prediction intervals to represent forecast uncertainty.
- Reconcile forecasts across multiple hierarchy levels.
- Optimize inventory order quantities by considering shortage and excess costs.
- Present results through an interactive Plotly Dash dashboard.

## 3. Dataset

**Dataset:** M5 Forecasting – Walmart Sales

**Source:** https://zenodo.org/records/10203108

Files used:
- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

Dataset hierarchy:
- Product / Item
- Store
- State
- National

The project processes a recent 393-day period from the source sales history. The forecasting experiments use a reproducible sample of 500 item-store series, rather than training all 30,490 available series.

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Forecast features | Lag and rolling statistics |
| Baseline forecasting | Naive Lag-7 |
| Statistical forecasting | Prophet |
| Machine learning | LightGBM |
| Deep learning | PyTorch Forecasting, Temporal Fusion Transformer |
| Hierarchical reconciliation | Darts |
| Optimization | Google OR-Tools |
| Visualization | Plotly |
| Dashboard | Dash |
| Testing | Pytest |

## 5. Project Workflow

1. Load and preprocess the M5 sales, calendar, and price data.
2. Convert sales data into a long-format time series.
3. Merge calendar events, SNAP indicators, and sell prices.
4. Generate lag, rolling, and calendar features.
5. Train and evaluate forecasting models.
6. Generate prediction intervals using validation residual quantiles.
7. Reconcile bottom-level forecasts into store, state, and national totals using Darts Bottom-Up reconciliation.
8. Use scenario-based demand estimates to optimize inventory orders with OR-Tools.
9. Display forecasts and recommendations in the Dash dashboard.

## 6. Forecasting Models and Results

Validation results from the current experiments:

| Model | WAPE | RMSSE |
|---|---:|---:|
| Naive Lag-7 | 88.5971% | 1.0247 |
| Prophet | 75.9410% | 0.7916 |
| LightGBM | 71.3071% | 0.7542 |
| Temporal Fusion Transformer | 77.2530% | 0.8326 |

LightGBM achieved the lowest WAPE and RMSSE in the reported experiments.

**Evaluation scope:** Naive, Prophet, and LightGBM results use 500 sampled series. TFT uses 100 sampled series due to computational constraints. Therefore, the model metrics are not a strictly like-for-like comparison. The validation evaluation uses rolling one-step-ahead predictions.

The RMSSE reported here is an unweighted mean of per-series scaled errors, not the official M5 WRMSSE metric.

## 7. Hierarchical Forecast Reconciliation

Implemented using Darts `BottomUpReconciliator`.

Hierarchy:

`Item-Store → Store → State → National`

Verified output:
- Bottom-level series: 500
- Store nodes: 10
- State nodes: 3
- National nodes: 1
- Forecast dates: 28
- Total hierarchy components: 514
- Output rows: 14,392
- Maximum aggregation difference: approximately 4.55 × 10⁻¹³
- Consistency check: PASSED

The reconciliation output is available in `outputs/darts_reconciled_forecasts.csv`.

## 8. Prediction Intervals

Prediction intervals are constructed from global validation residual quantiles.

- Lower residual quantile: -1.921
- Upper residual quantile: 2.49
- Empirical validation coverage: 90%

The lower prediction bound is clipped at zero to avoid negative demand.

These intervals are based on validation residuals and are not independently calibrated future-horizon guarantees.

## 9. Inventory Optimization

Google OR-Tools is used for scenario-based inventory order recommendations.

The optimizer considers:
- Forecast demand scenarios
- Scenario probabilities
- Initial inventory
- Ordering cost
- Holding cost
- Shortage cost
- Expected shortage and excess

Illustrative scenario assumptions:
- Initial stock: 5 units
- Holding cost: 1
- Ordering cost: 0.2
- Shortage cost: 3
- Scenario probabilities: 0.2, 0.6, 0.2

One reported run produced:
- Recommendations: 500
- Total order quantity: 3,176
- Expected shortage: 1,617.4
- Expected excess: 1,122.8

These costs and initial stock values are illustrative assumptions, not actual Walmart operational parameters.

## 10. Interactive Dashboard

The Plotly Dash application provides:
- State, store, and item filters
- Forecast performance metrics
- Actual versus predicted demand
- Prediction interval visualization
- Inventory order recommendations
- State-level sampled demand trends
- Hierarchical forecast visualization

Run the dashboard using:

`python app.py`

Open `http://127.0.0.1:8050` in a browser.

## 11. Testing

The project test suite was executed using:

`python -m pytest -v`

Reported result: **6 passed, 1 warning**.

Tests covered:
- Required output files
- Forecast output schema
- Forecast values
- Inventory recommendation output
- Model metrics
- LightGBM WAPE compared with the Naive baseline

The warning originated from a Dash testing plugin compatibility notice.

## 12. Project Structure

```text
Nationwide-Demand-Forecasting/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── forecasting/
│   ├── reconciliation/
│   └── optimization/
│
├── models/
├── outputs/
├── tests/
└── docs/
```

## 13. Limitations

- Forecasting experiments use a sample of 500 item-store series, not all 30,490 series.
- TFT was trained on a smaller sample of 100 series.
- Model metrics use different sample sizes and should not be treated as a fully controlled comparison.
- The validation procedure uses rolling one-step-ahead predictions.
- Prediction intervals use global residual quantiles rather than series-specific or horizon-specific calibration.
- Inventory costs and starting stock are illustrative assumptions.
- State and national reconciled forecasts represent the sampled hierarchy, not the entire Walmart business.
- The reported RMSSE is not official M5 WRMSSE.

## 14. Future Improvements

- Scale training to all available item-store series.
- Evaluate all models on the same fixed series subset.
- Add multi-horizon forecasting and rolling-origin backtesting.
- Calibrate prediction intervals by series and forecast horizon.
- Incorporate actual inventory, lead times, and operational costs.
- Evaluate official M5 WRMSSE.
- Add automated model retraining and monitoring.
- Improve dashboard reporting and deployment.

## 15. Conclusion

This project demonstrates a complete experimental retail demand forecasting and inventory optimization workflow. It integrates statistical, machine learning, and deep learning forecasting with hierarchical reconciliation and optimization, and presents the results through an interactive dashboard.

The experiments show that LightGBM improves on the Naive Lag-7 baseline for the sampled validation series. The project also demonstrates how forecasts can be aggregated consistently and translated into inventory recommendations.

## Documentation

- [Architecture Diagram](docs/architecture.md)
- [Installation and Execution Guide](docs/installation.md)
- [Dataset Documentation](docs/dataset_documentation.md)
- [Test Results](docs/test_results.md)
- [Limitations and Future Improvements](docs/limitations.md)
- [Presentation Script](docs/presentation_script.md)