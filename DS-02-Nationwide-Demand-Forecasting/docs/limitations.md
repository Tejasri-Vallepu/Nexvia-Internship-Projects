\# Limitations and Future Improvements



\## Current Limitations



\### 1. Sampled Dataset



The source dataset contains 30,490 item-store time series. The current forecasting experiments use 500 sampled series, so the results do not represent full-dataset training.



\### 2. TFT Training Scope



The Temporal Fusion Transformer experiment uses 100 sampled series because of computational constraints. Its metrics should not be directly compared as a controlled model benchmark against models trained on 500 series.



\### 3. Forecast Evaluation



The current validation process uses rolling one-step-ahead predictions. A fixed-origin multi-step evaluation would better represent operational forecasting.



\### 4. Prediction Intervals



Prediction intervals are based on global validation residual quantiles. They are not independently calibrated by product, store, or forecast horizon.



\### 5. Inventory Assumptions



Initial stock, ordering cost, holding cost, shortage cost, and scenario probabilities are illustrative values. Actual operational data would be required for deployment.



\### 6. Hierarchical Scope



The Darts reconciliation output is internally consistent for the sampled 500 item-store series. It does not represent all Walmart products and stores.



\### 7. Evaluation Metric



The reported RMSSE is an unweighted average of per-series scaled errors, not the official M5 WRMSSE metric.



\### 8. Deployment



The current dashboard is a local Dash application. A production API, authentication, automated retraining, and cloud deployment are not part of the verified implementation.



\## Future Improvements



1\. Train and evaluate on all 30,490 item-store series.

2\. Compare all forecasting models on the same sample and validation dates.

3\. Add multi-horizon forecasting and rolling-origin backtesting.

4\. Implement official M5 WRMSSE evaluation.

5\. Develop series-specific and horizon-specific prediction intervals.

6\. Integrate actual inventory, supplier lead times, and replenishment schedules.

7\. Tune optimization costs using business-provided values.

8\. Add a REST API for forecast and inventory services.

9\. Add automated retraining and model monitoring.

10\. Deploy the dashboard and services to a cloud platform.



\## Conclusion



The current implementation demonstrates the major stages of a retail demand forecasting and inventory optimization workflow. The next stage is to improve scalability, operational assumptions, and production integration.



