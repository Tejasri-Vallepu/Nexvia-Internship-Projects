\# Test Results Report



\## Test Execution



Command:



```bash

python -m pytest -v

```



\## Reported Result



\*\*6 passed, 1 warning\*\*



\## Test Coverage



| Test area | Result |

|---|---|

| Required output files | Passed |

| Forecast output schema | Passed |

| Forecast values | Passed |

| Inventory recommendations | Passed |

| Model metrics | Passed |

| LightGBM versus Naive WAPE check | Passed |



\## Additional Pipeline Verification



\### Forecasting



\- Naive Lag-7 WAPE: 88.5971%

\- Prophet WAPE: 75.9410%

\- LightGBM WAPE: 71.3071%

\- TFT WAPE: 77.2530%



\### Hierarchical Reconciliation



\- Output rows: 14,392

\- Components: 514

\- Maximum aggregation difference: 4.547473508864641e-13

\- Consistency check: PASSED



\### Inventory Optimization



\- Recommendations generated: 500

\- Optimizer: Google OR-Tools SCIP



\## Warning



One warning was reported from a Dash testing plugin regarding pytest compatibility. The six tests passed successfully.



\## Scope of Validation



The test suite verifies generated output artifacts and selected model behavior. It does not establish production readiness, full-dataset scalability, or performance under real operational inventory conditions.



\## Conclusion



The reported automated tests passed. Additional integration and deployment testing should be performed before production use.



