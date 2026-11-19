\# Dataset and Data Interface Documentation



\## Dataset Source



M5 Forecasting – Walmart Sales



https://zenodo.org/records/10203108



\## Source Files



| File | Purpose |

|---|---|

| sales\_train\_validation.csv | Historical daily sales by item and store |

| calendar.csv | Date, weekday, events, SNAP information |

| sell\_prices.csv | Weekly item-store selling prices |



\## Processing



The project processes a recent 393-day historical window, converts the sales data from wide format into long format, and joins calendar and price information.



The resulting processed sales dataset contains:



\- 11,982,570 rows

\- 25 columns

\- 500 sampled item-store combinations used in forecasting experiments



The feature dataset contains 182,500 rows before the train/validation split and 34 columns.



\## Feature Groups



\### Historical demand

\- sales

\- lag\_7

\- lag\_14

\- lag\_28

\- rolling\_mean\_7

\- rolling\_mean\_28



Rolling features are calculated using shifted observations to avoid using the current target in its own feature values.



\### Calendar

\- weekday

\- day\_of\_week

\- day\_of\_month

\- month

\- week\_of\_year

\- is\_event



\### Store and price

\- item\_id

\- store\_id

\- state\_id

\- sell\_price

\- price\_missing

\- SNAP indicators



\## Output Data Interfaces



\### forecast\_predictions.csv



Contains actual sales and model predictions for validation dates.



Key fields:

\- id

\- item\_id

\- store\_id

\- state\_id

\- date

\- sales

\- naive\_prediction

\- lightgbm\_prediction



\### forecast\_with\_intervals.csv



Contains point forecasts and interval estimates.



Key fields include:

\- Actual sales

\- Forecast values

\- Lower prediction bound

\- Upper prediction bound



\### prophet\_predictions.csv



Contains Prophet predictions for the sampled validation series.



\### tft\_predictions.csv



Contains Temporal Fusion Transformer predictions for the smaller TFT experiment sample.



\### darts\_reconciled\_forecasts.csv



Contains hierarchical reconciled forecasts.



| Field | Description |

|---|---|

| date | Forecast date |

| level | item\_store, store, state, or national |

| component | Darts hierarchy component |

| item\_id | Product identifier, where applicable |

| store\_id | Store identifier, where applicable |

| state\_id | State identifier, where applicable |

| forecast | Reconciled forecast value |



\### inventory\_recommendations.csv



Contains inventory decisions produced by the optimization module.



Fields include:

\- item\_id

\- store\_id

\- state\_id

\- forecast\_7\_day\_demand

\- lower\_demand

\- upper\_demand

\- assumed\_initial\_stock

\- order\_quantity

\- expected\_shortage

\- expected\_excess



\## API / Application Interface



The current project is a local batch-processing and dashboard application. It does not claim to expose a deployed REST API.



The dashboard is served by Plotly Dash:



`http://127.0.0.1:8050`



The dashboard reads the generated CSV and JSON files from the `outputs/` directory.



To integrate the forecasting system with an external application in the future, a REST API can expose:

\- Forecast generation

\- Model evaluation metrics

\- Hierarchical forecasts

\- Inventory recommendations



These REST endpoints are future work, not currently implemented API endpoints.



