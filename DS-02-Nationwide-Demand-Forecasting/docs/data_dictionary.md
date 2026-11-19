

\# Data Dictionary

\## Nationwide Demand Forecasting and Inventory Optimization



\## 1. Raw Datasets



\### sales\_train\_validation.csv



| Column | Description |

|---|---|

| id | Unique item-store identifier |

| item\_id | Product identifier |

| dept\_id | Department identifier |

| cat\_id | Product category |

| store\_id | Store identifier |

| state\_id | State identifier |

| d\_1 to d\_1913 | Daily historical sales |



\### calendar.csv



| Column | Description |

|---|---|

| date | Calendar date |

| wm\_yr\_wk | Walmart week identifier |

| weekday | Weekday name |

| month | Month |

| year | Year |

| event\_name\_1 | Primary event |

| event\_type\_1 | Primary event category |

| snap\_CA | California SNAP indicator |

| snap\_TX | Texas SNAP indicator |

| snap\_WI | Wisconsin SNAP indicator |



\### sell\_prices.csv



| Column | Description |

|---|---|

| store\_id | Store identifier |

| item\_id | Product identifier |

| wm\_yr\_wk | Walmart week identifier |

| sell\_price | Selling price |



\## 2. Engineered Features



| Feature | Description |

|---|---|

| lag\_7 | Sales seven days earlier |

| lag\_14 | Sales fourteen days earlier |

| lag\_28 | Sales twenty-eight days earlier |

| rolling\_mean\_7 | Seven-day rolling average |

| rolling\_mean\_28 | Twenty-eight-day rolling average |

| day\_of\_week | Day number |

| month | Calendar month |

| day\_of\_month | Day within month |

| week\_of\_year | Week number |

| is\_event | Event indicator |

| snap | SNAP indicator |

| sell\_price | Selling price |

| price\_missing | Missing price indicator |



\## 3. Forecast Output



\### forecast\_predictions.csv



| Column | Description |

|---|---|

| id | Item-store series identifier |

| item\_id | Product |

| store\_id | Store |

| state\_id | State |

| date | Forecast validation date |

| sales | Actual demand |

| naive\_prediction | Baseline prediction |

| lightgbm\_prediction | LightGBM prediction |



\### forecast\_with\_intervals.csv



Includes the forecast columns plus:



| Column | Description |

|---|---|

| lower\_90 | Lower empirical prediction bound |

| upper\_90 | Upper empirical prediction bound |



\## 4. Inventory Output



\### inventory\_recommendations.csv



| Column | Description |

|---|---|

| item\_id | Product identifier |

| store\_id | Store identifier |

| state\_id | State identifier |

| forecast\_7\_day\_demand | Aggregated demand estimate |

| lower\_demand | Lower scenario estimate |

| upper\_demand | Upper scenario estimate |

| assumed\_initial\_stock | Assumed starting inventory |

| order\_quantity | Optimized order quantity |

| expected\_shortage | Scenario-weighted shortage |

| expected\_excess | Scenario-weighted excess |



\## 5. Model Metrics



| Metric | Meaning |

|---|---|

| WAPE | Weighted Absolute Percentage Error |

| RMSSE | Root Mean Squared Scaled Error |



Lower values indicate better forecasting performance.



\## 6. Hierarchy



The M5 dataset contains the following hierarchy:



\- State

\- Store

\- Item



The current dashboard provides state-level aggregation of the selected item-store sample. Full hierarchical reconciliation is not yet implemented.



