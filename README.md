\# Nexvia Internship Projects



This repository contains two Data Science internship projects developed as part of the Nexvia internship assignment.



\## Projects



\### DS-02: Nationwide Demand Forecasting \& Inventory Optimization



A demand forecasting and inventory planning system using historical retail sales data from the M5 Forecasting dataset.



\*\*Key components:\*\*

\- Data preprocessing and feature engineering

\- Naive Lag-7 baseline and Prophet forecasting

\- LightGBM machine learning model

\- Temporal Fusion Transformer (TFT)

\- Prediction intervals

\- Hierarchical forecast reconciliation using Darts

\- Inventory optimization using Google OR-Tools

\- Interactive dashboard using Plotly Dash



\*\*Project results:\*\*

\- LightGBM WAPE: 71.31%

\- Prophet WAPE: 75.94%

\- TFT WAPE: 77.25% (100-series experiment)

\- 500 inventory recommendations generated

\- 6 automated tests passed



\[View DS-02 Project](DS-02-Nationwide-Demand-Forecasting/README.md)



\### DS-03: Large-Scale Social Media Intelligence Platform



A social media analytics project focused on extracting insights from online discussions and analyzing sentiment, topics, and relationships.



\*\*Project focus:\*\*

\- Social media data collection

\- Text preprocessing and multilingual sentiment analysis

\- Topic modeling using BERTopic

\- Network analysis using NetworkX and graph-based methods

\- Misinformation-related analysis



\[View DS-03 Project](DS-03-Social-Media-Intelligence/README.md)



\## Repository Structure



```text

Nexvia-Internship-Projects/

│

├── DS-02-Nationwide-Demand-Forecasting/

│   ├── src/

│   ├── data/

│   ├── models/

│   ├── outputs/

│   ├── tests/

│   ├── docs/

│   ├── app.py

│   └── README.md

│

├── DS-03-Social-Media-Intelligence/

│   ├── src/

│   ├── data/

│   ├── models/

│   ├── reports/

│   ├── notebooks/

│   ├── tests/

│   └── README.md

│

├── .gitignore

└── README.md

```



\## Technologies



\*\*DS-02:\*\* Python, Pandas, NumPy, LightGBM, Prophet, PyTorch Forecasting, Darts, OR-Tools, Plotly, Dash.



\*\*DS-03:\*\* Python, NLP, sentiment analysis, BERTopic, NetworkX, Neo4j.



\## Setup



Each project has its own README and requirements file. Refer to the individual project documentation for installation, configuration, and execution instructions.



\## Notes



\- Raw datasets are not included in this repository.

\- Download and configure the datasets as described in the respective project documentation.

\- API credentials and environment secrets are not committed.

\- Model and evaluation results should be interpreted with the experimental scope and assumptions documented in each project.



\## Internship Assignment



Nexvia Data Science Internship — DS-02 and DS-03.



