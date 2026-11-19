\# Installation and Execution Guide



\## Requirements



\- Python 3.11

\- Windows, Linux, or macOS

\- Internet connection for initial package installation and dataset acquisition

\- Sufficient memory and CPU resources for model training



\## Step 1: Clone the repository



```bash

git clone https://github.com/Tejasri-Vallepu/Nationwide-Demand-Forecasting.git

cd Nationwide-Demand-Forecasting

```



\## Step 2: Create a virtual environment



Windows:



```cmd

python -m venv venv

venv\\Scripts\\activate

```



Linux/macOS:



```bash

python3 -m venv venv

source venv/bin/activate

```



\## Step 3: Install dependencies



```bash

python -m pip install --upgrade pip

pip install -r requirements.txt

```



Some forecasting packages, particularly PyTorch Forecasting and Darts, may require additional platform-specific dependency resolution.



\## Step 4: Prepare dataset



Download the M5 dataset from:



https://zenodo.org/records/10203108



Place the required source files in:



```text

data/raw/

&#x20;   calendar.csv

&#x20;   sales\_train\_validation.csv

&#x20;   sell\_prices.csv

```



\## Step 5: Run preprocessing and feature engineering



Use the scripts available in the repository's `src` directories.



Run them in the documented pipeline order. Check each script's command-line arguments and output paths before execution.



Expected processed outputs include:



\- `data/processed/sales\_long.parquet`

\- `data/processed/forecast\_features.parquet`



\## Step 6: Run forecasting models



Execute the baseline, Prophet, LightGBM, and TFT scripts in `src/forecasting/`.



The current experiment outputs include:



\- `outputs/forecast\_predictions.csv`

\- `outputs/prophet\_predictions.csv`

\- `outputs/tft\_predictions.csv`

\- `outputs/model\_metrics.json`



\## Step 7: Generate prediction intervals



Run the prediction-interval generation script from the project source.



Expected output:



`outputs/forecast\_with\_intervals.csv`



\## Step 8: Run hierarchical reconciliation



```cmd

python src\\reconciliation\\darts\_reconciliation.py

```



Expected outputs:



\- `outputs/darts\_reconciled\_forecasts.csv`

\- `outputs/darts\_reconciliation\_summary.json`



\## Step 9: Run inventory optimization



```cmd

python src\\optimization\\inventory\_optimizer.py

```



Expected output:



`outputs/inventory\_recommendations.csv`



\## Step 10: Run tests



```cmd

python -m pytest -v

```



The reported test run completed with 6 passed tests and 1 warning.



\## Step 11: Launch dashboard



```cmd

python app.py

```



Open:



http://127.0.0.1:8050



The dashboard reads generated outputs. Model training is performed separately by the forecasting scripts.



\## Notes



\- The project uses a reproducible sample of 500 item-store series for most experiments.

\- TFT uses 100 sampled series in the reported experiment.

\- The current experiment outputs should be regenerated if source data, features, model settings, or assumptions change.

\- Large raw datasets and model checkpoints may be excluded from Git and documented separately.



