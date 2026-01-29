# Marketing Mix Revenue Optimiser
E-commerce marketing teams must decide how to allocate limited ad budgets across multiple channels but lack tools showing instant revenue impact of reallocation decisions. This project delivers a Streamlit dashboard where users set total weekly budget ($10k-$100k) and drag sliders across 5 channels (Paid Search, Email, Display, Social, Affiliate) to see live revenue predictions and channel contribution breakdowns.

## Data Collection & Inspection
Sourced from Kaggle Marketing Dataset containing 2M+ transaction events (events.csv) and 103k transactions (transactions.csv). Initial inspection showed messy timestamped events needing aggregation to weekly marketing spend by channel and corresponding total revenue. Dataset spans 157 weeks. Due to GitHub file size limits, the raw dataset is not included in this repository.
The cleaned and preprocessed dataset used for analysis is provided.
Raw data source: <https://www.kaggle.com/datasets/geethasagarbonthu/marketing-and-e-commerce-analytics-dataset>

## Data Cleaning & Preprocessing
Transformed raw events into star schema warehouse with 4 tables totaling 334 rows: dim_channel.csv (5 channels), fact_marketing.csv (172 spend records), fact_revenue.csv (157 revenue weeks), mmm_data.csv (101 weeks × 6 columns). Used pivot_table to convert long-format spend into wide-format weekly matrix with columns, merged with revenue, normalized spend to 0-1 scale.[33][34][35][36][37]

## Methodology
Trained Ridge regression (alpha=1.0) on 101 weekly observations with time-based 80/20 train/test split. X contains 5 normalized spend columns, y is weekly revenue ($46k-$54k range). Results show R² train: 0.033, test: -0.004. Channel coefficients per $1 spend are Paid Search: $4479, Affiliate: $4416 (strongest), others negative or weak.

## GUI & Results Evaluation
Streamlit dashboard provides budget allocation panel with total budget slider and 5 channel sliders that must sum exactly (green checkmark when balanced). Summary metrics show predicted revenue updating live. Channel breakdown table displays each channel's budget allocation and revenue contribution. Bar chart visualizes positive (green) vs negative (red) channel impact. 

## Constraints & Limitations
R² 0.033 explains only 3% of revenue variance - 97% driven by unmodeled factors including seasonality, pricing changes, competitors, organic traffic, and adstock/lagged effects. Negative test R² confirms overfitting on small 101-week dataset. Model provides directional channel ranking only, not precise ROI forecasts.

## Input vs Processed Data Comparison
Raw data contains 2M+ messy events. Star schema creates 334 structured rows across dimension and fact tables. Modeling matrix reduces to 101×6 weekly observations. Live predictions use 1×5 spend vector input.

## Validation of Improvements
Budget sliders enforce mathematical constraint (sum equals total). Model displays R² prominently with directional warning. Test R² computation confirms overfitting. Live predictions respond correctly to slider changes. Negative coefficients correctly identified as channels to reduce spend.

## Organized Outputs & Reproducibility
Production artifacts include model.pkl (trained Ridge model), data_processed folder (star schema CSVs), app.py (Streamlit dashboard), and notebook.ipynb (complete analysis). Run locally with pip install -r requirements.txt followed by streamlit run app.py. Processed CSVs enable instant reproduction without raw data download.



