## SalesDash – Retail Sales Forecasting Dashboard

**Author:** Podmaraj Boruah
**Project Type:** Machine Learning · Time Series Forecasting · Data Visualization
**Created On:** 28 December 2025

## Project Overview

**SalesDash** is an end-to-end **retail sales forecasting system** built using Python and Streamlit.
The project focuses on predicting **future retail sales trends** using **time-series models** and presenting insights through a **professionally designed interactive dashboard**.

The complete workflow—from raw data preprocessing to model evaluation and dashboard deployment—is implemented following **industry best practices**.

## Problem Statement

Retail businesses require accurate sales forecasting to:

* Optimize inventory planning
* Reduce overstock and stockouts
* Support data-driven decision-making

This project solves the problem by learning temporal patterns from historical retail sales data and forecasting future demand.

## Models Evaluated

Three time-series forecasting models were implemented and tested:

| Model       | Purpose                            |
| ----------- | ---------------------------------- |
| **ARIMA**   | Captures trend & autocorrelation   |
| **SARIMA**  | Handles seasonality explicitly     |
| **Prophet** | Flexible trend + seasonality model |

###  Model Selection Strategy

* **Train–test split** (last 30 days as test set)
* **Evaluation metric:** RMSE
* All models were trained and evaluated under identical conditions

###  Best Model

➡ **ARIMA** achieved the **lowest RMSE** and was selected for final deployment.

##  Tech Stack

### Programming & Frameworks

* **Python 3.x**
* **Streamlit** – interactive web dashboard

### Data Science & Time Series

* **Pandas** – data manipulation
* **NumPy** – numerical operations
* **Matplotlib** – data visualization
* **Statsmodels** – ARIMA & SARIMA
* **Prophet** – advanced forecasting
* **Scikit-learn** – model evaluation
* **Joblib** – model serialization

### Environment & Tooling

* **uv** – fast Python dependency & virtual environment manager

### Frontend Styling

* **Custom CSS** (`streamlit/assets/style.css`)
---

## 📂 Project Folder Structure

```
SalesDash/
│
├── data/
│   └── retail_store_inventory.csv      # Raw dataset (Kaggle)
│
├── streamlit/
│   ├── app.py                          # Streamlit dashboard
│   ├── cleaned_retail_sales.csv        # Cleaned daily sales data
│   ├── forecast_retail_sales_arima.csv # ARIMA forecast output
│   ├── arima_model.pkl                 # Trained ARIMA model
│   └── assets/
│       └── style.css                   # Custom UI styling
│
├── test.py                             # Data cleaning, EDA & model comparison
├── trained.py                          # Final ARIMA training & saving
├── main.py                             # Reserved / placeholder
└── README.md                           # Project documentation
```
## 🔄 Project Workflow

### 1️⃣ Data Cleaning & Model Comparison (`test.py`)

* Raw data ingestion
* Column standardization
* Date parsing & validation
* Missing value handling
* Daily sales aggregation
* Training & evaluation of:

  * ARIMA
  * SARIMA
  * Prophet
* RMSE-based model comparison

### 2️⃣ Final Model Training (`trained.py`)

* Train ARIMA on full cleaned dataset
* Generate **30-day sales forecast**
* Compute confidence intervals
* Save:

  * Cleaned dataset
  * Forecast CSV
  * Trained ARIMA model (`.pkl`)
* All outputs stored inside the `streamlit/` directory for deployment

### 3️⃣ Dashboard Deployment (`streamlit/app.py`)

The Streamlit dashboard includes:

## 📌 KPI Metrics

* Total Sales
* Average Daily Sales
* Peak Sales Value

## Visualizations

1. Historical Sales Trend
2. Historical vs 30-Day Forecast
3. Zoomed-In View (Recent Sales + Forecast)

## Forecast Table

* Clean tabular forecast output
* Enhanced readability with larger fonts
* Centered alignment for clarity

## UI Enhancements

* Custom CSS styling
* Professional typography
* Responsive layout
* High-quality, compact visualizations

##  How to Run the Project

### Step 1: Create Environment & Install Dependencies (using `uv`)

```bash
uv venv
uv pip install pandas numpy matplotlib streamlit statsmodels prophet scikit-learn joblib
```

### Step 2: Train Model & Generate Forecast

```bash
python trained.py
```

### Step 3: Launch Dashboard

```bash
streamlit run streamlit/app.py
```

---

## Forecast Details

* Forecast Horizon: **30 Days**
* Frequency: **Daily**
* Confidence Interval: **95%**
* Final Model: **ARIMA**

##  Conclusion

This project demonstrates:

* End-to-end time-series forecasting
* Model evaluation and selection
* Clean ML pipeline design
* Real-world dashboard deployment
* Professional project organization
##  Author

**Podmaraj Boruah**
