# SalesDash – Retail Sales Forecasting Dashboard

**Author:** Podmaraj Boruah
**Project Type:** Machine Learning · Time Series Forecasting · Data Visualization
**Created On:** 28 December 2025

---

## Project Overview

**SalesDash** is an end-to-end **retail sales forecasting system** built using **Python** and **Streamlit**.
The project predicts **future retail sales trends** using **time-series forecasting models** and presents insights through a **clean, professional, and interactive dashboard**.

The complete workflow—from **data cleaning and model comparison** to **final model deployment**—follows **industry-standard ML practices**.

---

## Problem Statement

Retail businesses need accurate sales forecasts to:

* Optimize inventory planning
* Reduce overstock and stockouts
* Improve supply chain efficiency
* Enable data-driven decision-making

SalesDash addresses this by learning temporal patterns from historical retail sales data and forecasting future demand.

---

## Models Evaluated

Three time-series forecasting models were implemented and evaluated:

| Model       | Description                             |
| ----------- | --------------------------------------- |
| **ARIMA**   | Captures trend and autocorrelation      |
| **SARIMA**  | Explicitly models seasonality           |
| **Prophet** | Flexible trend and seasonality modeling |

### Model Evaluation Strategy

* Train–test split (last **30 days** used as test set)
* Evaluation metric: **RMSE (Root Mean Squared Error)**
* All models trained and tested under identical conditions

### Best Model Selection

**ARIMA** achieved the **lowest RMSE** and was selected for final deployment.

---

## Tech Stack

### Programming & Frameworks

* **Python 3.x**
* **Streamlit** – interactive dashboard framework

### Data Science & Time Series

* **Pandas** – data manipulation
* **NumPy** – numerical operations
* **Matplotlib** – visualizations
* **Statsmodels** – ARIMA & SARIMA
* **Prophet** – advanced forecasting
* **Scikit-learn** – model evaluation
* **Joblib** – model serialization

### Environment & Tooling

* **uv** – fast Python dependency & virtual environment manager

---

## Project Folder Structure

```
Future_ML_01/
│
├── app.py                          # Streamlit dashboard application
├── test.py                         # Data cleaning, EDA & model comparison
├── trained.py                      # Final ARIMA training & forecast generation
├── main.py                         # Reserved / placeholder
│
├── cleaned_retail_sales.csv        # Cleaned daily sales dataset
├── forecast_retail_sales_arima.csv # 30-day ARIMA forecast output
├── arima_model.pkl                 # Trained ARIMA model
│
├── requirements.txt                # Project dependencies
├── pyproject.toml                  # uv environment configuration
└── README.md                       # Project documentation
```

---

## Project Workflow

### 1️⃣ Data Cleaning & Model Comparison (`test.py`)

* Load raw retail sales data
* Column standardization
* Date parsing and validation
* Missing value handling
* Daily sales aggregation
* Model training and evaluation:

  * ARIMA
  * SARIMA
  * Prophet
* RMSE-based model comparison

---

### 2️⃣ Final Model Training (`trained.py`)

* Train ARIMA model on full cleaned dataset
* Generate **30-day sales forecast**
* Calculate **95% confidence intervals**
* Save outputs:

  * Cleaned dataset
  * Forecast CSV file
  * Serialized ARIMA model (`.pkl`)

---

### 3️⃣ Dashboard Deployment (`app.py`)

The Streamlit dashboard provides:

#### KPI Metrics

* Total Sales
* Average Daily Sales
* Peak Sales Value

#### Visualizations

1. Historical Sales Trend
2. Historical vs 30-Day Forecast
3. Recent Sales with Forecast Overlay

#### Forecast Table

* Clean tabular forecast output
* Improved readability
* Center-aligned values

#### UI Enhancements

* Custom CSS styling
* Professional typography
* Responsive layout
* Compact, high-quality charts

---

## How to Run the Project

### Step 1: Create Environment & Install Dependencies

```bash
uv venv
uv pip install -r requirements.txt
```

### Step 2: Train Model & Generate Forecast

```bash
python trained.py
```

### Step 3: Launch Streamlit Dashboard

```bash
streamlit run app.py
```

---

## Forecast Details

* **Forecast Horizon:** 30 Days
* **Frequency:** Daily
* **Confidence Interval:** 95%
* **Final Model:** ARIMA

---

## Conclusion

This project demonstrates:

* End-to-end time-series forecasting
* Data cleaning and feature preparation
* Model comparison and selection
* Production-ready ML pipeline
* Interactive dashboard deployment
* Professional project organization

---

## Author

**Podmaraj Boruah**
