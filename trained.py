import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import joblib

plt.rcParams["figure.figsize"] = (12, 6)

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "retail_store_inventory.csv"

# Save inside streamlit folder
STREAMLIT_DIR = BASE_DIR / "streamlit"
CLEANED_DATA_PATH = STREAMLIT_DIR / "cleaned_retail_sales.csv"
FORECAST_PATH = STREAMLIT_DIR / "forecast_retail_sales_arima.csv"
MODEL_PATH = STREAMLIT_DIR / "arima_model.pkl"

# Make sure streamlit folder exists
STREAMLIT_DIR.mkdir(exist_ok=True)

# ---------------- Load Raw Data ----------------
df = pd.read_csv(RAW_DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")

# ---------------- Data Cleaning ----------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Fill missing numerical values
df["units_sold"] = df["units_sold"].fillna(0)
for col in ["inventory_level", "discount", "holiday_promotion"]:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# ---------------- Aggregate Daily Sales ----------------
daily_df = df.groupby("date", as_index=False)["units_sold"].mean()
daily_df.set_index("date", inplace=True)
daily_df = daily_df.asfreq("D")
daily_df["units_sold"] = daily_df["units_sold"].fillna(0)

# Save cleaned data inside streamlit folder
daily_df.to_csv(CLEANED_DATA_PATH)
print(f"✅ Cleaned data saved at {CLEANED_DATA_PATH}")

# ---------------- Train-Test Split ----------------
train = daily_df[:-30]
test = daily_df[-30:]

# ---------------- Train ARIMA Model ----------------
arima_model = ARIMA(train["units_sold"], order=(5,1,0)).fit()

# ---------------- Forecast ----------------
arima_forecast = arima_model.get_forecast(steps=30)
forecast_values = arima_forecast.predicted_mean
forecast_conf_int = arima_forecast.conf_int()

# Compute RMSE
rmse = np.sqrt(mean_squared_error(test["units_sold"], forecast_values))
print(f"ARIMA Test RMSE: {rmse:.2f}")

# ---------------- Save ARIMA Model ----------------
joblib.dump(arima_model, MODEL_PATH)
print(f"✅ ARIMA model saved at {MODEL_PATH}")

# ---------------- Prepare Forecast DataFrame ----------------
forecast_dates = pd.date_range(daily_df.index[-1] + pd.Timedelta(days=1), periods=30)
forecast_df = pd.DataFrame({
    "date": forecast_dates,
    "forecast_units_sold": forecast_values,
    "lower_ci": forecast_conf_int.iloc[:,0],
    "upper_ci": forecast_conf_int.iloc[:,1]
})

# Save forecast CSV inside streamlit folder
forecast_df.to_csv(FORECAST_PATH, index=False)
print(f"✅ Forecast saved at {FORECAST_PATH}")

# ---------------- Visualizations ----------------
# 1️⃣ Historical + Forecast
plt.figure(figsize=(14,6))
plt.plot(daily_df.index, daily_df["units_sold"], label="Historical Sales", color="black")
plt.plot(forecast_df["date"], forecast_df["forecast_units_sold"], linestyle="--", color="orange", label="Forecast")
plt.fill_between(forecast_df["date"], forecast_df["lower_ci"], forecast_df["upper_ci"], color="orange", alpha=0.2, label="95% CI")
plt.axvline(daily_df.index[-1], linestyle=":", color="gray", label="Forecast Start")
plt.title("Retail Sales: Historical + 30-Day Forecast (ARIMA)")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.show()

# 2️⃣ Zoomed last 60 days
plt.figure(figsize=(12,6))
plt.plot(daily_df.index[-60:], daily_df["units_sold"][-60:], label="Historical Sales")
plt.plot(forecast_df["date"], forecast_df["forecast_units_sold"], linestyle="--", color="orange", label="Forecast")
plt.fill_between(forecast_df["date"], forecast_df["lower_ci"], forecast_df["upper_ci"], color="orange", alpha=0.2)
plt.axvline(daily_df.index[-1], linestyle=":", color="gray")
plt.title("Zoomed: Recent Sales + Forecast (ARIMA)")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.show()
