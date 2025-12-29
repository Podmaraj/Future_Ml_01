import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error

plt.rcParams["figure.figsize"] = (12, 6)

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "retail_store_inventory.csv"

# ---------------- Load Data ----------------
df = pd.read_csv(RAW_DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df = df.sort_values("date")
df["units_sold"] = df["units_sold"].fillna(0)

# ---------------- Aggregate Daily Sales ----------------
daily_df = df.groupby("date", as_index=False)["units_sold"].mean()
daily_df.set_index("date", inplace=True)
daily_df = daily_df.asfreq('D')  # set daily frequency

# ---------------- Train-Test Split ----------------
train = daily_df[:-30]
test = daily_df[-30:]

# ---------------- ARIMA Model ----------------
arima_model = ARIMA(train["units_sold"], order=(5,1,0)).fit()
arima_forecast = arima_model.get_forecast(steps=30)
arima_values = arima_forecast.predicted_mean
arima_rmse = np.sqrt(mean_squared_error(test["units_sold"], arima_values))

# ---------------- SARIMA Model ----------------
sarima_model = SARIMAX(train["units_sold"], order=(1,1,1), seasonal_order=(1,1,1,7),
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
sarima_forecast = sarima_model.get_forecast(steps=30)
sarima_values = sarima_forecast.predicted_mean
sarima_rmse = np.sqrt(mean_squared_error(test["units_sold"], sarima_values))

# ---------------- Prophet Model ----------------
prophet_df = daily_df.reset_index()[["date","units_sold"]].rename(columns={"date":"ds","units_sold":"y"})
prophet_train = prophet_df.iloc[:-30]
prophet_test = prophet_df.iloc[-30:]

prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
prophet_model.fit(prophet_train)
future = prophet_model.make_future_dataframe(periods=30)
prophet_forecast = prophet_model.predict(future)
prophet_values = prophet_forecast['yhat'][-30:].values
prophet_rmse = np.sqrt(mean_squared_error(prophet_test['y'], prophet_values))

# ---------------- Compare Models ----------------
rmses = {"ARIMA": arima_rmse, "SARIMA": sarima_rmse, "Prophet": prophet_rmse}
best_model_name = min(rmses, key=rmses.get)
print("Best Model:", best_model_name)

# ---------------- Forecast using Best Model ----------------
forecast_dates = pd.date_range(daily_df.index[-1] + pd.Timedelta(days=1), periods=30)

if best_model_name == "ARIMA":
    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        "forecast_units_sold": arima_values,
        "lower_ci": arima_forecast.conf_int().iloc[:,0],
        "upper_ci": arima_forecast.conf_int().iloc[:,1]
    })
elif best_model_name == "SARIMA":
    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        "forecast_units_sold": sarima_values,
        "lower_ci": sarima_forecast.conf_int().iloc[:,0],
        "upper_ci": sarima_forecast.conf_int().iloc[:,1]
    })
else:
    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        "forecast_units_sold": prophet_values,
        "lower_ci": prophet_forecast['yhat_lower'][-30:].values,
        "upper_ci": prophet_forecast['yhat_upper'][-30:].values
    })

# ---------------- Visualization ----------------
plt.figure(figsize=(14,6))
plt.plot(daily_df.index, daily_df["units_sold"], label="Historical Sales", color="black")
plt.plot(forecast_df["date"], forecast_df["forecast_units_sold"], linestyle="--", color="orange", label=f"Forecast ({best_model_name})")
plt.fill_between(forecast_df["date"], forecast_df["lower_ci"], forecast_df["upper_ci"], color="orange", alpha=0.2, label="95% CI")
plt.axvline(daily_df.index[-1], linestyle=":", color="gray", label="Forecast Start")
plt.title(f"Retail Sales: Historical + 30-Day Forecast ({best_model_name})")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.show()

# Zoomed view - last 60 days
plt.figure(figsize=(12,6))
plt.plot(daily_df.index[-60:], daily_df["units_sold"].values[-60:], label="Historical Sales")
plt.plot(forecast_df["date"], forecast_df["forecast_units_sold"], linestyle="--", color="orange", label=f"Forecast ({best_model_name})")
plt.fill_between(forecast_df["date"], forecast_df["lower_ci"], forecast_df["upper_ci"], color="orange", alpha=0.2)
plt.axvline(daily_df.index[-1], linestyle=":", color="gray")
plt.title(f"Zoomed: Recent Sales + 30-Day Forecast ({best_model_name})")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.show()

# ---------------- Save Forecast ----------------
forecast_csv_path = BASE_DIR / f"forecast_retail_sales_{best_model_name.lower()}.csv"
forecast_df.to_csv(forecast_csv_path, index=False)
print("Forecast saved:", forecast_csv_path)
