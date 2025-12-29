import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Retail Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================
# FILE PATHS
# =========================
DATA_PATH = "cleaned_retail_sales.csv"
FORECAST_PATH = "forecast_retail_sales_arima.csv"
CSS_PATH = "assets/style.css"

# =========================
# LOAD CSS
# =========================
with open(CSS_PATH) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
hist_df = pd.read_csv(DATA_PATH)
forecast_df = pd.read_csv(FORECAST_PATH)

# Convert dates
hist_df["date"] = pd.to_datetime(hist_df["date"])
forecast_df["date"] = pd.to_datetime(forecast_df["date"])

# Fill NaN to avoid errors
hist_df["units_sold"] = hist_df["units_sold"].fillna(0)
forecast_df["forecast_units_sold"] = forecast_df["forecast_units_sold"].fillna(0)
forecast_df["lower_ci"] = forecast_df.get("lower_ci", 0)
forecast_df["upper_ci"] = forecast_df.get("upper_ci", 0)

# =========================
# HEADER
# =========================
st.markdown("""
<div style="text-align:center; padding:20px;">
    <h1 style="color:#1DB954; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        📈 Retail Sales Forecast Dashboard
    </h1>
    <p style="font-size:18px; color:#B0B0B0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        AI-powered sales prediction & analytics
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# KPI METRICS
# =========================
total_sales = int(hist_df["units_sold"].sum())
avg_sales = round(hist_df["units_sold"].mean(), 2)
max_sales = int(hist_df["units_sold"].max())

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"{total_sales}")
col2.metric("Average Daily Sales", f"{avg_sales}")
col3.metric("Highest Sales", f"{max_sales}")

# =========================
# HISTORICAL SALES CHART
# =========================
st.markdown("## 📊 Historical Sales Trend")
fig, ax = plt.subplots(figsize=(10,5), dpi=120)
ax.plot(hist_df["date"], hist_df["units_sold"], label="Historical Sales", color="#1DB954", marker='o', markersize=4)
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.set_title("Historical Sales Trend")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()
st.pyplot(fig)

# =========================
# FORECAST CHART
# =========================
st.markdown("## 🔮 30-Day Sales Forecast")
fig2, ax2 = plt.subplots(figsize=(10,5), dpi=120)
ax2.plot(hist_df["date"], hist_df["units_sold"], color="#1f77b4", label="Historical Sales", marker='o')
ax2.plot(forecast_df["date"], forecast_df["forecast_units_sold"], color="#ff7f0e", linestyle='--', marker='x', label="Forecasted Sales")
ax2.fill_between(forecast_df["date"], forecast_df["lower_ci"], forecast_df["upper_ci"], color='orange', alpha=0.2)
ax2.axvline(hist_df["date"].iloc[-1], color='gray', linestyle=':', linewidth=2)
ax2.set_xlabel("Date")
ax2.set_ylabel("Units Sold")
ax2.set_title("Historical vs Forecasted Sales")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()
ax2.tick_params(axis='x', rotation=45)
fig2.tight_layout()
st.pyplot(fig2)

# =========================
# ZOOMED 60-DAY CHART
# =========================
st.markdown("## 🔍 Zoomed View: Last 60 Days + Forecast")
fig3, ax3 = plt.subplots(figsize=(10,5), dpi=120)
last_60_hist = hist_df[-60:]
ax3.plot(last_60_hist["date"], last_60_hist["units_sold"], label="Historical Sales", color="#1f77b4", marker='o')
ax3.plot(forecast_df["date"], forecast_df["forecast_units_sold"], linestyle="--", color="#ff7f0e", marker='x', label="Forecasted Sales")
ax3.fill_between(forecast_df["date"], forecast_df["lower_ci"], forecast_df["upper_ci"], color='orange', alpha=0.2)
ax3.axvline(last_60_hist["date"].iloc[-1], color='gray', linestyle=':', linewidth=2)
ax3.set_xlabel("Date")
ax3.set_ylabel("Units Sold")
ax3.set_title("Zoomed View: Last 60 Days + Forecast")
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend()
ax3.tick_params(axis='x', rotation=45)
fig3.tight_layout()
st.pyplot(fig3)

# =========================
# FORECAST TABLE
# =========================
st.markdown("## 📋 Forecasted Sales Data")
forecast_df_display = forecast_df[["date", "forecast_units_sold"]].copy()
forecast_df_display["date"] = forecast_df_display["date"].dt.strftime("%Y-%m-%d")

st.dataframe(
    forecast_df_display.style.set_properties(**{
        "font-size": "18px",
        "text-align": "center",
        "font-family": "Segoe UI, Tahoma, Geneva, Verdana, sans-serif"
    }),
    use_container_width=True
)

# =========================
# FOOTER
# =========================
st.markdown("""
<div style="text-align:center; padding:15px; color:#888888; font-size:14px;">
    Built with ❤️ using Streamlit & Machine Learning<br>
    Created by: Podmaraj Boruah | Date: 28-12-2025
</div>
""", unsafe_allow_html=True)
