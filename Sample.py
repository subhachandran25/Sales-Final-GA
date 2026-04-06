import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Time-series forecasting
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
NUM_COLS = [
    "Calls_Dialed", "Call_Time_Mins", "New_Leads", "Disqualified",
    "No_Answer", "Qualified", "Converted", "Deals_Closed",
    "Followup_Leads", "Total_Revenue", "Avg_Unit_Value",
]
Q_MAP = {"Q1 2024": 1, "Q2 2024": 2, "Q3 2024": 3, "Q4 2024": 4}
QUARTER_ORDER = list(Q_MAP.keys())

# ── DATA LOADER ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    for col in [c for c in NUM_COLS if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Q_Num"] = df["Quarter"].map(Q_MAP).fillna(0).astype(int)
    df["Conversion_Rate"]  = df["Converted"] / df["Calls_Dialed"].replace(0, 1)
    df["Lead_Quality"]     = df["Qualified"] / df["New_Leads"].replace(0, 1)
    df["Revenue_Per_Call"] = df["Total_Revenue"] / df["Calls_Dialed"].replace(0, 1)
    return df

_csv_path = Path(__file__).parent / "sales_performance_data.csv"
if _csv_path.exists():
    df = load_data(_csv_path.read_bytes())
else:
    uploaded = st.file_uploader("Upload sales_performance_data.csv", type=["csv"])
    if uploaded is None:
        st.stop()
    df = load_data(uploaded.read())

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Filters")
    all_regions  = sorted(df["Region"].dropna().unique().tolist())
    all_quarters = [q for q in QUARTER_ORDER if q in df["Quarter"].unique()]
    sel_regions  = st.multiselect("Regions",  all_regions,  default=all_regions)
    sel_quarters = st.multiselect("Quarters", all_quarters, default=all_quarters)

    st.markdown("---")
    st.markdown("### 🤖 Predictive Model")
    model_choice = st.selectbox(
        "Algorithm",
        ["Random Forest", "Gradient Boosting", "Linear Regression", "Prophet (Time Series)", "ARIMA (Time Series)"]
    )
    st.markdown("---")
    st.caption("Sales Intelligence Dashboard v3.0")

fdf = df[df["Region"].isin(sel_regions) & df["Quarter"].isin(sel_quarters)].copy()
if fdf.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Descriptive", "🔍 Diagnostic", "💡 Prescriptive", "🤖 Predictive"])

# ── PREDICTIVE TAB ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Predictive Analysis</div>', unsafe_allow_html=True)

    if model_choice in ["Random Forest", "Gradient Boosting", "Linear Regression"]:
        # Feature-based ML prediction
        features = ["Calls_Dialed","Call_Time_Mins","New_Leads","Qualified","Converted","Deals_Closed"]
        X = fdf[features]
        y = fdf["Total_Revenue"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "Random Forest":
            model = RandomForestRegressor()
        elif model_choice == "Gradient Boosting":
            model = GradientBoostingRegressor()
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        st.write("R² Score:", r2_score(y_test, preds))
        st.write("MAE:", mean_absolute_error(y_test, preds))

    elif model_choice == "Prophet (Time Series)":
        ts_data = fdf.groupby("Quarter")["Total_Revenue"].sum().reset_index()
        ts_data["Quarter"] = pd.Categorical(ts_data["Quarter"], categories=QUARTER_ORDER, ordered=True)
        ts_data = ts_data.sort_values("Quarter")
        prophet_df = pd.DataFrame({
            "ds": pd.date_range(start="2024-01-01", periods=len(ts_data), freq="Q"),
            "y": ts_data["Total_Revenue"].values
        })
        m = Prophet(yearly_seasonality=True, daily_seasonality=False)
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=4, freq="Q")
        forecast = m.predict(future)
        fig_forecast = m.plot(forecast)
        st.pyplot(fig_forecast)

    elif model_choice == "ARIMA (Time Series)":
        ts_data = fdf.groupby("Quarter")["Total_Revenue"].sum().reset_index()
        ts_series = ts_data.set_index("Quarter")["Total_Revenue"]
        model = ARIMA(ts_series, order=(1,1,1))
        model_fit = model.fit()
        forecast_arima = model_fit.forecast(steps=4)
        st.write("ARIMA Forecast (next 4 quarters):", forecast_arima.values)
        st.line_chart(pd.Series(
            list(ts_series.values) + list(forecast_arima),
            index=pd.date_range(start="2024-01-01", periods=len(ts_series)+4, freq="Q")
        ))
