import streamlit as st
import pandas as pd

from app.services.mongo_service import MongoService
from app.services.aqi_utils import aqi_category

st.set_page_config(
    page_title="AQI Trends",
    layout="wide"
)

st.title("📊 AQI Trends & History")
st.caption("Last 48 hours air quality behavior")

st.divider()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
trend_data = MongoService.get_recent_features()

if not trend_data:
    st.warning("Not enough historical data available.")
    st.stop()

df = pd.DataFrame(trend_data)

# --------------------------------------------------
# 🔥 AUTO DETECT TIMESTAMP (VERY SENIOR MOVE)
# --------------------------------------------------
time_col = None

if "feature_generated_at" in df.columns:
    time_col = "feature_generated_at"

elif "timestamp" in df.columns:
    time_col = "timestamp"

elif "created_at" in df.columns:
    time_col = "created_at"

else:
    st.error("No timestamp column found in features collection.")
    st.write("Available columns:", df.columns)
    st.stop()

df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(time_col)

# --------------------------------------------------
# AQI TREND (TIME SERIES)
# --------------------------------------------------
st.subheader("AQI Trend (Last 48 Hours)")

st.line_chart(
    df.set_index(time_col)["aqi"]
)

st.divider()

# --------------------------------------------------
# AQI CATEGORY DISTRIBUTION
# (Much better than raw AQI histogram)
# --------------------------------------------------
st.subheader("Air Quality Category Distribution (Last 48 Hours)")

df["category"] = df["aqi"].apply(aqi_category)

cat_counts = df["category"].value_counts()

st.bar_chart(cat_counts)
