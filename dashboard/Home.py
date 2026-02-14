import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from app.services.mongo_service import MongoService

st.set_page_config(
    page_title="Live AQI Monitor",
    layout="wide"
)

st.title("🌍 Live AQI Monitor")

# -----------------------------
# FETCH DATA
# -----------------------------

prediction_log = MongoService.get_latest_prediction_log()
latest_features = MongoService.get_latest_features()

if prediction_log is None:
    st.error("No predictions available.")
    st.stop()

predictions = prediction_log.get("predictions", {})
created_at = prediction_log.get("created_at", "Unknown")

# Example structure assumption:
# predictions = { "day_1": 120, "day_2": 135, "day_3": 142 }

# -----------------------------
# AQI COLOR LOGIC
# -----------------------------

def get_aqi_status(aqi):
    if aqi <= 50:
        return "Good ✅"
    elif aqi <= 100:
        return "Moderate 🟡"
    elif aqi <= 150:
        return "Unhealthy for Sensitive ⚠️"
    else:
        return "Unhealthy 🔴"

# -----------------------------
# BIG CURRENT AQI
# -----------------------------

today_aqi = list(predictions.values())[0]

st.markdown("## Current Air Quality")

col1, col2 = st.columns([2,1])

with col1:
    st.metric(
        label="Today's AQI",
        value=today_aqi,
        delta=get_aqi_status(today_aqi)
    )

with col2:
    if today_aqi > 150:
        st.error("⚠️ Hazardous Air Quality! Avoid outdoor activity.")

# -----------------------------
# 3 DAY FORECAST
# -----------------------------

st.markdown("## 🔮 3-Day Forecast")

cols = st.columns(3)

for col, (day, value) in zip(cols, predictions.items()):
    col.metric(
        label=day.upper(),
        value=value,
        delta=get_aqi_status(value)
    )

# -----------------------------
# LAST UPDATED
# -----------------------------

st.caption(f"Last Updated: {created_at}")
