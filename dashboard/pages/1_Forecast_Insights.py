import streamlit as st
import pandas as pd
import plotly.express as px
from services.mongo_service import MongoService

st.set_page_config(layout="wide")

st.title("📊 Forecast & Model Insights")

prediction_log = MongoService.get_latest_prediction_log()

if prediction_log is None:
    st.error("No predictions found.")
    st.stop()

predictions = prediction_log.get("predictions", {})

# -----------------------------
# CREATE CHART DATA
# -----------------------------

df = pd.DataFrame({
    "Day": list(predictions.keys()),
    "Predicted AQI": list(predictions.values())
})

# -----------------------------
# FORECAST CHART
# -----------------------------

st.subheader("AQI Forecast Trend")

fig = px.line(
    df,
    x="Day",
    y="Predicted AQI",
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# SHAP IMAGE (if exists)
# -----------------------------

import os

if os.path.exists("../artifacts/shap_summary.png"):
    st.subheader("🔍 Model Explainability (SHAP)")
    st.image("../artifacts/shap_summary.png")
else:
    st.info("SHAP summary not found. Generate during training.")

# -----------------------------
# MODEL REGISTRY
# -----------------------------

models = MongoService.get_model_registry()

if models:
    latest_model = models[-1]

    st.subheader("🤖 Active Model")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", latest_model.get("model_name", "N/A"))
    col2.metric("RMSE", latest_model.get("rmse", "N/A"))
    col3.metric("MAE", latest_model.get("mae", "N/A"))
