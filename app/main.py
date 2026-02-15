import streamlit as st
from datetime import datetime, timedelta

from app.app_config import APP_CONFIG, CITY_CONFIG
from app.services.mongo_service import MongoService
from app.services.aqi_utils import (
    aqi_category,
    aqi_color_from_value,
    aqi_class_color,
    aqi_class_info
)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon=APP_CONFIG["icon"],
    layout="wide"
)

# =====================================================
# HELPERS
# =====================================================
def safe_round(val, n=2):
    try:
        return round(float(val), n)
    except:
        return None

def safe_str(val):
    return str(val) if val is not None else "N/A"

def format_date(dt_obj):
    if not dt_obj:
        return "N/A"
    try:
        if isinstance(dt_obj, str):
            dt_obj = datetime.fromisoformat(dt_obj)
        return dt_obj.strftime("%a, %d %b")
    except:
        return "N/A"

def chip(label: str, bg: str):
    if not label:
        label = "Unknown"
    if not bg:
        bg = "#95a5a6"
    return f"""
    <span style="
        display:inline-block;
        padding:6px 12px;
        border-radius:999px;
        font-size:14px;
        font-weight:700;
        background:{bg};
        color:white;
    ">
        {label}
    </span>
    """

# =====================================================
# HEADER
# =====================================================
st.title("🌫️ Pearls AQI Predictor")
st.caption(f"City: **{CITY_CONFIG['city']}**")
st.divider()

# =====================================================
# LOAD DATA
# =====================================================
latest_raw = MongoService.get_latest_raw()
latest_features = MongoService.get_latest_features()
latest_pred_log = MongoService.get_latest_prediction_log()

if not latest_raw:
    st.error("❌ No RAW weather data found. Run ingestion pipeline first.")
    st.stop()

if not latest_features:
    st.error("❌ No FEATURE data found. Run feature pipeline first.")
    st.stop()

if not latest_pred_log:
    st.error("❌ No prediction data found. Run prediction pipeline first.")
    st.stop()

# =====================================================
# CURRENT AQI + WEATHER
# =====================================================
current_aqi = latest_features.get("aqi")

# 🔥 IMPORTANT FIX (DATA TIME vs PIPELINE TIME)
data_time = latest_features.get("timestamp")
pipeline_time = latest_features.get("feature_generated_at")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Current AQI (Right Now)")

    aqi_val = safe_round(current_aqi)
    cat = aqi_category(current_aqi)

    if aqi_val is None:
        st.warning("AQI value missing in latest feature document.")
    else:
        st.metric("AQI", aqi_val)

    st.markdown(
        chip(cat, aqi_color_from_value(current_aqi)),
        unsafe_allow_html=True
    )

    # ✅ CLEAR PIPELINE VISIBILITY
    st.caption(
        f"Data time: {safe_str(data_time)} | "
        f"Pipeline run: {safe_str(pipeline_time)}"
    )

with col2:
    st.subheader("🌤️ Weather Snapshot")

    w1, w2 = st.columns(2)
    with w1:
        st.metric("Temperature (°C)", safe_round(latest_raw.get("temperature")))
        st.metric("Humidity (%)", safe_round(latest_raw.get("humidity")))
    with w2:
        st.metric("Wind Speed (m/s)", safe_round(latest_raw.get("wind_speed")))
        st.metric("Pressure (hPa)", safe_round(latest_raw.get("pressure")))

st.divider()

# =====================================================
# FORECAST SECTION
# =====================================================
st.subheader("📅 3-Day Forecast")

preds = latest_pred_log.get("predictions", {})

aqi_24h = preds.get("target_aqi_t_plus_24h")
class_24h = preds.get("target_aqi_class_t_plus_24h")
class_48h = preds.get("target_aqi_class_t_plus_48h")
class_72h = preds.get("target_aqi_class_t_plus_72h")

# =====================================================
# DATE LABELS
# =====================================================
now = datetime.now()
date_1 = now + timedelta(days=1)
date_2 = now + timedelta(days=2)
date_3 = now + timedelta(days=3)

# =====================================================
# FORECAST CARDS
# =====================================================
c1, c2, c3 = st.columns(3)

# ---------- DAY 1 ----------
with c1:
    st.markdown(f"### {format_date(date_1)}")

    aqi_24h_val = safe_round(aqi_24h)

    if aqi_24h_val is None:
        st.warning("Prediction missing")
        st.markdown(chip("Unknown", "#95a5a6"), unsafe_allow_html=True)
        st.caption("Expected range: N/A")
    else:
        label = aqi_category(aqi_24h_val)
        color = aqi_color_from_value(aqi_24h_val)

        st.metric("Predicted AQI", aqi_24h_val)
        st.markdown(chip(label, color), unsafe_allow_html=True)

        info = aqi_class_info(class_24h) or {}
        st.caption(f"Expected range: {info.get('range', 'N/A')}")

# ---------- DAY 2 ----------
with c2:
    st.markdown(f"### {format_date(date_2)}")

    info = aqi_class_info(class_48h) or {}
    st.markdown(
        chip(info.get("label", "Unknown"), aqi_class_color(class_48h)),
        unsafe_allow_html=True
    )
    st.write("")
    st.caption(f"Expected range: {info.get('range', 'N/A')}")

# ---------- DAY 3 ----------
with c3:
    st.markdown(f"### {format_date(date_3)}")

    info = aqi_class_info(class_72h) or {}
    st.markdown(
        chip(info.get("label", "Unknown"), aqi_class_color(class_72h)),
        unsafe_allow_html=True
    )
    st.write("")
    st.caption(f"Expected range: {info.get('range', 'N/A')}")

st.divider()
st.caption(f"Forecast generated at: {safe_str(latest_pred_log.get('created_at'))}")
