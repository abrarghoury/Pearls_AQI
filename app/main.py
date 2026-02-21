import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app_config import APP_CONFIG, CITY_CONFIG
from services.mongo_service import MongoService
from services.aqi_utils import (
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
# GLOBAL STYLES
# =====================================================
st.markdown("""
<style>
.big-aqi {
    font-size:72px;
    font-weight:800;
}
.hero-card {
    padding:40px;
    border-radius:24px;
    text-align:center;
    box-shadow:0 10px 40px rgba(0,0,0,0.25);
}
.small-text {
    font-size:14px;
    opacity:0.85;
}
.chip {
    padding:6px 14px;
    border-radius:999px;
    font-weight:700;
    font-size:14px;
    color:white;
    display:inline-block;
    margin-top:8px;
}
.forecast-chip {
    padding:6px 14px;
    border-radius:999px;
    font-weight:700;
    font-size:14px;
    color:white;
    display:inline-block;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HELPERS
# =====================================================
def safe_round(val, n=2):
    try:
        return round(float(val), n)
    except:
        return None

def format_relative_time(dt_obj):
    if not dt_obj:
        return "N/A"
    try:
        if isinstance(dt_obj, str):
            dt_obj = datetime.fromisoformat(dt_obj)
        diff = datetime.utcnow() - dt_obj
        mins = int(diff.total_seconds() / 60)
        if mins < 1:
            return "Just now"
        if mins < 60:
            return f"{mins} minutes ago"
        hrs = mins // 60
        if hrs < 24:
            return f"{hrs} hours ago"
        return f"{hrs//24} days ago"
    except:
        return "N/A"

def health_tip(category):
    return {
        "Good": "Perfect for outdoor activities 🌿",
        "Moderate": "Sensitive individuals should be cautious.",
        "Unhealthy (Sensitive)": "Limit prolonged outdoor exertion.",
        "Unhealthy": "Mask recommended outdoors 😷",
        "Very Unhealthy": "Avoid outdoor activities.",
        "Hazardous": "Stay indoors and keep windows closed."
    }.get(category, "")

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

if not (latest_raw and latest_features and latest_pred_log):
    st.error("Required data missing. Run pipelines first.")
    st.stop()

# =====================================================
# TODAY AQI HERO
# =====================================================
current_aqi = latest_features.get("aqi")
aqi_val = safe_round(current_aqi)
category = aqi_category(current_aqi)
color = aqi_color_from_value(current_aqi)

st.subheader("Today")

st.markdown(
    f"""
    <div class="hero-card" style="
        background:linear-gradient(135deg,{color}30,{color}10);
        border:2px solid {color};
    ">
        <div class="big-aqi">{aqi_val}</div>
        <div class="chip" style="background:{color};">{category}</div>
        <div class="small-text" style="margin-top:14px;">
            {health_tip(category)}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# WEATHER SNAPSHOT
# =====================================================
st.subheader("Weather Snapshot")

c1, c2, c3, c4 = st.columns(4)
c1.metric("🌡️ Temp (°C)", safe_round(latest_raw.get("temperature")))
c2.metric("💧 Humidity (%)", safe_round(latest_raw.get("humidity")))
c3.metric("💨 Wind (m/s)", safe_round(latest_raw.get("wind_speed")))
c4.metric("🧭 Pressure (hPa)", safe_round(latest_raw.get("pressure")))

st.divider()

# =====================================================
# 3-DAY FORECAST (FINAL UI)
# =====================================================
st.subheader("📅 3-Day Forecast")

preds = latest_pred_log.get("predictions", {})
now = datetime.now()

def forecast_card(title, date, color, content):
    st.markdown(
        f"""
        <div style="
            padding:30px;
            border-radius:22px;
            border:2px solid {color};
            background:linear-gradient(135deg,{color}25,{color}08);
            box-shadow:0 8px 28px rgba(0,0,0,0.25);
        ">
            <div style="font-size:15px; opacity:0.75;">{title}</div>
            <div style="font-size:22px; font-weight:800; margin-bottom:16px;">
                {date}
            </div>
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )

col1, col2, col3 = st.columns(3)

# Tomorrow (AQI value)
with col1:
    val = safe_round(preds.get("target_aqi_t_plus_24h"))
    col = aqi_color_from_value(val)
    lab = aqi_category(val)

    forecast_card(
        "Tomorrow",
        (now + timedelta(days=1)).strftime("%a, %d %b"),
        col,
        f"""
        <div style="font-size:52px; font-weight:900;">{val}</div>
        <div class="forecast-chip" style="background:{col};">{lab}</div>
        <div class="small-text" style="margin-top:12px;">
            {health_tip(lab)}
        </div>
        """
    )

# Day After
with col2:
    info = aqi_class_info(preds.get("target_aqi_class_t_plus_48h"))
    col = aqi_class_color(preds.get("target_aqi_class_t_plus_48h"))

    forecast_card(
        "Day After",
        (now + timedelta(days=2)).strftime("%a, %d %b"),
        col,
        f"""
        <div class="forecast-chip" style="background:{col};">
            {info['label']}
        </div>
        <div class="small-text" style="margin-top:12px;">
            Expected Range: <b>{info['range']}</b><br>
            {health_tip(info['label'])}
        </div>
        """
    )

# 3rd Day
with col3:
    info = aqi_class_info(preds.get("target_aqi_class_t_plus_72h"))
    col = aqi_class_color(preds.get("target_aqi_class_t_plus_72h"))

    forecast_card(
        "3rd Day",
        (now + timedelta(days=3)).strftime("%a, %d %b"),
        col,
        f"""
        <div class="forecast-chip" style="background:{col};">
            {info['label']}
        </div>
        <div class="small-text" style="margin-top:12px;">
            Expected Range: <b>{info['range']}</b><br>
            {health_tip(info['label'])}
        </div>
        """
    )

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption(
    f"Updated {format_relative_time(latest_features.get('timestamp'))} | "
    f"Forecast generated {format_relative_time(latest_pred_log.get('created_at'))}"
)

prob = latest_pred_log.get("class_probability_48h", 0)

if prob >= 0.8:
    confidence = "▮▮▮▮▮"
elif prob >= 0.65:
    confidence = "▮▮▮▮▯"
elif prob >= 0.5:
    confidence = "▮▮▮▯▯"
else:
    confidence = "▮▮▯▯▯"

st.caption(f"Classification Confidence (based on class probability): {confidence}")