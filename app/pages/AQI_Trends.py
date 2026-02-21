import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from app.services.mongo_service import MongoService
from app.services.aqi_utils import aqi_category

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AQI Trends",
    layout="wide"
)

# =====================================================
# GLOBAL STYLES (DARK MODE)
# =====================================================
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: #e5e7eb;
}
.block-container {
    padding-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.title("📊 AQI Trends & History")
st.caption("Last 48 hours air quality behavior")
st.divider()

# =====================================================
# LOAD DATA
# =====================================================
trend_data = MongoService.get_recent_features()

if not trend_data:
    st.warning("Not enough historical data available.")
    st.stop()

df = pd.DataFrame(trend_data)

# =====================================================
# AUTO DETECT TIMESTAMP
# =====================================================
time_col = next((c for c in ["feature_generated_at", "timestamp", "created_at"] if c in df.columns), None)
if not time_col:
    st.error("No timestamp column found.")
    st.stop()

df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(time_col)

# =====================================================
# ADD ROLLING AVERAGE
# =====================================================
df["aqi_ma"] = df["aqi"].rolling(3).mean()

# =====================================================
# MATPLOTLIB DARK THEME
# =====================================================
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#0f172a",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e5e7eb",
    "text.color": "#e5e7eb",
    "xtick.color": "#cbd5f5",
    "ytick.color": "#cbd5f5",
    "grid.color": "#334155",
    "legend.facecolor": "#0f172a",
    "legend.edgecolor": "#334155"
})

# =====================================================
# AQI TREND CHART
# =====================================================
st.subheader("AQI Trend (Last 48 Hours)")

fig, ax = plt.subplots(figsize=(14, 5))

# AQI background bands
band_colors = [
    (0, 50, "#2ecc71", 0.12),
    (51, 100, "#f1c40f", 0.12),
    (101, 150, "#e67e22", 0.12),
    (151, 200, "#e74c3c", 0.12),
    (201, 300, "#8e44ad", 0.12),
    (301, 500, "#7f0000", 0.12)
]

for low, high, color, alpha in band_colors:
    ax.axhspan(low, high, color=color, alpha=alpha)

# AQI line
ax.plot(df[time_col], df["aqi"], label="AQI", linewidth=2.5, marker="o", color="#e5e7eb")

# Rolling average
ax.plot(df[time_col], df["aqi_ma"], label="3-Hour Moving Avg", linestyle="--", linewidth=2, marker="o", color="#1abc9c")

ax.set_xlabel("Time")
ax.set_ylabel("AQI")
ax.set_title("AQI Trend with 3-Hour Moving Average", fontsize=14, weight="bold")
ax.xaxis.set_major_locator(MaxNLocator(integer=False))
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.35)
ax.legend()

# CENTER GRAPH
st.markdown("<div style='display:flex; justify-content:center'>", unsafe_allow_html=True)
st.pyplot(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# =====================================================
# AQI CATEGORY DISTRIBUTION
# =====================================================
st.subheader("Air Quality Category Distribution (Last 48 Hours)")

df["category"] = df["aqi"].apply(aqi_category)
cat_counts = df["category"].value_counts()

category_colors = {
    "Good": "#2ecc71",
    "Moderate": "#f1c40f",
    "Unhealthy (Sensitive)": "#e67e22",
    "Unhealthy": "#e74c3c",
    "Very Unhealthy": "#8e44ad",
    "Hazardous": "#7f0000"
}

ordered_categories = list(category_colors.keys())
counts = [cat_counts.get(cat, 0) for cat in ordered_categories]
colors = [category_colors[cat] for cat in ordered_categories]

fig2, ax2 = plt.subplots(figsize=(12, 4))

bars = ax2.bar(
    ordered_categories,
    counts,
    color=colors,
    edgecolor="#0f172a"
)

# VALUE LABELS ON TOP
for bar in bars:
    h = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.2,
        str(int(h)),
        ha="center",
        va="bottom",
        fontsize=10,
        color="#e5e7eb"
    )

ax2.set_ylabel("Frequency")
ax2.set_title("AQI Category Distribution", fontsize=14, weight="bold")
ax2.tick_params(axis='x', rotation=30)
ax2.grid(axis='y', alpha=0.35)

# CENTER GRAPH
st.markdown("<div style='display:flex; justify-content:center'>", unsafe_allow_html=True)
st.pyplot(fig2, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)