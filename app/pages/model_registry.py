import streamlit as st
import pandas as pd

from app.services.mongo_service import MongoService

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Model Registry",
    layout="wide"
)

# =====================================================
# GLOBAL STYLES
# =====================================================
st.markdown("""
<style>
/* Block container padding */
.block-container {
    padding-top: 1.5rem;
}

/* Center the table container */
.table-container {
    display: flex;
    justify-content: center;
    margin-top: 10px;
}

/* Table style */
table {
    border-collapse: collapse;
    width: 90%;
    max-width: 1200px;
    font-family: "Segoe UI", sans-serif;
    color: #f1f5f9;
    background-color: #1e293b;
    border-radius: 12px;
    overflow: hidden;
}

/* Table header */
table thead th {
    background-color: #111827 !important;
    color: #f1f5f9 !important;
    font-weight: bold !important;
    text-align: center !important;
    padding: 12px !important;
}

/* Table cells */
table tbody td {
    text-align: center !important;
    padding: 10px !important;
    border-bottom: 1px solid #334155;
}

/* Hover effect */
table tbody tr:hover {
    background-color: #374151;
}

/* Status chips */
.status-chip {
    padding: 6px 12px;
    border-radius: 12px;
    color: white;
    font-weight: 600;
    display: inline-block;
}
.active-chip { background-color: #10b981; }  /* Green */
.archived-chip { background-color: #f59e0b; } /* Amber */
.unknown-chip { background-color: #6b7280; }  /* Gray */

/* Metric cards */
.metric-card {
    padding: 20px;
    border-radius: 18px;
    background: linear-gradient(135deg, #1e293b, #111827);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    text-align: center;
    color: #f1f5f9;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
}
.metric-label {
    font-size: 14px;
    opacity: 0.75;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.title("📦 Model Registry")
st.caption("Enterprise-level ML model governance dashboard")
st.divider()

# =====================================================
# LOAD MODEL DATA
# =====================================================
model_data = MongoService.get_model_registry()

if not model_data:
    st.warning("No models found in registry.")
    st.stop()

df = pd.DataFrame(model_data)

# =====================================================
# CHECK REQUIRED COLUMNS
# =====================================================
required_cols = ["model_name", "version", "training_date", "rmse", "accuracy", "status"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing expected columns in model registry: {missing_cols}")
    st.stop()

# =====================================================
# FORMAT DATA
# =====================================================
df["training_date"] = pd.to_datetime(df["training_date"], errors="coerce")
df["training_date"] = df["training_date"].dt.strftime("%Y-%m-%d %H:%M")
df["status"] = df["status"].str.capitalize()

def status_chip(val):
    if val.lower() == "active":
        return f'<div class="status-chip active-chip">Active ✅</div>'
    elif val.lower() == "archived":
        return f'<div class="status-chip archived-chip">Archived ⚠️</div>'
    else:
        return f'<div class="status-chip unknown-chip">Unknown ❓</div>'

df["status_chip"] = df["status"].apply(status_chip)

# =====================================================
# METRICS SUMMARY CARDS
# =====================================================
total_models = len(df)
active_models = len(df[df["status"].str.lower() == "active"])
archived_models = len(df[df["status"].str.lower() == "archived"])

cols = st.columns(3)
with cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_models}</div>
        <div class="metric-label">Total Models</div>
    </div>
    """, unsafe_allow_html=True)
with cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{active_models}</div>
        <div class="metric-label">Active Models</div>
    </div>
    """, unsafe_allow_html=True)
with cols[2]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{archived_models}</div>
        <div class="metric-label">Archived Models</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =====================================================
# DISPLAY TABLE (CENTERED)
# =====================================================
st.subheader("Models Details")

display_cols = ["model_name", "version", "training_date", "rmse", "accuracy", "status_chip"]
df_display = df[display_cols].rename(columns={"status_chip": "Status"})

st.markdown(f"""
<div class="table-container">
    {df_display.to_html(escape=False, index=False)}
</div>
""", unsafe_allow_html=True)

st.caption("Legend: ✅ Active | ⚠️ Archived | ❓ Unknown")