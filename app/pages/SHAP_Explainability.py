import streamlit as st
import pandas as pd
import altair as alt

from models.shap_explainer import compute_global_shap, compute_local_shap
from app.app_config import APP_CONFIG

st.set_page_config(
    page_title="SHAP Explainability",
    page_icon="🧩",
    layout=APP_CONFIG.get("layout", "wide")
)

st.title("🧩 AQI SHAP Explainability")
st.caption("Global & Local Feature Contributions")

st.divider()

# =====================================================
# GLOBAL SHAP
# =====================================================
st.subheader("📊 Global Feature Importance (Top 15)")

try:
    global_shap = compute_global_shap(top_n=15)
    if global_shap.empty:
        st.warning("No SHAP global data available.")
    else:
        st.dataframe(global_shap, use_container_width=True)

        # Bar chart
        chart = alt.Chart(global_shap).mark_bar().encode(
            x=alt.X("importance", title="Mean |SHAP| Value"),
            y=alt.Y("feature", sort="-x", title="Feature"),
            tooltip=["feature", "importance"]
        )
        st.altair_chart(chart, use_container_width=True)
except Exception as e:
    st.error(f"Failed to load global SHAP: {e}")

st.divider()

# =====================================================
# LOCAL SHAP
# =====================================================
st.subheader("🧩 Local SHAP (Latest Feature Row)")

try:
    local_shap = compute_local_shap()
    prediction = local_shap["prediction"]
    base_value = local_shap["base_value"]
    shap_values = local_shap["shap_values"]

    st.metric("Predicted AQI", f"{prediction:.2f}", delta=f"{prediction - base_value:.2f}")

    # Prepare DataFrame for chart
    shap_df = pd.DataFrame({
        "feature": list(shap_values.keys()),
        "shap_value": list(shap_values.values())
    }).sort_values("shap_value", key=abs, ascending=False).head(10)

    st.write("Top 10 SHAP feature contributions:")

    st.dataframe(shap_df, use_container_width=True)

    # Altair bar chart
    chart_local = alt.Chart(shap_df).mark_bar().encode(
        x=alt.X("shap_value", title="SHAP Value"),
        y=alt.Y("feature", sort="-x", title="Feature"),
        color=alt.condition(
            alt.datum.shap_value > 0,
            alt.value("#e74c3c"),
            alt.value("#2ecc71")
        ),
        tooltip=["feature", "shap_value"]
    )
    st.altair_chart(chart_local, use_container_width=True)

except Exception as e:
    st.error(f"Failed to load local SHAP: {e}")

st.divider()
st.caption("Base value (expected AQI without feature effects) shown in metric delta.")
