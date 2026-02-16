import joblib
import shap
import numpy as np
import pandas as pd
from pymongo import MongoClient
from config.settings import settings
from config.constants import (
    MODEL_COLLECTION,
    FEATURE_COLLECTION
)

# =====================================================
# INTERNAL HELPERS
# =====================================================

def _get_db():
    client = MongoClient(settings.MONGO_URI)
    return client[settings.MONGO_DB_NAME]

def _load_best_regression_model():
    """
    Loads latest best regression model for 24h AQI.
    """
    db = _get_db()
    model_doc = db[MODEL_COLLECTION].find_one(
        {"target": "target_aqi_t_plus_24h"},
        sort=[("trained_at", -1)]
    )

    if not model_doc:
        raise ValueError("No trained regression model found in registry.")

    model_path = model_doc.get("model_path")
    if not model_path:
        raise ValueError("Model path missing in registry.")

    model = joblib.load(model_path)
    return model, model_doc

def _load_feature_dataframe(model_doc):
    """
    Loads cleaned feature dataset used for training and aligns columns with model.
    """
    db = _get_db()
    data = list(db[FEATURE_COLLECTION].find({}, {"_id": 0}))
    if not data:
        raise ValueError("No feature data found in FEATURE_COLLECTION.")

    df = pd.DataFrame(data)

    # Ensure features used for this model
    feature_cols = model_doc.get("features", [])
    X = df[feature_cols].copy()

    # Convert all to numeric and fill missing
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    return df, X, feature_cols

# =====================================================
# PUBLIC FUNCTIONS
# =====================================================

def compute_global_shap(top_n=15):
    """
    Returns global feature importance (mean |SHAP| values) for regression model.
    """
    model, model_doc = _load_best_regression_model()
    _, X, feature_cols = _load_feature_dataframe(model_doc)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute shap value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Safety check
    if len(feature_cols) != len(mean_abs_shap):
        raise ValueError(f"Feature count ({len(feature_cols)}) "
                         f"does not match SHAP values length ({len(mean_abs_shap)})")

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": mean_abs_shap
    }).sort_values("importance", ascending=False)

    return importance_df.head(top_n)

def compute_local_shap():
    """
    Returns SHAP explanation for latest feature row.
    """
    model, model_doc = _load_best_regression_model()
    _, X, feature_cols = _load_feature_dataframe(model_doc)

    latest_row = X.iloc[[-1]]  # Keep as DataFrame

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(latest_row)

    base_value = explainer.expected_value
    prediction = model.predict(latest_row)[0]

    shap_dict = dict(zip(feature_cols, shap_values[0]))

    return {
        "base_value": float(base_value),
        "prediction": float(prediction),
        "shap_values": shap_dict,
        "model_info": {
            "model_name": model_doc.get("model_name"),
            "trained_at": model_doc.get("trained_at"),
            "metrics": model_doc.get("metrics", {})
        }
    }
