# =====================================================
# models/shap_explainer.py
# =====================================================

import joblib
import shap
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient

from config.settings import settings
from config.constants import (
    MODEL_COLLECTION,
    CLEANED_FEATURE_COLLECTION   # e.g., "feature_cleaned"
)
from app.services.mongo_service import MongoService  # Added for wrapper

# =====================================================
# INTERNAL HELPERS
# =====================================================

def _get_db():
    """Returns MongoDB database connection."""
    client = MongoClient(settings.MONGO_URI)
    return client[settings.MONGO_DB_NAME]


def _load_best_regression_model():
    """
    Loads the latest GradientBoosting regression model for 24h AQI.
    """
    db = _get_db()

    model_doc = db[MODEL_COLLECTION].find_one(
        {"target": "target_aqi_t_plus_24h", "model_name": "GradientBoosting"},
        sort=[("trained_at", -1)]
    )

    if not model_doc:
        raise ValueError("No trained GradientBoosting model found in registry.")

    model_path = os.path.normpath(model_doc.get("model_path"))
    if not model_path or not os.path.exists(model_path):
        raise ValueError(f"Model file missing: {model_path}")

    model = joblib.load(model_path)
    return model, model_doc


def _load_feature_dataframe(model_doc):
    """
    Loads cleaned feature dataset used for training.
    """
    db = _get_db()
    data = list(db[CLEANED_FEATURE_COLLECTION].find({}, {"_id": 0}))
    if not data:
        raise ValueError("No feature data found in CLEANED_FEATURE_COLLECTION.")

    df = pd.DataFrame(data)
    feature_cols = model_doc.get("features", [])
    missing_cols = [f for f in feature_cols if f not in df.columns]
    if missing_cols:
        raise ValueError(f"Features missing in cleaned dataset: {missing_cols}")

    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    return df, X, feature_cols


# =====================================================
# GLOBAL SHAP
# =====================================================

def compute_global_shap(top_n=15, sample_size=None):
    """
    Returns global feature importance (mean |SHAP| values) for regression model.
    Optional: sample_size to speed up calculation.
    """
    model, model_doc = _load_best_regression_model()
    df, X, feature_cols = _load_feature_dataframe(model_doc)

    if sample_size:
        X = X.sample(min(sample_size, len(X)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": mean_abs_shap
    }).sort_values("importance", ascending=False)

    return importance_df.head(top_n)


# =====================================================
# LOCAL SHAP (FROM DASHBOARD FEATURES)
# =====================================================

def compute_local_shap_from_dashboard(latest_features: dict):
    """
    Returns SHAP explanation for the latest feature row from dashboard.
    latest_features: dictionary from MongoService.get_latest_features()
    """
    if not latest_features:
        raise ValueError("latest_features is empty. Cannot compute local SHAP.")

    model, model_doc = _load_best_regression_model()
    feature_cols = model_doc.get("features", [])

    missing_cols = [f for f in feature_cols if f not in latest_features]
    if missing_cols:
        raise ValueError(f"Missing features in latest_features: {missing_cols}")

    # Convert to DataFrame
    X = pd.DataFrame([latest_features[feat] if feat in latest_features else 0 for feat in feature_cols], index=feature_cols).T
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    prediction_value = latest_features.get("aqi") or model.predict(X)[0]
    base_value = explainer.expected_value

    shap_dict = dict(zip(feature_cols, shap_values[0]))

    return {
        "base_value": float(base_value),
        "prediction": float(prediction_value),
        "shap_values": shap_dict,
        "model_info": {
            "model_name": model_doc.get("model_name"),
            "trained_at": model_doc.get("trained_at"),
            "metrics": model_doc.get("metrics", {})
        }
    }


# =====================================================
# WRAPPER FOR DASHBOARD
# =====================================================

def compute_local_shap():
    """
    Fetch latest features from MongoService and return local SHAP.
    Dashboard calls this function directly.
    """
    latest_features = MongoService.get_latest_features()
    return compute_local_shap_from_dashboard(latest_features)
