# =====================================================
# models/shap_explainer.py (FULL UPDATED FINAL)
# =====================================================

import os
import joblib
import shap
import numpy as np
import pandas as pd
from pymongo import MongoClient
from glob import glob

from config.settings import settings
from config.constants import (
    MODEL_COLLECTION,
    CLEANED_FEATURE_COLLECTION
)
from app.services.mongo_service import MongoService

# =====================================================
# INTERNAL HELPERS
# =====================================================

def _get_db():
    """Returns MongoDB database connection."""
    client = MongoClient(settings.MONGO_URI)
    return client[settings.MONGO_DB_NAME]


def _load_latest_regression_model(target: str = "target_aqi_t_plus_24h"):
    """
    Fetch the latest active regression model for a given target.
    If the model path in MongoDB is missing, fallback to the latest file in artifacts/models/
    """
    db = _get_db()
    model_doc = db[MODEL_COLLECTION].find_one(
        {"target": target, "task_type": "regression", "status": "active"},
        sort=[("trained_at", -1)]
    )
    if not model_doc:
        raise ValueError(f"No active regression model found for target {target}")

    model_path = model_doc.get("model_path")

    # Fallback if registry path missing
    if not model_path or not os.path.exists(model_path):
        print(f"Model file missing at registry path: {model_path}")
        files = glob(f"artifacts/models/{target}_*.joblib")
        if not files:
            raise ValueError(f"No model files found locally for target {target}")
        model_path = max(files, key=os.path.getctime)
        print(f"Falling back to latest local model: {model_path}")

    model_path = os.path.normpath(model_path)
    model = joblib.load(model_path)
    return model, model_doc


def _load_feature_dataframe(model_doc):
    """
    Loads cleaned feature dataset used for training.
    Ensures feature columns match the model.
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

def compute_global_shap(top_n=15, sample_size=None, target: str = "target_aqi_t_plus_24h"):
    """
    Returns global SHAP importance (mean |SHAP| values) for regression model.
    Optional sample_size to speed up calculation.
    """
    model, model_doc = _load_latest_regression_model(target)
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

def compute_local_shap_from_dashboard(latest_features: dict, target: str = "target_aqi_t_plus_24h"):
    """
    Returns SHAP explanation for latest dashboard feature row.
    latest_features: dict from MongoService.get_latest_features()
    """
    if not latest_features:
        raise ValueError("latest_features is empty. Cannot compute local SHAP.")

    model, model_doc = _load_latest_regression_model(target)
    feature_cols = model_doc.get("features", [])

    # Fill missing features with 0
    for f in feature_cols:
        if f not in latest_features:
            latest_features[f] = 0

    # Convert to single-row DataFrame
    X = pd.DataFrame([{feat: latest_features.get(feat, 0) for feat in feature_cols}])
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Prediction value (regression)
    try:
        prediction_value = float(model.predict(X)[0])
    except Exception:
        prediction_value = latest_features.get("aqi", 0)

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

def compute_local_shap(target: str = "target_aqi_t_plus_24h"):
    """
    Fetch latest features from MongoService and return local SHAP.
    Dashboard calls this function directly.
    """
    latest_features = MongoService.get_latest_features()
    return compute_local_shap_from_dashboard(latest_features, target)