# =====================================================
# PEARLS AQI — ADVANCED PREDICTION PIPELINE (FINAL)
# Load Active Models + Predict + Save to MongoDB Atlas
# Fully Compatible with Updated Training Pipeline
# =====================================================

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

from config.settings import settings
from config.constants import (
    CLEANED_FEATURE_COLLECTION,
    MODEL_COLLECTION,
    PREDICTION_COLLECTION
)

# -----------------------------------------------------
# LOAD ENV
# -----------------------------------------------------
load_dotenv()
print("========== STARTING PREDICTION PIPELINE ==========")

# -----------------------------------------------------
# CONNECT TO MONGODB ATLAS
# -----------------------------------------------------
client = MongoClient(settings.MONGO_URI)
db = client[settings.MONGO_DB_NAME]

feature_col = db[CLEANED_FEATURE_COLLECTION]
registry_col = db[MODEL_COLLECTION]
pred_col = db[PREDICTION_COLLECTION]

print("Connected to MongoDB Atlas")

# -----------------------------------------------------
# TARGET DEFINITIONS
# -----------------------------------------------------
TARGETS = [
    "target_aqi_t_plus_24h",        # Regression
    "target_aqi_class_t_plus_24h",  # Classification
    "target_aqi_class_t_plus_48h",
    "target_aqi_class_t_plus_72h"
]

# -----------------------------------------------------
# LOAD LATEST FEATURE ROW
# -----------------------------------------------------
latest_row = feature_col.find_one(
    sort=[("feature_generated_at", -1)],
    projection={"_id": 0}
)

if latest_row is None:
    raise ValueError("No cleaned features found. Run feature pipeline first.")

df_latest = pd.DataFrame([latest_row])

print("Latest feature row loaded")
print("Feature timestamp:", latest_row.get("feature_generated_at"))
print("Total available features:", len(df_latest.columns))

# -----------------------------------------------------
# PREDICTION LOOP
# -----------------------------------------------------
predictions = {}
meta_info = {}

for target in TARGETS:

    print(f"\n--- Predicting: {target} ---")

    # -------------------------------------------------
    # LOAD ACTIVE MODEL FROM REGISTRY
    # -------------------------------------------------
    model_doc = registry_col.find_one(
        {"target": target, "status": "active"}
    )

    if model_doc is None:
        print(f"No active model found for {target}")
        continue

    model_path = model_doc.get("model_path")
    features_used = model_doc.get("features", [])
    model_name = model_doc.get("model_name", "unknown")
    task_type = model_doc.get("task_type", "unknown")
    version = model_doc.get("version")
    pipeline_version = model_doc.get("pipeline_version")
    training_date = model_doc.get("training_date")

    # -------------------------------------------------
    # VALIDATE MODEL FILE
    # -------------------------------------------------
    if not model_path or not os.path.exists(model_path):
        print(f"Model file missing: {model_path}")
        continue

    # -------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        continue

    # -------------------------------------------------
    # FEATURE VALIDATION
    # -------------------------------------------------
    if not features_used:
        print("No feature list found in registry.")
        continue

    missing_features = [f for f in features_used if f not in df_latest.columns]
    if missing_features:
        print(f"Missing features ({len(missing_features)}): {missing_features[:5]}...")
        continue

    X_latest = df_latest[features_used].copy()

    # Ensure numeric + safe
    X_latest = X_latest.apply(pd.to_numeric, errors="coerce").fillna(0)

    # -------------------------------------------------
    # MAKE PREDICTION
    # -------------------------------------------------
    try:
        pred = model.predict(X_latest)
    except Exception as e:
        print(f"Prediction failed: {e}")
        continue

    if isinstance(pred, (list, np.ndarray)):
        pred = pred[0]

    # -------------------------------------------------
    # SAFE TYPE CAST
    # -------------------------------------------------
    try:
        pred_value = float(pred)
    except Exception:
        pred_value = str(pred)

    if "class" in target:
        try:
            pred_value = int(round(float(pred_value)))
        except Exception:
            pass

    predictions[target] = pred_value

    # -------------------------------------------------
    # META INFO (GOVERNANCE READY)
    # -------------------------------------------------
    meta_info[target] = {
        "model_name": model_name,
        "task_type": task_type,
        "version": version,
        "pipeline_version": pipeline_version,
        "training_date": training_date,
        "model_path": model_path
    }

    print(f"Model Used: {model_name} (v{version})")
    print(f"Prediction: {pred_value}")

# -----------------------------------------------------
# SAVE PREDICTIONS + FEATURE SNAPSHOT
# -----------------------------------------------------
if predictions:

    prediction_document = {
        "city": settings.CITY,
        "created_at": datetime.utcnow(),
        "feature_generated_at": latest_row.get("feature_generated_at"),
        "predictions": predictions,
        "meta": meta_info,
        "feature_snapshot": df_latest.to_dict(orient="records")[0],
        "prediction_pipeline_version": "v2_production"
    }

    pred_col.insert_one(prediction_document)

    print("\nPredictions + feature snapshot saved to MongoDB:", PREDICTION_COLLECTION)

else:
    print("\nNo predictions generated.")

# -----------------------------------------------------
# FINAL OUTPUT
# -----------------------------------------------------
print("\n========== FINAL PREDICTIONS ==========")
for k, v in predictions.items():
    print(f"{k} --> {v}")

print("\n========== PREDICTION PIPELINE COMPLETED ==========")