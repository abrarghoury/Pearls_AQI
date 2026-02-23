# =====================================================
# PEARLS AQI — ADVANCED PREDICTION PIPELINE (INFERENCE READY)
# MongoDB-safe: Converts all NumPy/Pandas types to Python
# =====================================================

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from io import BytesIO
import gridfs

from config.settings import settings
from config.constants import (
    CLEANED_FEATURE_COLLECTION,
    MODEL_COLLECTION,
    PREDICTION_COLLECTION
)

# -------------------------------
# HELPER: Convert all types recursively
# -------------------------------
def to_python_types(obj):
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(x) for x in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    else:
        return obj

# -------------------------------
# LOAD ENV & CONNECT
# -------------------------------
load_dotenv()
print("========== STARTING PREDICTION PIPELINE ==========")

client = MongoClient(settings.MONGO_URI)
db = client[settings.MONGO_DB_NAME]

feature_col = db[CLEANED_FEATURE_COLLECTION]
registry_col = db[MODEL_COLLECTION]
pred_col = db[PREDICTION_COLLECTION]

fs = gridfs.GridFS(db)
print("Connected to MongoDB Atlas")

# -------------------------------
# TARGETS
# -------------------------------
TARGETS = [
    "target_aqi_t_plus_24h",
    "target_aqi_class_t_plus_24h",
    "target_aqi_class_t_plus_48h",
    "target_aqi_class_t_plus_72h"
]

# -------------------------------
# LOAD HISTORICAL FEATURES
# -------------------------------
MAX_LAG = 72
history_cursor = feature_col.find(
    sort=[("feature_generated_at", -1)],
    limit=MAX_LAG + 1,
    projection={"_id": 0}
)

df_history = pd.DataFrame(list(history_cursor))
if df_history.empty:
    raise ValueError("No cleaned features found. Run feature pipeline first.")

df_history["timestamp"] = pd.to_datetime(df_history["timestamp"])
df_history = df_history.sort_values("timestamp").reset_index(drop=True)
print("Loaded historical rows:", len(df_history))

# -------------------------------
# LATEST ROW
# -------------------------------
df_latest = df_history.iloc[[-1]].copy()

for col in df_latest.columns:
    if col != "timestamp":
        df_latest[col] = pd.to_numeric(df_latest[col], errors="coerce").fillna(0)

print("Latest feature row ready:", df_latest["timestamp"].iloc[0])
print("Total features:", len(df_latest.columns))

# -------------------------------
# PREDICTION LOOP
# -------------------------------
predictions = {}
meta_info = {}

for target in TARGETS:
    print(f"\n--- Predicting: {target} ---")

    model_doc = registry_col.find_one({"target": target, "status": "active"})
    if not model_doc:
        print(f"No active model for {target}")
        continue

    gridfs_file_id = model_doc.get("gridfs_file_id")
    features_used = model_doc.get("features", [])
    model_name = model_doc.get("model_name", "unknown")
    task_type = model_doc.get("task_type", "unknown")
    version = model_doc.get("version")
    pipeline_version = model_doc.get("pipeline_version")
    training_date = model_doc.get("training_date")

    if not gridfs_file_id:
        print(f"No GridFS file ID for {target}. Skipping.")
        continue

    # Load model
    try:
        model_bytes = fs.get(gridfs_file_id).read()
        model = joblib.load(BytesIO(model_bytes))
    except Exception as e:
        print(f"Failed to load model: {e}")
        continue

    # Ensure all features exist
    missing_features = [f for f in features_used if f not in df_latest.columns]
    if missing_features:
        print(f"Missing features ({len(missing_features)}): {missing_features[:5]}...")
        for f in missing_features:
            df_latest[f] = 0

    X_latest = df_latest[features_used].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Predict
    try:
        pred = model.predict(X_latest)
        if isinstance(pred, (list, np.ndarray)):
            pred = pred[0]
    except Exception as e:
        print(f"Prediction failed: {e}")
        continue

    # Cast types
    if "class" in target:
        try:
            pred_value = int(round(float(pred)))
        except:
            pred_value = str(pred)
    else:
        try:
            pred_value = float(pred)
        except:
            pred_value = str(pred)

    predictions[target] = pred_value

    meta_info[target] = {
        "model_name": model_name,
        "task_type": task_type,
        "version": version,
        "pipeline_version": pipeline_version,
        "training_date": training_date,
        "gridfs_file_id": str(gridfs_file_id)
    }

    print(f"Model Used: {model_name} (v{version}) | Prediction: {pred_value}")

# -------------------------------
# SAVE TO MONGO (ALL TYPES SAFE)
# -------------------------------
if predictions:
    feature_snapshot_safe = to_python_types(df_latest.to_dict(orient="records")[0])
    feature_generated_at_safe = to_python_types(df_latest["feature_generated_at"].iloc[0])

    prediction_document = {
        "city": settings.CITY,
        "predicted_at": datetime.utcnow(),
        "created_at": datetime.utcnow(),
        "feature_generated_at": feature_generated_at_safe,
        "predictions": predictions,
        "meta": meta_info,
        "feature_snapshot": feature_snapshot_safe,
        "prediction_pipeline_version": "v2_production"
    }

    pred_col.insert_one(prediction_document)
    print("\nPredictions + feature snapshot saved to MongoDB")

else:
    print("\nNo predictions generated.")

# -------------------------------
# FINAL OUTPUT
# -------------------------------
print("\n========== FINAL PREDICTIONS ==========")
for k, v in predictions.items():
    print(f"{k} --> {v}")

print("\n========== PREDICTION PIPELINE COMPLETED ==========")
