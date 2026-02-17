# =====================================================
# PEARLS AQI — PREDICTION PIPELINE
# Load Best Models + Predict + Save to MongoDB Atlas
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
    CLEANED_FEATURE_COLLECTION,  # ✅ use cleaned features
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

feature_col = db[CLEANED_FEATURE_COLLECTION]  # Use cleaned features for prediction
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
    raise ValueError("❌ No features found. Run feature pipeline first!")

df_latest = pd.DataFrame([latest_row])
print("Latest feature row loaded")
print("Feature timestamp:", latest_row.get("feature_generated_at"))
print("Total features:", len(df_latest.columns))

# -----------------------------------------------------
# PREDICTION LOOP
# -----------------------------------------------------
predictions = {}
meta_info = {}

for target in TARGETS:
    print(f"\n--- Predicting: {target} ---")

    # Fetch latest trained model for this target
    model_doc = registry_col.find_one(
        {"target": target},
        sort=[("trained_at", -1)]
    )

    if model_doc is None:
        print(f"No trained model found for {target}")
        continue

    model_path = model_doc.get("model_path")
    features_used = model_doc.get("features", [])
    model_name = model_doc.get("model_name", "unknown")
    task_type = model_doc.get("task_type", "unknown")

    if not model_path or not os.path.exists(model_path):
        print(f"Model file missing: {model_path}")
        continue

    # Load model
    model = joblib.load(model_path)

    # Validate features
    missing_features = [f for f in features_used if f not in df_latest.columns]
    if missing_features:
        print(f"❌ Missing features ({len(missing_features)}): {missing_features[:5]}...")
        continue

    X_latest = df_latest[features_used].copy()

    # Convert all features to numeric
    X_latest = X_latest.apply(pd.to_numeric, errors="coerce").fillna(0)

    # -------------------------------------------------
    # SCALER SUPPORT (IF USED DURING TRAINING)
    # -------------------------------------------------
    metrics = model_doc.get("metrics", {})
    scaler_path = metrics.get("scaler_path")

    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_input = scaler.transform(X_latest)
        print("Scaler applied")
    else:
        X_input = X_latest

    # -------------------------------------------------
    # MAKE PREDICTION
    # -------------------------------------------------
    pred = model.predict(X_input)

    if isinstance(pred, (list, np.ndarray)):
        pred = pred[0]

    # -------------------------------------------------
    # JSON SAFE PREDICTION
    # -------------------------------------------------
    try:
        pred_value = float(pred)
    except:
        pred_value = str(pred)

    if "class" in target:
        try:
            pred_value = int(pred_value)
        except:
            pass

    predictions[target] = pred_value
    meta_info[target] = {
        "model_name": model_name,
        "task_type": task_type,
        "model_path": model_path,
        "trained_at": model_doc.get("trained_at")
    }

    print(f"Model Used: {model_name}")
    print(f"Prediction: {pred_value}")

# -----------------------------------------------------
# SAVE PREDICTIONS + FEATURE SNAPSHOT TO MONGODB
# -----------------------------------------------------
if predictions:
    doc = {
        "city": settings.CITY,
        "created_at": datetime.utcnow(),
        "feature_generated_at": latest_row.get("feature_generated_at"),
        "predictions": predictions,
        "meta": meta_info,
        "features_used": df_latest.to_dict(orient="records")[0]  # 🔥 IMPORTANT: STORE FEATURE SNAPSHOT
    }

    pred_col.insert_one(doc)
    print("\n✅ Predictions + feature snapshot saved to MongoDB Atlas collection:", PREDICTION_COLLECTION)

# -----------------------------------------------------
# FINAL OUTPUT
# -----------------------------------------------------
print("\n========== FINAL PREDICTIONS ==========")
for k, v in predictions.items():
    print(f"{k} --> {v}")

print("\n========== PREDICTION PIPELINE COMPLETED ==========")
