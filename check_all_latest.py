# =====================================================
# Pearls AQI — MongoDB Collection Health Check (Updated)
# =====================================================

import os
from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
from config.settings import settings
from config.constants import (
    CLEANED_FEATURE_COLLECTION,
    MODEL_COLLECTION,
    PREDICTION_COLLECTION,
)

# -----------------------------------------------------
# LOAD ENV
# -----------------------------------------------------
load_dotenv()
client = MongoClient(settings.MONGO_URI)
db = client[settings.MONGO_DB_NAME]

# -----------------------------------------------------
# COLLECTIONS TO CHECK
# -----------------------------------------------------
collections_to_check = [
    "raw_aqi_data",
    "aqi_features",
    "clean_aqi",
    "feature_cleaned",
    CLEANED_FEATURE_COLLECTION,
    MODEL_COLLECTION,
    PREDICTION_COLLECTION,
    "fs.files",
    "fs.chunks"
]

print("\n========== MONGO DB COLLECTION HEALTH CHECK ==========")

for col_name in collections_to_check:
    col = db[col_name]
    total_docs = col.count_documents({})

    # Fetch latest doc safely
    latest_doc = col.find_one(sort=[("timestamp", -1)]) or col.find_one(sort=[("feature_generated_at", -1)]) or col.find_one()
    
    if latest_doc:
        # Determine timestamp / unique id
        ts = latest_doc.get("timestamp") or latest_doc.get("feature_generated_at") or latest_doc.get("_id")
        if hasattr(ts, "generation_time"):  # ObjectId case
            ts = ts.generation_time

        # Count NaN / None values
        if isinstance(latest_doc, dict):
            nan_count = sum(pd.isna(v) for v in latest_doc.values())
        else:
            nan_count = "N/A"
    else:
        ts = "No documents"
        nan_count = "N/A"

    print(f"\nCollection: {col_name}")
    print(f"  Total Documents       : {total_docs}")
    print(f"  Latest Timestamp / ID : {ts}")
    print(f"  NaN / None values     : {nan_count}")

# -----------------------------------------------------
# CHECK ACTIVE MODELS
# -----------------------------------------------------
active_models = db[MODEL_COLLECTION].find({"status": "active"})
print("\n========== ACTIVE MODELS ==========")
if active_models.count() == 0:
    print("No active models found.")
else:
    for m in active_models:
        print(
            f"  Target: {m.get('target')} | "
            f"Model: {m.get('model_name')} | "
            f"Version: {m.get('version')} | "
            f"Trained at: {m.get('trained_at')} | "
            f"Features used: {len(m.get('features', []))}"
        )

# -----------------------------------------------------
# CHECK LATEST PREDICTIONS
# -----------------------------------------------------
latest_pred = db[PREDICTION_COLLECTION].find_one(sort=[("predicted_at", -1)])
print("\n========== LATEST PREDICTIONS ==========")
if latest_pred:
    pred_ts = latest_pred.get("predicted_at")
    pred_targets = latest_pred.get("predictions", {})
    print(f"Latest prediction saved at: {pred_ts}")
    print(f"Predicted targets: {list(pred_targets.keys())}")
    # Optional: print prediction values
    for k, v in pred_targets.items():
        print(f"  {k} --> {v}")
else:
    print("No predictions found in PREDICTION_COLLECTION")