# =====================================================
# PM2.5 MULTI-HORIZON TRAINING PIPELINE (LOCAL ONLY)
# =====================================================

import os
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import warnings

from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config.settings import settings
from config.constants import FEATURE_COLLECTION

load_dotenv()

print("========== STARTING PM2.5 TRAINING (LOCAL MODE) ==========")

# -----------------------------------------------------
# LOAD DATA FROM MONGO (READ ONLY)
# -----------------------------------------------------
client = MongoClient(settings.MONGO_URI)
db = client[settings.MONGO_DB_NAME]
feature_col = db[FEATURE_COLLECTION]

data = list(feature_col.find({}, {"_id": 0}))
df = pd.DataFrame(data)

print("Loaded shape:", df.shape)

# -----------------------------------------------------
# TARGETS
# -----------------------------------------------------
TARGETS = [
    "target_pm2_5_t_plus_24h",
    "target_pm2_5_t_plus_48h",
    "target_pm2_5_t_plus_72h"
]

# -----------------------------------------------------
# SORT (VERY IMPORTANT FOR TIME SERIES)
# -----------------------------------------------------
df = df.sort_values("timestamp").reset_index(drop=True)

# -----------------------------------------------------
# DROP NON-FEATURES
# -----------------------------------------------------
DROP_COLS = [
    "timestamp",
    "feature_generated_at",
    "target_aqi_t_plus_24h",
    "target_aqi_class_t_plus_24h",
    "target_aqi_class_t_plus_48h",
    "target_aqi_class_t_plus_72h"
] + TARGETS

FEATURE_COLS = [
    c for c in df.columns
    if c not in DROP_COLS
    and pd.api.types.is_numeric_dtype(df[c])
]

X = df[FEATURE_COLS]

warnings.filterwarnings("ignore")

# -----------------------------------------------------
# LOCAL MODEL DIR
# -----------------------------------------------------
MODEL_DIR = "artifacts/models"
os.makedirs(MODEL_DIR, exist_ok=True)

metrics_summary = {}

# -----------------------------------------------------
# TRAIN LOOP
# -----------------------------------------------------
for target in TARGETS:

    print(f"\n------ TRAINING {target} ------")

    if target not in df.columns:
        print("Target missing → skipping")
        continue

    y = df[target]

    valid_idx = y.notna()
    X_t = X.loc[valid_idx]
    y_t = y.loc[valid_idx]

    split = int(len(X_t) * 0.8)

    X_train = X_t.iloc[:split]
    X_test = X_t.iloc[split:]

    y_train = y_t.iloc[:split]
    y_test = y_t.iloc[split:]

    # ✅ CREATE FRESH MODEL (IMPORTANT)
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )

    # TRAIN
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R2  : {r2:.3f}")

    # -------------------------------------------------
    # SAVE LOCALLY
    # -------------------------------------------------
    model_path = os.path.join(MODEL_DIR, f"{target}_rf.joblib")

    joblib.dump({
        "model": model,
        "features": FEATURE_COLS,
        "target": target,
        "metrics": {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
    }, model_path)

    print("✅ MODEL SAVED:", model_path)

    metrics_summary[target] = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

# -----------------------------------------------------
# SAVE METRICS FILE (SUPER USEFUL)
# -----------------------------------------------------
metrics_path = os.path.join(MODEL_DIR, "training_metrics.json")

import json
with open(metrics_path, "w") as f:
    json.dump(metrics_summary, f, indent=4)

print("\n✅ Metrics saved:", metrics_path)

print("\n========== PM2.5 TRAINING COMPLETE ==========")
