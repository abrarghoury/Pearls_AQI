# =====================================================
# MULTI-DAY AQI MODEL TRAINING PIPELINE (GRIDFS / ATLAS)
# PRODUCTION-READY | DYNAMIC VERSIONING | ACTIVE MODEL
# =====================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from io import BytesIO

from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.exceptions import ConvergenceWarning

from config.settings import settings
from config.constants import CLEANED_FEATURE_COLLECTION, MODEL_COLLECTION

# =====================================================
# ENV + DB CONNECTION
# =====================================================
load_dotenv()

mongo_client = MongoClient(settings.MONGO_URI)
db = mongo_client[settings.MONGO_DB_NAME]

clean_feature_collection = db[CLEANED_FEATURE_COLLECTION]
model_registry_collection = db[MODEL_COLLECTION]

# GridFS instance
fs = gridfs.GridFS(db)

print("========== STARTING CASE B TRAINING PIPELINE (GRIDFS) ==========")

# =====================================================
# LOAD CLEANED FEATURES
# =====================================================
records = list(clean_feature_collection.find({}, {"_id": 0}))
if not records:
    raise ValueError("No cleaned features found.")

df = pd.DataFrame(records)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
print(f"Loaded dataset shape: {df.shape}")

# =====================================================
# TARGET CONFIG
# =====================================================
REGRESSION_TARGET = "target_aqi_t_plus_24h"
CLASSIFICATION_TARGETS = [
    "target_aqi_class_t_plus_24h",
    "target_aqi_class_t_plus_48h",
    "target_aqi_class_t_plus_72h"
]
ALL_TARGETS = [REGRESSION_TARGET] + CLASSIFICATION_TARGETS

# =====================================================
# FEATURE SELECTION
# =====================================================
META_COLUMNS = ["timestamp", "feature_generated_at", "rows_after_cleaning"]
DROP_COLUMNS = META_COLUMNS + ALL_TARGETS
FEATURE_COLUMNS = [c for c in df.columns if c not in DROP_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
X_full = df[FEATURE_COLUMNS]
print(f"Total features used: {len(FEATURE_COLUMNS)}")

# =====================================================
# DEFINE MODELS
# =====================================================
REGRESSION_MODELS = {
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=18, min_samples_split=5, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, max_depth=8, learning_rate=0.05, min_samples_leaf=20, random_state=42)
}

CLASSIFICATION_MODELS = {
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_split=5, class_weight="balanced", random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, eval_metric="mlogloss", random_state=42)
}

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# =====================================================
# TRAIN LOOP
# =====================================================
for target_name in ALL_TARGETS:
    print(f"\n--- Training for {target_name} ---")
    if target_name not in df.columns:
        print(f"Skipping {target_name} (not found)")
        continue

    y_full = df[target_name]
    valid_idx = y_full.notna()
    X_full_valid = X_full.loc[valid_idx]
    y_full = y_full.loc[valid_idx]
    df_valid = df.loc[valid_idx]

    if len(df_valid) < 50:
        print("Not enough data. Skipping.")
        continue

    # Detect task type
    task_type = "classification" if "class" in target_name else "regression"
    models = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS

    # Time-series split
    df_sorted = df_valid.sort_values("timestamp")
    X_sorted = X_full_valid.loc[df_sorted.index]
    y_sorted = y_full.loc[df_sorted.index]
    split_index = int(len(df_sorted) * 0.8)
    X_train, X_test = X_sorted.iloc[:split_index], X_sorted.iloc[split_index:]
    y_train, y_test = y_sorted.iloc[:split_index], y_sorted.iloc[split_index:]

    # Handle imbalance
    if task_type == "classification":
        ros = RandomOverSampler(random_state=42)
        y_train_xgb = y_train - 1
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
        X_train_res_xgb, y_train_res_xgb = ros.fit_resample(X_train, y_train_xgb)
    else:
        X_train_res, y_train_res = X_train, y_train

    best_model, best_score, best_metrics, best_model_name = None, -np.inf, None, None
    training_start_time = datetime.utcnow()

    # =====================================================
    # MODEL TRAINING & EVALUATION
    # =====================================================
    for model_name, model in models.items():
        if task_type == "classification" and model_name == "XGBoost":
            model.fit(X_train_res_xgb, y_train_res_xgb)
            preds = model.predict(X_test) + 1
        else:
            model.fit(X_train_res, y_train_res)
            preds = model.predict(X_test)

        if task_type == "regression":
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            score = r2
            metrics = {"rmse": rmse, "mae": mae, "r2": r2}
        else:
            accuracy = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            score = f1
            metrics = {"accuracy": accuracy, "f1_weighted": f1}

        print(f"{model_name} | {metrics}")
        if score > best_score:
            best_score, best_model, best_model_name, best_metrics = score, model, model_name, metrics

    # =====================================================
    # SAVE BEST MODEL TO GRIDFS
    # =====================================================
    model_bytes = BytesIO()
    joblib.dump(best_model, model_bytes)
    model_bytes.seek(0)
    gridfs_filename = f"{target_name}_{best_model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.joblib"
    file_id = fs.put(model_bytes, filename=gridfs_filename)

    # =====================================================
    # ARCHIVE PREVIOUS ACTIVE MODELS
    # =====================================================
    model_registry_collection.update_many(
        {"target": target_name, "status": "active"},
        {"$set": {"status": "archived"}}
    )

    # =====================================================
    # DYNAMIC VERSIONING
    # =====================================================
    last_model = model_registry_collection.find({"target": target_name}).sort("version", -1).limit(1)
    last_version_num = 0
    for m in last_model:
        v_str = m.get("version", "v0").lstrip("v").split(".")[0]
        if v_str.isdigit():
            last_version_num = int(v_str)
    new_version = f"v{last_version_num + 1}"

    # =====================================================
    # REGISTER NEW ACTIVE MODEL
    # =====================================================
    model_registry_collection.insert_one({
        "target": target_name,
        "model_name": best_model_name,
        "task_type": task_type,
        "version": new_version,
        "metrics": best_metrics,
        "training_date": datetime.utcnow(),
        "trained_at": datetime.utcnow(),
        "training_duration_seconds": (datetime.utcnow() - training_start_time).total_seconds(),
        "status": "active",
        "gridfs_file_id": file_id,
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "data_rows_used": len(df_valid),
        "pipeline_version": "case_b_v2",
        "created_at": datetime.utcnow()
    })

    print(f"BEST MODEL: {best_model_name} | Task={task_type} | Score={best_score:.3f} | Version={new_version}")

print("\n========== TRAINING PIPELINE COMPLETED ==========")
