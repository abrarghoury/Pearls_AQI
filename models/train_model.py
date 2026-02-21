# =====================================================
# MULTI-DAY AQI MODEL TRAINING PIPELINE (CASE B – UPDATED FINAL)
# Logic Same | Structure Improved | Production Hardened
# =====================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from pymongo import MongoClient
from dotenv import load_dotenv

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.exceptions import ConvergenceWarning

from config.settings import settings
from config.constants import (
    CLEANED_FEATURE_COLLECTION,
    MODEL_COLLECTION,
    TARGET_COLUMN
)

# =====================================================
# ENV + DB CONNECTION
# =====================================================
load_dotenv()

mongo_client = MongoClient(settings.MONGO_URI)
database = mongo_client[settings.MONGO_DB_NAME]

clean_feature_collection = database[CLEANED_FEATURE_COLLECTION]
model_registry_collection = database[MODEL_COLLECTION]

print("========== STARTING CASE B TRAINING PIPELINE (FINAL) ==========")

# =====================================================
# LOAD CLEANED FEATURES
# =====================================================
records = list(clean_feature_collection.find({}, {"_id": 0}))
if not records:
    raise ValueError("No cleaned features found.")

df = pd.DataFrame(records)
print(f"Loaded dataset shape: {df.shape}")

# =====================================================
# TIMESTAMP SAFETY CHECK
# =====================================================
if "timestamp" not in df.columns:
    raise ValueError("timestamp column missing for time-series split.")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])

# =====================================================
# TARGET CONFIG (Case B)
# =====================================================
REGRESSION_TARGET = "target_aqi_t_plus_24h"

CLASSIFICATION_TARGETS = [
    "target_aqi_class_t_plus_24h",
    "target_aqi_class_t_plus_48h",
    "target_aqi_class_t_plus_72h"
]

ALL_TARGETS = [REGRESSION_TARGET] + CLASSIFICATION_TARGETS

# =====================================================
# FEATURE SELECTION (Numeric Only + Drop Meta)
# =====================================================
META_COLUMNS = [
    "timestamp",
    "feature_generated_at",
    "rows_after_cleaning"
]

DROP_COLUMNS = META_COLUMNS + ALL_TARGETS

FEATURE_COLUMNS = [
    col for col in df.columns
    if col not in DROP_COLUMNS
    and pd.api.types.is_numeric_dtype(df[col])
]

X_full = df[FEATURE_COLUMNS]

print(f"Total features used: {len(FEATURE_COLUMNS)}")

# =====================================================
# DEFINE MODELS
# =====================================================
REGRESSION_MODELS = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=42
    )
}

CLASSIFICATION_MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        eval_metric="mlogloss",
        random_state=42
    )
}

# =====================================================
# SUPPRESS WARNINGS
# =====================================================
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

    # Drop NaN target rows safely
    valid_idx = y_full.notna()
    X_full_valid = X_full.loc[valid_idx]
    y_full = y_full.loc[valid_idx]
    df_valid = df.loc[valid_idx]

    if len(df_valid) < 50:
        print("Not enough data after NaN filtering. Skipping.")
        continue

    # Detect Task Type
    if "class" in target_name:
        task_type = "classification"
        models = CLASSIFICATION_MODELS
    else:
        task_type = "regression"
        models = REGRESSION_MODELS

    # -------------------------------------------------
    # Time-Series Aware Split (No Shuffle)
    # -------------------------------------------------
    df_sorted = df_valid.sort_values("timestamp")
    X_sorted = X_full_valid.loc[df_sorted.index]
    y_sorted = y_full.loc[df_sorted.index]

    split_index = int(len(df_sorted) * 0.8)

    X_train = X_sorted.iloc[:split_index]
    X_test = X_sorted.iloc[split_index:]
    y_train = y_sorted.iloc[:split_index]
    y_test = y_sorted.iloc[split_index:]

    # -------------------------------------------------
    # Handle Imbalance (Classification)
    # -------------------------------------------------
    if task_type == "classification":
        ros = RandomOverSampler(random_state=42)

        y_train_xgb = y_train - 1

        X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
        X_train_res_xgb, y_train_res_xgb = ros.fit_resample(X_train, y_train_xgb)
    else:
        X_train_res, y_train_res = X_train, y_train

    best_model = None
    best_score = -np.inf
    best_metrics = None
    best_model_name = None

    training_start_time = datetime.utcnow()

    # -------------------------------------------------
    # Model Selection Loop
    # -------------------------------------------------
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

            print(f"{model_name} | RMSE={rmse:.2f} | R2={r2:.3f}")

        else:
            accuracy = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            score = f1
            metrics = {"accuracy": accuracy, "f1_weighted": f1}

            print(f"{model_name} | Accuracy={accuracy:.3f} | F1={f1:.3f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = model_name
            best_metrics = metrics

    training_end_time = datetime.utcnow()
    training_duration_seconds = (
        training_end_time - training_start_time
    ).total_seconds()

    # =================================================
    # SAVE BEST MODEL
    # =================================================
    model_directory = "artifacts/models"
    os.makedirs(model_directory, exist_ok=True)

    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    model_path = os.path.join(
        model_directory,
        f"{target_name}_{best_model_name}_{timestamp_str}.joblib"
    )

    joblib.dump(best_model, model_path)

    # =================================================
    # ARCHIVE PREVIOUS ACTIVE MODEL
    # =================================================
    model_registry_collection.update_many(
        {"target": target_name, "status": "active"},
        {"$set": {"status": "archived"}}
    )

    # =================================================
    # REGISTER NEW ACTIVE MODEL
    # =================================================
    model_registry_collection.update_one(
        {"target": target_name, "version": "v2.0"},
        {
            "$set": {
                "target": target_name,
                "model_name": best_model_name,
                "task_type": task_type,
                "version": "v2.0",

                "rmse": best_metrics.get("rmse"),
                "mae": best_metrics.get("mae"),
                "r2": best_metrics.get("r2"),
                "accuracy": best_metrics.get("accuracy"),
                "f1_weighted": best_metrics.get("f1_weighted"),

                "training_date": datetime.utcnow(),
                "training_duration_seconds": training_duration_seconds,
                "status": "active",
                "model_path": model_path,
                "feature_count": len(FEATURE_COLUMNS),
                "features": FEATURE_COLUMNS,
                "data_rows_used": len(df_valid),
                "pipeline_version": "case_b_v2",
                "created_at": datetime.utcnow()
            }
        },
        upsert=True
    )

    print(
        f" BEST MODEL: {best_model_name} | "
        f"Task={task_type} | Score={best_score:.3f}"
    )

print("\n========== TRAINING PIPELINE COMPLETED ==========")