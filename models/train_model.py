# =====================================================
# MULTI-DAY AQI MODEL TRAINING PIPELINE (BEST MODEL ONLY)
# =====================================================

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score
)
from sklearn.exceptions import ConvergenceWarning

from config.settings import settings
from config.constants import FEATURE_COLLECTION, MODEL_COLLECTION

# -----------------------------------------------------
# LOAD ENV
# -----------------------------------------------------
load_dotenv()

# -----------------------------------------------------
# CONNECT DB
# -----------------------------------------------------
client = MongoClient(settings.MONGO_URI)
db = client[settings.MONGO_DB_NAME]

feature_col = db[FEATURE_COLLECTION]
registry_col = db[MODEL_COLLECTION]

print("========== STARTING CASE B TRAINING PIPELINE ==========")

# -----------------------------------------------------
# LOAD FEATURES
# -----------------------------------------------------
data = list(feature_col.find({}, {"_id": 0}))
df = pd.DataFrame(data)
print(f"Loaded data shape: {df.shape}")

# -----------------------------------------------------
# TARGETS
# -----------------------------------------------------
TARGETS = [
    "target_aqi_t_plus_24h",        # regression
    "target_aqi_class_t_plus_24h",  # classification
    "target_aqi_class_t_plus_48h",
    "target_aqi_class_t_plus_72h"
]

# -----------------------------------------------------
# DROP NON-FEATURE COLUMNS AND NON-NUMERIC
# -----------------------------------------------------
DROP_COLS = ["timestamp", "feature_generated_at"] + TARGETS
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(df[c])]
X = df[FEATURE_COLS]

# -----------------------------------------------------
# DEFINE MODELS
# -----------------------------------------------------
REGRESSORS = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=18, min_samples_split=5,
        random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
    ),
    "Ridge": Ridge(alpha=1.0)
}

CLASSIFIERS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=18, min_samples_split=5,
        random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=2000, solver='lbfgs'
    )
}

# -----------------------------------------------------
# SUPPRESS WARNINGS
# -----------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -----------------------------------------------------
# TRAIN LOOP
# -----------------------------------------------------
for target in TARGETS:
    print(f"\n--- Training for {target} ---")
    
    if target not in df.columns:
        print(f"Skipping {target} (not found in dataset)")
        continue

    y = df[target]

    # Detect task type
    if 'class' in target or y.dtype == object:
        task_type = 'classification'
        models = CLASSIFIERS
    else:
        task_type = 'regression'
        models = REGRESSORS

    # Split (time-series aware)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    best_model = None
    best_score = -999
    best_metrics = None
    best_name = None

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if task_type == 'regression':
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            score = r2
            metrics = {"rmse": rmse, "mae": mae, "r2": r2}
            print(f"{model_name} | RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.3f}")
        else:
            accuracy = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            score = f1
            metrics = {"accuracy": accuracy, "f1_weighted": f1}
            print(f"{model_name} | Accuracy={accuracy:.3f} | F1={f1:.3f}")

        # Track best model
        if score > best_score:
            best_score = score
            best_model = model
            best_name = model_name
            best_metrics = metrics

    # -------------------------------------------------
    # SAVE BEST MODEL ONLY
    # -------------------------------------------------
    model_dir = "artifacts/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{target}_{best_name}.joblib")
    joblib.dump(best_model, model_path)

    # -------------------------------------------------
    # SAVE TO MONGO MODEL REGISTRY
    # -------------------------------------------------
    registry_col.insert_one({
        "target": target,
        "model_name": best_name,
        "model_path": model_path,
        "metrics": best_metrics,
        "trained_at": datetime.utcnow(),
        "features": FEATURE_COLS,
        "task_type": task_type
    })

    print(f"  BEST MODEL for {target}: {best_name} | Task={task_type} | Score={best_score:.3f}")

print("\n========== TRAINING PIPELINE COMPLETED ==========")
