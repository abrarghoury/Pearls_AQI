# =========================================================
# FEATURE SANITY + CLEANING PIPELINE – TRAINING + INFERENCE
# ---------------------------------------------------------
# Loads:   FEATURE_COLLECTION  (aqi_features)
# Writes:  CLEANED_FEATURE_COLLECTION
# ---------------------------------------------------------
# Cleans:
# - strict hourly continuity (training only)
# - drops bad columns
# - fills missing values safely (time-series aware)
# - removes constant features
# - validates targets exist (training only)
# - inference mode: keep latest row for dashboard
# =========================================================

import numpy as np
import pandas as pd
from datetime import datetime

from config.mongo import get_database
from config.logging import logger
from config.constants import (
    FEATURE_COLLECTION,
    CLEANED_FEATURE_COLLECTION,
    TARGET_COLUMN
)
from config.settings import settings

# =========================================================
# CONFIG
# =========================================================
MAX_MISSING_COL_RATIO = 0.35

AQI_COL = TARGET_COLUMN

# CASE B TARGETS
REGRESSION_TARGET = "target_aqi_t_plus_24h"

CLASS_TARGETS = [
    "target_aqi_class_t_plus_24h",
    "target_aqi_class_t_plus_48h",
    "target_aqi_class_t_plus_72h"
]

ALL_TARGETS = [REGRESSION_TARGET] + CLASS_TARGETS

# =========================================================
# HELPERS
# =========================================================
def enforce_hourly_index(df: pd.DataFrame) -> pd.DataFrame:
    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(full_range)
    df.index.name = "timestamp"

    missing_rows = df.isna().all(axis=1).sum()
    if missing_rows > 0:
        logger.warning(f"{missing_rows} completely missing hours detected after reindex")

    return df.reset_index()


def drop_high_missing_columns(df: pd.DataFrame):
    missing_ratio = df.isna().mean()
    drop_cols = missing_ratio[missing_ratio > MAX_MISSING_COL_RATIO].index.tolist()
    for protected in ["timestamp"] + ALL_TARGETS:
        if protected in drop_cols:
            drop_cols.remove(protected)
    if drop_cols:
        logger.warning(f"Dropping high-missing columns: {drop_cols}")
    return df.drop(columns=drop_cols), drop_cols


def smart_fill(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].ffill().bfill()
        for col in num_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
    return df


def remove_constant_features(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    protected = set(ALL_TARGETS + [AQI_COL])
    drop_cols = [c for c in num_cols if c not in protected and df[c].nunique(dropna=True) <= 1]
    if drop_cols:
        logger.warning(f"Dropping constant columns: {drop_cols}")
    return df.drop(columns=drop_cols), drop_cols


def validate_targets_exist(df: pd.DataFrame):
    missing = [t for t in ALL_TARGETS if t not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required targets in FEATURE_COLLECTION: {missing}. Run feature_engineering again (Case B)."
        )


# =========================================================
# MAIN PIPELINE
# =========================================================
def clean_features_case_b():
    mode = getattr(settings, "PIPELINE_MODE", "training")
    logger.info(f"========== FEATURE CLEANING STARTED (CASE B) | MODE={mode} ==========")

    db = get_database()
    src_col = db[FEATURE_COLLECTION]
    dst_col = db[CLEANED_FEATURE_COLLECTION]

    # -----------------------------------------------------
    # LOAD FEATURES
    # -----------------------------------------------------
    data = list(src_col.find({}, {"_id": 0}))
    if not data:
        raise ValueError("Feature collection empty")

    df = pd.DataFrame(data)
    logger.info(f"Feature rows loaded: {len(df)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # -----------------------------------------------------
    # TRAINING MODE
    # -----------------------------------------------------
    if mode == "training":
        validate_targets_exist(df)
        df = enforce_hourly_index(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=ALL_TARGETS)
    else:
        # -------------------------------------------------
        # INFERENCE MODE – keep only latest row
        # -------------------------------------------------
        df = df.sort_values("timestamp").iloc[[-1]]
        df = df.replace([np.inf, -np.inf], np.nan)

    # -----------------------------------------------------
    # Clip AQI / targets
    # -----------------------------------------------------
    if AQI_COL in df.columns:
        df[AQI_COL] = pd.to_numeric(df[AQI_COL], errors="coerce").clip(0, 500)

    if REGRESSION_TARGET in df.columns:
        df[REGRESSION_TARGET] = pd.to_numeric(df[REGRESSION_TARGET], errors="coerce").clip(0, 500)

    for t in CLASS_TARGETS:
        if t in df.columns:
            df[t] = pd.to_numeric(df[t], errors="coerce").clip(1, 5)

    # -----------------------------------------------------
    # Drop sparse columns
    # -----------------------------------------------------
    df, dropped_missing = drop_high_missing_columns(df)

    # -----------------------------------------------------
    # Smart fill remaining NaNs
    # -----------------------------------------------------
    df = smart_fill(df)

    # -----------------------------------------------------
    # Remove constant features
    # -----------------------------------------------------
    df, dropped_constant = remove_constant_features(df)

    # -----------------------------------------------------
    # FINAL CHECK
    # -----------------------------------------------------
    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"NaNs still present after cleaning. Bad cols: {bad_cols[:30]}")

    df["validation_done_at"] = datetime.utcnow()
    df["rows_after_cleaning"] = int(len(df))
    df = df.replace({np.nan: None})

    logger.info("Dropping old CLEANED_FEATURE_COLLECTION and inserting fresh dataset...")
    dst_col.drop()
    dst_col.insert_many(df.to_dict("records"))

    logger.info(f"========== FEATURE CLEANING COMPLETE | ROWS={len(df)} | MODE={mode} ==========")
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "dropped_high_missing": dropped_missing,
        "dropped_constant": dropped_constant
    }


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    summary = clean_features_case_b()
    print("\n===== FEATURE CLEANING SUMMARY (CASE B) =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("============================================")
