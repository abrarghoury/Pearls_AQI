# =========================================================
# FEATURE SANITY + CLEANING PIPELINE 
# ---------------------------------------------------------
# Loads:   FEATURE_COLLECTION  (aqi_features)
# Writes:  CLEANED_FEATURE_COLLECTION
# ---------------------------------------------------------
# Cleans:
# - strict hourly continuity
# - drops bad columns
# - fills missing values safely (time-series aware)
# - removes constant features
# - validates targets exist
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
    """
    Reindex to strict hourly timeline.
    This is safer than dropping missing timestamps.
    """
    full_range = pd.date_range(
        df["timestamp"].min(),
        df["timestamp"].max(),
        freq="h"
    )

    df = df.set_index("timestamp").reindex(full_range)
    df.index.name = "timestamp"

    missing_rows = df.isna().all(axis=1).sum()
    if missing_rows > 0:
        logger.warning(f"{missing_rows} completely missing hours detected after reindex")

    return df.reset_index()


def drop_high_missing_columns(df: pd.DataFrame):
    missing_ratio = df.isna().mean()
    drop_cols = missing_ratio[missing_ratio > MAX_MISSING_COL_RATIO].index.tolist()

    # Never drop timestamp
    if "timestamp" in drop_cols:
        drop_cols.remove("timestamp")

    # Never drop targets even if missing
    for t in ALL_TARGETS:
        if t in drop_cols:
            drop_cols.remove(t)

    if drop_cols:
        logger.warning(f"Dropping high-missing columns: {drop_cols}")

    return df.drop(columns=drop_cols), drop_cols


def smart_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time-series safe filling:
    - forward fill
    - backward fill
    - median fallback
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if num_cols:
        df[num_cols] = df[num_cols].ffill()
        df[num_cols] = df[num_cols].bfill()

        for col in num_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    return df


def remove_constant_features(df: pd.DataFrame):
    """
    Remove numeric columns with 0 variance.
    These hurt ML training.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # never drop targets
    protected = set(ALL_TARGETS + [AQI_COL])

    drop_cols = []
    for c in num_cols:
        if c in protected:
            continue
        if df[c].nunique(dropna=True) <= 1:
            drop_cols.append(c)

    if drop_cols:
        logger.warning(f"Dropping constant columns: {drop_cols}")

    return df.drop(columns=drop_cols), drop_cols


def validate_targets_exist(df: pd.DataFrame):
    missing = [t for t in ALL_TARGETS if t not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required targets in FEATURE_COLLECTION: {missing}. "
            f"Run feature_engineering again (Case B)."
        )


# =========================================================
# MAIN PIPELINE
# =========================================================
def clean_features_for_training_case_b():
    logger.info("========== FEATURE CLEANING STARTED (CASE B) ==========")

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

    # -----------------------------------------------------
    # TIMESTAMP
    # -----------------------------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------------------------------
    # TARGET VALIDATION
    # -----------------------------------------------------
    validate_targets_exist(df)

    # -----------------------------------------------------
    # STRICT hourly continuity
    # -----------------------------------------------------
    df = enforce_hourly_index(df)

    # -----------------------------------------------------
    # Replace inf
    # -----------------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)

    # -----------------------------------------------------
    # Drop rows where targets missing
    # (CRITICAL)
    # -----------------------------------------------------
    df = df.dropna(subset=ALL_TARGETS)
    logger.info(f"Rows after target filtering: {len(df)}")

    # -----------------------------------------------------
    # Clip AQI numeric range (EPA)
    # -----------------------------------------------------
    if AQI_COL in df.columns:
        df[AQI_COL] = pd.to_numeric(df[AQI_COL], errors="coerce").clip(0, 500)

    if REGRESSION_TARGET in df.columns:
        df[REGRESSION_TARGET] = pd.to_numeric(df[REGRESSION_TARGET], errors="coerce").clip(0, 500)

    # classification targets must be ints 1..5
    for t in CLASS_TARGETS:
        if t in df.columns:
            df[t] = pd.to_numeric(df[t], errors="coerce").clip(1, 5)

    # -----------------------------------------------------
    # Drop very sparse columns
    # -----------------------------------------------------
    df, dropped_missing = drop_high_missing_columns(df)

    # -----------------------------------------------------
    # SMART FILL
    # -----------------------------------------------------
    df = smart_fill(df)

    # -----------------------------------------------------
    # Remove constant junk features
    # -----------------------------------------------------
    df, dropped_constant = remove_constant_features(df)

    # -----------------------------------------------------
    # FINAL CHECK
    # -----------------------------------------------------
    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"NaNs still present after cleaning. Bad cols: {bad_cols[:30]}")

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------
    df["validation_done_at"] = datetime.utcnow()
    df["rows_after_cleaning"] = int(len(df))

    df = df.replace({np.nan: None})

    logger.info("Dropping old CLEANED_FEATURE_COLLECTION and inserting fresh dataset...")
    dst_col.drop()
    dst_col.insert_many(df.to_dict("records"))

    logger.info("========== FEATURE CLEANING COMPLETE (CASE B) ==========")
    logger.info(f"Final rows: {len(df)}")
    logger.info(f"Final columns: {len(df.columns)}")

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
    summary = clean_features_for_training_case_b()

    print("\n===== FEATURE CLEANING SUMMARY (CASE B) =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("============================================")
