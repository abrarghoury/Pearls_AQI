# =========================================================
# FEATURE SANITY + CLEANING PIPELINE – TRAINING + INFERENCE
# CASE B – PRODUCTION SAFE VERSION
# =========================================================

import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import UpdateOne

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
    df = df.sort_values("timestamp")
    full_range = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq="h"
    )
    df = df.set_index("timestamp").reindex(full_range)
    df.index.name = "timestamp"

    missing_rows = df.isna().all(axis=1).sum()
    if missing_rows > 0:
        logger.warning(f"{missing_rows} completely missing hourly rows created during reindex")

    return df.reset_index()


def drop_high_missing_columns(df: pd.DataFrame):
    missing_ratio = df.isna().mean()
    drop_cols = missing_ratio[missing_ratio > MAX_MISSING_COL_RATIO].index.tolist()

    protected_cols = ["timestamp", AQI_COL] + ALL_TARGETS
    drop_cols = [c for c in drop_cols if c not in protected_cols]

    if drop_cols:
        logger.warning(f"Dropping high-missing columns: {drop_cols}")

    return df.drop(columns=drop_cols), drop_cols


def smart_fill(df: pd.DataFrame):
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

    drop_cols = [
        c for c in num_cols
        if c not in protected and df[c].nunique(dropna=True) <= 1
    ]

    if drop_cols:
        logger.warning(f"Dropping constant columns: {drop_cols}")

    return df.drop(columns=drop_cols), drop_cols


def validate_targets_exist(df: pd.DataFrame):
    missing = [t for t in ALL_TARGETS if t not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required targets: {missing}. "
            f"Run feature_engineering Case B first."
        )


# =========================================================
# MAIN PIPELINE
# =========================================================

def clean_features_case_b():
    mode = getattr(settings, "PIPELINE_MODE", "training").lower()

    logger.info(f"========== FEATURE CLEANING STARTED | MODE={mode.upper()} ==========")

    db = get_database()
    src_col = db[FEATURE_COLLECTION]
    dst_col = db[CLEANED_FEATURE_COLLECTION]

    # -----------------------------------------------------
    # LOAD FEATURES
    # -----------------------------------------------------
    data = list(src_col.find({}, {"_id": 0}))
    if not data:
        raise ValueError("Feature collection is empty")

    df = pd.DataFrame(data)
    logger.info(f"Loaded feature rows: {len(df)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Remove duplicate timestamps
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    # -----------------------------------------------------
    # TRAINING MODE
    # -----------------------------------------------------
    if mode == "training":
        validate_targets_exist(df)

        df = enforce_hourly_index(df)
        df = df.replace([np.inf, -np.inf], np.nan)

        # Remove rows where targets missing
        df = df.dropna(subset=ALL_TARGETS)

    # -----------------------------------------------------
    # INFERENCE MODE
    # -----------------------------------------------------
    else:
        df = df.sort_values("timestamp").iloc[[-1]]
        df = df.replace([np.inf, -np.inf], np.nan)

    # -----------------------------------------------------
    # Clip AQI + Targets
    # -----------------------------------------------------
    if AQI_COL in df.columns:
        df[AQI_COL] = pd.to_numeric(df[AQI_COL], errors="coerce").clip(0, 500)

    if REGRESSION_TARGET in df.columns:
        df[REGRESSION_TARGET] = pd.to_numeric(
            df[REGRESSION_TARGET], errors="coerce"
        ).clip(0, 500)

    for t in CLASS_TARGETS:
        if t in df.columns:
            df[t] = pd.to_numeric(df[t], errors="coerce").clip(1, 5)

    # -----------------------------------------------------
    # Drop Sparse Columns
    # -----------------------------------------------------
    df, dropped_missing = drop_high_missing_columns(df)

    # -----------------------------------------------------
    # Fill Remaining NaNs
    # -----------------------------------------------------
    df = smart_fill(df)

    # -----------------------------------------------------
    # Remove Constant Columns
    # -----------------------------------------------------
    df, dropped_constant = remove_constant_features(df)

    # -----------------------------------------------------
    # Final Validation
    # -----------------------------------------------------
    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"NaNs still present after cleaning. Bad cols: {bad_cols[:20]}")

    df["feature_generated_at"] = datetime.utcnow()
    df["rows_after_cleaning"] = int(len(df))

    df = df.replace({np.nan: None})

    # -----------------------------------------------------
    # BULK UPSERT (FASTER)
    # -----------------------------------------------------
    records = df.to_dict("records")

    operations = [
        UpdateOne(
            {"timestamp": rec["timestamp"]},
            {"$set": rec},
            upsert=True
        )
        for rec in records
    ]

    if operations:
        result = dst_col.bulk_write(operations)
        logger.info(
            f"Bulk upsert complete | "
            f"Inserted: {result.upserted_count} | "
            f"Modified: {result.modified_count}"
        )

    logger.info(
        f"========== FEATURE CLEANING COMPLETE | "
        f"ROWS={len(df)} | MODE={mode.upper()} =========="
    )

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
