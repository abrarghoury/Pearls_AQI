# =========================================================
# FEATURE ENGINEERING PIPELINE (PRODUCTION SAFE)
# =========================================================

import pandas as pd
import numpy as np
from datetime import datetime

from config.mongo import get_database
from config.logging import logger
from config.constants import (
    CLEAN_COLLECTION,
    FEATURE_COLLECTION,
    POLLUTANT_FEATURES,
    WEATHER_FEATURES,
    TARGET_COLUMN
)
from config.settings import settings


# =========================================================
# AQI BREAKPOINT TABLES (US EPA STANDARD)
# =========================================================
AQI_BREAKPOINTS = {
    "pm2_5": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100),
              (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200),
              (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100),
             (155, 254, 101, 150), (255, 354, 151, 200),
             (355, 424, 201, 300), (425, 604, 301, 500)],
    "no2": [(0, 53, 0, 50), (54, 100, 51, 100),
            (101, 360, 101, 150), (361, 649, 151, 200),
            (650, 1249, 201, 300), (1250, 2049, 301, 500)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100),
            (76, 185, 101, 150), (186, 304, 151, 200),
            (305, 604, 201, 300), (605, 1004, 301, 500)],
    "o3": [(0, 54, 0, 50), (55, 70, 51, 100),
           (71, 85, 101, 150), (86, 105, 151, 200),
           (106, 200, 201, 300)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100),
           (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200),
           (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500)]
}


# =========================================================
# AQI CALCULATION
# =========================================================
def calculate_sub_index(conc, pollutant):
    if conc is None or pd.isna(conc):
        return None
    for bp_l, bp_h, i_l, i_h in AQI_BREAKPOINTS[pollutant]:
        if bp_l <= conc <= bp_h:
            return ((i_h - i_l) / (bp_h - bp_l)) * (conc - bp_l) + i_l
    return None


def calculate_aqi_numeric(row):
    sub_indices = []
    for p in POLLUTANT_FEATURES:
        idx = calculate_sub_index(row.get(p), p)
        if idx is not None:
            sub_indices.append(idx)
    return max(sub_indices) if sub_indices else None


def aqi_numeric_to_class(aqi_value):
    if aqi_value is None or pd.isna(aqi_value):
        return None
    if aqi_value <= 50:
        return 1
    elif aqi_value <= 100:
        return 2
    elif aqi_value <= 150:
        return 3
    elif aqi_value <= 200:
        return 4
    else:
        return 5


# =========================================================
# FEATURE PIPELINE
# =========================================================
def run_feature_pipeline():
    logger.info("========== FEATURE PIPELINE STARTED ==========")

    db = get_database()
    clean_col = db[CLEAN_COLLECTION]
    feature_col = db[FEATURE_COLLECTION]

    # -----------------------------------------------------
    # LOAD CLEAN DATA
    # -----------------------------------------------------
    data = list(clean_col.find({}, {"_id": 0}))
    if not data:
        raise ValueError("CLEAN collection is empty")

    df = pd.DataFrame(data)
    logger.info(f"CLEAN rows loaded: {len(df)}")

    # -----------------------------------------------------
    # STRICT TIME HANDLING
    # -----------------------------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.reset_index(drop=True)
    df = df.set_index("timestamp")

    # -----------------------------------------------------
    # SAFE NUMERIC CONVERSION
    # -----------------------------------------------------
    for col in POLLUTANT_FEATURES + WEATHER_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(method="time").ffill().bfill()

    # -----------------------------------------------------
    # CURRENT AQI NUMERIC
    # -----------------------------------------------------
    df[TARGET_COLUMN] = df.apply(calculate_aqi_numeric, axis=1)
    df = df.dropna(subset=[TARGET_COLUMN])

    # -----------------------------------------------------
    # TIME FEATURES
    # -----------------------------------------------------
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # -----------------------------------------------------
    # LAG FEATURES
    # -----------------------------------------------------
    for lag in [1, 3, 6, 12, 24, 48, 72]:
        df[f"aqi_lag_{lag}h"] = df[TARGET_COLUMN].shift(lag)

    for p in POLLUTANT_FEATURES:
        if p in df.columns:
            for lag in [1, 3, 6, 12, 24]:
                df[f"{p}_lag_{lag}h"] = df[p].shift(lag)

    for w in WEATHER_FEATURES:
        if w in df.columns:
            for lag in [1, 24]:
                df[f"{w}_lag_{lag}h"] = df[w].shift(lag)

    # -----------------------------------------------------
    # ROLLING FEATURES
    # -----------------------------------------------------
    df["aqi_roll_mean_6h"] = df[TARGET_COLUMN].rolling(6).mean()
    df["aqi_roll_mean_12h"] = df[TARGET_COLUMN].rolling(12).mean()
    df["aqi_roll_mean_24h"] = df[TARGET_COLUMN].rolling(24).mean()
    df["aqi_roll_std_12h"] = df[TARGET_COLUMN].rolling(12).std()
    df["aqi_roll_std_24h"] = df[TARGET_COLUMN].rolling(24).std()

    # -----------------------------------------------------
    # DELTA FEATURES
    # -----------------------------------------------------
    for delta in [1, 3, 6, 24]:
        df[f"aqi_delta_{delta}h"] = df[TARGET_COLUMN] - df[TARGET_COLUMN].shift(delta)

    # -----------------------------------------------------
    # RATIOS
    # -----------------------------------------------------
    if "pm2_5" in df.columns and "pm10" in df.columns:
        df["pm2_5_pm10_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)

    if "no2" in df.columns and "o3" in df.columns:
        df["no2_o3_ratio"] = df["no2"] / (df["o3"] + 1e-6)

    # =====================================================
    # TARGET SHIFT (SAFE)
    # =====================================================
    mode = getattr(settings, "PIPELINE_MODE", "training").lower()

    if mode == "training":
        # Generate targets only if enough rows exist
        n_rows = len(df)
        max_shift = 72
        if n_rows > max_shift:
            df["target_aqi_t_plus_24h"] = df[TARGET_COLUMN].shift(-24)
            df["target_aqi_class_t_plus_24h"] = df["target_aqi_t_plus_24h"].apply(aqi_numeric_to_class)
            df["target_aqi_class_t_plus_48h"] = df[TARGET_COLUMN].shift(-48).apply(aqi_numeric_to_class)
            df["target_aqi_class_t_plus_72h"] = df[TARGET_COLUMN].shift(-72).apply(aqi_numeric_to_class)

            # Drop rows where targets missing
            df = df.dropna(subset=[
                "target_aqi_t_plus_24h",
                "target_aqi_class_t_plus_24h",
                "target_aqi_class_t_plus_48h",
                "target_aqi_class_t_plus_72h"
            ])
        else:
            logger.warning("Not enough rows to generate 24/48/72h targets. Skipping target shift.")
    else:
        # Inference → last row only, no drop
        df = df.iloc[[-1]]

    # -----------------------------------------------------
    # FINAL CLEANUP
    # -----------------------------------------------------
    df = df.reset_index()
    df["timestamp"] = df["timestamp"].apply(lambda x: x.to_pydatetime())
    df["feature_generated_at"] = datetime.utcnow()
    df = df.replace({np.nan: None, pd.NaT: None})

    # -----------------------------------------------------
    # UPSERT (SAFE FOR HOURLY RUN)
    # -----------------------------------------------------
    records = df.to_dict("records")

    for rec in records:
        feature_col.update_one(
            {"timestamp": rec["timestamp"]},
            {"$set": rec},
            upsert=True
        )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------
    total_cols = df.shape[1]
    column_names = df.columns.tolist()

    logger.info(
        f"========== FEATURE PIPELINE COMPLETED | MODE={mode} | ROWS={len(df)} | COLUMNS={total_cols} =========="
    )

    print("\n===== FEATURE PIPELINE SUMMARY =====")
    print(f"Rows processed: {len(df)}")
    print(f"Columns count: {total_cols}")
    print("Column names:", column_names)
    print("=====================================")


if __name__ == "__main__":
    run_feature_pipeline()