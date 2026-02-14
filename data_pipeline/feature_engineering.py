# =========================================================
# FEATURE ENGINEERING PIPELINE (CASE B HYBRID)
# ---------------------------------------------------------
# Features: based ONLY on time t and past history
# Targets:
#   - Regression: AQI numeric at t+24h
#   - Classification: AQI class at t+24h, t+48h, t+72h
# ---------------------------------------------------------
# 
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

# =========================================================
# AQI BREAKPOINT TABLES (US EPA STANDARD)
# =========================================================
AQI_BREAKPOINTS = {
    "pm2_5": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
              (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
             (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
    "no2": [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
            (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
            (186, 304, 151, 200), (305, 604, 201, 300), (605, 1004, 301, 500)],
    "o3": [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
           (86, 105, 151, 200), (106, 200, 201, 300)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
           (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500)]
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
    """
    Numeric AQI computed at CURRENT TIME (t).
    Uses max sub-index across pollutants (US EPA).
    """
    sub_indices = []

    for p in POLLUTANT_FEATURES:
        idx = calculate_sub_index(row.get(p), p)
        if idx is not None:
            sub_indices.append(idx)

    return max(sub_indices) if sub_indices else None


def aqi_numeric_to_class(aqi_value: float):
    """
    Convert numeric AQI -> class (1-5)
    Class mapping:
        1 = Good (0-50)
        2 = Moderate (51-100)
        3 = Unhealthy for Sensitive (101-150)
        4 = Unhealthy (151-200)
        5 = Very Unhealthy / Hazardous (201+)
    """
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
# MAIN FEATURE PIPELINE
# =========================================================
def run_feature_pipeline_case_b():
    logger.info("========== FEATURE PIPELINE STARTED (CASE B HYBRID) ==========")

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
    # STRICT TIME CLEANING
    # -----------------------------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.reset_index(drop=True)

    # -----------------------------------------------------
    # NUMERIC CONVERSION
    # -----------------------------------------------------
    for col in POLLUTANT_FEATURES + WEATHER_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------------------------------
    # CURRENT AQI NUMERIC (time t)
    # -----------------------------------------------------
    df[TARGET_COLUMN] = df.apply(calculate_aqi_numeric, axis=1)
    df = df.dropna(subset=[TARGET_COLUMN])

    # -----------------------------------------------------
    # TIME FEATURES (SAFE)
    # -----------------------------------------------------
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical time encoding (helps ML)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # -----------------------------------------------------
    # LAG FEATURES (PAST ONLY)
    # -----------------------------------------------------
    # AQI lags (very important for regression + class)
    for lag in [1, 3, 6, 12, 24, 48, 72]:
        df[f"aqi_lag_{lag}h"] = df[TARGET_COLUMN].shift(lag)

    # Pollutant lags (strongest features)
    pollutant_lags = [1, 3, 6, 12, 24]
    for p in POLLUTANT_FEATURES:
        for lag in pollutant_lags:
            df[f"{p}_lag_{lag}h"] = df[p].shift(lag)

    # Weather lags (light but helpful)
    weather_lags = [1, 24]
    for w in WEATHER_FEATURES:
        if w not in df.columns:
            continue
        for lag in weather_lags:
            df[f"{w}_lag_{lag}h"] = df[w].shift(lag)

    # -----------------------------------------------------
    # ROLLING FEATURES (PAST WINDOW ONLY)
    # -----------------------------------------------------
    # AQI rolling
    df["aqi_roll_mean_6h"] = df[TARGET_COLUMN].rolling(6, min_periods=3).mean()
    df["aqi_roll_mean_12h"] = df[TARGET_COLUMN].rolling(12, min_periods=6).mean()
    df["aqi_roll_mean_24h"] = df[TARGET_COLUMN].rolling(24, min_periods=12).mean()

    df["aqi_roll_std_12h"] = df[TARGET_COLUMN].rolling(12, min_periods=6).std()
    df["aqi_roll_std_24h"] = df[TARGET_COLUMN].rolling(24, min_periods=12).std()

    # Pollutant rolling means (best signal)
    for p in ["pm2_5", "pm10"]:
        if p not in df.columns:
            continue

        df[f"{p}_roll_mean_3h"] = df[p].rolling(3, min_periods=2).mean()
        df[f"{p}_roll_mean_6h"] = df[p].rolling(6, min_periods=3).mean()
        df[f"{p}_roll_mean_12h"] = df[p].rolling(12, min_periods=6).mean()
        df[f"{p}_roll_mean_24h"] = df[p].rolling(24, min_periods=12).mean()

        df[f"{p}_roll_std_12h"] = df[p].rolling(12, min_periods=6).std()
        df[f"{p}_roll_std_24h"] = df[p].rolling(24, min_periods=12).std()

    # -----------------------------------------------------
    # TREND / CHANGE RATE FEATURES
    # -----------------------------------------------------
    df["aqi_delta_1h"] = df[TARGET_COLUMN] - df["aqi_lag_1h"]
    df["aqi_delta_3h"] = df[TARGET_COLUMN] - df["aqi_lag_3h"]
    df["aqi_delta_6h"] = df[TARGET_COLUMN] - df["aqi_lag_6h"]
    df["aqi_delta_24h"] = df[TARGET_COLUMN] - df["aqi_lag_24h"]

    # pollutant deltas
    if "pm2_5" in df.columns:
        df["pm2_5_delta_1h"] = df["pm2_5"] - df["pm2_5_lag_1h"]
        df["pm2_5_delta_3h"] = df["pm2_5"] - df["pm2_5_lag_3h"]
        df["pm2_5_delta_6h"] = df["pm2_5"] - df["pm2_5_lag_6h"]

    if "pm10" in df.columns:
        df["pm10_delta_1h"] = df["pm10"] - df["pm10_lag_1h"]
        df["pm10_delta_3h"] = df["pm10"] - df["pm10_lag_3h"]

    # -----------------------------------------------------
    # DERIVED RATIOS (SAFE)
    # -----------------------------------------------------
    if "pm2_5" in df.columns and "pm10" in df.columns:
        df["pm2_5_pm10_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)

    if "no2" in df.columns and "o3" in df.columns:
        df["no2_o3_ratio"] = df["no2"] / (df["o3"] + 1e-6)

    # -----------------------------------------------------
    # TARGETS (FUTURE ONLY) - CASE B
    # -----------------------------------------------------
    # Regression target (only day 1)
    df["target_aqi_t_plus_24h"] = df[TARGET_COLUMN].shift(-24)

    # Classification targets (3 days)
    df["target_aqi_class_t_plus_24h"] = df[TARGET_COLUMN].shift(-24).apply(aqi_numeric_to_class)
    df["target_aqi_class_t_plus_48h"] = df[TARGET_COLUMN].shift(-48).apply(aqi_numeric_to_class)
    df["target_aqi_class_t_plus_72h"] = df[TARGET_COLUMN].shift(-72).apply(aqi_numeric_to_class)

    # -----------------------------------------------------
    # DROP ROWS THAT WOULD CAUSE LEAKAGE / NA
    # -----------------------------------------------------
    # We must drop rows where:
    #   - lag features missing (start)
    #   - future targets missing (end)
    df = df.dropna()

    # -----------------------------------------------------
    # FINAL CLEANUP FOR MONGO
    # -----------------------------------------------------
    df["timestamp"] = df["timestamp"].apply(lambda x: x.to_pydatetime())
    df["feature_generated_at"] = datetime.utcnow()

    df = df.replace({np.nan: None, pd.NaT: None})

    # -----------------------------------------------------
    # SAVE TO FEATURE STORE (MongoDB)
    # -----------------------------------------------------
    logger.info("Dropping old FEATURE_COLLECTION and inserting fresh dataset...")

    feature_col.drop()
    feature_col.insert_many(df.to_dict("records"))

    logger.info(
        f"========== FEATURE PIPELINE COMPLETED (CASE B) | "
        f"ROWS={len(df)} | COLS={len(df.columns)} =========="
    )

    # quick summary
    sample = feature_col.find_one({}, {"_id": 0})
    if sample:
        logger.info(f"Sample feature keys: {list(sample.keys())[:20]} ...")


if __name__ == "__main__":
    run_feature_pipeline_case_b()
