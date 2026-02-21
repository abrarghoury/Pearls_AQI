import pandas as pd
import numpy as np
from config.mongo import get_database
from config.constants import RAW_COLLECTION, CLEAN_COLLECTION
from config.logging import logger

def build_clean_dataset():
    logger.info("========== BUILD CLEAN DATASET STARTED ==========")

    db = get_database()
    raw_col = db[RAW_COLLECTION]
    clean_col = db[CLEAN_COLLECTION]

    # -----------------------------
    # LOAD RAW DATA (memory safe)
    # -----------------------------
    cursor = raw_col.find({}, {"_id": 0})
    df = pd.DataFrame(list(cursor))
    if df.empty:
        raise ValueError("RAW collection is empty")
    logger.info(f"RAW rows loaded: {len(df)}")

    # -----------------------------
    # TIMESTAMP VALIDATION + TIMEZONE AWARE
    # -----------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df[df["timestamp"].notna()]
    df = df.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------
    # DROP DUPLICATES
    # -----------------------------
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    # -----------------------------
    # NUMERIC COLUMNS
    # -----------------------------
    numeric_cols = [
        "pm2_5", "pm10", "no2", "o3", "co", "so2",
        "temperature", "humidity", "pressure", "wind_speed",
        "wind_direction", "precipitation"
    ]

    df = df.set_index("timestamp")

    # -----------------------------
    # CIRCULAR INTERPOLATION for wind_direction
    # -----------------------------
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if col == "wind_direction":
            # Convert to radians
            angles = np.deg2rad(df[col].values.astype(float))
            sin_vals = np.sin(angles)
            cos_vals = np.cos(angles)
            sin_vals = pd.Series(sin_vals, index=df.index).interpolate(method="time").ffill().bfill()
            cos_vals = pd.Series(cos_vals, index=df.index).interpolate(method="time").ffill().bfill()
            df[col] = np.rad2deg(np.arctan2(sin_vals, cos_vals)) % 360
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(method="time").ffill().bfill()

    # -----------------------------
    # STRICT HOURLY CONTINUITY
    # -----------------------------
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="H", tz='UTC')
    missing_hours = full_range.difference(df.index)
    if len(missing_hours) > 0:
        logger.warning(f"Missing timestamps detected: {len(missing_hours)} hours")
        df = df.reindex(full_range)
        for col in numeric_cols:
            if col in df.columns:
                if col == "wind_direction":
                    angles = np.deg2rad(df[col].values.astype(float))
                    sin_vals = np.sin(angles)
                    cos_vals = np.cos(angles)
                    sin_vals = pd.Series(sin_vals, index=df.index).interpolate(method="time").ffill().bfill()
                    cos_vals = pd.Series(cos_vals, index=df.index).interpolate(method="time").ffill().bfill()
                    df[col] = np.rad2deg(np.arctan2(sin_vals, cos_vals)) % 360
                else:
                    df[col] = df[col].interpolate(method="time").ffill().bfill()

    df = df.reset_index().rename(columns={"index": "timestamp"})

    # -----------------------------
    # REPLACE NaN / NaT with None for Mongo
    # -----------------------------
    df = df.replace({np.nan: None, pd.NaT: None})

    # -----------------------------
    # SAVE TO TEMP COLLECTION AND SWAP
    # -----------------------------
    temp_collection = CLEAN_COLLECTION + "_tmp"
    temp_col = db[temp_collection]

    temp_col.delete_many({})
    records = df.to_dict("records")
    if records:
        temp_col.insert_many(records)

    # Atomic swap: drop old clean collection and rename tmp
    clean_col.drop()
    db[temp_collection].rename(CLEAN_COLLECTION)

    # -----------------------------
    # SUMMARY
    # -----------------------------
    total_rows = db[CLEAN_COLLECTION].count_documents({})
    sample = db[CLEAN_COLLECTION].find_one({}, {"_id": 0})
    total_cols = len(sample.keys()) if sample else 0

    print("\n========== CLEAN DATASET READY ==========")
    print("Rows:", total_rows)
    print("Columns:", total_cols)
    print("========================================")

    logger.info("========== BUILD CLEAN DATASET COMPLETED ==========")


if __name__ == "__main__":
    build_clean_dataset()
