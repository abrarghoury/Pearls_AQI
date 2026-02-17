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
    # LOAD RAW DATA
    # -----------------------------
    data = list(raw_col.find({}, {"_id": 0}))
    if not data:
        raise ValueError("RAW collection is empty")

    df = pd.DataFrame(data)
    logger.info(f"RAW rows loaded: {len(df)}")

    # -----------------------------
    # TIMESTAMP VALIDATION
    # -----------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()]
    df = df.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------
    # DROP DUPLICATES
    # -----------------------------
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    # -----------------------------
    # CRITICAL NUMERIC COLUMNS
    # -----------------------------
    numeric_cols = [
        "pm2_5", "pm10", "no2", "o3", "co", "so2",
        "temperature", "humidity", "pressure", "wind_speed",
        "wind_direction", "precipitation"
    ]

    # Set DatetimeIndex for time interpolation
    df = df.set_index("timestamp")

    for col in numeric_cols:
        if col in df.columns:
            # Time-aware interpolation
            df[col] = df[col].interpolate(method="time")
            # Fill remaining NaNs at start/end
            df[col] = df[col].ffill().bfill()

    # -----------------------------
    # STRICT HOURLY CONTINUITY
    # -----------------------------
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="H")
    missing_hours = full_range.difference(df.index)

    if len(missing_hours) > 0:
        logger.warning(f"Missing timestamps detected: {len(missing_hours)} hours")
        df = df.reindex(full_range)
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method="time").ffill().bfill()

    df = df.reset_index().rename(columns={"index": "timestamp"})

    # -----------------------------
    # FINAL CLEANUP FOR MONGO
    # -----------------------------
    df = df.replace({np.nan: None, pd.NaT: None})

    logger.info(f"CLEAN rows ready: {len(df)}")

    records = df.to_dict("records")

    # -----------------------------
    # SAVE TO CLEAN COLLECTION
    # -----------------------------
    logger.info("Dropping old CLEAN_COLLECTION and inserting fresh data...")
    clean_col.delete_many({})
    if records:
        clean_col.insert_many(records)

    # -----------------------------
    # SUMMARY
    # -----------------------------
    total_rows = clean_col.count_documents({})
    sample = clean_col.find_one({}, {"_id": 0})
    total_cols = len(sample.keys()) if sample else 0

    print("\n========== CLEAN DATASET READY ==========")
    print("Rows:", total_rows)
    print("Columns:", total_cols)
    print("========================================")

    logger.info("========== BUILD CLEAN DATASET COMPLETED ==========")


if __name__ == "__main__":
    build_clean_dataset()
