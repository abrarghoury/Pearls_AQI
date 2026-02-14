import pandas as pd
import numpy as np
from config.mongo import get_database
from config.constants import RAW_COLLECTION
from config.logging import logger

CLEAN_COLLECTION = "clean_aqi"

def build_clean_dataset():
    logger.info("Building CLEAN dataset from RAW (production-grade)...")

    db = get_database()
    raw_col = db[RAW_COLLECTION]
    clean_col = db[CLEAN_COLLECTION]

    # -----------------------------
    # LOAD RAW
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
    numeric_cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2",
                    "temperature", "humidity", "pressure", "wind_speed",
                    "wind_direction", "precipitation"]

    for col in numeric_cols:
        if col in df.columns:
            # Forward fill then backward fill to handle isolated NaNs
            df[col] = df[col].ffill().bfill()

    # -----------------------------
    # CHECK CONTINUITY (hourly)
    # -----------------------------
    df = df.set_index("timestamp")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="H")
    missing_hours = full_range.difference(df.index)

    if len(missing_hours) > 0:
        logger.warning(f"Missing timestamps detected: {len(missing_hours)} hours")
        # Insert NaN rows for missing hours to maintain continuity
        df = df.reindex(full_range)
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)

    # -----------------------------
    # FINAL CLEANUP
    # -----------------------------
    # Replace remaining NaN / NaT → None for Mongo
    df = df.replace({np.nan: None, pd.NaT: None})

    logger.info(f"CLEAN rows ready: {len(df)}")

    records = df.to_dict("records")

    # -----------------------------
    # WRITE CLEAN COLLECTION
    # -----------------------------
    clean_col.drop()
    if records:
        clean_col.insert_many(records)

    print("\n========== CLEAN DATASET READY ==========")
    print("Rows:", clean_col.count_documents({}))
    sample = clean_col.find_one({}, {"_id": 0})
    print("Columns:", len(sample.keys()))
    print("========================================")

if __name__ == "__main__":
    build_clean_dataset()
