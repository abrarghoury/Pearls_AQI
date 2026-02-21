import requests
import pandas as pd

from datetime import datetime, timedelta
from time import sleep

from config.settings import settings
from config.logging import logger
from config.constants import HISTORICAL_MONTHS, RAW_COLLECTION

from config.mongo import get_database
from pymongo import UpdateOne


# =========================================================
# AIR POLLUTION
# =========================================================

def fetch_air_pollution(start_ts: int, end_ts: int, retries: int = 3, delay: int = 5):
    logger.info("Fetching Air Pollution historical data...")

    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": settings.LAT,
        "lon": settings.LON,
        "start": start_ts,
        "end": end_ts,
        "appid": settings.OPENWEATHER_API_KEY
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                sleep(delay)
            else:
                raise

    data = response.json().get("list", [])
    if not data:
        raise ValueError("No AQI data returned")

    records = []
    for item in data:
        ts = pd.to_datetime(item["dt"], unit="s").floor("H")
        comp = item.get("components", {})
        records.append({
            "timestamp": ts,
            "pm2_5": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "no2": comp.get("no2"),
            "o3": comp.get("o3"),
            "co": comp.get("co"),
            "so2": comp.get("so2"),
        })

    df = pd.DataFrame(records)
    before_dup = len(df)
    df.drop_duplicates(subset="timestamp", inplace=True)
    after_dup = len(df)
    if before_dup - after_dup > 0:
        logger.info(f"Removed {before_dup - after_dup} duplicate AQI rows")
    df.sort_values("timestamp", inplace=True)

    logger.info(f"AQI rows fetched: {len(df)}")
    return df


# =========================================================
# WEATHER
# =========================================================

def fetch_weather(start_date: str, end_date: str, retries: int = 3, delay: int = 5):
    logger.info("Fetching Weather historical data...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": settings.LAT,
        "longitude": settings.LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation"
        ],
        "timezone": "UTC"
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                sleep(delay)
            else:
                raise

    hourly = response.json().get("hourly", {})
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly.get("time")),
        "temperature": hourly.get("temperature_2m"),
        "humidity": hourly.get("relative_humidity_2m"),
        "pressure": hourly.get("pressure_msl"),
        "wind_speed": hourly.get("wind_speed_10m"),
        "wind_direction": hourly.get("wind_direction_10m"),
        "precipitation": hourly.get("precipitation"),
    })

    before_dup = len(df)
    df.drop_duplicates(subset="timestamp", inplace=True)
    after_dup = len(df)
    if before_dup - after_dup > 0:
        logger.info(f"Removed {before_dup - after_dup} duplicate Weather rows")
    df.sort_values("timestamp", inplace=True)

    logger.info(f"Weather rows fetched: {len(df)}")
    return df


# =========================================================
# TIME ALIGNMENT (VERY IMPORTANT)
# =========================================================

def align_and_merge(aqi_df, weather_df):
    logger.info("Aligning timestamps on full hourly grid...")

    start = min(aqi_df.timestamp.min(), weather_df.timestamp.min())
    end = max(aqi_df.timestamp.max(), weather_df.timestamp.max())
    full_range = pd.date_range(start=start, end=end, freq="H")

    # Reindex
    aqi_df = aqi_df.set_index("timestamp").reindex(full_range)
    weather_df = weather_df.set_index("timestamp").reindex(full_range)

    # Interpolate AQI numeric columns
    numeric_cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2"]
    for col in numeric_cols:
        if col in aqi_df.columns:
            aqi_df[col] = aqi_df[col].interpolate(method="time").ffill().bfill()

    # Drop rows with missing pm2_5
    before_drop = len(aqi_df)
    aqi_df = aqi_df.dropna(subset=["pm2_5"])
    after_drop = len(aqi_df)
    logger.info(f"Dropped empty AQI rows: {before_drop - after_drop}")

    # Forward fill weather (limit 3 hours)
    weather_df = weather_df.ffill(limit=3)
    # Interpolate remaining numeric weather columns
    weather_numeric_cols = ["temperature", "humidity", "pressure", "wind_speed", "wind_direction", "precipitation"]
    for col in weather_numeric_cols:
        if col in weather_df.columns:
            weather_df[col] = weather_df[col].interpolate(method="time").ffill().bfill()

    df = aqi_df.join(weather_df, how="left")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)

    logger.info(f"Final merged rows: {len(df)}")
    return df


# =========================================================
# BACKFILL
# =========================================================

def run_backfill():
    logger.info("Starting AQI historical backfill pipeline...")

    db = get_database()
    collection = db[RAW_COLLECTION]

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=HISTORICAL_MONTHS * 30)

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    start_date = start_dt.date().isoformat()
    end_date = end_dt.date().isoformat()
    logger.info(f"Backfill window: {start_date} -> {end_date}")

    # Fetch
    aqi_df = fetch_air_pollution(start_ts, end_ts)
    weather_df = fetch_weather(start_date, end_date)

    # ALIGN
    df = align_and_merge(aqi_df, weather_df)

    # Prepare Mongo Records
    records = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        rec["city"] = settings.CITY
        rec["source"] = "openweather + openmeteo"
        rec["ingested_at"] = datetime.utcnow()
        records.append(rec)

    # Bulk Upsert
    operations = []
    for r in records:
        operations.append(
            UpdateOne(
                {"timestamp": r["timestamp"], "city": settings.CITY},
                {"$set": r},
                upsert=True
            )
        )

    if operations:
        result = collection.bulk_write(operations)
        logger.info(f"Records upserted: {result.upserted_count + result.modified_count}")

    # SUMMARY
    total_rows = collection.count_documents({"city": settings.CITY})
    sample = collection.find_one({"city": settings.CITY}, {"_id": 0})
    columns = list(sample.keys()) if sample else []

    print("\n========== BACKFILL SUMMARY ==========")
    print("City:", settings.CITY)
    print("Rows in DB:", total_rows)
    print("Columns:", len(columns))
    print("Column Names:")
    for c in columns:
        print("-", c)
    print("======================================")

    logger.info("Backfill pipeline completed.")


if __name__ == "__main__":
    run_backfill()
