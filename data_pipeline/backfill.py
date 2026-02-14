import requests
import pandas as pd

from datetime import datetime, timedelta

from config.settings import settings
from config.logging import logger
from config.constants import HISTORICAL_MONTHS, RAW_COLLECTION

from config.mongo import get_database


# =========================================================
# AIR POLLUTION
# =========================================================

def fetch_air_pollution(start_ts: int, end_ts: int):

    logger.info("Fetching Air Pollution historical data...")

    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"

    params = {
        "lat": settings.LAT,
        "lon": settings.LON,
        "start": start_ts,
        "end": end_ts,
        "appid": settings.OPENWEATHER_API_KEY
    }

    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()

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
            "so2": comp.get("so2"),  # optional but safe
        })

    df = pd.DataFrame(records)

    df.drop_duplicates(subset="timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)

    logger.info(f"AQI rows fetched: {len(df)}")

    return df


# =========================================================
# WEATHER
# =========================================================

def fetch_weather(start_date: str, end_date: str):

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

    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()

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

    df.drop_duplicates(subset="timestamp", inplace=True)
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

    # Set index
    aqi_df = aqi_df.set_index("timestamp").reindex(full_range)
    weather_df = weather_df.set_index("timestamp").reindex(full_range)

    # AQI must exist (target safety)
    before_drop = len(aqi_df)
    aqi_df = aqi_df.dropna(subset=["pm2_5"])
    after_drop = len(aqi_df)

    logger.info(f"Dropped empty AQI rows: {before_drop - after_drop}")

    # Weather forward fill (max 3 hours)
    weather_df = weather_df.ffill(limit=3)

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

    # ALIGN (critical step)
    df = align_and_merge(aqi_df, weather_df)

    # Mongo Records
    records = []

    for _, row in df.iterrows():

        rec = row.to_dict()

        rec["city"] = settings.CITY
        rec["source"] = "openweather + openmeteo"
        rec["ingested_at"] = datetime.utcnow()

        records.append(rec)

    # UPSERT
    upserted = 0

    for r in records:

        result = collection.update_one(
            {"timestamp": r["timestamp"], "city": settings.CITY},
            {"$set": r},
            upsert=True
        )

        if result.upserted_id:
            upserted += 1

    logger.info(f"Records upserted: {upserted}")

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


if __name__ == "__main__":
    run_backfill()
