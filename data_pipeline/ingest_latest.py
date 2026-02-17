import requests
import pandas as pd

from datetime import datetime, timedelta

from config.settings import settings
from config.logging import logger
from config.constants import RAW_COLLECTION
from config.mongo import get_database


# =====================================================
# FETCH AIR POLLUTION (LAST HOURS)
# =====================================================

def fetch_latest_air_pollution():

    logger.info("Fetching latest AQI data...")

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(hours=3)

    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"

    params = {
        "lat": settings.LAT,
        "lon": settings.LON,
        "start": int(start_dt.timestamp()),
        "end": int(end_dt.timestamp()),
        "appid": settings.OPENWEATHER_API_KEY
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json().get("list", [])

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

    return pd.DataFrame(records)


# =====================================================
# FETCH WEATHER
# =====================================================

def fetch_latest_weather():

    logger.info("Fetching latest weather data...")

    today = datetime.utcnow().date().isoformat()

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": settings.LAT,
        "longitude": settings.LON,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation"
        ],
        "timezone": "UTC",
        "start_date": today,
        "end_date": today
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    hourly = response.json()["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["pressure_msl"],
        "wind_speed": hourly["wind_speed_10m"],
        "wind_direction": hourly["wind_direction_10m"],
        "precipitation": hourly["precipitation"],
    })

    return df


# =====================================================
# MERGE + UPSERT
# =====================================================

def run_latest_ingestion():

    logger.info("========== LATEST INGESTION START ==========")

    db = get_database()
    collection = db[RAW_COLLECTION]

    aqi_df = fetch_latest_air_pollution()
    weather_df = fetch_latest_weather()

    df = pd.merge(aqi_df, weather_df, on="timestamp", how="left")

    upserted = 0

    for _, row in df.iterrows():

        rec = row.to_dict()

        rec["city"] = settings.CITY
        rec["source"] = "openweather + openmeteo"
        rec["ingested_at"] = datetime.utcnow()

        result = collection.update_one(
            {"timestamp": rec["timestamp"], "city": settings.CITY},
            {"$set": rec},
            upsert=True
        )

        if result.upserted_id:
            upserted += 1

    logger.info(f"Latest rows upserted: {upserted}")
    logger.info("========== INGESTION COMPLETE ==========")


if __name__ == "__main__":
    run_latest_ingestion()
