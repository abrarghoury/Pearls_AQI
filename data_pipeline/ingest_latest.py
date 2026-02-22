import requests
import pandas as pd

from datetime import datetime, timedelta
from pymongo import UpdateOne
from pymongo.errors import PyMongoError

from config.settings import settings
from config.logging import logger
from config.constants import RAW_COLLECTION
from config.mongo import get_database


# =====================================================
# FETCH AIR POLLUTION (LAST 3 HOURS)
# =====================================================

def fetch_latest_air_pollution():
    try:
        logger.info("Fetching latest AQI data...")

        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(hours=3)

        url = "https://api.openweathermap.org/data/2.5/air_pollution/history"

        params = {
            "lat": settings.LAT,
            "lon": settings.LON,
            "start": int(start_dt.timestamp()),
            "end": int(end_dt.timestamp()),
            "appid": settings.OPENWEATHER_API_KEY
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        payload = response.json()
        data = payload.get("list", [])

        if not data:
            logger.warning("No AQI records returned from API.")
            return pd.DataFrame()

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
        df.drop_duplicates(subset="timestamp", inplace=True)
        df.sort_values("timestamp", inplace=True)

        return df

    except requests.RequestException as e:
        logger.error(f"AQI API request failed: {e}")
        return pd.DataFrame()

    except Exception as e:
        logger.exception(f"Unexpected AQI fetch error: {e}")
        return pd.DataFrame()


# =====================================================
# FETCH WEATHER
# =====================================================

def fetch_latest_weather():
    try:
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

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        payload = response.json()

        if "hourly" not in payload:
            logger.warning("Weather API returned no hourly data.")
            return pd.DataFrame()

        hourly = payload["hourly"]

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"]),
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "pressure": hourly["pressure_msl"],
            "wind_speed": hourly["wind_speed_10m"],
            "wind_direction": hourly["wind_direction_10m"],
            "precipitation": hourly["precipitation"],
        })

        df.drop_duplicates(subset="timestamp", inplace=True)
        df.sort_values("timestamp", inplace=True)

        return df

    except requests.RequestException as e:
        logger.error(f"Weather API request failed: {e}")
        return pd.DataFrame()

    except Exception as e:
        logger.exception(f"Unexpected weather fetch error: {e}")
        return pd.DataFrame()


# =====================================================
# ALIGN + MERGE
# =====================================================

def align_and_merge_latest(aqi_df, weather_df):
    logger.info("Aligning timestamps...")

    if aqi_df.empty:
        logger.warning("AQI DataFrame empty. Skipping merge.")
        return pd.DataFrame()

    start = aqi_df.timestamp.min()
    end = aqi_df.timestamp.max()

    full_range = pd.date_range(start=start, end=end, freq="H")

    aqi_df = aqi_df.set_index("timestamp").reindex(full_range)
    weather_df = weather_df.set_index("timestamp").reindex(full_range)

    # Interpolate AQI
    aqi_df = aqi_df.interpolate(method="time").ffill().bfill()

    # Interpolate weather
    weather_df = weather_df.interpolate(method="time").ffill().bfill()

    df = aqi_df.join(weather_df, how="left")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)

    return df


# =====================================================
# MERGE + BULK UPSERT
# =====================================================

def run_latest_ingestion():
    logger.info("========== LATEST INGESTION START ==========")

    try:
        db = get_database()
        collection = db[RAW_COLLECTION]

        aqi_df = fetch_latest_air_pollution()
        weather_df = fetch_latest_weather()

        df = align_and_merge_latest(aqi_df, weather_df)

        if df.empty:
            logger.warning("No data to upsert.")
            return

        # Replace NaN with None for MongoDB
        df = df.where(pd.notnull(df), None)

        operations = []

        for _, row in df.iterrows():
            rec = row.to_dict()
            rec["city"] = settings.CITY
            rec["source"] = "openweather + openmeteo"
            rec["ingested_at"] = datetime.utcnow()

            operations.append(
                UpdateOne(
                    {"timestamp": rec["timestamp"], "city": settings.CITY},
                    {"$set": rec},
                    upsert=True
                )
            )

        if operations:
            result = collection.bulk_write(operations)
            logger.info(f"Inserted: {result.upserted_count}")
            logger.info(f"Modified: {result.modified_count}")

        logger.info("========== INGESTION COMPLETE ==========")

    except PyMongoError as e:
        logger.error(f"MongoDB error: {e}")

    except Exception as e:
        logger.exception(f"Unexpected ingestion error: {e}")


if __name__ == "__main__":
    run_latest_ingestion()