import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    Central configuration for AQI system
    """

    # ENV
    ENV = os.getenv("ENV", "dev")
    TIMEZONE = os.getenv("TIMEZONE", "Asia/Karachi")

    # API
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

    # DATABASE
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "pearls_aqi")

    # LOCATION
    CITY = os.getenv("CITY", "Karachi")
    LAT = float(os.getenv("LAT", "24.8607"))
    LON = float(os.getenv("LON", "67.0011"))

    # PIPELINE
    PIPELINE_MODE = os.getenv("PIPELINE_MODE", "training")


settings = Settings()


# Fail Fast
if not settings.OPENWEATHER_API_KEY:
    raise ValueError("Missing OPENWEATHER_API_KEY in .env")

if not settings.MONGO_URI:
    raise ValueError("Missing MONGO_URI in .env")

if settings.PIPELINE_MODE not in ["training", "inference"]:
    raise ValueError("PIPELINE_MODE must be training or inference")
