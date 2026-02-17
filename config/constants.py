# =========================
# AQI PIPELINE CONSTANTS
# =========================

AQI_THRESHOLDS = {
    "good": 50,
    "moderate": 100,
    "unhealthy_sensitive": 150,
    "unhealthy": 200,
    "very_unhealthy": 300,
    "hazardous": 500
}

# =========================
# DATA CONFIG
# =========================

HISTORICAL_MONTHS = 6
LIVE_FETCH_INTERVAL_HOURS = 1
PREDICTION_HORIZON_HOURS = 72

# =========================
# DATABASE SCHEMA
# =========================

TARGET_COLUMN = "aqi"
TIMESTAMP_COLUMN = "timestamp"

MODEL_COLLECTION = "model_registry"
CLEANED_FEATURE_COLLECTION = "feature_cleaned"
RAW_COLLECTION = "raw_aqi_data"
CLEAN_COLLECTION = "clean_aqi"
FEATURE_COLLECTION = "aqi_features"
PREDICTION_COLLECTION = "prediction_logs"
LATEST_PREDICTION_COLLECTION = "predictions"

# =========================
# FEATURE CONFIG
# =========================

POLLUTANT_FEATURES = [
    "pm2_5",
    "pm10",
    "no2",
    "so2",
    "o3",
    "co"
]

WEATHER_FEATURES = [
    "temperature",
    "wind_speed",
    "wind_direction",
    "humidity",
    "pressure",
    "precipitation"
]

TIME_FEATURES = [
    "hour",
    "day_of_week",
    "month"
]
