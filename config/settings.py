import os
from dotenv import load_dotenv

# -----------------------------------------------------
# LOAD LOCAL .env IF EXISTS
# -----------------------------------------------------
load_dotenv()  # safe: only affects local dev

class Settings:
    """
    Central configuration for AQI system.
    Works for both local dev (.env) and CI/CD (GitHub Actions Secrets)
    """

    # ---------------- ENV & LOCATION ----------------
    ENV = os.getenv("ENV", "dev")
    TIMEZONE = os.getenv("TIMEZONE", "Asia/Karachi")
    CITY = os.getenv("CITY", "Karachi")
    LAT = float(os.getenv("LAT", "24.8607"))
    LON = float(os.getenv("LON", "67.0011"))

    # ---------------- API ----------------
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

    # ---------------- DATABASE ----------------
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "pearls_aqi")

    # ---------------- PIPELINE ----------------
    # training or inference
    PIPELINE_MODE = os.getenv("PIPELINE_MODE", "training")

# ---------------- CREATE INSTANCE ----------------
settings = Settings()

# ---------------- FAIL FAST ----------------
missing_vars = []

if not settings.OPENWEATHER_API_KEY:
    missing_vars.append("OPENWEATHER_API_KEY")

if not settings.MONGO_URI:
    missing_vars.append("MONGO_URI")

if settings.PIPELINE_MODE not in ["training", "inference"]:
    raise ValueError("PIPELINE_MODE must be 'training' or 'inference'")

if missing_vars:
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")