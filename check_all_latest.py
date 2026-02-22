# check_model_debug.py
import os
import joblib
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# MongoDB Connection
# -------------------------------
client = MongoClient(os.getenv("MONGO_URI"))
db = client["pearls_aqi"]

# -------------------------------
# Get latest features
# -------------------------------
latest_feature = db.feature_cleaned.find_one(sort=[("validation_done_at", -1)])
if latest_feature is None:
    raise ValueError("No latest feature row found in feature_cleaned collection!")

df_latest = pd.DataFrame([latest_feature])
print("Latest feature timestamp:", latest_feature.get("validation_done_at"))

# -------------------------------
# Get active model
# -------------------------------
active_model = db.model_registry.find_one({"status": "active"})
if active_model is None:
    raise ValueError("No active model found in model_registry!")

model_path = active_model.get("model_path")
features = active_model.get("features", [])

print("\nModel path from DB:", model_path)
print("Features used in model:", features[:5], "...")  # show first 5 features

# -------------------------------
# Check file existence
# -------------------------------
full_model_path = os.path.abspath(model_path)
print("Full absolute path:", full_model_path)
print("File exists:", os.path.exists(full_model_path))

if not os.path.exists(full_model_path):
    raise FileNotFoundError(f"Model file not found at: {full_model_path}")

# -------------------------------
# Fill missing features
# -------------------------------
for f in features:
    if f not in df_latest.columns:
        df_latest[f] = 0

# -------------------------------
# Load model & predict
# -------------------------------
try:
    model = joblib.load(full_model_path)
    pred = model.predict(df_latest[features])
    print("\n✅ Prediction:", pred)
except Exception as e:
    print("\n❌ Error loading model or predicting:", str(e))
