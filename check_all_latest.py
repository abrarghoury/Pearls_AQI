import joblib
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["pearls_aqi"]

latest_feature = db.feature_cleaned.find_one(sort=[("validation_done_at", -1)])
df_latest = pd.DataFrame([latest_feature])

active_model = db.model_registry.find_one({"status": "active"})
model_path = active_model["model_path"]
features = active_model["features"]

# Fill missing features with 0
for f in features:
    if f not in df_latest.columns:
        df_latest[f] = 0

model = joblib.load(model_path)
pred = model.predict(df_latest[features])
print("Prediction:", pred)
