# check_latest_aqi.py
from pymongo import MongoClient
from config.settings import settings
import pprint

def main():
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]

    pp = pprint.PrettyPrinter(indent=2)

    # ===============================
    # Latest RAW Weather
    # ===============================
    print("\n=== Latest RAW Weather ===")
    latest_raw_cursor = db.raw_aqi_data.find().sort("timestamp", -1).limit(1)
    latest_raw = list(latest_raw_cursor)
    if not latest_raw:
        print("No RAW data found!")
    else:
        for doc in latest_raw:
            pp.pprint(doc)

    # ===============================
    # Latest Feature AQI
    # ===============================
    print("\n=== Latest Feature AQI ===")
    latest_features_cursor = db.feature_cleaned.find().sort("feature_generated_at", -1).limit(1)
    latest_features = list(latest_features_cursor)
    if not latest_features:
        print("No FEATURE data found!")
    else:
        for doc in latest_features:
            pp.pprint(doc)

    # ===============================
    # Latest Prediction AQI
    # ===============================
    print("\n=== Latest Prediction AQI ===")
    latest_preds_cursor = db.prediction_logs.find().sort("created_at", -1).limit(1)
    latest_preds = list(latest_preds_cursor)
    if not latest_preds:
        print("No PREDICTION data found!")
    else:
        for doc in latest_preds:
            pp.pprint(doc)

if __name__ == "__main__":
    main()
