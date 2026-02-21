# check_all_latest.py

from pymongo import MongoClient
from datetime import datetime
from pprint import pprint
from config.settings import settings

# ----------------------------
# COLLECTION CONFIG
# ----------------------------

COLLECTION_CONFIG = {
    "raw_aqi_data": "timestamp",
    "clean_aqi": "timestamp",
    "aqi_features": "feature_generated_at",
    "feature_cleaned": "validation_done_at",
    "model_registry": "trained_at",
    "prediction_logs": "predicted_at",
    "predictions": "predicted_at",
}


def check_collections():
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]

    print("\n========== PIPELINE DEBUG STATUS ==========\n")

    for coll_name, time_field in COLLECTION_CONFIG.items():
        collection = db[coll_name]

        total_docs = collection.count_documents({})
        print(f"\n--- {coll_name} ---")
        print(f"Total Documents: {total_docs}")

        if total_docs == 0:
            print("⚠️ Collection is EMPTY")
            continue

        # Get latest doc based on time field
        latest = collection.find().sort(time_field, -1).limit(1)
        latest = list(latest)

        if not latest:
            print("⚠️ Could not fetch latest document")
            continue

        doc = latest[0]

        print(f"Sorted by field: {time_field}")

        # Build summary for display
        summary = {
            "_id": doc.get("_id"),
            "city": doc.get("city"),
            time_field: doc.get(time_field),
            "timestamp": doc.get("timestamp"),
            "aqi": doc.get("aqi"),
        }

        pprint(summary)

        # Date check
        dt_value = doc.get(time_field)
        updated_today = False

        if isinstance(dt_value, datetime):
            if dt_value.date() == datetime.utcnow().date():
                print("✅ UPDATED TODAY")
                updated_today = True
            else:
                print(f"⚠️ NOT updated today (latest: {dt_value.date()})")
        else:
            print("⚠️ No valid datetime found")

        # Extra check for predictions collection
        if coll_name == "predictions":
            if not doc.get("predictions"):
                print("⚠️ Predictions field is empty! Pipeline might not have run.")
            elif updated_today:
                print("✅ Predictions generated today.")
            else:
                print("⚠️ Predictions are outdated.")

    print("\n===========================================\n")


if __name__ == "__main__":
    check_collections()