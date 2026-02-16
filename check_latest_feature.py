# check_latest_feature.py
from pymongo import MongoClient
from config.settings import settings
from pprint import pprint
from datetime import datetime

def main():
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]

    # --- Collection to check ---
    feature_coll_name = "aqi_features"

    print(f"\n=== Checking Latest Feature in '{feature_coll_name}' ===")
    collection = db[feature_coll_name]

    latest_doc_cursor = collection.find().sort("feature_generated_at", -1).limit(1)
    latest_docs = list(latest_doc_cursor)

    if not latest_docs:
        print("No documents found in this collection.")
        return

    doc = latest_docs[0]

    pprint({
        "_id": doc.get("_id"),
        "feature_generated_at": doc.get("feature_generated_at"),
        "timestamp": doc.get("timestamp"),
        "aqi": doc.get("aqi"),
        "city": doc.get("city")
    })

    # Check if feature_generated_at is today
    if doc.get("feature_generated_at") and doc.get("feature_generated_at").date() < datetime.now().date():
        print("⚠️ Warning: Latest feature is older than today. Pipeline may not have run properly.")
    else:
        print("✅ Latest feature is up-to-date with today.")

if __name__ == "__main__":
    main()
