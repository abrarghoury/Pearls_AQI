# 🌫️ Pearls AQI Predictor

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)](https://www.python.org/) 
[![MongoDB](https://img.shields.io/badge/mongodb-6.0-green?logo=mongodb)](https://www.mongodb.com/) 
[![Streamlit](https://img.shields.io/badge/streamlit-1.24-orange?logo=streamlit)](https://streamlit.io/)

**Pearls AQI Predictor** is an advanced Air Quality Index forecasting tool for urban environments. It leverages **real-time air pollution data**, a **Gradient Boosting regression model**, and **SHAP-based interpretability** to provide actionable insights for 24–72 hour AQI predictions.

---

## 🚀 Features

- 🔹 Real-time AQI dashboard with latest raw and cleaned data  
- 🔹 24-hour, 48-hour, and 72-hour AQI forecasts with color-coded risk levels  
- 🔹 Weather integration: temperature, humidity, wind, and pressure  
- 🔹 Explainable AI using **SHAP** to identify key contributors to AQI  
- 🔹 Modular pipeline: ingestion → feature engineering → prediction → dashboard  
- 🔹 MongoDB backend for historical and latest data storage  

---

## 🗂️ Project Structure

Pearls_AQI/
│

├── requirements.txt

├── pyproject.toml

├── check_latest_features.py

│

├── venv/ # Virtual environment (not pushed to GitHub)

│

├── models/

│ ├── train_model.py

│ └── predict.py

│
├── eda/

│ ├── export_clean_features_to_csv.py

│ ├── aqi_EDA.ipynb

│ └── clean_feature.csv

│
├── data_pipeline/

│ ├── ingest_latest.py

│ ├── feature_engineering.py

│ ├── build_clean_dataset.py

│ ├── backfill.py

│ ├── clear_database.py

│ └── sanity_check_feature.py

│
├── config/

│ ├── settings.py

│ ├── constants.py

│ ├── logging.py

│ └── mongo.py

│
├── artifacts/

│ └── models/

│ └── (saved trained models)
│
├── app/

│ ├── main.py

│ ├── app_config.py

│ │
│ ├── services/

│ │ ├── aqi_utils.py

│ │ └── mongo_service.py

│ │
│ └── pages/

│ └── AQI_Trends.py

│
└── .streamlit/

└── config.toml


---

## ⚙️ System Architecture



Data Source
↓
Ingestion Pipeline (ingest_latest.py)
↓
Feature Engineering (feature_engineering.py)
↓
Clean Dataset Builder (build_clean_dataset.py)
↓
Model Training (train_model.py)
↓
Prediction (predict.py)
↓
MongoDB Storage
↓
Streamlit Dashboard (main.py)


---

## 🚀 Key Components

### 🔹 Data Pipeline
Located in `data_pipeline/`

- `ingest_latest.py` → Fetches and stores latest AQI data  
- `feature_engineering.py` → Creates time-based & pollutant features  
- `build_clean_dataset.py` → Produces ML-ready dataset  
- `backfill.py` → Historical data processing  
- `sanity_check_feature.py` → Data validation  
- `clear_database.py` → Reset utility  

---

### 🔹 Model Layer
Located in `models/`

- `train_model.py` → Trains Gradient Boosting regression model  
- `predict.py` → Generates 24h / 48h / 72h predictions  

Saved trained models are stored in:



artifacts/models/


---

### 🔹 EDA Layer
Located in `eda/`

- `aqi_EDA.ipynb` → Exploratory Data Analysis notebook  
- `export_clean_features_to_csv.py` → Export utility  
- `clean_feature.csv` → Clean dataset snapshot  

---

### 🔹 Configuration Layer
Located in `config/`

- `settings.py` → Environment & MongoDB configs  
- `constants.py` → Collection names & constants  
- `logging.py` → Logging setup  
- `mongo.py` → Mongo connection wrapper  

---

### 🔹 Dashboard (Streamlit App)
Located in `app/`

- `main.py` → Main dashboard entry point  
- `app_config.py` → App UI settings  
- `services/` → Backend logic for dashboard  
- `pages/AQI_Trends.py` → Multi-page analytics view  

Streamlit configuration:



.streamlit/config.toml


---

## 📊 Features

- ✅ Real-time AQI display
- ✅ 3-day AQI Forecast
- ✅ Weather Snapshot Integration
- ✅ SHAP Explainability (Global + Local)
- ✅ MongoDB Atlas backend
- ✅ Modular production-ready architecture

---

## 💻 Run Locally

### 1️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run data pipeline
python -m data_pipeline.ingest_latest
python -m data_pipeline.feature_engineering
python -m data_pipeline.build_clean_dataset

4️⃣ Train model
python models/train_model.py

5️⃣ Run dashboard
streamlit run app/main.py

Perfect 👍 samajh gaya — SHAP remove kar dete hain aur ending ko proper professional close dete hain.

Main tumhe final clean ending section de raha hoon jo tum README ke end me paste kar sakte ho 👇

---

## 🔮 Future Enhancements

The following improvements are planned to further enhance the system:

- Cloud deployment (AWS / Streamlit Cloud)
- Alert system for hazardous AQI levels
- Historical AQI trend analytics dashboard

---

## 📊 Project Status  
This project is under continuous improvement and optimization.

---

## 🤝 Contribution

Contributions, suggestions, and improvements are welcome.

If you'd like to contribute:
