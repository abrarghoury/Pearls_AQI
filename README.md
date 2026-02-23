# 🌫️ Pearls AQI Predictor

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)](https://www.python.org/) 
[![MongoDB](https://img.shields.io/badge/mongodb-6.0-green?logo=mongodb)](https://www.mongodb.com/) 
[![Streamlit](https://img.shields.io/badge/streamlit-1.24-orange?logo=streamlit)](https://streamlit.io/)

# 🌫 Pearls AQI Predictor

Production-Grade, Serverless Multi-Day AQI Forecasting System

🔗 **Live Dashboard:** [https://pearls-aqi.streamlit.app/](https://pearls-aqi.streamlit.app/)
📍 Deployment Target: **Karachi**

---

# 📖 Overview

**Pearls AQI Predictor** is an end-to-end machine learning system that forecasts Air Quality Index (AQI) for the next **24, 48, and 72 hours** using a fully automated, serverless architecture.

This project collects **historical and real-time air quality and weather data from external APIs.**
AQI data is obtained from **OpenWeather**, while meteorological data is sourced from **Open-Meteo**.

Both datasets are aligned on an **hourly timestamp** and integrated into a unified dataset for downstream feature engineering and model training.

The system implements a complete MLOps lifecycle:

* Automated hourly data ingestion
* Production-safe feature engineering
* Historical backfilling
* Multi-model training (regression + classification)
* Dynamic model versioning using GridFS
* Daily retraining
* Real-time prediction pipeline
* SHAP explainability
* Streamlit cloud deployment
* CI/CD via GitHub Actions

---

# 🏗 System Architecture

### Data Flow

External APIs
→ Raw MongoDB Collection
→ Clean Dataset
→ Feature Engineering
→ Feature Sanity Validation
→ Model Training
→ Model Registry (GridFS)
→ Prediction Pipeline
→ Streamlit Dashboard

The system runs entirely in the cloud with no dedicated server.

---

# 📡 Data Collection

### 1️⃣ Air Quality Data

* Source: OpenWeather Air Pollution API
* Pollutants:

  * PM2.5
  * PM10
  * NO₂
  * SO₂
  * O₃
  * CO

### 2️⃣ Meteorological Data

* Source: Open-Meteo API
* Weather features:

  * Temperature
  * Humidity
  * Wind speed
  * Pressure
  * Other atmospheric variables

### 3️⃣ Data Alignment Strategy

* All records are converted to UTC
* Hourly timestamp standardization
* Duplicate timestamp removal
* Missing value interpolation (time-based)
* Forward/backward fill as fallback
* Strict time sorting to preserve time-series integrity

---

# 🧠 Feature Engineering

The feature pipeline is designed to be **production-safe**, preventing data leakage and ensuring compatibility between training and inference modes.

---

## 🔹 AQI Calculation (US EPA Standard)

* Sub-index calculated for each pollutant using breakpoint tables
* Final AQI = maximum sub-index
* AQI class mapping:

  * 1 → Good
  * 2 → Moderate
  * 3 → Unhealthy for Sensitive Groups
  * 4 → Unhealthy
  * 5 → Hazardous

---

## 🔹 Time-Based Features

* Hour
* Day of week
* Month
* Weekend indicator
* Cyclical encoding:

  * hour_sin / hour_cos
  * dow_sin / dow_cos

These capture seasonality and periodicity patterns.

---

## 🔹 Lag Features

AQI Lags:

* 1h, 3h, 6h, 12h, 24h, 48h, 72h

Pollutant Lags:

* 1h, 3h, 6h, 12h, 24h

Weather Lags:

* 1h, 24h

Lag features allow the model to learn temporal dependencies.

---

## 🔹 Rolling Statistics

* 6h rolling mean
* 12h rolling mean
* 24h rolling mean
* 12h rolling standard deviation
* 24h rolling standard deviation

Captures short-term trends and volatility.

---

## 🔹 Delta Features

Change rates:

* AQI delta (1h, 3h, 6h, 24h)

Measures momentum and rapid pollution spikes.

---

## 🔹 Ratio Features

* PM2.5 / PM10
* NO₂ / O₃

Helps capture pollutant interaction patterns.

---

## 🔹 Target Engineering (Training Mode Only)

Generated only when `PIPELINE_MODE=training`:

* `target_aqi_t_plus_24h`
* `target_aqi_class_t_plus_24h`
* `target_aqi_class_t_plus_48h`
* `target_aqi_class_t_plus_72h`

Rows with missing future targets are dropped safely.
During inference mode, no target shifting occurs.

---

# 🤖 Model Training Pipeline

The training pipeline is fully automated and runs daily.

---

## Tasks

### Regression

Predict numeric AQI for +24h.

### Classification

Predict AQI category for:

* +24h
* +48h
* +72h

---

## Models Implemented

Regression:

* Random Forest Regressor
* Gradient Boosting Regressor
* HistGradientBoosting Regressor

Classification:

* Random Forest Classifier
* Gradient Boosting Classifier
* XGBoost Classifier

---

## Training Strategy

* Strict time-series split (80% train / 20% test)
* Oversampling for class imbalance (RandomOverSampler)
* Best model selected based on:

  * R² (regression)
  * Weighted F1-score (classification)

---

## Model Registry (GridFS)

Each best-performing model is:

* Serialized with joblib
* Stored in MongoDB GridFS
* Versioned dynamically (v1, v2, v3…)
* Marked as active
* Previous active models automatically archived

Stored metadata includes:

* Metrics
* Feature list
* Training duration
* Data rows used
* Pipeline version

---

# 🔁 CI/CD Automation

## Hourly Workflow

Runs every hour:

* Data ingestion
* Clean dataset build
* Feature engineering
* Sanity checks

## Daily Workflow

Runs once per day:

* Training pipeline (training mode)
* Prediction pipeline (inference mode)
* Updates active models
* Stores new predictions

---

# 📊 Exploratory Data Analysis

Includes:

* AQI trend visualization
* Seasonal pattern detection
* Pollutant correlation analysis
* Volatility assessment
* Clean feature export for analysis
* Notebook: `aqi_EDA.ipynb`

---

# 🌐 Live Dashboard Features

* 3-day AQI forecast
* Historical AQI trends
* Model performance metrics
* SHAP-based feature importance
* Hazard alerts
* Interactive visualizations

---

# 📂 Project Structure

```
Pearls_AQI/
│
├── models/
│   ├── train_model.py
│   └── predict.py
│
├── data_pipeline/
│   ├── ingest_latest.py
│   ├── build_clean_dataset.py
│   ├── feature_engineering.py
│   ├── backfill.py
│   └── sanity_check_feature.py
│
├── config/
│   ├── settings.py
│   ├── constants.py
│   ├── logging.py
│   └── mongo.py
│
├── app/
│   ├── main.py
│   ├── services/
│   └── pages/
│
└── .github/workflows/
```

---

# 🚀 How to Run Locally

```bash
git clone <repo-url>
cd Pearls_AQI
pip install -r requirements.txt

# Training
PIPELINE_MODE=training python -m models.train_model

# Prediction
PIPELINE_MODE=inference python -m models.predict

# Run dashboard
streamlit run app/main.py
```

---

# 🎯 Key Highlights

* Fully automated serverless ML system
* Time-series safe architecture
* Multi-horizon forecasting
* Dynamic model versioning
* GridFS production storage
* CI/CD integration
* Explainable AI with SHAP
* Real-world deployment

---

# 🌍 Impact

Air pollution poses a significant public health challenge in **Karachi**.

This system enables:

* Early warnings for hazardous AQI
* Data-driven environmental monitoring
* Scalable forecasting to additional cities


---

## 📊 Project Status  
This project is under continuous improvement and optimization.

---


