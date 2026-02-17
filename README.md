🌫️ Air Quality Index (AQI) Prediction System

An end-to-end MLOps-based Air Quality Index (AQI) Prediction System that estimates real-time AQI and provides a 3-day forecast using machine learning and deep learning models.
The system automates data collection, feature engineering, model training, deployment, and visualization using a modern production-ready stack.

📌 Project Overview

This project predicts AQI for Lahore, Pakistan using historical and real-time weather and air pollution data.
It follows a full ML lifecycle, from raw data ingestion to deployment with CI/CD automation.

Key highlights:

Real-time AQI prediction

3-day autoregressive AQI forecasting

Automated model selection

Interactive dashboard for visualization

Production-grade MLOps workflow

🏗️ System Architecture

The project is modular and follows a complete MLOps pipeline:

Data Collection

Feature Engineering

Model Training & Evaluation

Model Registry & Selection

API Deployment

Visualization Dashboard

Core Platform: Hopsworks
Backend: FastAPI
Frontend: Streamlit

📊 Data Collection

Hourly weather and air pollution data is fetched using Open-Meteo APIs for:

Location: Lahore, Pakistan

Latitude: 31.558

Longitude: 74.3507

Time Range: May 2024 – November 2025

Data Sources

Weather API

Relative Humidity

Air Quality API

PM10

PM2.5

O₃ (Ozone)

NO₂ (Nitrogen Dioxide)

SO₂ (Sulphur Dioxide)

CO (Carbon Monoxide)

The merged raw dataset is stored in the Hopsworks Feature Store as:

aqi_raw_weather_pollution (v1)

⚙️ Feature Engineering

Key processing steps:

Exploratory Data Analysis (EDA)

Correlation analysis

Temporal importance analysis

Feature importance evolution

AQI sub-index computation → final AQI

Missing value handling

Robust scaling using RobustScaler

Hyperparameter tuning & early stopping

Temporal Feature Engineering

Hour: Cyclic encoding (hour_sin, hour_cos)

Day of Week: One-hot encoding (dow_0 … dow_6)

Month → Season: One-hot encoded seasons (winter, spring, summer, autumn)

Final features are stored in:

aqi_hourly_features (version 4)

🤖 Model Development

Three models were trained and evaluated:

Model	Strength
XGBoost	Handles non-linear relationships
Ridge Regression	Simple and interpretable
LSTM	Captures temporal dependencies
Training Details

Time-series cross-validation (5 folds)

Temporal train-test split (80/20)

LSTM trained with a 24-hour sliding window

Metrics used:

RMSE

MAE

R² Score

📈 Model Performance
Model	RMSE	MAE	R²
XGBoost	7.18	4.05	0.992
Ridge Regression	36.38	27.96	0.795
LSTM	0.045	0.031	0.932

➡️ XGBoost achieved the best overall performance and is selected dynamically in production.

🚀 Backend API (FastAPI)

The FastAPI backend:

Connects to Hopsworks

Retrieves all registered models

Automatically selects the best model based on R² score

API Endpoints
GET  /                → Summary & best model info
POST /predict         → Current AQI prediction
GET  /forecast_3day   → 3-day AQI forecast

📊 Frontend & Visualization

A Streamlit dashboard provides:

Current AQI View

Real-time AQI value

Color-coded AQI category (Good, Moderate, Unhealthy, etc.)

3-Day Forecast

Interactive bar chart using Plotly

🌍 Deployment

Backend (FastAPI): Render Cloud

Frontend (Streamlit): Render Cloud

Feature Store & Model Registry: Hopsworks

Live Links

Streamlit Dashboard:
https://aqipredictor-gmnlqqv2zx3hsjqcamaq9l.streamlit.app/

FastAPI Backend:
https://aqi-fastapi-backend.onrender.com/

🔄 Automation & MLOps

Automated data ingestion

Automated model training & registration

Automated model comparison and selection

CI/CD pipelines for continuous updates

Versioned feature groups and models

🧾 Conclusion

This AQI Prediction System demonstrates a production-ready ML solution, combining:

Strong feature engineering

Robust model evaluation

Automated deployment

User-friendly visualization

It is suitable for real-time air quality monitoring, forecasting, and future scalability.
