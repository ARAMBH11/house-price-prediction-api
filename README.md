# 🏠 House Price Prediction API

An end-to-end **Machine Learning project** that predicts house prices based on property features.  
The trained ML model is deployed as a **FastAPI REST API** and hosted publicly using **Render**.

---

## 🚀 Live API

🔗 **Base URL:**  
https://house-price-prediction-api-4gac.onrender.com

📄 **Swagger UI (API Docs):**  
https://house-price-prediction-api-4gac.onrender.com/docs

---

## 📌 Features

- End-to-end ML pipeline (EDA → Training → Evaluation → Deployment)
- FastAPI-based REST API
- Publicly deployed on Render
- Real-time predictions
- Automatic API documentation using Swagger UI

---

## 📥 Input Parameters

The API accepts the following inputs:

| Feature        | Type    | Description |
|---------------|---------|-------------|
| Location      | string  | Property location |
| Size          | float   | House size (sq ft) |
| Bedrooms      | int     | Number of bedrooms |
| Bathrooms     | int     | Number of bathrooms |
| Year_Built    | int     | Year the house was built |
| Condition     | string  | Property condition |
| Type          | string  | Property type |
| sold_year     | int     | Year sold |
| sold_month    | int     | Month sold |

---

## 🧪 Sample Request (JSON)

```json
{
  "Location": "New York",
  "Size": 1200,
  "Bedrooms": 3,
  "Bathrooms": 2,
  "Year_Built": 2015,
  "Condition": "Good",
  "Type": "Apartment",
  "sold_year": 2024,
  "sold_month": 6
}

## 📂 **Project Structure**

🏠 house-price-prediction-api/
│
├── 📁 api/
│ └── 🚀 main.py # FastAPI application (API entry point)
│
├── 📁 models/
│ └── 🤖 house_price_model.joblib # Trained ML model
│
├── 📄 data_loading.py # Data loading utilities
├── 📊 eda.py # Exploratory Data Analysis
├── 🧹 preprocessing.py # Feature engineering & preprocessing
├── 🏋️ training.py # Model training & selection
├── 📈 evaluation.py # Model evaluation metrics
├── 🔮 inference.py # Prediction logic
│
├── 📒 House_Price_Notebook.ipynb # Complete EDA & experimentation notebook
├── 📦 requirements.txt # Project dependencies
├── 🚫 .gitignore # Ignored files & folders
└── 📘 README.md # Project documentation




