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
