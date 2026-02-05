from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import os

app = FastAPI(title="House Price Prediction API")

# Robust model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "house_price_model.joblib")
model = load(MODEL_PATH)


class HouseInput(BaseModel):
    Location: str
    Size: float
    Bedrooms: int
    Bathrooms: int
    Year_Built: int
    Condition: str
    Type: str
    sold_year: int
    sold_month: int


@app.post("/predict")
def predict_price(data: HouseInput):
    """
    Predict house price from input features
    """
    input_data = data.dict()

    # Fix column name to match training data
    input_data["Year Built"] = input_data.pop("Year_Built")

    df = pd.DataFrame([input_data])
    price = model.predict(df)[0]

    return {"predicted_price": round(float(price), 2)}
