from joblib import load
import pandas as pd

# Load model once (efficient for APIs)
MODEL_PATH = "models/house_price_model.joblib"
model = load(MODEL_PATH)


def predict(input_data: dict) -> float:
    """
    Generate house price prediction from input features.

    Args:
        input_data (dict): Feature values

    Returns:
        float: Predicted house price
    """
    try:
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)
        return float(prediction[0])
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")
