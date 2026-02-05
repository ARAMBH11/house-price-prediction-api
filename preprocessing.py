import pandas as pd
from typing import Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def preprocess_data(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Preprocess raw real estate data.

    Steps:
    - Drop non-informative ID column
    - Remove rows with missing target values
    - Extract year and month from Date Sold
    - Handle missing values in features
    - Scale numerical features
    - One-hot encode categorical features

    Returns:
    - X: Feature dataframe
    - y: Target series
    - preprocessor: ColumnTransformer
    """

    df = df.copy()

    # Drop ID column if present
    df.drop(columns=["Property ID"], inplace=True, errors="ignore")

    # Drop rows with missing target (CRITICAL FIX)
    df = df.dropna(subset=["Price"])

    # Date feature engineering (safe parsing)
    df["Date Sold"] = pd.to_datetime(df["Date Sold"], errors="coerce")
    df["sold_year"] = df["Date Sold"].dt.year
    df["sold_month"] = df["Date Sold"].dt.month
    df.drop(columns=["Date Sold"], inplace=True)

    # Separate features and target
    X = df.drop("Price", axis=1)
    y = df["Price"]

    # Identify column types
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    # Numerical preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return X, y, preprocessor
