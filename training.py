from typing import Tuple
import numpy as np
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump


def train_model(X, y, preprocessor) -> Tuple[Pipeline, str, any, any]:
    """
    Train multiple regression models, evaluate them, and select the best model.

    Returns:
    - best_model: Trained sklearn Pipeline
    - best_model_name: Name of selected model
    - X_test: Test features
    - y_test: Test target
    """

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models to compare
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    # Hyperparameter grid
    param_grid = {
        "Random Forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10]
        }
    }

    best_model = None
    best_model_name = None
    best_r2 = float("-inf")

    for name, model in models.items():

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Hyperparameter tuning (only for RF)
        if name in param_grid:
            search = GridSearchCV(
                pipeline,
                param_grid[name],
                cv=5,
                scoring="r2",
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            trained_model = search.best_estimator_
        else:
            trained_model = pipeline.fit(X_train, y_train)

        # Evaluate on TEST set
        y_pred = trained_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n{name} Performance:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE : {mae:.2f}")
        print(f"R2  : {r2:.4f}")

        # Select best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = trained_model
            best_model_name = name

    # Ensure model directory exists
    os.makedirs("models", exist_ok=True)

    # Save best model
    dump(best_model, "models/house_price_model.joblib")
    print(f"\nFinal selected model: {best_model_name}")
    print("Best model saved as models/house_price_model.joblib")

    return best_model, best_model_name, X_test, y_test
