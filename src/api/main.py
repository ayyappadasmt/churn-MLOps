import json
import os
import pandas as pd
import mlflow.xgboost
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")


# ---------------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------------

# Build absolute paths based on where this file lives.
# This prevents the API from depending on the terminal's
# current working directory.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = .../churn-mlops/src/api

PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
# PROJECT_ROOT = .../churn-mlops

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "churn_model"
)

COLUMNS_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "feature_columns.json"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

# Load the trained XGBoost model ONCE when the server starts.
# We do not reload it for every prediction request.
model = mlflow.xgboost.load_model(MODEL_PATH)


# Load the exact feature-column order used during training.
with open(COLUMNS_PATH, "r") as f:
    FEATURE_COLUMNS = json.load(f)


# ---------------------------------------------------------
# REQUEST SCHEMA
# ---------------------------------------------------------

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerData):

    # Convert incoming JSON into a one-row DataFrame.
    input_df = pd.DataFrame([customer.model_dump()])

    # Apply one-hot encoding.
    input_encoded = pd.get_dummies(input_df)

    # Force the request to have exactly the same
    # feature columns and ordering as the training data.
    input_final = input_encoded.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    )

    # Generate prediction.
    prediction = model.predict(input_final)[0]

    # Get probability of the predicted class.
    probabilities = model.predict_proba(input_final)[0]
    probability = probabilities[int(prediction)]

    return {
        "churn_prediction": bool(prediction),
        "confidence": round(float(probability), 4)
    }
@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running",
        "docs": "/docs",
        "health": "/health"
    }
