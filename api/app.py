from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import List
import os
import psycopg2
from datetime import datetime
import json

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API with Logging",
    description="API for predicting diabetes risk with database logging",
    version="1.0.0"
)

# Database connection settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "diabetes_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_PORT = os.getenv("DB_PORT", "5432")

# Load the trained model
MODEL_PATH = "models/diabetes_logistic_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


def log_prediction(input_data, prediction, probability, risk_level):
    """Log prediction to database"""
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO prediction_logs
                (timestamp, input_data, prediction, probability, risk_level)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                datetime.now(),
                json.dumps(input_data),
                prediction,
                probability,
                risk_level
            ))
            conn.commit()
            cur.close()
            conn.close()
    except Exception as e:
        print(f"Failed to log prediction: {e}")

# Define input schema


class PredictionInput(BaseModel):
    number_of_times_pregnant: float
    plasma_glucose_concentration: float
    diastolic_blood_pressure: float
    triceps_skin_fold_thickness: float
    serum_insulin: float
    body_mass_index: float
    diabetes_pedigree_function: float
    age_years: float

    class Config:
        json_schema_extra = {
            "example": {
                "number_of_times_pregnant": 2.0,
                "plasma_glucose_concentration": 120.0,
                "diastolic_blood_pressure": 70.0,
                "triceps_skin_fold_thickness": 25.0,
                "serum_insulin": 80.0,
                "body_mass_index": 25.5,
                "diabetes_pedigree_function": 0.5,
                "age_years": 30.0
            }
        }


class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    risk_level: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Diabetes Prediction API with Database Logging",
        "status": "healthy",
        "model_loaded": model is not None,
        "database_connected": get_db_connection() is not None
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make a prediction and log it to database"""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame with correct column names
        input_dict = {
            "Number of times pregnant": input_data.number_of_times_pregnant,
            "Plasma glucose concentration a 2 hours in an oral glucose tolerance test": input_data.plasma_glucose_concentration,
            "Diastolic blood pressure (mm Hg)": input_data.diastolic_blood_pressure,
            "Triceps skin fold thickness (mm)": input_data.triceps_skin_fold_thickness,
            "2-Hour serum insulin (mu U/ml)": input_data.serum_insulin,
            "Body mass index (weight in kg/(height in m)^2)": input_data.body_mass_index,
            "Diabetes pedigree function": input_data.diabetes_pedigree_function,
            "Age (years)": input_data.age_years
        }

        input_df = pd.DataFrame([input_dict])

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Log the prediction
        log_prediction(input_data.dict(), int(prediction),
                       float(probability), risk_level)

        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level
        )

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/prediction-history")
async def get_prediction_history(limit: int = 10):
    """Get recent prediction history from database"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(
                status_code=503, detail="Database not available")

        cur = conn.cursor()
        cur.execute("""
            SELECT timestamp, prediction, probability, risk_level
            FROM prediction_logs
            ORDER BY timestamp DESC
            LIMIT %s
        """, (limit,))

        results = cur.fetchall()
        cur.close()
        conn.close()

        history = []
        for row in results:
            history.append({
                "timestamp": row[0].isoformat(),
                "prediction": row[1],
                "probability": row[2],
                "risk_level": row[3]
            })

        return {"history": history, "count": len(history)}

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
