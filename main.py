from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# --- 1. SETUP ---
app = FastAPI()
# Load model into memory ONCE when the server starts
model = joblib.load("linear_regression_model.pkl")

# --- 2. THE SCHEMA ---
# This defines what the "Request Body" must look like
class PredictionInput(BaseModel):
    years_experience: float 

# --- 3. THE ENDPOINT ---
@app.post("/predict")
def predict_salary(data: PredictionInput):
    # Convert input to the 2D array format sklearn expects: [[val]]
    feature_array = np.array([[data.years_experience]])
    
    # Make prediction
    prediction = model.predict(feature_array)
    
    # Return as JSON
    return {"status": "success", "predicted_salary": float(prediction[0])}

