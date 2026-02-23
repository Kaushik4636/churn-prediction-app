from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# 1. Create the App
app = FastAPI()

# 2. Load the "Brain" (model.pkl)
model = joblib.load('model.pkl')

# 3. Define what the input data should look like
# For now, we only use the numeric columns we trained on
# In main.py
class CustomerData(BaseModel):
    SeniorCitizen: int  # Add this line!
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {"message": "Churn Prediction API is Running!"}

@app.post("/predict")
def predict(data: CustomerData):
    # Convert input data to a DataFrame for the model
    input_df = pd.DataFrame([data.dict()])
    
    # Get prediction (0 or 1)
    prediction = model.predict(input_df)[0]
    
    # Return result
    status = "Will Churn" if prediction == 1 else "Will Stay"
    return {"prediction": status}