import streamlit as st
import joblib
import pandas as pd

st.title("📊 Customer Churn Predictor")

# 1. Load the model directly 
model = joblib.load('model.pkl')

# 2. Inputs
senior_citizen = st.selectbox("Senior Citizen (1=Yes, 0=No)", [0, 1])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
total_charges = st.number_input("Total Charges ($)", 0.0, 8000.0, 500.0)

if st.button("Predict"):
    # Match the column names exactly as they were during training
    input_data = pd.DataFrame([[senior_citizen, tenure, monthly_charges, total_charges]], 
                              columns=['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'])
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("Prediction: Will Churn ⚠️")
    else:
        st.success("Prediction: Will Stay ✅")