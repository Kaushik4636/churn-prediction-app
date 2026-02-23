import pandas as pd
import numpy as np

# 1. Load the dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# 2. Basic Cleaning
# TotalCharges has some empty strings. We convert them to numeric and fill with 0.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Drop CustomerID as it has no predictive power
df.drop('customerID', axis=1, inplace=True)

# 3. Label Encoding (Target variable)
# Change 'Yes'/'No' to 1/0
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

print("Data Loaded and Cleaned! Here is a preview:")
print(df.head())