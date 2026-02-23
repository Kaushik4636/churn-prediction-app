import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load the pieces we just made
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

# 2. Simple fix: Only use numbers for now so it doesn't crash
# We will add the "pro" text handling later
X_train_numeric = X_train.select_dtypes(include=['number'])

# 3. Train the Brain
model = RandomForestClassifier()
model.fit(X_train_numeric, y_train.values.ravel())

# 4. Save the Brain
joblib.dump(model, 'model.pkl')
print("!!! SUCCESS: model.pkl is now in your folder !!!")