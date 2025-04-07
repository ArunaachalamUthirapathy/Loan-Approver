import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
file_path = "loan_data.csv"
df = pd.read_csv(file_path)

# Encode categorical features
le = LabelEncoder()
categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Target encoding
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Split features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Calculate feature importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# Save feature importance values
importances = result.importances_mean
feature_names = X.columns

importance_data = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

# Save for use in app
joblib.dump(importance_data, "feature_importance.pkl")


# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved.")
