import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")
st.write("Enter your loan application details below:")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
    Loan_Amount_Term = st.selectbox("Loan Term (months)", [360, 120, 180, 240, 300])
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert to numerical
def encode_input():
    gender = 1 if Gender == "Male" else 0
    married = 1 if Married == "Yes" else 0
    dependents = {"0": 0, "1": 1, "2": 2, "3+": 3}[Dependents]
    education = 1 if Education == "Graduate" else 0
    self_employed = 1 if Self_Employed == "Yes" else 0
    property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[Property_Area]

    return [
        gender, married, dependents, education, self_employed,
        ApplicantIncome, CoapplicantIncome, LoanAmount,
        Loan_Amount_Term, Credit_History, property_area
    ]

if st.button("Predict Loan Approval"):
    input_features = encode_input()
    input_data = np.array([input_features])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prediction_text = "✅ Loan is likely to be Approved!" if prediction == 1 else "❌ Loan is likely to be Rejected."

    # Show result
    st.markdown("### Result:")
    if prediction == 1:
        st.success(prediction_text)
    else:
        st.error(prediction_text)

    # Prepare data for download
    result_dict = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area,
        "Prediction": "Approved" if prediction == 1 else "Rejected"
    }
    result_df = pd.DataFrame([result_dict])

    # Download button
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Report (CSV)", csv, file_name="loan_prediction_report.csv", mime="text/csv")
