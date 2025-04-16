import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

# Header
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🏦 Loan Approval Prediction App</h1>
    <p style='text-align: center;'>Enter details to check loan eligibility</p>
    <hr>
""", unsafe_allow_html=True)

# Form
with st.form("loan_form"):
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

    submitted = st.form_submit_button("Predict")

if submitted:
    # Encode inputs
    gender = 1 if Gender == "Male" else 0
    married = 1 if Married == "Yes" else 0
    education = 1 if Education == "Graduate" else 0
    self_employed = 1 if Self_Employed == "Yes" else 0
    credit_history = Credit_History  # Already encoded correctly
    dependents = 3 if Dependents == "3+" else int(Dependents)
    property_dict = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_area = property_dict[Property_Area]

    features = np.array([[gender, married, dependents, education, self_employed,
                          ApplicantIncome, CoapplicantIncome, LoanAmount,
                          Loan_Amount_Term, credit_history, property_area]])

    # Scale features using the loaded scaler
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]

    # Display results
    if prediction == 1:
        st.success("✅ Loan is likely to be Approved!")
        st.markdown("### 😍 Congrats your loan is approved by team 💰?")
        st.write(f"""
            Based on the provided information, your loan is likely to be approved because:
            - **Credit History**: {Credit_History}, a good repayment history.
            - **Education**: {Education} (Graduate), which is favorable for approval.
            - **Property Area**: {Property_Area}, an urban area generally increases chances of approval.
            - **Applicant Income**: {ApplicantIncome}, shows financial stability.
            - **Coapplicant Income**: {CoapplicantIncome}, further supports your financial background.
        """)
    else:
        st.error("❌ Loan is likely to be Rejected.")
        st.markdown("### 😓Why is the loan rejected?")
        st.write(f"""
            Based on the provided information, your loan is likely to be rejected due to one or more of the following reasons:
            - **Credit History**: {Credit_History}, which is unfavorable.
            - **Loan Amount**: {LoanAmount}, might be too high compared to your income.
            - **Applicant Income**: {ApplicantIncome}, might not meet the required threshold for approval.
            - **Dependents**: {Dependents}, additional financial obligations can affect approval chances.
            - **Self-Employed**: {Self_Employed}, which could be a risk factor in some cases.
        """)

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
    st.download_button("📥 Download Your Report as (CSV)", csv, file_name="loan_prediction_report.csv", mime="text/csv")
