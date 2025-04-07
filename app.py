import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")
importance_data = joblib.load("feature_importance.pkl")


# Set page config
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

# Header
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🏦 Loan Approval Prediction App</h1>
    <p style='text-align: center;'>Enter applicant details to check loan eligibility</p>
    <hr>
""", unsafe_allow_html=True)

# Form
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])

    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
        loan_amount = st.number_input("Loan Amount (in Thousands)", min_value=0)
        loan_amount_term = st.number_input("Loan Term (in days)", min_value=0)
        credit_history = st.selectbox("Credit History", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Encode inputs
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = 1 if credit_history == "Yes" else 0
    dependents = 3 if dependents == "3+" else int(dependents)
    property_dict = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_area = property_dict[property_area]

    features = np.array([[gender, married, dependents, education, self_employed,
                          applicant_income, coapplicant_income, loan_amount,
                          loan_amount_term, credit_history, property_area]])

    # Predict
    prediction = model.predict(features)[0]
    
st.markdown("---")
st.subheader("🔍 Feature Influence (Top 5)")

for name, score in importance_data[:5]:
    st.write(f"• **{name}** – importance: `{score:.4f}`")


    # Show result
    if prediction == 1:
        st.success("✅ Loan is likely to be Approved!")
    else:
        st.error("❌ Loan is likely to be Rejected.")
