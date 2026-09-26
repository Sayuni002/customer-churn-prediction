"""
Customer Churn Prediction Web Application

This Streamlit application allows users to enter customer
information and estimate customer churn risk using a trained
machine learning model.
"""

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
model_columns = joblib.load("model/model_columns.pkl")
numerical_columns = joblib.load("model/numerical_columns.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Churn Prediction")

st.write(
    "Enter the customer's information below to estimate their "
    "risk of churn."
)

st.subheader("Customer Information")

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

st.subheader("Services")

phone_service = st.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

st.subheader("Account Information")

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)



import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
model_columns = joblib.load("model/model_columns.pkl")
numerical_columns = joblib.load("model/numerical_columns.pkl")



predict_button = st.button("Predict Churn")

if predict_button:
    customer_data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([customer_data])

    
    input_encoded = pd.DataFrame(
    0,
    index=[0],
    columns=model_columns,
    dtype=float
)

# Add numerical values
    for col in numerical_columns:
       if col in customer_data:
        input_encoded.at[0, col] = customer_data[col]

# Encode categorical values
    for col, value in customer_data.items():
       if col not in numerical_columns:
        dummy_column = f"{col}_{value}"

        if dummy_column in model_columns:
            input_encoded.at[0, dummy_column] = 1

    
    input_encoded[numerical_columns] = scaler.transform(
    input_encoded[numerical_columns]
    )

    prediction = model.predict(input_encoded)[0]

    probability = model.predict_proba(input_encoded)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is predicted to CHURN")
    else:
        st.success("✅ Customer is predicted to NOT CHURN")

    st.write(
        f"Estimated Churn Probability: **{probability * 100:.2f}%**"
    )

    st.progress(float(probability))
st.divider()

st.caption(
    "Model: Tuned Random Forest | "
    "This application is a machine learning portfolio project."
)