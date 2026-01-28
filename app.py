import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

@st.cache_resource
def load_artifacts():
    model = keras.models.load_model("models/churn_ann_final.keras")
    preprocessor = joblib.load("models/preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_artifacts()

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn")
with st.form("churn_form"):
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0)

    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    senior = st.selectbox("Senior Citizen", [0, 1])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    online_security = st.selectbox("Online Security", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

    submit = st.form_submit_button("Predict Churn")

if submit:
    input_df = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'PaperlessBilling': paperless,
        'SeniorCitizen': senior,
        'Contract': contract,
        'InternetService': internet,
        'PaymentMethod': payment,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies
    }])

    processed_input = preprocessor.transform(input_df)
    prob = model.predict(processed_input)[0][0]
    churn_pred = int(prob > 0.35)

    st.subheader("🔍 Prediction Result")
    st.write(f"**Churn Probability:** {prob:.2f}")

    if churn_pred == 1:
        st.error("⚠️ Customer is likely to churn")
        st.write("👉 Recommended action: Offer retention incentives")
    else:
        st.success("✅ Customer is likely to stay")
