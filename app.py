# Updated app.py for Streamlit deployment again - May 2

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model.h5")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("🫀 Predicting Heart Disease Risk")
st.markdown("""
Provide the patient's health data below to estimate the likelihood of coronary heart disease.
""")

# Input fields
sbp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80.0, max_value=250.0, value=120.0)
tobacco = st.number_input("Cumulative Tobacco (kg)", min_value=0.0, max_value=30.0, value=5.0)
ldl = st.number_input("LDL Cholesterol Level", min_value=0.0, max_value=20.0, value=3.0)
adiposity = st.number_input("Adiposity (Fat Index)", min_value=5.0, max_value=50.0, value=25.0)
famhist = st.selectbox("Family History of Heart Disease", ["Absent", "Present"])
typea = st.slider("Type-A Behavior Score", 0, 100, 50)
obesity = st.number_input("Obesity (Fat % Index)", min_value=5.0, max_value=50.0, value=25.0)
alcohol = st.number_input("Current Alcohol Consumption", min_value=0.0, max_value=150.0, value=10.0)
age = st.number_input("Age (years)", min_value=10, max_value=100, value=50)

# Preprocess input
famhist_encoded = 1 if famhist == "Present" else 0

user_input = np.array([[sbp, tobacco, ldl, adiposity, famhist_encoded, typea, obesity, alcohol, age]])

# Predict
if st.button("Predict Heart Disease Risk"):
    prediction = model.predict(user_input)[0][0]
    risk_percentage = round(prediction * 100, 2)

    st.subheader("🧾 Prediction Result")
    st.write(f"Estimated risk of heart disease: **{risk_percentage}%**")

    if risk_percentage > 50:
        st.error("⚠️ High risk. Please consult a medical professional.")
    else:
        st.success("✅ Low risk. Maintain healthy habits!")
