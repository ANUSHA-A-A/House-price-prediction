

import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("gradient_boost_model.pkl")

st.title("House Price Prediction")

# Input fields
area = st.number_input("Enter Area")

bedrooms = st.number_input("Enter Bedrooms")

bathrooms = st.number_input("Enter Bathrooms")

# Prediction button
if st.button("Predict Price"):

    features = np.array([[area, bedrooms, bathrooms]])

    prediction = model.predict(features)

    st.success(f"Predicted House Price: ₹ {prediction[0]}")

