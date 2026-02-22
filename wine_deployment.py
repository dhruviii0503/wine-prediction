import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("dtc_model.pkl")

st.title("Wine Type Prediction")

# User Inputs
fixed_acidity = st.number_input("fixed_acidity")
volatile_acidity = st.number_input("volatile_acidity")
citric_acid = st.number_input("citric_acid")
residual_sugar = st.number_input("residual_sugar")
chlorides = st.number_input("chlorides")
free_sulfur_dioxide = st.number_input("free_sulfur_dioxide")
total_sulfur_dioxide = st.number_input("total_sulfur_dioxide")
density = st.number_input("density")
pH = st.number_input("pH")
sulphates = st.number_input("sulphates")
alcohol = st.number_input("alcohol")
quality = st.number_input("quality")

# Create DataFrame
input_data = pd.DataFrame({
    "fixed_acidity": [fixed_acidity],
    "volatile_acidity": [volatile_acidity],
    "citric_acid": [citric_acid],
    "residual_sugar": [residual_sugar],
    "chlorides": [chlorides],
    "free_sulfur_dioxide": [free_sulfur_dioxide],
    "total_sulfur_dioxide": [total_sulfur_dioxide],
    "density": [density],
    "pH": [pH],
    "sulphates": [sulphates],
    "alcohol": [alcohol],
    "quality": [quality]
})

# Prediction Button
if st.button("Predict"):
    try:
        # Match column order exactly like training
        input_data = input_data[model.feature_names_in_]

        prediction = model.predict(input_data)[0]

        if prediction == "white":
            st.success("White Wine 🍾")
        else:
            st.error("Red Wine 🍷")

    except Exception:
        st.error("Feature mismatch with trained model.")
        st.write("Model expects these features:")
        st.write(model.feature_names_in_)








