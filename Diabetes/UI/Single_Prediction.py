import streamlit as st
import pickle
import numpy as np
import os

st.title("Welcome to Diabetes Test")
st.header("Please fill in your details")

# Input fields
preg = st.number_input("Enter Pregnancies", 0, 10, 5)
glucose = st.number_input("Enter Glucose level", 0, 200)
blood_pressure = st.number_input("Enter Blood Pressure", 0, 200)
skin = st.number_input("Enter Skin Thickness", 0, 200)
insulin = st.number_input("Enter Insulin level", 0, 200)
bmi = st.number_input("Enter BMI", 0, 200)
pedigreefn = st.number_input("Enter Diabetes Pedigree Function", 0, 200)
age = st.slider("Select your Age", 0, 100, 25)

# Button to predict
if st.button("Get the prediction"):
    try:
        # Always resolve path relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "model.pkl")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Prepare data
        data = np.array([[preg, glucose, blood_pressure, skin, insulin, bmi, pedigreefn, age]])

        # Predict
        result = model.predict(data)

        if result[0] == 1:
            st.error("The outcome for your diabetes test is: Diabetic")
        else:
            st.success("The outcome for your diabetes test is: Non-Diabetic")

    except FileNotFoundError:
        st.error(f"❌ model.pkl not found at {model_path}. Please check your repository.")
    except Exception as e:
        st.error(f"⚠️ An error occurred while making prediction: {e}")






