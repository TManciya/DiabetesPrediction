import streamlit as st
import pickle
import numpy as np
import os

st.title("Welcome to Diabetes Test")
st.header("Please fill in your details")

preg = st.number_input("Enter Pregnancies", 0, 10, 5)
glucose = st.number_input("Enter glucose level", 0, 200)
blood_pressure = st.number_input("Enter blood pressure", 0, 200)
skin = st.number_input("Enter skin thickness", 0, 200)
insulin = st.number_input("Enter insulin level", 0, 200)
bmi = st.number_input("Enter bmi", 0, 200)
pedigreefn = st.number_input("Enter DiabetesPedigreeFunction", 0, 200)
age = st.slider('Select your age:', 0, 100, 25)

# Create a button to predict output
predict_clicked = st.button("Get the prediction")

if predict_clicked:
    try:
        # Use relative path - model.pkl should be in the same directory as your script
        # or adjust the path based on your repository structure
        with open("model.pkl", 'rb') as file:
            model = pickle.load(file)
        
        # Load the test data into numpy array
        data = np.array([[preg, glucose, blood_pressure, skin, insulin, bmi, pedigreefn, age]])
        
        # Call the model to predict
        result = model.predict(data)
        
        if result[0] == 1:
            result_string = "Diabetic"
            st.error("The outcome for your diabetes test is " + result_string)
        else:
            result_string = "Non-Diabetic"
            st.success("The outcome for your diabetes test is " + result_string)
            
    except FileNotFoundError:
        st.error("❌ Model file not found. Please check if 'model.pkl' is in your repository.")
        # Debug: Show what files are available
        st.write("Available files in current directory:")
        try:
            files = os.listdir(".")
            for file in files:
                st.write(f"- {file}")
        except:
            st.write("Could not list directory contents")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


