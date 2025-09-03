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
        # Since your file is in UI folder, you need to go up one level then into Model_Development
        model_path = "../Model_Development/model.pkl"
        
        # Alternative: Use os.path.join for cross-platform compatibility
        # model_path = os.path.join("..", "Model_Development", "model.pkl")
        
        with open(model_path, 'rb') as file:
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
        st.error("❌ Model file not found.")
        # Debug information
        st.write("Current working directory:", os.getcwd())
        st.write("Looking for model at:", os.path.abspath("../Model_Development/model.pkl"))
        
        # Show directory contents for debugging
        st.write("Contents of parent directory:")
        try:
            parent_files = os.listdir("..")
            for item in parent_files:
                st.write(f"- {item}")
                if os.path.isdir(f"../{item}"):
                    st.write(f"  Contents of {item}:")
                    try:
                        sub_files = os.listdir(f"../{item}")
                        for sub_item in sub_files:
                            st.write(f"    - {sub_item}")
                    except:
                        st.write("    Could not read directory")
        except Exception as e:
            st.write(f"Could not list parent directory: {e}")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write(f"Error type: {type(e).__name__}")




