import streamlit as st
import pickle
import numpy as np
import os

# Explicitly import sklearn first
try:
    import sklearn
    import sklearn.ensemble  # This might be needed for your model
    import sklearn.linear_model
    import sklearn.tree
    st.write(f"✅ scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    st.error(f"❌ Failed to import scikit-learn: {e}")
    st.error("Please make sure 'scikit-learn' is in your requirements.txt file")
    st.stop()

st.title("Welcome to Diabetes Test")
st.header("Please fill in your details")

# Input fields
preg = st.number_input("Enter Pregnancies", 0, 10, 5)
glucose = st.number_input("Enter Glucose level", 0, 200, 120)
blood_pressure = st.number_input("Enter Blood Pressure", 0, 200, 80)
skin = st.number_input("Enter Skin Thickness", 0, 200, 20)
insulin = st.number_input("Enter Insulin level", 0, 200, 80)
bmi = st.number_input("Enter BMI", 0.0, 50.0, 25.0)
pedigreefn = st.number_input("Enter Diabetes Pedigree Function", 0.0, 5.0, 0.5)
age = st.slider("Select your Age", 0, 100, 25)

# Button to predict
if st.button("Get the prediction"):
    try:
        # Show current working directory and files
        current_dir = os.getcwd()
        st.write(f"Current directory: {current_dir}")
        
        # List all files in current directory
        files = os.listdir(current_dir)
        st.write(f"Files in directory: {files}")
        
        # Try different possible paths for the model
        possible_paths = [
            "model.pkl",
            "./model.pkl", 
            os.path.join(current_dir, "model.pkl")
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            st.error("❌ model.pkl not found in any of the expected locations:")
            for path in possible_paths:
                st.write(f"- {path}")
            st.stop()
        
        st.write(f"Loading model from: {model_path}")
        
        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        st.write(f"✅ Model loaded successfully: {type(model)}")
        
        # Prepare data
        data = np.array([[preg, glucose, blood_pressure, skin, insulin, bmi, pedigreefn, age]])
        st.write(f"Input data shape: {data.shape}")
        st.write(f"Input data: {data}")
        
        # Predict
        result = model.predict(data)
        probability = None
        
        # Try to get prediction probability if available
        try:
            probability = model.predict_proba(data)
            st.write(f"Prediction probabilities: {probability}")
        except:
            pass
        
        st.write(f"Prediction result: {result}")
        
        if result[0] == 1:
            st.error("🔴 The outcome for your diabetes test is: **Diabetic**")
            if probability is not None:
                confidence = probability[0][1] * 100
                st.write(f"Confidence: {confidence:.1f}%")
        else:
            st.success("🟢 The outcome for your diabetes test is: **Non-Diabetic**")
            if probability is not None:
                confidence = probability[0][0] * 100
                st.write(f"Confidence: {confidence:.1f}%")
            
    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        
        # Show detailed traceback for debugging
        import traceback
        st.text("Full traceback:")
        st.text(traceback.format_exc())








