import streamlit as st
import numpy as np
import os

# Try to import required packages with fallback
try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False
    st.warning("Pickle not available")

try:
    import sklearn
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("❌ scikit-learn is not installed. Please check your requirements.txt file.")

st.title("Welcome to Diabetes Test")
st.header("Please fill in your details")

# Input fields
preg = st.number_input("Enter Pregnancies", 0, 10, 0)
glucose = st.number_input("Enter Glucose level", 0, 200, 100)
blood_pressure = st.number_input("Enter Blood Pressure", 0, 200, 70)
skin = st.number_input("Enter Skin Thickness", 0, 100, 20)
insulin = st.number_input("Enter Insulin level", 0, 200, 80)
bmi = st.number_input("Enter BMI", 0.0, 70.0, 25.0)
pedigreefn = st.number_input("Enter Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Select your Age", 0, 100, 25)

# Button to predict
if st.button("Get the prediction"):
    try:
        # Check if sklearn is available
        if not SKLEARN_AVAILABLE:
            st.error("""
            **scikit-learn is not installed.**
            
            Please make sure your requirements.txt file contains:
            ```
            streamlit
            numpy
            pandas
            scikit-learn==1.2.2
            ```
            
            And that you've redeployed your app after updating the file.
            """)
            
            # Show demo result based on inputs
            risk_score = (glucose > 140) * 0.4 + (bmi > 30) * 0.3 + (age > 45) * 0.3
            if risk_score > 0.5:
                st.warning("Demo result: Potentially Diabetic (based on input values)")
            else:
                st.info("Demo result: Likely Non-Diabetic (based on input values)")
                
            return
        
        # If sklearn is available, try to load the model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "model.pkl")

        if not os.path.exists(model_path):
            st.error(f"❌ model.pkl not found at {model_path}. Please check your repository.")
            
            # Create a simple model for demonstration
            st.info("Creating a demonstration model...")
            model = LogisticRegression()
            # Train with dummy data
            X_dummy = np.random.rand(10, 8)
            y_dummy = np.random.randint(0, 2, 10)
            model.fit(X_dummy, y_dummy)
            st.success("Using demonstration model")
        else:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            st.success("✅ Model loaded successfully")

        # Prepare data
        data = np.array([[preg, glucose, blood_pressure, skin, insulin, bmi, pedigreefn, age]])

        # Predict
        result = model.predict(data)

        if result[0] == 1:
            st.error("The outcome for your diabetes test is: Diabetic")
        else:
            st.success("The outcome for your diabetes test is: Non-Diabetic")

    except Exception as e:
        st.error(f"⚠️ An error occurred while making prediction: {e}")

# Debug information
with st.expander("Debug Information"):
    st.write("Python version:", os.sys.version)
    st.write("sklearn available:", SKLEARN_AVAILABLE)
    if SKLEARN_AVAILABLE:
        st.write("sklearn version:", sklearn.__version__)
    
    st.write("Current directory files:", os.listdir('.'))
    
    st.code("""
# Your requirements.txt should contain:
streamlit
numpy
pandas
scikit-learn==1.2.2  # Note the double equals sign
    """)

# Footer
st.markdown("---")
st.caption("Note: This is a demonstration app. For actual medical diagnosis, please consult a healthcare professional.")






