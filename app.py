import streamlit as st
import numpy as np

# Main Title
st.title("Medical Insurance Prediction Model")
import joblib  # This might already be in your code

# Load the model
try:
    lg = joblib.load("medical_insurance_model.pkl")  # Ensure that this file exists in your project folder
except FileNotFoundError:
    st.error("Model file not found! Please ensure 'medical_insurance_model.pkl' exists in the project folder.")
    lg = None  # Set lg to None if the model cannot be loaded

# Introduction and Explanation Section
st.subheader("Predict Medical Insurance Cost")
st.write("""
    Welcome to the Medical Insurance Prediction tool! This tool helps you estimate the medical insurance cost based on various personal details.
    
    Please fill in the following details accurately:
    - **Age**: Enter your age in years.
    - **Gender**: Enter 0 for Male and 1 for Female.
    - **BMI**: Enter your Body Mass Index (BMI) value. BMI should typically be between 10.0 and 50.0.
        **Formula**:  
        BMI = Weight (kg) / [Height (m)]² 1 Foot = 0.3048 Meter
""")

# Add a BMI category table
st.write("""
    **BMI Categories**:
    | Category       | BMI Range (kg/m²)         |
    |----------------|---------------------------|
    | Underweight    | Less than 18.5            |
    | Normal weight  | 18.5 – 24.9               |
    | Overweight     | 25.0 – 29.9               |
    | Obesity        | 30.0 and above            |
""")

st.write("""
    - **Children**: Enter the number of children you have.
    - **Smoker**: Enter 0 if you are a smoker, and 1 if you are not.
    - **Region**: Enter your residential region (0=Southeast, 1=Southwest, 2=Northeast, 3=Northwest).
        - **Southeast**: Includes states like Florida, Georgia, Alabama.  
        - **Southwest**: Includes states like Texas, Arizona, New Mexico.  
        - **Northeast**: Includes states like New York, Pennsylvania, Massachusetts.  
        - **Northwest**: Includes states like Washington, Oregon, Idaho.  
""")

# Input fields with integer values directly
try:
    age = int(st.number_input("Age (Example: 30)", min_value=1, max_value=120, value=33, help="Enter your age (e.g., 25, 30)"))
    sex = int(st.selectbox("Gender (0=Male, 1=Female)", [0, 1], help="Enter 0 for Male and 1 for Female"))
    bmi = float(st.number_input("BMI (Example: 22.5)", min_value=10.0, max_value=50.0, value=22.705, help="Enter your BMI value (e.g., 18.5, 25.0)"))
    children = int(st.number_input("Number of Children (Example: 1)", min_value=0, max_value=10, value=0, help="Enter number of children you have (e.g., 0, 1, 2)"))
    smoker = int(st.selectbox("Smoker Status (0=Yes, 1=No)", [0, 1], help="Enter 0 if you are a smoker, 1 if not"))
    region = int(st.selectbox("Region (0=Southeast, 1=Southwest, 2=Northeast, 3=Northwest)", [0, 1, 2, 3], help="Enter region code"))

    # Combine all inputs into an array for prediction
    input_features = np.array([age, sex, bmi, children, smoker, region])

    # Predict button with explanatory text
    if st.button("Predict Insurance Cost"):
        # Predict based on input features
        prediction = lg.predict(input_features.reshape(1, -1))
        st.subheader("Prediction Result")
        st.write("The estimated medical insurance cost for the entered details is: ₹", round(prediction[0], 2), "per year.")
        st.write("""
            **Note**: This is an estimated annual (per year) medical insurance cost based on the input details and may vary. 
            For more accurate information, please consult an insurance provider.
        """)

        # Thank You message after prediction
        st.write("""
            Thank you for using the Medical Insurance Prediction Model! We hope this tool has helped you make an informed decision
            about your health and medical insurance needs. Stay healthy, take care, and remember that your well-being matters the
            most! If you need to check anyone else's medical insurance cost later, feel free to use our web page again. 😊
        """)
except ValueError as e:
    # Display error message if input is invalid
    st.error("Invalid input detected. Please follow the instructions and try again.")
