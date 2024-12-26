import streamlit as st
import pandas as pd
import joblib
import json
import os

# Load the saved pipeline
final_pipe = joblib.load("Desktop/Credit_risk_ai/final_best_model.pkl")

# File to store input data for future predictions
input_file = "loan_application_data.json"

# App Header
st.title("Applicant Information Input Form")

# Create input fields in the sidebar for user input
st.sidebar.header("Applicant Information")
person_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=25, step=1)
person_income = st.sidebar.number_input("Income (in USD)", min_value=4000, value=50000, step=1000)
person_home_ownership = st.sidebar.selectbox("Home Ownership", ["OWN", "RENT", "MORTGAGE", "OTHER"])
person_emp_length = st.sidebar.slider("Employment Length (in years)", 0, 70, 5)  # Employment Length with scrollbar
loan_intent = st.sidebar.selectbox("Loan Intent", ["PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION", "MEDICAL", "VENTURE", "EDUCATION"])
loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input("Loan Amount (in USD)", min_value=500,max_value=35000, value=10000, step=100)
loan_int_rate = st.sidebar.slider("Interest Rate (in %)", 5.0, 100.0, 10.5, 0.1)  # Interest rate with scrollbar
loan_percent_income = st.sidebar.slider("Loan Percentage of Income", 0.0, 1.0, 0.2, 0.01)  # Loan Percentage with scrollbar
cb_person_default_on_file = st.sidebar.selectbox("Default on File", ["Y", "N"])
cb_person_cred_hist_length = st.sidebar.slider("Credit History Length (in years)", 0, 50, 10)  # Credit History Length with scrollbar

# Button to predict loan eligibility and save the input data
if st.sidebar.button("Submit"):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        "person_age": [person_age],
        "person_income": [person_income],
        "person_home_ownership": [person_home_ownership],
        "person_emp_length": [person_emp_length],
        "loan_intent": [loan_intent],
        "loan_grade": [loan_grade],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_default_on_file": [cb_person_default_on_file],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length],
    })
    
    # Display the input data in JSON format in the main area above the result
    st.header("Submitted Loan Applicant Details:")
    st.json(input_data.to_dict(orient="records")[0])

    # Save the input data as a JSON file
    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    
    # Append the new input data to the existing data
    existing_data.append(input_data.to_dict(orient="records")[0])
    
    # Write the updated data back to the JSON file
    with open(input_file, 'w') as f:
        json.dump(existing_data, f, indent=4)
    
    # Make prediction using the saved pipeline
    prediction = final_pipe.predict(input_data)[0]
    
    # Display the result below the JSON data in the main area
    st.header("Prediction Result")
    if prediction == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Denied!")
else:
    st.info("Please complete the form and click Submit.")