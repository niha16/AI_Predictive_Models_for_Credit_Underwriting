from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json
import os

# Initialize Flask app
app = Flask(__name__)

# Load the saved pipeline
model_path = "Desktop/Credit_risk_ai/final_best_model.pkl"
final_pipe = joblib.load(model_path)

# File to store input data for future predictions
input_file = "loan_application_data.json"


# Home route for rendering the form (GET) and processing the prediction (POST)
@app.route("/predict", methods=["GET", "POST"])
def index():
    prediction = None
    json_data = None

    # If the form is submitted (POST method)
    if request.method == "POST":
        # Collect input data from the JSON request
        data = request.get_json()

        # Ensure all required fields are provided
        required_fields = [
            "person_age", "person_income", "person_home_ownership", "person_emp_length",
            "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_percent_income",
            "cb_person_default_on_file", "cb_person_cred_hist_length"
        ]

        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        # Prepare input data as a DataFrame for prediction
        input_data = pd.DataFrame([data])

        # Save input data to JSON for future predictions
        input_data_dict = data
        if os.path.exists(input_file):
            with open(input_file, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        
        # Append the new input data and save back
        existing_data.append(input_data_dict)
        with open(input_file, "w") as f:
            json.dump(existing_data, f, indent=4)

        # Make prediction
        prediction_value = final_pipe.predict(input_data)[0]
        prediction = "Loan Approved!" if prediction_value == 1 else "Loan Denied!"

        # Pass the input JSON and prediction to the response
        response = {
            "prediction": prediction,
            "input_data": input_data_dict
        }

        return jsonify(response)

    # If GET request, simply render the form page
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)