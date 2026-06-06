"""
app.py - Main Flask Web Application
=====================================
This is the entry point of the Lung Cancer Prediction web application.
It handles:
  1. Serving the HTML web pages to the browser
  2. Receiving patient data submitted via the form
  3. Processing and validating that data
  4. Calling the AI model to make a prediction
  5. Sending the result back to the browser
"""

# --- Standard library imports ---
import os                              # Used to read environment variables (e.g., PORT)
from pathlib import Path               # Used to build file paths in a cross-platform way

# --- Flask imports ---
# Flask   : The web framework that runs our server
# render_template : Loads and returns an HTML page
# request  : Lets us access the data submitted by the user via the form
# jsonify  : Converts Python dicts to JSON (not used directly here but imported for extensibility)
from flask import Flask, render_template, request, jsonify

# --- Local module imports ---
from validators import validate_input_data  # Our custom input validation function
from inference import CancerPredictor       # Our AI model wrapper class

import traceback   # Used to print detailed error messages if something goes wrong
import pandas as pd  # Data manipulation library (used in inference.py)


# ─────────────────────────────────────────────────────
# HELPER FUNCTIONS: Convert raw numeric values into
# categorical labels that the AI model understands.
# These categories match what the model was trained on.
# ─────────────────────────────────────────────────────

def categorize_radon(v):
    """
    Converts a raw Radon level (in Becquerels) into a risk category.
    Radon is a radioactive gas that can cause lung cancer.
      < 10  → "None"
      10-49 → "Low"
      50-99 → "Moderate"
      >= 100 → "High"
    """
    if v < 10: return "None"
    if v < 50: return "Low"
    if v < 100: return "Moderate"
    return "High"


def categorize_asbestos(v):
    """
    Converts an Asbestos Exposure Index value into a risk category.
    Asbestos is a toxic mineral linked to lung cancer.
      < 20  → "None"
      20-59 → "Low"
      60-99 → "Moderate"
      >= 100 → "High"
    """
    if v < 20: return "None"
    if v < 60: return "Low"
    if v < 100: return "Moderate"
    return "High"


def categorize_bmi(v):
    """
    Converts a raw BMI (Body Mass Index) number into a health category.
    BMI is a measure of body fat based on height and weight.
      < 18.5 → "Underweight"
      18.5-24.9 → "Normal"
      25-29.9 → "Overweight"
      >= 30 → "Obese"
    """
    if v < 18.5: return "Underweight"
    if v < 25: return "Normal"
    if v < 30: return "Overweight"
    return "Obese"


def categorize_oxygen(v):
    """
    Converts a raw blood Oxygen Saturation percentage into a health category.
    Normal blood oxygen is >= 95%.
      >= 95 → "Normal"
      90-94 → "Slight Drop"
      85-89 → "Low"
      < 85  → "Critical"
    """
    if v >= 95: return "Normal"
    if v >= 90: return "Slight Drop"
    if v >= 85: return "Low"
    return "Critical"


def categorize_pack_years(v):
    """
    Converts a Pack-Years value into a smoking risk category.
    Pack-Years = (cigarettes per day / 20) × years smoked.
    NOTE: This function is defined but not currently used by the model.
      < 5   → "Low"
      5-19  → "Moderate"
      20-39 → "High"
      >= 40 → "Very High"
    """
    if v < 5: return "Low"
    if v < 20: return "Moderate"
    if v < 40: return "High"
    return "Very High"


def categorize_age(age_years):
    """
    Converts an age in years into an age group category.
    The model was trained using these broad age groups instead of exact ages.
      <= 15 → "Young"
      16-30 → "Middle"
      31-50 → "Senior"
      > 50  → "Elder"
    """
    age_years = float(age_years)
    if age_years <= 15:
        return "Young"
    elif age_years <= 30:
        return "Middle"
    elif age_years <= 50:
        return "Senior"
    else:
        return "Elder"


# ─────────────────────────────────────────────────────
# FLASK APP SETUP
# ─────────────────────────────────────────────────────

# Create the Flask application instance
app = Flask(__name__)

# Get the absolute path to the directory where this file (app.py) is located.
# This is used to build correct paths to the model files regardless of where
# the application is run from.
BASE_DIR = Path(__file__).resolve().parent

# Load the trained AI model from disk when the server starts.
# The model is a calibrated stacking ensemble (combination of multiple models).
# The features file tells us the exact order of input columns the model expects.
predictor = CancerPredictor(
    model_path=str(BASE_DIR / 'models' / 'calibrated_stacking_model.joblib'),
    features_path=str(BASE_DIR / 'models' / 'calibrated_stacking_original_features.json')
)


# ─────────────────────────────────────────────────────
# ROUTES: URL endpoints that the browser can visit
# ─────────────────────────────────────────────────────

@app.route('/')
def index():
    """
    HOME PAGE ROUTE
    When a user visits http://localhost:5000/ in their browser,
    this function runs and returns the main patient input form (index.html).
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    PREDICTION ROUTE
    When the user fills in the form and clicks 'Submit',
    the browser sends all the form data here via an HTTP POST request.
    This function:
      1. Reads all the form fields from the request
      2. Validates them (checks for missing/invalid values)
      3. Converts strings to numbers where needed
      4. Derives extra category fields from the raw numeric values
      5. Feeds the processed data into the AI model
      6. Returns the prediction result page (result.html)
    """
    try:
        # ── Step 1: Read all 17 input fields from the HTML form ──
        # request.form.get('FieldName') reads a value submitted in the form.
        # All values arrive as strings by default.
        input_data = {
            'Gender': request.form.get('Gender'),
            'Smoking_Status': request.form.get('Smoking_Status'),
            'Smoking_Intensity': request.form.get('Smoking_Intensity'),
            'Smoking_Frequency': request.form.get('Smoking_Frequency'),
            'Years_Smoked': request.form.get('Years_Smoked'),
            'Cigarettes_Per_Day': request.form.get('Cigarettes_Per_Day'),
            'PM25_Level': request.form.get('PM25_Level'),               # Air pollution level
            'Asbestos_Exposure_Index': request.form.get('Asbestos_Exposure_Index'),
            'Oxygen_Percentage': request.form.get('Oxygen_Percentage'), # Blood oxygen %
            'BMI_Value': request.form.get('BMI_Value'),
            'Breathlessness': request.form.get('Breathlessness'),
            'Coughing': request.form.get('Coughing'),
            'Smoking_Risk': request.form.get('Smoking_Risk'),
            'Age_Years': request.form.get('Age_Years'),                 # Patient's age in years
            'Has_Cough': request.form.get('Has_Cough'),                 # 1 = Yes, 0 = No
            'Has_Breathlessness': request.form.get('Has_Breathlessness'), # 1 = Yes, 0 = No
            'Radon_Level_Bq': request.form.get('Radon_Level_Bq')       # Radon exposure in Becquerels
        }

        # ── Step 2: Validate all inputs using our validators.py module ──
        # This checks for missing fields, out-of-range numbers,
        # and logical contradictions (e.g., smoking fields filled for a non-smoker).
        is_valid, errors = validate_input_data(input_data)
        if not is_valid:
            # If validation fails, reload the form page and show the error messages
            return render_template('index.html', errors=errors, input_data=input_data)

        # ── Step 3: Convert text strings to Python numbers ──
        # The AI model needs actual float numbers, not text strings.
        numeric_fields = [
            'Years_Smoked', 'Cigarettes_Per_Day', 'PM25_Level',
            'Asbestos_Exposure_Index', 'Oxygen_Percentage', 'BMI_Value', 'Smoking_Risk',
            'Radon_Level_Bq', 'Age_Years'
        ]
        for field in numeric_fields:
            input_data[field] = float(input_data[field])  # Convert "25.5" → 25.5

        # Convert yes/no checkboxes (0 or 1) from string to integer
        binary_fields = ['Has_Cough', 'Has_Breathlessness']
        for field in binary_fields:
            input_data[field] = int(input_data[field])  # Convert "1" → 1

        # ── Step 4: Derive additional categorical features ──
        # The model was trained with these derived categories, so we must create them
        # from the raw numeric values the user entered.
        input_data['Radon_Exposure'] = categorize_radon(input_data['Radon_Level_Bq'])
        input_data['Asbestos_Exposure'] = categorize_asbestos(input_data['Asbestos_Exposure_Index'])
        input_data['BMI_Category'] = categorize_bmi(input_data['BMI_Value'])
        input_data['Oxygen_Saturation'] = categorize_oxygen(input_data['Oxygen_Percentage'])
        input_data['Age_Group'] = categorize_age(input_data['Age_Years'])  # e.g., 45 → "Senior"

        # Note: Pack_Years could be computed here but is not needed by this model version.

        # ── Step 5: Run the AI model prediction ──
        # predictor.predict() returns:
        #   prediction  → "Cancer" or "No Cancer"
        #   probability → a float between 0.0 and 1.0 (e.g., 0.87 = 87% cancer risk)
        prediction, probability = predictor.predict(input_data)

        # Convert probability from decimal to a rounded percentage (e.g., 0.873 → 87.3)
        prob_percentage = round(probability * 100, 2)

        # ── Step 6: Show the result page with the prediction ──
        return render_template(
            'result.html',
            prediction=prediction,       # "Cancer" or "No Cancer"
            probability=prob_percentage, # e.g., 87.3
            input_data=input_data        # Pass back the inputs so result.html can display them
        )

    except Exception as e:
        # If anything goes wrong, print the full error to the console for debugging
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        # Show a user-friendly error message on the form page
        error_msg = f"An error occurred during prediction: {str(e)}"
        return render_template('index.html', errors=[error_msg])


# ─────────────────────────────────────────────────────
# START THE SERVER
# ─────────────────────────────────────────────────────
if __name__ == '__main__':
    # Read the PORT from environment variables, default to 5000 for local development.
    # This allows platforms like PythonAnywhere to inject their own port number.
    port = int(os.getenv('PORT', '5000'))
    # host='0.0.0.0' makes the server accessible from other devices on the same network
    # debug=False is safer for production/demo use
    app.run(debug=False, host='0.0.0.0', port=port)
