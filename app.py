import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from validators import validate_input_data
from inference import CancerPredictor
import traceback
import pandas as pd


def categorize_radon(v):
    if v < 10: return "None"
    if v < 50: return "Low"
    if v < 100: return "Moderate"
    return "High"


def categorize_asbestos(v):
    if v < 20: return "None"
    if v < 60: return "Low"
    if v < 100: return "Moderate"
    return "High"


def categorize_bmi(v):
    if v < 18.5: return "Underweight"
    if v < 25: return "Normal"
    if v < 30: return "Overweight"
    return "Obese"


def categorize_oxygen(v):
    if v >= 95: return "Normal"
    if v >= 90: return "Slight Drop"
    if v >= 85: return "Low"
    return "Critical"


def categorize_pack_years(v):
    if v < 5: return "Low"
    if v < 20: return "Moderate"
    if v < 40: return "High"
    return "Very High"


def categorize_age(age_years):
    """Convert age in years to age group category"""
    age_years = float(age_years)
    if age_years <= 15:
        return "Young"
    elif age_years <= 30:
        return "Middle"
    elif age_years <= 50:
        return "Senior"
    else:
        return "Elder"


app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Initialize the cancer predictor
predictor = CancerPredictor(
    model_path=str(BASE_DIR / 'models' / 'calibrated_stacking_model.joblib'),
    features_path=str(BASE_DIR / 'models' / 'calibrated_stacking_original_features.json')
)

@app.route('/')
def index():
    """Render the main input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        input_data = {
            'Gender': request.form.get('Gender'),
            'Smoking_Status': request.form.get('Smoking_Status'),
            'Smoking_Intensity': request.form.get('Smoking_Intensity'),
            'Smoking_Frequency': request.form.get('Smoking_Frequency'),
            'Years_Smoked': request.form.get('Years_Smoked'),
            'Cigarettes_Per_Day': request.form.get('Cigarettes_Per_Day'),
            'PM25_Level': request.form.get('PM25_Level'),
            'Asbestos_Exposure_Index': request.form.get('Asbestos_Exposure_Index'),
            'Oxygen_Percentage': request.form.get('Oxygen_Percentage'),
            'BMI_Value': request.form.get('BMI_Value'),
            'Breathlessness': request.form.get('Breathlessness'),
            'Coughing': request.form.get('Coughing'),
            'Smoking_Risk': request.form.get('Smoking_Risk'),
            'Age_Years': request.form.get('Age_Years'),  # Changed from Age_Group to Age_Years
            'Has_Cough': request.form.get('Has_Cough'),
            'Has_Breathlessness': request.form.get('Has_Breathlessness'),
            'Radon_Level_Bq': request.form.get('Radon_Level_Bq')
        }
        
        # Validate inputs
        is_valid, errors = validate_input_data(input_data)
        if not is_valid:
            return render_template('index.html', errors=errors, input_data=input_data)
        
        # Convert numeric values to appropriate types
        numeric_fields = [
            'Years_Smoked', 'Cigarettes_Per_Day', 'PM25_Level', 
            'Asbestos_Exposure_Index', 'Oxygen_Percentage', 'BMI_Value', 'Smoking_Risk',
            'Radon_Level_Bq', 'Age_Years'
        ]
        
        for field in numeric_fields:
            input_data[field] = float(input_data[field])
        
        binary_fields = ['Has_Cough', 'Has_Breathlessness']
        for field in binary_fields:
            input_data[field] = int(input_data[field])
        
        # Compute derived categorical fields based on numeric inputs
        input_data['Radon_Exposure'] = categorize_radon(input_data['Radon_Level_Bq'])
        input_data['Asbestos_Exposure'] = categorize_asbestos(input_data['Asbestos_Exposure_Index'])
        input_data['BMI_Category'] = categorize_bmi(input_data['BMI_Value'])
        input_data['Oxygen_Saturation'] = categorize_oxygen(input_data['Oxygen_Percentage'])
        input_data['Age_Group'] = categorize_age(input_data['Age_Years'])  # Convert age in years to age group
        
        # Note: Pack_Years is computed but not used as it's not in the model's expected features
        
        # Make prediction
        prediction, probability = predictor.predict(input_data)
        
        # Format probability as percentage
        prob_percentage = round(probability * 100, 2)
        
        return render_template(
            'result.html', 
            prediction=prediction, 
            probability=prob_percentage,
            input_data=input_data
        )
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        error_msg = f"An error occurred during prediction: {str(e)}"
        return render_template('index.html', errors=[error_msg])

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    app.run(debug=False, host='0.0.0.0', port=port)
