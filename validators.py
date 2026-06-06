"""
validators.py - Input Validation Module
=========================================
This module checks all the patient data submitted through the web form
BEFORE it is passed to the AI model.

Why is validation important?
  - The AI model was trained on data within specific ranges. Feeding it values
    far outside those ranges could produce unreliable or nonsensical predictions.
  - It catches user mistakes like leaving fields blank or entering text in a
    number field.
  - It enforces medical logic (e.g., a non-smoker cannot have 'Years_Smoked' > 0).
"""

# Type hints for cleaner, more readable function signatures
from typing import Dict, List, Tuple, Any


def validate_input_data(input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates all patient input fields from the web form.

    Checks performed:
      1. All required fields are present and not empty
      2. Categorical (text) fields have only allowed values
      3. Numeric fields are within their valid clinical ranges
      4. Binary fields (yes/no) are exactly 0 or 1
      5. Logical consistency across related fields

    Args:
        input_data: Dictionary of all form fields submitted by the user.
                    Example: {'Gender': 'Male', 'Age_Years': '45', ...}

    Returns:
        A tuple: (is_valid, list_of_errors)
          - is_valid  : True if all checks pass, False if any check fails
          - list_of_errors: A list of human-readable error messages (empty if valid)
    """
    errors = []  # Collect all error messages here; return all at once

    # ─────────────────────────────────────────────────────────────────────
    # RULE TABLES: Define the rules for each field
    # ─────────────────────────────────────────────────────────────────────

    # Fields that must be one of a specific set of text values
    ALLOWED_VALUES = {
        'Gender':            ['Male', 'Female'],
        'Smoking_Status':    ['Never', 'Former', 'Current'],
        'Smoking_Intensity': ['None', 'Light', 'Moderate', 'Heavy'],
        'Smoking_Frequency': ['None', 'Occasional', 'Daily', 'Chain Smoker'],
        'Breathlessness':    ['None', 'Mild', 'Moderate', 'High', 'Severe'],
        'Coughing':          ['None', 'Mild', 'Moderate', 'High', 'Severe']
    }

    # Numeric fields and their (minimum, maximum) valid ranges
    # These ranges match the data the model was trained on.
    NUMERIC_RANGES = {
        'Years_Smoked':           (0, 40),
        'Cigarettes_Per_Day':     (0, 40),
        'PM25_Level':             (10, 300),   # Air pollution (μg/m³)
        'Asbestos_Exposure_Index':(0, 150),
        'Oxygen_Percentage':      (80, 100),   # Blood oxygen saturation %
        'BMI_Value':              (15, 40),
        'Smoking_Risk':           (0, float('inf')),  # Must be >= 0, no upper limit
        'Radon_Level_Bq':         (10, 150),   # Radon gas (Becquerels per cubic meter)
        'Age_Years':              (10, 100)
    }

    # Fields that must be exactly 0 (No) or 1 (Yes)
    BINARY_FIELDS = ['Has_Cough', 'Has_Breathlessness']

    # ─────────────────────────────────────────────────────────────────────
    # THE COMPLETE LIST OF REQUIRED FIELDS
    # Every single one of these must be present and valid.
    # ─────────────────────────────────────────────────────────────────────
    required_fields = [
        'Gender', 'Smoking_Status', 'Smoking_Intensity', 'Smoking_Frequency',
        'Years_Smoked', 'Cigarettes_Per_Day', 'PM25_Level',
        'Asbestos_Exposure_Index', 'Oxygen_Percentage', 'BMI_Value',
        'Breathlessness', 'Coughing', 'Smoking_Risk',
        'Has_Cough', 'Has_Breathlessness', 'Radon_Level_Bq', 'Age_Years'
    ]

    # ─────────────────────────────────────────────────────────────────────
    # CHECK 1 & 2 & 3: Loop through every required field and validate it
    # ─────────────────────────────────────────────────────────────────────
    for field in required_fields:

        # CHECK 1: Is the field present and not None/empty?
        if field not in input_data or input_data[field] is None:
            errors.append(f"Missing required field: {field}")
            continue  # Skip further checks for this field since it's missing

        value = input_data[field]

        # CHECK 2: For categorical fields, is the value in the allowed list?
        if field in ALLOWED_VALUES:
            if value not in ALLOWED_VALUES[field]:
                errors.append(
                    f"Invalid value for {field}: '{value}'. "
                    f"Allowed values: {ALLOWED_VALUES[field]}"
                )

        # CHECK 3: For numeric fields, is the value a valid number within range?
        elif field in NUMERIC_RANGES:
            try:
                num_value = float(value)  # Try to convert string to number
                min_val, max_val = NUMERIC_RANGES[field]

                if num_value < min_val:
                    errors.append(
                        f"Value for {field} ({num_value}) is below minimum ({min_val})"
                    )
                if max_val != float('inf') and num_value > max_val:
                    errors.append(
                        f"Value for {field} ({num_value}) is above maximum ({max_val})"
                    )
            except (ValueError, TypeError):
                # The value could not be converted to a number (e.g., it was text)
                errors.append(f"Invalid numeric value for {field}: '{value}'")

        # CHECK 4: For binary fields, is the value exactly 0 or 1?
        elif field in BINARY_FIELDS:
            try:
                bin_value = int(value)
                if bin_value not in [0, 1]:
                    errors.append(
                        f"Invalid binary value for {field}: '{value}'. Must be 0 or 1"
                    )
            except (ValueError, TypeError):
                errors.append(
                    f"Invalid binary value for {field}: '{value}'. Must be 0 or 1"
                )

    # ─────────────────────────────────────────────────────────────────────
    # CHECK 5: LOGICAL CONSISTENCY RULES
    # These rules check that related fields make medical sense together.
    # ─────────────────────────────────────────────────────────────────────

    # Rule: A "Never" smoker cannot have any smoking-related values > 0
    if 'Smoking_Status' in input_data and 'Years_Smoked' in input_data:
        if input_data['Smoking_Status'] == 'Never':

            # A never-smoker cannot have smoked for any years
            if str(input_data['Years_Smoked']) != '0':
                errors.append("Years_Smoked must be 0 when Smoking_Status is 'Never'")

            # A never-smoker cannot smoke any cigarettes per day
            if 'Cigarettes_Per_Day' in input_data and str(input_data['Cigarettes_Per_Day']) != '0':
                errors.append("Cigarettes_Per_Day must be 0 when Smoking_Status is 'Never'")

            # A never-smoker cannot have a smoking intensity
            if 'Smoking_Intensity' in input_data and input_data['Smoking_Intensity'] != 'None':
                errors.append("Smoking_Intensity must be 'None' when Smoking_Status is 'Never'")

            # A never-smoker cannot have a smoking frequency
            if 'Smoking_Frequency' in input_data and input_data['Smoking_Frequency'] != 'None':
                errors.append("Smoking_Frequency must be 'None' when Smoking_Status is 'Never'")

            # A never-smoker must have a smoking risk score of 0
            if 'Smoking_Risk' in input_data:
                try:
                    if float(input_data['Smoking_Risk']) != 0:
                        errors.append("Smoking_Risk must be 0 when Smoking_Status is 'Never'")
                except (ValueError, TypeError):
                    errors.append("Invalid numeric value for Smoking_Risk")

    # Rule: If the patient has no cough (Has_Cough=0), the cough severity must be 'None'
    if 'Has_Cough' in input_data and 'Coughing' in input_data:
        try:
            has_cough = int(input_data['Has_Cough'])
            if has_cough == 0 and input_data['Coughing'] != 'None':
                errors.append("Coughing must be 'None' when Has_Cough is 0")
        except (ValueError, TypeError):
            pass  # This error was already captured in the binary field check above

    # Rule: If the patient has no breathlessness (Has_Breathlessness=0), severity must be 'None'
    if 'Has_Breathlessness' in input_data and 'Breathlessness' in input_data:
        try:
            has_breathlessness = int(input_data['Has_Breathlessness'])
            if has_breathlessness == 0 and input_data['Breathlessness'] != 'None':
                errors.append("Breathlessness must be 'None' when Has_Breathlessness is 0")
        except (ValueError, TypeError):
            pass  # This error was already captured in the binary field check above

    # ─────────────────────────────────────────────────────────────────────
    # RETURN RESULT
    # Returns True (valid) only if zero errors were found.
    # Returns False + the list of errors if any check failed.
    # ─────────────────────────────────────────────────────────────────────
    return len(errors) == 0, errors
