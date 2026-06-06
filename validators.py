"""Validation module for cancer prediction app"""

from typing import Dict, List, Tuple, Any

def validate_input_data(input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate input data against required constraints.
    
    Args:
        input_data: Dictionary containing input values
        
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    
    # Define allowed categorical values
    ALLOWED_VALUES = {
        'Gender': ['Male', 'Female'],
        'Smoking_Status': ['Never', 'Former', 'Current'],
        'Smoking_Intensity': ['None', 'Light', 'Moderate', 'Heavy'],
        'Smoking_Frequency': ['None', 'Occasional', 'Daily', 'Chain Smoker'],
        'Breathlessness': ['None', 'Mild', 'Moderate', 'High', 'Severe'],
        'Coughing': ['None', 'Mild', 'Moderate', 'High', 'Severe']
    }
    
    # Define numeric ranges
    NUMERIC_RANGES = {
        'Years_Smoked': (0, 40),
        'Cigarettes_Per_Day': (0, 40),
        'PM25_Level': (10, 300),
        'Asbestos_Exposure_Index': (0, 150),
        'Oxygen_Percentage': (80, 100),
        'BMI_Value': (15, 40),
        'Smoking_Risk': (0, float('inf')),  # >= 0
        'Radon_Level_Bq': (10, 150),
        'Age_Years': (10, 100)
    }
    
    # Define binary fields
    BINARY_FIELDS = ['Has_Cough', 'Has_Breathlessness']
    
    # Check all required fields are present
    required_fields = [
        'Gender', 'Smoking_Status', 'Smoking_Intensity', 'Smoking_Frequency',
        'Years_Smoked', 'Cigarettes_Per_Day', 'PM25_Level', 
        'Asbestos_Exposure_Index', 'Oxygen_Percentage', 'BMI_Value',
        'Breathlessness', 'Coughing', 'Smoking_Risk',
        'Has_Cough', 'Has_Breathlessness', 'Radon_Level_Bq', 'Age_Years'
    ]
    
    for field in required_fields:
        if field not in input_data or input_data[field] is None:
            errors.append(f"Missing required field: {field}")
            continue
            
        value = input_data[field]
        
        # Validate categorical fields
        if field in ALLOWED_VALUES:
            if value not in ALLOWED_VALUES[field]:
                errors.append(f"Invalid value for {field}: {value}. Allowed values: {ALLOWED_VALUES[field]}")
        
        # Validate numeric fields
        elif field in NUMERIC_RANGES:
            try:
                num_value = float(value)
                min_val, max_val = NUMERIC_RANGES[field]
                
                if num_value < min_val:
                    errors.append(f"Value for {field} ({num_value}) is below minimum ({min_val})")
                if max_val != float('inf') and num_value > max_val:
                    errors.append(f"Value for {field} ({num_value}) is above maximum ({max_val})")
            except (ValueError, TypeError):
                errors.append(f"Invalid numeric value for {field}: {value}")
        
        # Validate binary fields
        elif field in BINARY_FIELDS:
            try:
                bin_value = int(value)
                if bin_value not in [0, 1]:
                    errors.append(f"Invalid binary value for {field}: {value}. Must be 0 or 1")
            except (ValueError, TypeError):
                errors.append(f"Invalid binary value for {field}: {value}. Must be 0 or 1")
    
    # Check logical consistency rules
    if 'Smoking_Status' in input_data and 'Years_Smoked' in input_data:
        if input_data['Smoking_Status'] == 'Never':
            if str(input_data['Years_Smoked']) != '0':
                errors.append("Years_Smoked must be 0 when Smoking_Status is 'Never'")
            
            if 'Cigarettes_Per_Day' in input_data and str(input_data['Cigarettes_Per_Day']) != '0':
                errors.append("Cigarettes_Per_Day must be 0 when Smoking_Status is 'Never'")
            
            if 'Smoking_Intensity' in input_data and input_data['Smoking_Intensity'] != 'None':
                errors.append("Smoking_Intensity must be 'None' when Smoking_Status is 'Never'")
            
            if 'Smoking_Frequency' in input_data and input_data['Smoking_Frequency'] != 'None':
                errors.append("Smoking_Frequency must be 'None' when Smoking_Status is 'Never'")
            
            if 'Smoking_Risk' in input_data:
                try:
                    if float(input_data['Smoking_Risk']) != 0:
                        errors.append("Smoking_Risk must be 0 when Smoking_Status is 'Never'")
                except (ValueError, TypeError):
                    errors.append("Invalid numeric value for Smoking_Risk")
    
    if 'Has_Cough' in input_data and 'Coughing' in input_data:
        try:
            has_cough = int(input_data['Has_Cough'])
            if has_cough == 0 and input_data['Coughing'] != 'None':
                errors.append("Coughing must be 'None' when Has_Cough is 0")
        except (ValueError, TypeError):
            pass  # Error already caught above
    
    if 'Has_Breathlessness' in input_data and 'Breathlessness' in input_data:
        try:
            has_breathlessness = int(input_data['Has_Breathlessness'])
            if has_breathlessness == 0 and input_data['Breathlessness'] != 'None':
                errors.append("Breathlessness must be 'None' when Has_Breathlessness is 0")
        except (ValueError, TypeError):
            pass  # Error already caught above
    
    return len(errors) == 0, errors
