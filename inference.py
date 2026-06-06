"""Inference module for cancer prediction app"""

import joblib
import json
import pandas as pd
from typing import Dict, Tuple, Any

class CancerPredictor:
    def __init__(self, model_path: str, features_path: str):
        """
        Initialize the cancer predictor with a trained model and feature list.
        
        Args:
            model_path: Path to the trained model file
            features_path: Path to the JSON file with ordered feature list
        """
        self.model_path = model_path
        model_data = joblib.load(model_path)
        self.label_encoders = {}

        # Handle different model saving formats
        if isinstance(model_data, dict):
            if 'model' in model_data:
                self.model = model_data['model']
            else:
                self.model = model_data
            self.label_encoders = model_data.get('label_encoders', {})
        else:
            self.model = model_data
            
        with open(features_path, 'r') as f:
            self.feature_list = json.load(f)
    
    def prepare_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare input data for prediction by encoding and ordering features.
        
        Args:
            input_data: Dictionary containing input values
            
        Returns:
            DataFrame with properly encoded and ordered features
        """
        # Create a copy to avoid modifying the original
        data = input_data.copy()
        
        # Create a DataFrame with one row
        df = pd.DataFrame([data])
        
        if self.label_encoders:
            # Apply label encoding to categorical features that were encoded during training
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    # Convert the value to string first (in case it's numeric)
                    val_str = str(df[col].iloc[0])
                    
                    # Check if the value exists in the encoder's classes
                    if val_str in encoder.classes_:
                        # Transform the value using the fitted encoder
                        df[col] = encoder.transform([val_str])[0]
                    else:
                        # If the value is not in the training classes, use the most common class (index 0)
                        # or handle as needed
                        df[col] = 0  # default to first class index
        
        # Ensure all required features are present and in the correct order
        # Add missing columns with default value 0
        for feature in self.feature_list:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select only the features that were used during training and in the correct order
        df_final = df[self.feature_list]
        
        return df_final
    
    def predict(self, input_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Make a prediction on input data.
        
        Args:
            input_data: Dictionary containing input values
            
        Returns:
            Tuple of (prediction_label, probability)
        """
        # Prepare features
        features_df = self.prepare_features(input_data)
        
        # Make prediction
        probabilities = self.model.predict_proba(features_df)[0]
        
        # Get the probability of the positive class (cancer)
        cancer_probability = probabilities[1]  # Index 1 is for cancer class
        
        # Determine prediction based on threshold
        prediction = "Cancer" if cancer_probability >= 0.5 else "No Cancer"
        
        return prediction, cancer_probability
