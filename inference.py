"""
inference.py - AI Model Inference Module
==========================================
This module contains the CancerPredictor class which is responsible for:
  1. Loading the pre-trained machine learning model from disk
  2. Preparing/formatting the raw patient data so the model can understand it
  3. Running the prediction and returning the result

The model used is a Calibrated Stacking Ensemble — a combination of multiple
machine learning models (XGBoost, CatBoost, Random Forest, etc.) whose outputs
are combined by a Logistic Regression "meta-learner" to produce a final decision.
"""

# joblib: Used to load the saved AI model file (.joblib format is efficient for sklearn models)
import joblib

# json: Used to load the feature names list from a .json file
import json

# pandas: Used to create a DataFrame (a table) from the patient's input data.
# The AI model requires data in DataFrame format.
import pandas as pd

# Type hints: these make the code easier to read by declaring what types of
# values a function expects and returns.
from typing import Dict, Tuple, Any


class CancerPredictor:
    """
    A wrapper class for the trained cancer prediction model.

    This class handles everything needed to go from raw patient input data
    to a final cancer risk prediction.
    """

    def __init__(self, model_path: str, features_path: str):
        """
        CONSTRUCTOR — runs automatically when CancerPredictor() is called.
        Loads the model and the feature list from disk into memory.

        Args:
            model_path   : Full file path to the saved model (.joblib file)
            features_path: Full file path to the JSON file containing the
                           ordered list of feature (column) names
        """
        self.model_path = model_path

        # Load the model file from disk.
        # The model can be saved in two formats:
        #   Format A: Just the model object directly
        #   Format B: A dictionary containing {'model': ..., 'label_encoders': ...}
        model_data = joblib.load(model_path)

        # Initialize label encoders as empty (will be populated below if present)
        self.label_encoders = {}

        # Handle both saving formats
        if isinstance(model_data, dict):
            # Format B: The file contains a dictionary
            if 'model' in model_data:
                self.model = model_data['model']   # Extract just the model
            else:
                self.model = model_data            # The dict itself IS the model
            # Extract label encoders if they were saved (used for categorical columns)
            self.label_encoders = model_data.get('label_encoders', {})
        else:
            # Format A: The file directly contains the model object
            self.model = model_data

        # Load the ordered list of feature names from the JSON file.
        # This list tells us exactly which columns the model expects,
        # and in what order. Example: ['Age_Group', 'BMI_Category', 'Gender', ...]
        with open(features_path, 'r') as f:
            self.feature_list = json.load(f)

    def prepare_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Converts the raw patient input dictionary into a DataFrame that the
        model can make a prediction on.

        This involves:
          - Encoding categorical (text) values into numbers using Label Encoders
          - Adding any missing columns with a default value of 0
          - Reordering all columns to exactly match the training order

        Args:
            input_data: A Python dictionary of patient values, e.g.:
                        {'Gender': 'Male', 'Age_Years': 45.0, ...}

        Returns:
            A single-row pandas DataFrame with all columns in the correct order.
        """
        # Make a copy so we don't accidentally modify the original input dictionary
        data = input_data.copy()

        # Convert the dictionary into a single-row DataFrame.
        # pd.DataFrame([data]) creates a table with one row from the dict.
        df = pd.DataFrame([data])

        # ── Apply Label Encoding (if the model was saved with encoders) ──
        # Label Encoding converts text categories to numbers.
        # Example: 'Male' → 0, 'Female' → 1 (based on what was used during training)
        if self.label_encoders:
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    # Get the value as a string (ensures compatibility)
                    val_str = str(df[col].iloc[0])

                    if val_str in encoder.classes_:
                        # Value is known → encode it using the trained encoder
                        df[col] = encoder.transform([val_str])[0]
                    else:
                        # Value is unknown (not seen during training) → default to 0
                        df[col] = 0

        # ── Ensure all required columns are present ──
        # If the model needs a column that wasn't provided, add it with value 0
        # (This is a safety measure to prevent "KeyError" crashes)
        for feature in self.feature_list:
            if feature not in df.columns:
                df[feature] = 0

        # ── Select and reorder columns to match training order exactly ──
        # The model is sensitive to column order — it must be EXACTLY the same
        # as when the model was trained.
        df_final = df[self.feature_list]

        return df_final

    def predict(self, input_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        The main prediction function — takes patient data and returns a diagnosis.

        Steps:
          1. Prepares the data using prepare_features()
          2. Calls the model's predict_proba() to get cancer probability
          3. Applies a 0.5 decision threshold to label it "Cancer" or "No Cancer"

        Args:
            input_data: A Python dictionary of patient values (raw + derived fields)

        Returns:
            A tuple: ("Cancer" or "No Cancer", probability_as_float)
            Example: ("Cancer", 0.87) means 87% cancer probability
        """
        # Step 1: Format and encode the data into a model-ready DataFrame
        features_df = self.prepare_features(input_data)

        # Step 2: Get the probability scores for each class.
        # predict_proba() returns an array like [[0.13, 0.87]]
        #   Index 0 → probability of "No Cancer" (class 0)
        #   Index 1 → probability of "Cancer" (class 1)
        probabilities = self.model.predict_proba(features_df)[0]

        # Extract just the cancer probability (the positive class)
        cancer_probability = probabilities[1]

        # Step 3: Apply the decision threshold.
        # If the model is >= 50% confident of cancer, predict "Cancer".
        # A higher threshold (e.g., 0.3) would be more sensitive (catch more cases).
        prediction = "Cancer" if cancer_probability >= 0.5 else "No Cancer"

        return prediction, cancer_probability
