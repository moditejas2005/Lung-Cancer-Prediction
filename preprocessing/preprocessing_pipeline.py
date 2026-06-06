"""
preprocessing_pipeline.py - Data Scaling & Encoding Pipeline
==============================================================
Before feeding data into ML models, we need to standardize it:

  1. StandardScaler for numeric columns:
     - Converts each number to z-score: (value - mean) / std_deviation
     - Result: most values fall between -3 and +3
     - Why? Many models (SVM, Logistic Regression) perform poorly with unscaled data.

  2. OneHotEncoder for categorical columns:
     - Converts text categories to binary columns.
     - E.g., Gender: "Male" → [1, 0], "Female" → [0, 1]

CRITICAL RULE: The scaler must be "fitted" ONLY on training data.
We then use that fitted scaler to transform BOTH train and validation data.
If we fit on all data, validation data "leaks" into training — this is data leakage!

This module wraps sklearn's ColumnTransformer to handle both numeric
and categorical columns in one pipeline object.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib  # Save/load the fitted pipeline to disk

logger = logging.getLogger("PreprocessingPipeline")

class MedicalPreprocessingPipeline:
    """
    A reusable preprocessing pipeline for medical patient data.
    
    Usage:
        pipeline = MedicalPreprocessingPipeline(numeric_cols=[...], categorical_cols=[...])
        X_train_scaled, _ = pipeline.fit_transform(X_train)   # Fit + transform training data
        X_val_scaled = pipeline.transform(X_val)               # Only transform validation data
        pipeline.save_pipeline("pipeline.joblib")              # Save for later use in app.py
    """
    
    def __init__(self, numeric_cols, categorical_cols):
        """
        Initialize with the list of column names to scale/encode.
        
        Args:
            numeric_cols    : List of numeric column names to StandardScale
            categorical_cols: List of text column names to OneHotEncode
        """
        self.numeric_cols = list(numeric_cols)       # Columns to scale
        self.categorical_cols = list(categorical_cols)  # Columns to one-hot encode
        self.preprocessor = None   # Will be set after build_pipeline()
        self.is_fitted = False     # Tracks whether fit_transform() has been called
        
    def build_pipeline(self):
        """
        Creates the ColumnTransformer that applies:
          - StandardScaler   to numeric columns
          - OneHotEncoder    to categorical columns
        """
        # Numeric pipeline: just scaling
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())   # Standardize: (x - mean) / std
        ])
        
        # Categorical pipeline: one-hot encoding
        # handle_unknown='ignore': if a new category appears at inference time, ignore it
        # sparse_output=False: return a regular dense numpy array (not a sparse matrix)
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # ColumnTransformer applies different transformations to different columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_cols),   # Scale these columns
                ('cat', categorical_transformer, self.categorical_cols)  # Encode these columns
            ]
        )
        logger.info("Built column-wise preprocessing transformer pipeline.")
        
    def fit_transform(self, X):
        """
        Fits the pipeline on X (learns mean/std/categories) and transforms X.
        
        IMPORTANT: Call this ONLY on training data.
        
        Args:
            X: Training feature DataFrame
        
        Returns:
            (X_transformed_df, preprocessor): Transformed DataFrame + the fitted preprocessor object
        """
        if self.preprocessor is None:
            self.build_pipeline()  # Build if not already done
        logger.info("Fitting and transforming data through pipeline...")
        X_transformed = self.preprocessor.fit_transform(X)  # Learn + apply transformations
        self.is_fitted = True
        
        # Reconstruct column names (ColumnTransformer loses them during transformation)
        feature_names = self._get_feature_names()
        return pd.DataFrame(X_transformed, columns=feature_names), self.preprocessor
        
    def transform(self, X):
        """
        Transforms X using an ALREADY-FITTED pipeline (does NOT re-learn parameters).
        
        Use this for validation/test data after calling fit_transform() on training data.
        
        Args:
            X: Feature DataFrame to transform
        
        Returns:
            Transformed DataFrame with same columns as the training output
        """
        if not self.is_fitted:
            raise ValueError("The preprocessing pipeline has not been fitted yet!")
        X_transformed = self.preprocessor.transform(X)  # Only apply, don't re-learn
        feature_names = self._get_feature_names()
        return pd.DataFrame(X_transformed, columns=feature_names)
        
    def _get_feature_names(self):
        """
        Internal helper: extracts the output column names from the ColumnTransformer.
        
        Numeric column names stay the same.
        OneHot columns get renamed to: OriginalColumn_CategoryValue
        """
        names = []
        names.extend(self.numeric_cols)  # Numeric columns keep their original names
        
        # Add one-hot encoded column names
        if self.categorical_cols:
            cat_trans = self.preprocessor.named_transformers_['cat']
            if 'onehot' in cat_trans.named_steps:
                encoder = cat_trans.named_steps['onehot']
                # get_feature_names_out creates names like "Gender_Male", "Gender_Female"
                cat_features = encoder.get_feature_names_out(self.categorical_cols)
                names.extend(cat_features)
            else:
                names.extend(self.categorical_cols)
            
        return names
        
    def save_pipeline(self, filepath):
        """
        Saves the fitted pipeline to a .joblib file so it can be reused later
        without re-fitting (e.g., when the web app loads at startup).
        """
        logger.info(f"Saving fitted preprocessing pipeline to {filepath}...")
        joblib.dump(self, filepath)
        
    @staticmethod
    def load_pipeline(filepath):
        """
        Loads a previously saved pipeline from disk.
        The loaded pipeline is already fitted and ready to transform new data.
        """
        logger.info(f"Loading preprocessing pipeline from {filepath}...")
        return joblib.load(filepath)
