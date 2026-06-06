import pandas as pd
import numpy as np
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

logger = logging.getLogger("PreprocessingPipeline")

class MedicalPreprocessingPipeline:
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.preprocessor = None
        self.is_fitted = False
        
    def build_pipeline(self):
        """
        Creates scaling for numeric features and one-hot-encoding for categorical features.
        """
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )
        logger.info("Built column-wise preprocessing transformer pipeline.")
        
    def fit_transform(self, X):
        """
        Fits and transforms dataset X.
        """
        if self.preprocessor is None:
            self.build_pipeline()
        logger.info("Fitting and transforming data through pipeline...")
        X_transformed = self.preprocessor.fit_transform(X)
        self.is_fitted = True
        
        # Get feature names
        feature_names = self._get_feature_names()
        return pd.DataFrame(X_transformed, columns=feature_names), self.preprocessor
        
    def transform(self, X):
        """
        Transforms dataset X using the fitted pipeline.
        """
        if not self.is_fitted:
            raise ValueError("The preprocessing pipeline has not been fitted yet!")
        X_transformed = self.preprocessor.transform(X)
        feature_names = self._get_feature_names()
        return pd.DataFrame(X_transformed, columns=feature_names)
        
    def _get_feature_names(self):
        """
        Helper to extract names from ColumnTransformer.
        """
        names = []
        # Get numerical feature names
        names.extend(self.numeric_cols)
        
        # Get one-hot categorical feature names
        if self.categorical_cols:
            cat_trans = self.preprocessor.named_transformers_['cat']
            if 'onehot' in cat_trans.named_steps:
                encoder = cat_trans.named_steps['onehot']
                cat_features = encoder.get_feature_names_out(self.categorical_cols)
                names.extend(cat_features)
            else:
                names.extend(self.categorical_cols)
            
        return names
        
    def save_pipeline(self, filepath):
        """
        Serializes pipeline object.
        """
        logger.info(f"Saving fitted preprocessing pipeline to {filepath}...")
        joblib.dump(self, filepath)
        
    @staticmethod
    def load_pipeline(filepath):
        """
        Deserializes pipeline object.
        """
        logger.info(f"Loading preprocessing pipeline from {filepath}...")
        return joblib.load(filepath)
