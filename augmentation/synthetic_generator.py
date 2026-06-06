import os
import pandas as pd
import logging
from synthetic_medical_ai.augmentation.ctgan_trainer import load_ctgan_model

logger = logging.getLogger("SyntheticGenerator")

def generate_synthetic_data(model_path, n_samples=50000):
    """
    Loads trained CTGAN and samples n_samples synthetic patient records.
    """
    logger.info(f"Loading CTGAN model checkpoint from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CTGAN model file not found at {model_path}!")
        
    model = load_ctgan_model(model_path)
    
    logger.info(f"Sampling {n_samples} high-fidelity synthetic patient records from CTGAN...")
    synthetic_df = model.sample(n_samples)
    
    logger.info(f"Successfully generated synthetic dataset of size {synthetic_df.shape}.")
    return synthetic_df
