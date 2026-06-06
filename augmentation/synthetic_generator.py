"""
synthetic_generator.py - Synthetic Data Generator
====================================================
A simple utility module that:
  1. Loads a previously trained CTGAN model from disk
  2. Uses it to sample (generate) a specified number of synthetic patient records
  3. Returns the records as a pandas DataFrame

This is the "production" interface for generating synthetic data.
It is a thin wrapper around the CTGAN model's .sample() method.
"""

import os
import pandas as pd
import logging
from augmentation.ctgan_trainer import load_ctgan_model  # Function to load saved CTGAN

logger = logging.getLogger("SyntheticGenerator")

def generate_synthetic_data(model_path, n_samples=50000):
    """
    Loads a saved CTGAN model and generates n_samples synthetic patient records.

    Args:
        model_path : Path to the saved CTGAN .joblib file
        n_samples  : Number of synthetic records to generate (default 50,000)

    Returns:
        synthetic_df: A pandas DataFrame of n_samples rows of synthetic patient data

    Raises:
        FileNotFoundError: If the CTGAN model file doesn't exist at model_path
    """
    logger.info(f"Loading CTGAN model checkpoint from {model_path}...")
    
    # Check the model file exists before trying to load it
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CTGAN model file not found at {model_path}!")
        
    # Load the trained CTGAN model from disk
    model = load_ctgan_model(model_path)
    
    logger.info(f"Sampling {n_samples} high-fidelity synthetic patient records from CTGAN...")
    
    # Generate synthetic patient records by sampling from the learned distribution
    synthetic_df = model.sample(n_samples)
    
    logger.info(f"Successfully generated synthetic dataset of size {synthetic_df.shape}.")
    return synthetic_df
