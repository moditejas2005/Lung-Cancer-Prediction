"""
ctgan_trainer.py - CTGAN Model Training Module
================================================
CTGAN = Conditional Tabular GAN (Generative Adversarial Network for table data).

What is a GAN?
  A GAN has two neural networks that compete with each other:
    - Generator : Creates fake patient records
    - Discriminator: Tries to tell real records from fake ones
  
  Over time, the Generator gets so good that the Discriminator can't tell the
  difference — meaning the fake data looks statistically real.

Why CTGAN specifically?
  Standard GANs work on images. CTGAN is specially designed for tabular data
  (tables with mixed numeric + categorical columns), making it ideal for
  generating synthetic patient records.

This module handles:
  - train_ctgan()  : Trains a new CTGAN model from scratch on the dataset
  - save_ctgan_model(): Saves the trained CTGAN to a file
  - load_ctgan_model(): Loads a previously trained CTGAN from a file
"""

import os
import torch    # PyTorch — CTGAN uses this deep learning library internally
import logging
import joblib   # For saving/loading the trained CTGAN object
from ctgan import CTGAN  # The CTGAN library

logger = logging.getLogger("CtganTrainer")

def train_ctgan(df, categorical_cols, epochs=15, batch_size=500, generator_dim=(256, 256), discriminator_dim=(256, 256), verbose=True):
    """
    Trains a CTGAN model on the given dataset.

    Args:
        df              : The cleaned training DataFrame to learn from
        categorical_cols: List of column names that are categorical (text, not numbers)
        epochs          : Number of training iterations (more = better quality but slower)
        batch_size      : How many records to process at once during training
        generator_dim   : Neural network architecture for the Generator (256 units × 2 layers)
        discriminator_dim: Neural network architecture for the Discriminator
        verbose         : If True, print training progress to the console

    Returns:
        ctgan_model: A fully trained CTGAN object ready to generate synthetic data
    """
    logger.info("Initializing CTGAN model training...")
    
    # Check if a CUDA GPU is available — CTGAN trains MUCH faster on GPU
    use_gpu = torch.cuda.is_available()
    device = "CUDA GPU" if use_gpu else "CPU"
    logger.info(f"Using device for CTGAN training: {device} (GPU Available: {use_gpu})")
    
    # Create the CTGAN model with the specified architecture
    ctgan_model = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        generator_dim=generator_dim,          # Generator network size
        discriminator_dim=discriminator_dim,  # Discriminator network size
        verbose=verbose,                       # Show training progress
        cuda=use_gpu                           # Use GPU if available
    )
    
    logger.info(f"Fitting CTGAN on {len(df)} records for {epochs} epochs...")
    # "fit" trains the GAN: Generator and Discriminator compete for `epochs` rounds
    # discrete_columns tells CTGAN which columns are categorical (not continuous numbers)
    ctgan_model.fit(df, discrete_columns=categorical_cols)
    
    logger.info("CTGAN model training successfully completed.")
    return ctgan_model

def save_ctgan_model(model, filepath):
    """
    Saves the trained CTGAN model to a .joblib file so it can be loaded later
    without re-training (training takes many minutes).
    """
    logger.info(f"Saving CTGAN model checkpoint to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Create folder if needed
    joblib.dump(model, filepath)  # Serialize and write the model object to disk

def load_ctgan_model(filepath):
    """
    Loads a previously trained and saved CTGAN model from disk.
    Use this to generate more synthetic data without re-training.
    """
    logger.info(f"Loading CTGAN model checkpoint from {filepath}...")
    return joblib.load(filepath)  # Deserialize and return the model object
