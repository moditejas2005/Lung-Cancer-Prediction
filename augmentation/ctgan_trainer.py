import os
import torch
import logging
import joblib
from ctgan import CTGAN

logger = logging.getLogger("CtganTrainer")

def train_ctgan(df, categorical_cols, epochs=15, batch_size=500, generator_dim=(256, 256), discriminator_dim=(256, 256), verbose=True):
    """
    Trains a CTGAN model on the raw base dataset to learn joint distributions.
    """
    logger.info("Initializing CTGAN model training...")
    
    # Auto-detect CUDA GPU
    use_gpu = torch.cuda.is_available()
    device = "CUDA GPU" if use_gpu else "CPU"
    logger.info(f"Using device for CTGAN training: {device} (GPU Available: {use_gpu})")
    
    # Initialize CTGAN
    ctgan_model = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        generator_dim=generator_dim,
        discriminator_dim=discriminator_dim,
        verbose=verbose,
        cuda=use_gpu
    )
    
    logger.info(f"Fitting CTGAN on {len(df)} records for {epochs} epochs...")
    ctgan_model.fit(df, discrete_columns=categorical_cols)
    
    logger.info("CTGAN model training successfully completed.")
    return ctgan_model

def save_ctgan_model(model, filepath):
    """
    Saves trained CTGAN model checkpoint.
    """
    logger.info(f"Saving CTGAN model checkpoint to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_ctgan_model(filepath):
    """
    Loads trained CTGAN model checkpoint.
    """
    logger.info(f"Loading CTGAN model checkpoint from {filepath}...")
    return joblib.load(filepath)
