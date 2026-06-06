"""
conditional_sampling.py - Targeted Synthetic Data Sampler
===========================================================
Sometimes we need synthetic data for a specific subgroup of patients.
For example: "Generate 500 synthetic Female cancer patients."

The problem: CTGAN's .sample() generates random data matching the overall distribution.
It doesn't guarantee that exactly N records will match a specific condition.

Solution: REJECTION SAMPLING
  1. Ask CTGAN for 2.5× more records than needed
  2. Filter to keep only records matching our conditions
  3. Repeat until we have enough records (up to max_attempts times)

This is called "rejection sampling" — we generate many and reject those that don't fit.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("ConditionalSampling")

def sample_conditionally(ctgan_model, conditions, n_samples=1000, max_attempts=10):
    """
    Generates synthetic patient records that match specific conditions using rejection sampling.

    Args:
        ctgan_model  : A trained CTGAN model object
        conditions   : Dict of conditions to filter by, e.g., {'Gender': 'Female', 'Diagnosis': 1}
        n_samples    : How many matching records we need
        max_attempts : Maximum number of sampling rounds before giving up

    Returns:
        A pandas DataFrame with exactly n_samples rows (or fewer if max_attempts reached)

    Example:
        # Generate 500 female cancer patients
        df = sample_conditionally(ctgan_model, {'Gender': 'Female', 'Diagnosis': 1}, n_samples=500)
    """
    logger.info(f"Targeting subpopulation conditional sampling with: {conditions}...")
    
    collected = []   # List of DataFrames collected across attempts
    attempts = 0
    total_needed = n_samples
    
    while len(collected) < total_needed and attempts < max_attempts:
        attempts += 1
        
        # Sample 2.5× more than needed to account for records that don't match conditions
        sample_chunk = ctgan_model.sample(int(total_needed * 2.5))
        
        # Filter: keep only rows where all conditions are satisfied
        filtered_chunk = sample_chunk.copy()
        for col, val in conditions.items():
            if col in filtered_chunk.columns:
                filtered_chunk = filtered_chunk[filtered_chunk[col] == val]  # Keep only matching rows
                
        collected.append(filtered_chunk)
        
        # Count total collected records so far
        current_len = sum(len(c) for c in collected)
        logger.info(f"Attempt {attempts}/{max_attempts}: collected {current_len}/{total_needed} matched records.")
        
        if current_len >= total_needed:
            break  # We have enough records — stop early
            
    if not collected:
        logger.warning("Could not match any records under the specified conditions.")
        return pd.DataFrame()  # Return empty DataFrame if nothing collected
        
    # Combine all collected chunks and take exactly n_samples rows
    df_result = pd.concat(collected, ignore_index=True).head(total_needed)
    logger.info(f"Conditional sampling successful. Generated {len(df_result)} records.")
    return df_result
