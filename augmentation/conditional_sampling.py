import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("ConditionalSampling")

def sample_conditionally(ctgan_model, conditions, n_samples=1000, max_attempts=10):
    """
    Samples from CTGAN ensuring that specific condition constraints are met.
    Implements rejection sampling for maximum reliability across ctgan package versions.
    
    conditions: dict, e.g., {'Gender': 'Female', 'Diagnosis': 1}
    """
    logger.info(f"Targeting subpopulation conditional sampling with: {conditions}...")
    
    collected = []
    attempts = 0
    total_needed = n_samples
    
    while len(collected) < total_needed and attempts < max_attempts:
        attempts += 1
        # Sample slightly more to account for rejections
        sample_chunk = ctgan_model.sample(int(total_needed * 2.5))
        
        # Filter according to conditions
        filtered_chunk = sample_chunk.copy()
        for col, val in conditions.items():
            if col in filtered_chunk.columns:
                filtered_chunk = filtered_chunk[filtered_chunk[col] == val]
                
        collected.append(filtered_chunk)
        current_len = sum(len(c) for c in collected)
        logger.info(f"Attempt {attempts}/{max_attempts}: collected {current_len}/{total_needed} matched records.")
        
        if current_len >= total_needed:
            break
            
    if not collected:
        logger.warning("Could not match any records under the specified conditions.")
        return pd.DataFrame()
        
    df_result = pd.concat(collected, ignore_index=True).head(total_needed)
    logger.info(f"Conditional sampling successful. Generated {len(df_result)} records.")
    return df_result
