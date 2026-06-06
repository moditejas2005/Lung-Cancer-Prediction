import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("DriftDetector")

def calculate_psi(expected, actual, num_bins=10):
    """
    Calculates the Population Stability Index (PSI) between two 1D series.
    
    PSI < 0.1: No significant change / drift.
    0.1 <= PSI < 0.2: Moderate change.
    PSI >= 0.2: Significant change / drift.
    """
    # Remove nulls
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # Get bins from expected
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    # Ensure unique bin edges
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
        
    # Adjust outer bounds
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    # Calculate counts in bins
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    # Frequencies
    expected_freq = expected_counts / len(expected)
    actual_freq = actual_counts / len(actual)
    
    # Add small value to avoid division by zero
    expected_freq = np.where(expected_freq == 0, 1e-4, expected_freq)
    actual_freq = np.where(actual_freq == 0, 1e-4, actual_freq)
    
    # PSI Formula
    psi_value = np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))
    return psi_value

def detect_drift(df_real, df_synthetic):
    """
    Computes PSI for all continuous columns to audit feature drift between raw and synthetic.
    """
    logger.info("Performing Population Stability Index (PSI) feature drift audits...")
    
    continuous_cols = df_real.select_dtypes(include=[np.number]).columns
    drift_results = {}
    
    for col in continuous_cols:
        if col == "Diagnosis":
            continue
        psi_score = calculate_psi(df_real[col].values, df_synthetic[col].values)
        
        if psi_score < 0.1:
            status = "Minimal (Stable)"
        elif psi_score < 0.2:
            status = "Moderate (Warning)"
        else:
            status = "Significant (Drifted)"
            
        drift_results[col] = {
            "PSI": psi_score,
            "Drift_Status": status
        }
        
    drift_df = pd.DataFrame.from_dict(drift_results, orient="index").reset_index()
    drift_df.columns = ["Feature", "PSI_Score", "Drift_Status"]
    
    logger.info("Feature drift audit completed.")
    return drift_df
