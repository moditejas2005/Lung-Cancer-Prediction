"""
drift_detector.py - Feature Drift Detection Module
=====================================================
This module checks whether the synthetic data "drifted away" from the real data.

What is data drift?
  If the CTGAN generates patients with very different age distributions or smoking
  patterns compared to the original dataset, that is called "drift". Drifted data
  would train a biased, unreliable model.

How do we measure drift?
  We use PSI = Population Stability Index — a number that measures how much
  the distribution of a feature changed between real and synthetic data.
  
  PSI < 0.1  → Stable (no drift) ✓
  PSI < 0.2  → Moderate drift (warning) ⚠
  PSI >= 0.2 → Significant drift (problem) ✗
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("DriftDetector")

def calculate_psi(expected, actual, num_bins=10):
    """
    Calculates the Population Stability Index (PSI) between two 1D data series.

    PSI measures how much the distribution of 'actual' differs from 'expected'.
    Think of it like: "Did the synthetic data follow the same statistical pattern
    as the real data for this feature?"

    Args:
        expected  : The 'real' data values (our ground truth)
        actual    : The 'synthetic' data values (what we want to compare)
        num_bins  : How many equal-frequency bins to split the data into

    Returns:
        A single PSI float value (higher = more drift)
    """
    # Remove NaN (missing) values before calculating
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    # Can't calculate PSI on empty data
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # Create equal-frequency bins based on the expected (real) data
    percentiles = np.linspace(0, 100, num_bins + 1)  # e.g., [0, 10, 20, ... 100]
    bins = np.percentile(expected, percentiles)         # The actual bin boundary values
    bins = np.unique(bins)  # Remove duplicate bin edges (can happen with discrete data)
    if len(bins) < 2:
        return 0.0  # Not enough bin variety — return 0 (no drift measurable)
        
    # Set outer bounds to infinity to capture all values including extremes
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    # Count how many values fall in each bin for both datasets
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    # Convert counts to proportions (percentages of total)
    expected_freq = expected_counts / len(expected)
    actual_freq = actual_counts / len(actual)
    
    # Avoid log(0) errors by replacing zero proportions with a tiny value
    expected_freq = np.where(expected_freq == 0, 1e-4, expected_freq)
    actual_freq = np.where(actual_freq == 0, 1e-4, actual_freq)
    
    # PSI Formula: sum of (actual - expected) * log(actual / expected)
    psi_value = np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))
    return psi_value

def detect_drift(df_real, df_synthetic):
    """
    Runs PSI drift detection on every numeric column in the dataset.
    Compares the real patient data against the CTGAN-generated synthetic data.

    Args:
        df_real      : The cleaned real patient dataset (pandas DataFrame)
        df_synthetic : The cleaned synthetic patient dataset (pandas DataFrame)

    Returns:
        A DataFrame with columns: Feature, PSI_Score, Drift_Status
        One row per numeric feature.
    """
    logger.info("Performing Population Stability Index (PSI) feature drift audits...")
    
    # Get all numeric columns from the real data
    continuous_cols = df_real.select_dtypes(include=[np.number]).columns
    drift_results = {}  # Will store {column_name: {PSI: ..., Drift_Status: ...}}
    
    for col in continuous_cols:
        if col == "Diagnosis":
            continue  # Skip the target column — we only audit input features
            
        # Calculate PSI for this column
        psi_score = calculate_psi(df_real[col].values, df_synthetic[col].values)
        
        # Classify drift severity based on PSI value
        if psi_score < 0.1:
            status = "Minimal (Stable)"      # Good — synthetic matches real
        elif psi_score < 0.2:
            status = "Moderate (Warning)"    # Some drift — worth monitoring
        else:
            status = "Significant (Drifted)" # Problem — synthetic drifted far from real
            
        drift_results[col] = {
            "PSI": psi_score,
            "Drift_Status": status
        }
        
    # Convert the results dictionary into a neat DataFrame table
    drift_df = pd.DataFrame.from_dict(drift_results, orient="index").reset_index()
    drift_df.columns = ["Feature", "PSI_Score", "Drift_Status"]
    
    logger.info("Feature drift audit completed.")
    return drift_df
