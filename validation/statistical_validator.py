"""
statistical_validator.py - Statistical Distribution Similarity Checker
========================================================================
After CTGAN generates synthetic patient data, we need to confirm that
the synthetic data follows the same statistical distributions as the
original real data.

This module uses two types of statistical tests:

  For NUMERIC columns (e.g., Age, BMI, Pack_Years):
    → KS Test (Kolmogorov-Smirnov Test): Checks if two distributions are the same.
      p-value > 0.05 means we cannot prove they are different (good!).
    → Wasserstein Distance: Measures how "far apart" the two distributions are.
      0 = identical, higher = more different.

  For CATEGORICAL columns (e.g., Gender, Smoking_Status):
    → Chi-Square Test: Checks if the category proportions are the same.
      p-value > 0.05 means proportions are statistically similar (good!).
"""

import pandas as pd
import numpy as np
import logging
from scipy.stats import ks_2samp, chi2_contingency    # Statistical test functions
from scipy.stats import wasserstein_distance            # Earth mover's distance

logger = logging.getLogger("StatisticalValidator")

def validate_distributions(df_real, df_synthetic):
    """
    Runs statistical similarity tests on every column comparing real vs synthetic data.

    Args:
        df_real      : The original real patient dataset (pandas DataFrame)
        df_synthetic : The CTGAN-generated synthetic patient dataset (pandas DataFrame)

    Returns:
        A DataFrame with one row per feature containing:
          Feature, Type, Test_Statistic, P_Value, Wasserstein_Distance, Passed_Statistical_Identical
    """
    logger.info("Conducting statistical similarity analysis...")
    
    results = []  # Collect one result dict per column
    
    # ── Numeric Columns: KS Test + Wasserstein Distance ──
    # Find columns that are numeric in BOTH real and synthetic datasets
    real_numeric = df_real.select_dtypes(include=[np.number]).columns
    synth_numeric = df_synthetic.select_dtypes(include=[np.number]).columns
    common_numeric = list(set(real_numeric) & set(synth_numeric))  # Only columns in both
    
    for col in common_numeric:
        if col == "Diagnosis":
            continue  # Skip the target variable
            
        # KS Test: Compares the cumulative distribution functions of two samples.
        # Returns: ks_stat (test statistic), ks_pval (p-value)
        ks_stat, ks_pval = ks_2samp(df_real[col], df_synthetic[col])
        
        # Wasserstein Distance: How much "work" is needed to transform one distribution into the other
        w_dist = wasserstein_distance(df_real[col], df_synthetic[col])
        
        # p > 0.05 means we can't prove the distributions are different → they're statistically similar
        passed = ks_pval > 0.05
        
        results.append({
            "Feature": col,
            "Type": "Numerical",
            "Test_Statistic": ks_stat,
            "P_Value": ks_pval,
            "Wasserstein_Distance": w_dist,
            "Passed_Statistical_Identical": passed
        })
        
    # ── Categorical Columns: Chi-Square Test ──
    # Find columns that are text/category type in BOTH datasets
    real_cat = df_real.select_dtypes(include=["object", "category"]).columns
    synth_cat = df_synthetic.select_dtypes(include=["object", "category"]).columns
    common_cat = list(set(real_cat) & set(synth_cat))  # Only columns in both
    
    for col in common_cat:
        # Build a contingency table: rows = Real/Synthetic, columns = each category value
        real_counts = df_real[col].value_counts().sort_index()
        synth_counts = df_synthetic[col].value_counts().sort_index()
        
        # Align both series on the same index (union of all category values)
        combined = pd.DataFrame({"Real": real_counts, "Synthetic": synth_counts}).fillna(0)
        
        try:
            # Chi-Square Test: Tests if the category proportions match
            chi2, p_val, dof, ex = chi2_contingency(combined.values.T)
            passed = p_val > 0.05  # p > 0.05 = similar proportions
        except Exception as e:
            # If the test fails (e.g., all values the same), default to failed
            chi2, p_val = 0.0, 0.0
            passed = False
            
        results.append({
            "Feature": col,
            "Type": "Categorical",
            "Test_Statistic": chi2,
            "P_Value": p_val,
            "Wasserstein_Distance": np.nan,  # Not applicable for categorical data
            "Passed_Statistical_Identical": passed
        })
        
    results_df = pd.DataFrame(results)
    
    # Summary: average p-value across all features (higher is better)
    avg_similarity = results_df["P_Value"].mean()
    logger.info(f"Statistical validation completed. Mean Feature Similarity (p-value): {avg_similarity:.4f}")
    
    return results_df
