import pandas as pd
import numpy as np
import logging
from scipy.stats import ks_2samp, chi2_contingency
from scipy.stats import wasserstein_distance

logger = logging.getLogger("StatisticalValidator")

def validate_distributions(df_real, df_synthetic):
    """
    Performs statistical similarity checks between real and synthetic data distributions.
    Computes KS tests (continuous) and Chi-Square tests (categorical).
    """
    logger.info("Conducting statistical similarity analysis...")
    
    results = []
    
    # 1. Continuous Columns (KS Test & Wasserstein Distance)
    real_numeric = df_real.select_dtypes(include=[np.number]).columns
    synth_numeric = df_synthetic.select_dtypes(include=[np.number]).columns
    
    common_numeric = list(set(real_numeric) & set(synth_numeric))
    
    for col in common_numeric:
        if col == "Diagnosis":
            continue
        ks_stat, ks_pval = ks_2samp(df_real[col], df_synthetic[col])
        w_dist = wasserstein_distance(df_real[col], df_synthetic[col])
        
        # KS p-value > 0.05 indicates the distributions are statistically identical
        passed = ks_pval > 0.05
        
        results.append({
            "Feature": col,
            "Type": "Numerical",
            "Test_Statistic": ks_stat,
            "P_Value": ks_pval,
            "Wasserstein_Distance": w_dist,
            "Passed_Statistical_Identical": passed
        })
        
    # 2. Categorical Columns (Chi-Square Contingency Test)
    real_cat = df_real.select_dtypes(include=["object", "category"]).columns
    synth_cat = df_synthetic.select_dtypes(include=["object", "category"]).columns
    
    common_cat = list(set(real_cat) & set(synth_cat))
    
    for col in common_cat:
        # Construct contingency table
        real_counts = df_real[col].value_counts().sort_index()
        synth_counts = df_synthetic[col].value_counts().sort_index()
        
        # Align indices
        combined = pd.DataFrame({"Real": real_counts, "Synthetic": synth_counts}).fillna(0)
        
        try:
            chi2, p_val, dof, ex = chi2_contingency(combined.values.T)
            passed = p_val > 0.05
        except Exception as e:
            chi2, p_val = 0.0, 0.0
            passed = False
            
        results.append({
            "Feature": col,
            "Type": "Categorical",
            "Test_Statistic": chi2,
            "P_Value": p_val,
            "Wasserstein_Distance": np.nan,
            "Passed_Statistical_Identical": passed
        })
        
    results_df = pd.DataFrame(results)
    
    # Calculate global score
    avg_similarity = results_df["P_Value"].mean()
    logger.info(f"Statistical validation completed. Mean Feature Similarity (p-value): {avg_similarity:.4f}")
    
    return results_df
