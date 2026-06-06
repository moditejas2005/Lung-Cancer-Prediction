import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("QualityScoring")

def compute_quality_score(statistical_df, drift_df):
    """
    Computes an aggregate tabular Quality Score (0-100%) by combining
    statistical similarity metrics and feature drift metrics.
    """
    logger.info("Computing aggregate quality score for synthetic dataset...")
    
    # 1. Statistical similarity component (KS/Chi-Square tests passed)
    # The percentage of features that did not significantly drift (KS or Chi2 p-val > 0.01)
    stat_passed = (statistical_df["P_Value"] > 0.01).mean()
    
    # 2. PSI component (stable features)
    # Percentage of features with PSI < 0.2
    stable_features = (drift_df["PSI_Score"] < 0.2).mean()
    
    # 3. Wasserstein distance penalty (if any Wasserstein distances are very large)
    # Normalized Wasserstein component
    numeric_stats = statistical_df[statistical_df["Type"] == "Numerical"]
    w_distances = numeric_stats["Wasserstein_Distance"].dropna()
    if len(w_distances) > 0:
        w_score = 1.0 / (1.0 + w_distances.mean())
    else:
        w_score = 1.0
        
    # Aggregate weighted score
    aggregate_score = (0.4 * stat_passed + 0.4 * stable_features + 0.2 * w_score) * 100.0
    
    # Clip between 0 and 100
    aggregate_score = np.clip(aggregate_score, 0.0, 100.0)
    
    logger.info(f"Aggregate Synthetic Data Quality Score: {aggregate_score:.2f}%")
    return aggregate_score
