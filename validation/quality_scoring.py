"""
quality_scoring.py - Synthetic Data Quality Score Calculator
==============================================================
This module computes a single overall quality score (0% to 100%) for the
synthetic dataset by combining three different metrics:

  1. Statistical Similarity (40% weight)
     → How many features passed the KS/Chi2 statistical test?
     → A feature "passes" if its p-value > 0.01 (distributions are similar)

  2. PSI Stability (40% weight)
     → How many features have PSI < 0.2 (not significantly drifted)?

  3. Wasserstein Distance Penalty (20% weight)
     → The average "Earth Mover's Distance" between real and synthetic distributions.
     → A smaller distance = more similar distributions = higher score.

A score close to 100% means the synthetic data is statistically indistinguishable
from the real data — which is exactly what we want!
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("QualityScoring")

def compute_quality_score(statistical_df, drift_df):
    """
    Computes a single aggregate Quality Score (0-100%) for the synthetic dataset.

    Args:
        statistical_df : DataFrame from validate_distributions() — KS/Chi2 test results
        drift_df       : DataFrame from detect_drift() — PSI drift scores

    Returns:
        aggregate_score: A float between 0.0 and 100.0 representing overall quality
    """
    logger.info("Computing aggregate quality score for synthetic dataset...")
    
    # ── Component 1: Statistical Similarity (KS/Chi2 pass rate) ──
    # Count the fraction of features whose p-value > 0.01
    # (p > 0.01 means we cannot statistically prove the distributions differ)
    stat_passed = (statistical_df["P_Value"] > 0.01).mean()  # Value between 0.0 and 1.0
    
    # ── Component 2: PSI Stability Score ──
    # Count the fraction of features with PSI < 0.2 (stable / no major drift)
    stable_features = (drift_df["PSI_Score"] < 0.2).mean()   # Value between 0.0 and 1.0
    
    # ── Component 3: Wasserstein Distance Penalty ──
    # Wasserstein distance = "how far" two distributions are from each other.
    # A higher distance = more different = lower score.
    # We convert it to a 0-1 score: 1 / (1 + avg_distance)
    numeric_stats = statistical_df[statistical_df["Type"] == "Numerical"]  # Only numeric columns have this metric
    w_distances = numeric_stats["Wasserstein_Distance"].dropna()            # Drop NaN values
    if len(w_distances) > 0:
        w_score = 1.0 / (1.0 + w_distances.mean())  # Range: (0, 1] — smaller distance → higher score
    else:
        w_score = 1.0  # If no distances available, assume perfect score
        
    # ── Combine all 3 components into a single weighted score ──
    # Weights: 40% stat similarity + 40% PSI stability + 20% Wasserstein score
    aggregate_score = (0.4 * stat_passed + 0.4 * stable_features + 0.2 * w_score) * 100.0
    
    # Clamp the final score between 0% and 100%
    aggregate_score = np.clip(aggregate_score, 0.0, 100.0)
    
    logger.info(f"Aggregate Synthetic Data Quality Score: {aggregate_score:.2f}%")
    return aggregate_score
