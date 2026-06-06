"""
threshold_optimizer.py - Decision Threshold Optimizer
=======================================================
Normally, ML models classify "Cancer" if cancer probability >= 0.5 (50%).
But in medical diagnosis, MISSING a cancer case (False Negative) is far more
dangerous than a false alarm (False Positive).

This module searches for the best threshold (e.g., 0.30 instead of 0.50)
that guarantees the model catches at least 95% of all real cancer cases
(Recall >= 95%), while still trying to maximize Precision.

Example:
  - At threshold 0.50: Recall = 88%, Precision = 92%  (misses 12% of cancer cases)
  - At threshold 0.28: Recall = 96%, Precision = 80%  (misses only 4% — much safer!)
"""

import numpy as np
import logging
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

logger = logging.getLogger("ThresholdOptimizer")

def optimize_decision_threshold(y_true, probs, target_recall=0.95):
    """
    Searches all possible thresholds from 0.01 to 0.99 to find the optimal one
    that satisfies Recall >= target_recall (default 95%) AND maximizes Precision.

    Args:
        y_true        : True labels (0 = No Cancer, 1 = Cancer) — array
        probs         : Model's predicted cancer probabilities — array of floats [0,1]
        target_recall : Minimum acceptable recall (default 0.95 = 95%)

    Returns:
        best_thresh     : The optimal threshold value (float)
        results_dict    : Dict with Recall, Precision, F1_Score at the optimal threshold
    """
    logger.info(f"Initiating threshold search to satisfy Recall >= {target_recall * 100}%...")
    
    # Test 99 threshold values evenly spaced between 0.01 and 0.99
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5   # Default fallback
    best_recall = 0.0
    best_precision = 0.0
    best_f1 = 0.0
    
    candidates = []  # Store all thresholds that satisfy the recall requirement
    
    for thresh in thresholds:
        # Convert probabilities to 0/1 predictions using this threshold
        preds = (probs >= thresh).astype(int)
        
        # Calculate metrics at this threshold
        rec = recall_score(y_true, preds, zero_division=0)     # Fraction of true cancer cases caught
        prec = precision_score(y_true, preds, zero_division=0)  # Fraction of positive predictions that are correct
        f1 = f1_score(y_true, preds, zero_division=0)           # Harmonic mean of Precision and Recall
        
        # Keep this threshold only if it meets the minimum recall requirement
        if rec >= target_recall:
            candidates.append((thresh, rec, prec, f1))
            
    if candidates:
        # Among all valid candidates, pick the one with the highest Precision.
        # This minimizes false alarms while still catching >= 95% of cancer cases.
        candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
        best_thresh, best_recall, best_precision, best_f1 = candidates[0]
    else:
        # Fallback: if NO threshold achieves 95% recall, just pick the one with highest recall
        logger.warning(f"No threshold satisfied Recall >= {target_recall * 100}%. Falling back to maximum recall.")
        sorted_by_recall = sorted(
            [(t, recall_score(y_true, (probs >= t).astype(int), zero_division=0),
              precision_score(y_true, (probs >= t).astype(int), zero_division=0),
              f1_score(y_true, (probs >= t).astype(int), zero_division=0))
             for t in thresholds],
            key=lambda x: x[1], reverse=True
        )
        best_thresh, best_recall, best_precision, best_f1 = sorted_by_recall[0]
        
    logger.info(f"Optimal Threshold Identified: {best_thresh:.4f}")
    logger.info(f"Optimized Performance metrics -> Recall: {best_recall * 100:.2f}%, Precision: {best_precision * 100:.2f}%, F1: {best_f1 * 100:.2f}%")
    
    # Return the best threshold AND a summary of the metrics at that threshold
    return best_thresh, {
        "Recall": best_recall,
        "Precision": best_precision,
        "F1_Score": best_f1
    }
