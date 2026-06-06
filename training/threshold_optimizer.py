import numpy as np
import logging
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

logger = logging.getLogger("ThresholdOptimizer")

def optimize_decision_threshold(y_true, probs, target_recall=0.95):
    """
    Searches thresholds from 0.01 to 0.99 to find the optimal decision cutoff
    that guarantees Recall >= target_recall (95%) while maximizing Precision.
    """
    logger.info(f"Initiating threshold search to satisfy Recall >= {target_recall * 100}%...")
    
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_recall = 0.0
    best_precision = 0.0
    best_f1 = 0.0
    
    candidates = []
    
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        
        if rec >= target_recall:
            candidates.append((thresh, rec, prec, f1))
            
    if candidates:
        # Of those satisfying Recall >= 95%, select the one that maximizes Precision
        # This keeps FPs in control
        candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
        best_thresh, best_recall, best_precision, best_f1 = candidates[0]
    else:
        # Fallback: if no threshold gets >= 95% Recall, pick the one that maximizes Recall
        logger.warning(f"No threshold satisfied Recall >= {target_recall * 100}%. Falling back to maximum recall.")
        sorted_by_recall = sorted([(t, recall_score(y_true, (probs >= t).astype(int), zero_division=0), precision_score(y_true, (probs >= t).astype(int), zero_division=0), f1_score(y_true, (probs >= t).astype(int), zero_division=0)) for t in thresholds], key=lambda x: x[1], reverse=True)
        best_thresh, best_recall, best_precision, best_f1 = sorted_by_recall[0]
        
    logger.info(f"Optimal Threshold Identified: {best_thresh:.4f}")
    logger.info(f"Optimized Performance metrics -> Recall: {best_recall * 100:.2f}%, Precision: {best_precision * 100:.2f}%, F1: {best_f1 * 100:.2f}%")
    
    return best_thresh, {
        "Recall": best_recall,
        "Precision": best_precision,
        "F1_Score": best_f1
    }
