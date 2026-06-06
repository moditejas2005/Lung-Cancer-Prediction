"""
calibration_pipeline.py - Probability Calibration Module
==========================================================
After training, many models (especially ensemble models) output probabilities
that are NOT well-calibrated. For example:
  - A model might say "70% cancer probability" but only 50% of such patients actually have cancer.
  - A well-calibrated model means: when it says 70%, ~70% of those patients truly have cancer.

This module uses sklearn's CalibratedClassifierCV to fix this:
  - method="isotonic" : Non-parametric calibration (more flexible, better for more data)
  - method="sigmoid"  : Platt scaling (simpler, good for small datasets)

After calibration, the model's probability outputs are much more trustworthy
for clinical decision-making.
"""

import logging
import os
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend (saves plots to files, no display popup)
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

logger = logging.getLogger("CalibrationPipeline")

def calibrate_model(model, X_val, y_val, method="isotonic"):
    """
    Wraps a pre-fitted model in a calibration layer using validation data.

    How it works:
      - CalibratedClassifierCV with cv="prefit" assumes the model is ALREADY trained.
      - It then fits a calibration mapping ON TOP of the model using X_val/y_val.
      - After calibration, model.predict_proba() outputs calibrated probabilities.

    Args:
        model   : A pre-fitted sklearn-compatible model
        X_val   : Validation feature DataFrame (used to learn the calibration mapping)
        y_val   : Validation true labels (0/1)
        method  : "isotonic" or "sigmoid" (default: "isotonic" — more flexible)

    Returns:
        calibrated_model: A new model object that outputs calibrated probabilities
    """
    logger.info(f"Calibrating fitted model probabilities using {method} scaling...")
    
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv="prefit"   # "prefit" means: the base model is already trained, just calibrate outputs
    )
    calibrated_model.fit(X_val, y_val)  # Learn the calibration mapping from validation data
    
    logger.info("Probability calibration successful.")
    return calibrated_model

def plot_calibration_curves(models_dict, X_val, y_val, save_dir="data/reports/plots"):
    """
    Plots reliability (calibration) curves for all models.
    
    A "perfectly calibrated" model follows the diagonal line y=x.
    Points above the diagonal = model underestimates confidence.
    Points below the diagonal = model overestimates confidence.

    Args:
        models_dict : Dict of {model_name: model_object}
        X_val       : Validation features
        y_val       : Validation true labels
        save_dir    : Directory to save the plot image
    """
    logger.info("Plotting reliability calibration curves...")
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")  # Diagonal = perfect calibration reference line
    
    colors = ["#d9534f", "#0275d8", "#5cb85c", "#f0ad4e", "#5bc0de", "#292b2c"]
    
    for idx, (name, model) in enumerate(models_dict.items()):
        if not hasattr(model, "predict_proba"):
            continue  # Skip models without probability output
            
        # Get model's predicted probabilities for the positive class (Cancer)
        probs = model.predict_proba(X_val)[:, 1]
        
        # calibration_curve bins the probabilities and computes fraction of actual positives per bin
        fraction_of_positives, mean_predicted_value = calibration_curve(y_val, probs, n_bins=10)
        
        color = colors[idx % len(colors)]
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", color=color, label=f"{name}")
        
    plt.ylabel("Fraction of positive diagnosis", fontsize=12)
    plt.xlabel("Mean predicted probability", fontsize=12)
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right", fontsize=11)
    plt.title("Clinical Reliability Curve (Probability Calibration)", fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "calibration_reliability_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    logger.info(f"Calibration plot exported to {plot_path}.")
