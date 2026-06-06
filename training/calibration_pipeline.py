import logging
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

logger = logging.getLogger("CalibrationPipeline")

def calibrate_model(model, X_val, y_val, method="isotonic"):
    """
    Calibrates prediction probabilities of a prefitted model using Isotonic Regression or Platt scaling.
    """
    logger.info(f"Calibrating fitted model probabilities using {method} scaling...")
    
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv="prefit"
    )
    calibrated_model.fit(X_val, y_val)
    
    logger.info("Probability calibration successful.")
    return calibrated_model

def plot_calibration_curves(models_dict, X_val, y_val, save_dir="data/reports/plots"):
    """
    Plots probability reliability curves comparing calibrated vs. uncalibrated models.
    """
    logger.info("Plotting reliability calibration curves...")
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    colors = ["#d9534f", "#0275d8", "#5cb85c", "#f0ad4e", "#5bc0de", "#292b2c"]
    
    for idx, (name, model) in enumerate(models_dict.items()):
        if not hasattr(model, "predict_proba"):
            continue
            
        probs = model.predict_proba(X_val)[:, 1]
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
