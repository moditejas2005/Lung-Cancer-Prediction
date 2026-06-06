"""
feature_importance.py - Feature Importance Comparison Module
=============================================================
This module compares how different ML models rank the importance of each
patient feature. Each model has its own way of measuring "importance":

  - XGBoost / Random Forest / GradientBoosting: "Feature Importance" attribute
    → How much did each feature reduce impurity across all tree splits?
    → Higher = more useful to the model

  - Logistic Regression: "Coefficients"
    → How much does each feature change the log-odds of cancer?
    → Higher absolute value = stronger influence

By comparing importance across models, we can:
  1. Identify which features ALL models agree are important (robust features)
  2. Detect features that only one model relies on (potentially overfitting)
  3. Remove features that no model finds important (reduce noise)
"""

import pandas as pd
import numpy as np
import logging
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("FeatureImportance")

def compare_feature_importances(fitted_models, feature_names, save_dir="data/reports/plots"):
    """
    Extracts and audits global feature importance scores across Random Forest, XGBoost, and CatBoost models.
    Exports comparison spreadsheets and charts.
    """
    logger.info("Extracting model-based feature importances...")
    os.makedirs(save_dir, exist_ok=True)
    
    importance_data = {"Feature": feature_names}
    
    # Random Forest Importance
    if "Random Forest" in fitted_models:
        rf = fitted_models["Random Forest"]
        importance_data["Random_Forest"] = rf.feature_importances_
        
    # XGBoost Importance
    if "XGBoost" in fitted_models:
        xgb = fitted_models["XGBoost"]
        importance_data["XGBoost"] = xgb.feature_importances_
        
    # CatBoost Importance
    if "CatBoost" in fitted_models:
        cat = fitted_models["CatBoost"]
        importance_data["CatBoost"] = cat.feature_importances_ / cat.feature_importances_.sum()
        
    importance_df = pd.DataFrame(importance_data)
    
    # Calculate average rank
    importance_df["Mean_Importance"] = importance_df.select_dtypes(include=[np.number]).mean(axis=1)
    importance_df = importance_df.sort_values(by="Mean_Importance", ascending=False).reset_index(drop=True)
    
    # Plot top 15 features
    top_n = importance_df.head(15)
    
    plt.figure(figsize=(12, 7))
    top_n_melted = pd.melt(top_n, id_vars=["Feature"], value_vars=[col for col in importance_df.columns if col != "Feature" and col != "Mean_Importance"])
    
    sns.barplot(x="value", y="Feature", hue="variable", data=top_n_melted, palette="muted")
    plt.title("Cross-Model Feature Importance Audit (Top 15 Predictors)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.ylabel("Predictor Variable", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="ML Algorithm", fontsize=10)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "feature_importance_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    logger.info(f"Cross-model feature importance comparison plot saved to {plot_path}.")
    return importance_df
