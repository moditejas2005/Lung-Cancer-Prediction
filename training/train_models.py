"""
train_models.py - Multi-Model Training with Cross-Validation
=============================================================
This module trains SEVEN different machine learning models and evaluates
each one using Stratified K-Fold Cross-Validation.

What is Stratified K-Fold CV?
  - The data is split into K equal parts (folds).
  - Each fold is used once as the test set while the rest are used for training.
  - "Stratified" means each fold keeps the same Cancer/No-Cancer ratio.
  - This gives a more reliable evaluation than a single train/test split.

What is SMOTE?
  - SMOTE = Synthetic Minority Oversampling Technique.
  - Cancer patients are rarer than non-cancer patients (imbalanced data).
  - SMOTE creates fake cancer patient samples to balance the training data.
  - It is only applied to TRAINING folds — never to the validation set.

Models trained:
  1. Logistic Regression   — Simple linear model
  2. Random Forest          — Many decision trees combined
  3. Decision Tree          — A single decision tree
  4. Gradient Boosting      — Trees built one after another to fix errors
  5. Support Vector Machine — Finds the best boundary between classes
  6. XGBoost                — Fast, optimized gradient boosting
  7. CatBoost               — Gradient boosting optimized for categorical data
"""

import numpy as np
import pandas as pd
import logging
import time                                   # Measure training and inference speed
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE       # Balances the Cancer/No-Cancer ratio

logger = logging.getLogger("TrainModels")

def get_base_models(random_seed=42):
    """
    Creates and returns a dictionary of all ML model objects.
    
    Each model uses:
      - class_weight="balanced"  → gives more weight to the minority class (Cancer)
      - random_state=random_seed → ensures reproducibility (same results every run)
      - n_jobs=-1                → use all CPU cores for faster training

    Returns:
        A dict: {model_name: model_object}
    """
    models = {
        # Simple linear model — fast but less powerful for complex patterns
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed),
        # 100 decision trees averaged — robust and handles overfitting well
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_seed, n_jobs=-1),
        # A single tree — interpretable but can overfit
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=random_seed),
        # Boosted trees — each tree fixes the errors of the previous one
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_seed),
        # SVM with probability output — effective for smaller datasets
        "Support Vector Machine": SVC(probability=True, class_weight="balanced", random_state=random_seed),
        # Highly efficient XGBoost — typically the best single model
        "XGBoost": XGBClassifier(n_estimators=100, random_state=random_seed, eval_metric="logloss", n_jobs=-1),
        # CatBoost — handles mixed-type data well (verbose=False suppresses training output)
        "CatBoost": CatBoostClassifier(iterations=100, random_seed=random_seed, verbose=False)
    }
    return models

def train_and_evaluate_cv(X, y, random_seed=42, n_splits=5):
    """
    Trains all 7 models using Stratified K-Fold Cross-Validation with SMOTE.

    For each model and each fold:
      1. Split data into training and validation folds
      2. Apply SMOTE to the training fold (balance classes)
      3. Train the model on the balanced training fold
      4. Evaluate on the validation fold (unmodified, real proportions)
      5. Record Accuracy, Precision, Recall, F1, ROC-AUC, and timing

    After CV, each model is re-trained on ALL data (with SMOTE) for deployment.

    Args:
        X           : Feature DataFrame (already scaled)
        y           : Target Series (0 = No Cancer, 1 = Cancer)
        random_seed : Random seed for reproducibility
        n_splits    : Number of K-Fold splits (default 5)

    Returns:
        results      : Dict of {model_name: {metric: value, ...}}
        fitted_models: Dict of {model_name: trained_model_object}
    """
    logger.info(f"Initiating {n_splits}-fold Stratified CV multi-model training pipeline...")
    
    models = get_base_models(random_seed)              # Get all model instances
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)  # Set up CV splitter
    
    results = {}        # Final averaged metrics per model
    fitted_models = {}  # Fully trained model objects
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Lists to collect per-fold metrics (we average them at the end)
        accs, precs, recs, f1s, aucs = [], [], [], [], []
        train_times, infer_times = [], []
        
        # Out-of-fold probabilities: predictions on validation folds (never seen during training)
        oof_probs = np.zeros(len(X))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # Split this fold's data
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Apply SMOTE only to training data — never to validation data!
            # SMOTE creates synthetic samples of the minority class to balance the dataset.
            smote = SMOTE(random_state=random_seed)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            
            # Train the model and measure how long it takes
            start_time = time.time()
            model.fit(X_train_res, y_train_res)
            train_times.append(time.time() - start_time)
            
            # Run inference and measure how fast it is per sample
            start_infer = time.time()
            if hasattr(model, "predict_proba"):
                # Most models: get probability of cancer (class 1)
                probs = model.predict_proba(X_val)[:, 1]
            else:
                # SVM in some modes uses decision_function instead of predict_proba
                probs = model.decision_function(X_val)
                # Normalize decision scores to [0, 1] range so they act like probabilities
                probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-5)
            infer_times.append((time.time() - start_infer) / len(X_val))  # Time per patient
            
            # Store probabilities for this validation fold
            oof_probs[val_idx] = probs
            
            # Convert probabilities to hard predictions (threshold = 0.5)
            preds = (probs >= 0.5).astype(int)
            
            # Calculate and store all metrics for this fold
            accs.append(accuracy_score(y_val, preds))
            precs.append(precision_score(y_val, preds, zero_division=0))
            recs.append(recall_score(y_val, preds, zero_division=0))
            f1s.append(f1_score(y_val, preds, zero_division=0))
            aucs.append(roc_auc_score(y_val, probs))
            
        # Final training: fit on ALL data with SMOTE for the deployable model
        smote = SMOTE(random_state=random_seed)
        X_res, y_res = smote.fit_resample(X, y)
        model.fit(X_res, y_res)
        fitted_models[name] = model  # Store the trained model
        
        # Average all per-fold metrics to get the final CV score
        results[name] = {
            "Accuracy": np.mean(accs),
            "Precision": np.mean(precs),
            "Recall": np.mean(recs),
            "F1_Score": np.mean(f1s),
            "ROC_AUC": np.mean(aucs),
            "Training_Time_Sec": np.mean(train_times),
            "Inference_Time_Sec": np.mean(infer_times),
            "OOF_Probs": oof_probs
        }
        
        logger.info(f"{name} CV Results -> Accuracy: {np.mean(accs):.4f}, Recall: {np.mean(recs):.4f}, Precision: {np.mean(precs):.4f}, F1: {np.mean(f1s):.4f}, AUC: {np.mean(aucs):.4f}")
        
    return results, fitted_models
