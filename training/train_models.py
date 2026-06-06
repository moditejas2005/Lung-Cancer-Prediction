import numpy as np
import pandas as pd
import logging
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

logger = logging.getLogger("TrainModels")

def get_base_models(random_seed=42):
    """
    Instantiates all classical and advanced boosting models.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_seed, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=random_seed),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_seed),
        "Support Vector Machine": SVC(probability=True, class_weight="balanced", random_state=random_seed),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=random_seed, eval_metric="logloss", n_jobs=-1),
        "CatBoost": CatBoostClassifier(iterations=100, random_seed=random_seed, verbose=False)
    }
    return models

def train_and_evaluate_cv(X, y, random_seed=42, n_splits=5):
    """
    Trains all models using Stratified K-Fold Cross Validation.
    Applies SMOTE to handle class imbalance during training folds.
    """
    logger.info(f"Initiating {n_splits}-fold Stratified CV multi-model training pipeline...")
    
    models = get_base_models(random_seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        accs, precs, recs, f1s, aucs = [], [], [], [], []
        train_times, infer_times = [], []
        
        # Array to store out-of-fold probability predictions
        oof_probs = np.zeros(len(X))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Apply SMOTE only to training set
            smote = SMOTE(random_state=random_seed)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            
            start_time = time.time()
            model.fit(X_train_res, y_train_res)
            train_times.append(time.time() - start_time)
            
            # Measure inference time
            start_infer = time.time()
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_val)[:, 1]
            else:
                probs = model.decision_function(X_val)
                # Normalize decision function to [0,1]
                probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-5)
            infer_times.append((time.time() - start_infer) / len(X_val))
            
            oof_probs[val_idx] = probs
            
            preds = (probs >= 0.5).astype(int)
            
            accs.append(accuracy_score(y_val, preds))
            precs.append(precision_score(y_val, preds, zero_division=0))
            recs.append(recall_score(y_val, preds, zero_division=0))
            f1s.append(f1_score(y_val, preds, zero_division=0))
            aucs.append(roc_auc_score(y_val, probs))
            
        # Fit final model on all data
        smote = SMOTE(random_state=random_seed)
        X_res, y_res = smote.fit_resample(X, y)
        model.fit(X_res, y_res)
        fitted_models[name] = model
        
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
