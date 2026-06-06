import optuna
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score, precision_score
from imblearn.over_sampling import SMOTE

logger = logging.getLogger("HyperparameterTuning")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_xgb(X, y, n_trials=20, random_seed=42):
    """
    Tunes XGBoost hyperparameters using Optuna.
    Custom objective: maximize Recall with a penalty if Precision is < 0.90.
    """
    logger.info(f"Starting Optuna tuning for XGBoost ({n_trials} trials)...")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "random_state": random_seed,
            "eval_metric": "logloss",
            "n_jobs": -1
        }
        
        recs, precs = [], []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            smote = SMOTE(random_state=random_seed)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            
            model = XGBClassifier(**params)
            model.fit(X_train_res, y_train_res)
            
            probs = model.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)
            
            recs.append(recall_score(y_val, preds, zero_division=0))
            precs.append(precision_score(y_val, preds, zero_division=0))
            
        mean_rec = np.mean(recs)
        mean_prec = np.mean(precs)
        
        # Soft penalty constraint: keep precision >= 90%
        score = mean_rec
        if mean_prec < 0.90:
            score -= (0.90 - mean_prec) * 2.0
            
        return score
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best XGBoost Optuna Score: {study.best_value:.4f}")
    logger.info(f"Best XGBoost Parameters: {study.best_params}")
    return study.best_params

def tune_catboost(X, y, n_trials=15, random_seed=42):
    """
    Tunes CatBoost hyperparameters using Optuna.
    """
    logger.info(f"Starting Optuna tuning for CatBoost ({n_trials} trials)...")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
    
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 50, 150),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_seed": random_seed,
            "verbose": False
        }
        
        recs, precs = [], []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            smote = SMOTE(random_state=random_seed)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            
            model = CatBoostClassifier(**params)
            model.fit(X_train_res, y_train_res)
            
            probs = model.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)
            
            recs.append(recall_score(y_val, preds, zero_division=0))
            precs.append(precision_score(y_val, preds, zero_division=0))
            
        mean_rec = np.mean(recs)
        mean_prec = np.mean(precs)
        
        # Soft penalty constraint: keep precision >= 90%
        score = mean_rec
        if mean_prec < 0.90:
            score -= (0.90 - mean_prec) * 2.0
            
        return score
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best CatBoost Optuna Score: {study.best_value:.4f}")
    logger.info(f"Best CatBoost Parameters: {study.best_params}")
    return study.best_params
