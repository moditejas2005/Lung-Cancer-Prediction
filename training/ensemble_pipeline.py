"""
ensemble_pipeline.py - Ensemble Model Builder
===============================================
An "ensemble" combines multiple ML models to produce a stronger prediction
than any single model could alone. This module builds two types:

  1. Voting Classifier (Soft Voting):
     → Each model votes with its cancer probability score.
     → The final prediction is the weighted average of all votes.
     → Like asking 5 doctors and averaging their opinions.

  2. Stacking Classifier:
     → "Base models" (XGBoost, CatBoost, RF, GradientBoosting) each make predictions.
     → Their predictions are fed as INPUTS to a "meta-learner" (Logistic Regression).
     → The meta-learner learns which base models to trust more.
     → This is more powerful than simple voting.
"""

import logging
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

logger = logging.getLogger("EnsemblePipeline")

def build_voting_ensemble(models):
    """
    Creates a Soft Voting ensemble from a dictionary of already-trained models.
    
    Soft Voting: Each model outputs a probability (not just 0/1).
    The final prediction = average of all model probabilities.
    
    Args:
        models: Dict of {name: trained_model_object}

    Returns:
        A VotingClassifier that combines all provided models.
    """
    logger.info("Assembling multi-model Voting Classifier (Soft Voting)...")
    
    # Only include models that can output probabilities (predict_proba)
    estimators = []
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            # Convert name to a valid identifier (no spaces)
            estimators.append((name.lower().replace(" ", "_"), model))
            
    voting_ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",    # Use probability averaging, not majority vote
        n_jobs=-1          # Use all CPU cores
    )
    return voting_ensemble

def build_stacking_ensemble(random_seed=42):
    """
    Builds a Stacking Ensemble with 4 base models and 1 meta-learner.

    Architecture:
      Level 0 (Base Models): XGBoost, CatBoost, Random Forest, Gradient Boosting
      Level 1 (Meta-Learner): Logistic Regression

    How it works:
      1. All 4 base models are trained on the data using cross-validation.
      2. Their out-of-fold probability predictions are used as NEW features.
      3. The Logistic Regression meta-learner is trained on these new features.
      4. At inference time: base models predict → meta-learner makes final decision.

    Args:
        random_seed: Seed for reproducibility

    Returns:
        A StackingClassifier ready to be fitted.
    """
    logger.info("Architecting advanced Stacking Ensemble...")
    
    # The 4 base models — each is strong in different ways
    base_estimators = [
        ("xgb", XGBClassifier(n_estimators=100, random_state=random_seed, eval_metric="logloss", n_jobs=-1)),
        ("cat", CatBoostClassifier(iterations=100, random_seed=random_seed, verbose=False)),
        ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_seed, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=random_seed))
    ]
    
    # The meta-learner: learns from the base models' mistakes
    meta_learner = LogisticRegression(class_weight="balanced", random_state=random_seed)
    
    stacking_ensemble = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,   # Meta-learner combines base model outputs
        cv=5,                            # Use 5-fold CV to generate base model predictions
        n_jobs=-1                        # Parallel training for speed
    )
    
    return stacking_ensemble
