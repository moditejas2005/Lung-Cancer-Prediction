import logging
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

logger = logging.getLogger("EnsemblePipeline")

def build_voting_ensemble(models):
    """
    Creates a soft voting ensemble from dictionary of trained models.
    """
    logger.info("Assembling multi-model Voting Classifier (Soft Voting)...")
    
    # Filter estimators that support predict_proba
    estimators = []
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            estimators.append((name.lower().replace(" ", "_"), model))
            
    voting_ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=-1
    )
    return voting_ensemble

def build_stacking_ensemble(random_seed=42):
    """
    Creates a standard Stacking Ensemble.
    Base Estimators: XGBoost, CatBoost, Random Forest, Gradient Boosting
    Meta Learner: Logistic Regression
    """
    logger.info("Architecting advanced Stacking Ensemble...")
    
    base_estimators = [
        ("xgb", XGBClassifier(n_estimators=100, random_state=random_seed, eval_metric="logloss", n_jobs=-1)),
        ("cat", CatBoostClassifier(iterations=100, random_seed=random_seed, verbose=False)),
        ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_seed, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=random_seed))
    ]
    
    meta_learner = LogisticRegression(class_weight="balanced", random_state=random_seed)
    
    stacking_ensemble = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    return stacking_ensemble
