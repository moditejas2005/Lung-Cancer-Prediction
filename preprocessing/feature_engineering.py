import numpy as np
import pandas as pd
import logging
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger("FeatureEngineering")

def engineer_features(df):
    """
    Computes all complex derived features, clinical interactions, and physiological risk indexes.
    """
    logger.info("Engineering advanced interaction and composite clinical features...")
    df_feat = df.copy()
    
    # 1. Advanced interactions
    df_feat["Smoking_Intensity_Years"] = df_feat["Smoking_Intensity"] * df_feat["Years_Smoked"]
    df_feat["Cigarettes_Pack_Years"] = df_feat["Cigarettes_Per_Day"] * df_feat["Pack_Years"]
    df_feat["PM25_Oxygen_Ratio"] = df_feat["PM25_Level"] / (df_feat["Oxygen_Saturation"] + 1e-5)
    df_feat["Radon_Asbestos_Interaction"] = df_feat["Radon_Exposure"] * df_feat["Asbestos_Exposure"]
    df_feat["BMI_Smoking_Intensity"] = df_feat["BMI"] * df_feat["Smoking_Intensity"]
    df_feat["Age_Smoking_Risk"] = df_feat["Age"] * df_feat["Pack_Years"]
    
    # 2. Composite risk indicators
    df_feat["Environmental_Exposure_Index"] = (
        (df_feat["PM25_Level"] / 120.0) + 
        (df_feat["Radon_Exposure"] / 15.0) + 
        df_feat["Asbestos_Exposure"]
    )
    
    df_feat["Symptom_Severity_Index"] = (
        df_feat["Breathlessness"] + 
        df_feat["Chronic_Cough"] + 
        df_feat["Chest_Pain"] + 
        df_feat["Wheezing"] + 
        (df_feat["Fatigue_Level"] / 5.0)
    )
    
    df_feat["Respiratory_Risk_Index"] = (
        (100.0 - df_feat["Oxygen_Saturation"]) * 
        (df_feat["Symptom_Severity_Index"] + 1.0)
    )
    
    df_feat["Composite_Cancer_Risk_Score"] = (
        0.3 * np.log1p(df_feat["Age_Smoking_Risk"]) + 
        0.4 * df_feat["Environmental_Exposure_Index"] + 
        0.3 * df_feat["Respiratory_Risk_Index"]
    )
    
    df_feat["Smoking_Damage_Index"] = df_feat["Pack_Years"] * (df_feat["Years_Smoked"] + 1)
    df_feat["Oxygen_Deficiency_Risk"] = (100.0 - df_feat["Oxygen_Saturation"]) / 100.0
    
    df_feat["Pulmonary_Stress_Index"] = (
        (df_feat["Heart_Rate"] / 100.0) * 
        (df_feat["Breathlessness"] + df_feat["Wheezing"] + 1.0) / 
        (df_feat["Oxygen_Saturation"] / 100.0)
    )
    
    logger.info("Feature engineering completed. Derived columns created.")
    return df_feat

def perform_feature_selection(df, target_col="Diagnosis", correlation_threshold=0.95, max_features=25):
    """
    Performs mutual information, correlation filtering, and model-based feature importance
    to select the best clinical features.
    """
    logger.info("Initiating advanced feature selection system...")
    
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    
    # 1. Correlation Filtering
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    logger.info(f"Correlation Filter: Dropping {len(to_drop)} redundant features (> {correlation_threshold}): {to_drop}")
    X_filtered = X.drop(columns=to_drop)
    
    # 2. Mutual Information Scoring
    mi_scores = mutual_info_classif(X_filtered, y, random_state=42)
    mi_df = pd.DataFrame({"Feature": X_filtered.columns, "Mutual_Information": mi_scores})
    mi_df = mi_df.sort_values(by="Mutual_Information", ascending=False)
    
    # 3. Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_filtered, y)
    rf_df = pd.DataFrame({"Feature": X_filtered.columns, "RF_Importance": rf.feature_importances_})
    
    # Merge metrics
    importance_df = pd.merge(mi_df, rf_df, on="Feature")
    importance_df["Combined_Score"] = (
        (importance_df["Mutual_Information"] / importance_df["Mutual_Information"].max()) * 0.5 + 
        (importance_df["RF_Importance"] / importance_df["RF_Importance"].max()) * 0.5
    )
    importance_df = importance_df.sort_values(by="Combined_Score", ascending=False)
    
    # Select top features
    selected_features = list(importance_df["Feature"].head(max_features))
    logger.info(f"Feature Selection completed. Selected {len(selected_features)} primary predictors.")
    
    return selected_features, importance_df
