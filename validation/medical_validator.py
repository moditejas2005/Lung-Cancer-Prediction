import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("MedicalValidator")

def validate_medical_records(df):
    """
    Applies medical validation checks on synthetic data.
    Filters or cleans medically impossible relationships.
    """
    logger.info("Starting clinical rules medical validation on synthetic dataset...")
    df_val = df.copy()
    
    n_original = len(df_val)
    
    # Rule 1: Never smokers must have 0 smoking metrics
    never_smokers = df_val["Smoking_Status"] == "Never"
    
    # Count anomalies
    anomalies_smokers = never_smokers & (
        (df_val["Cigarettes_Per_Day"] > 0) | 
        (df_val["Years_Smoked"] > 0) | 
        (df_val["Pack_Years"] > 0.0) | 
        (df_val["Smoking_Intensity"] > 0) | 
        (df_val["Smoking_Frequency"] != "None")
    )
    n_anomalies_smokers = anomalies_smokers.sum()
    
    # Correct Never smokers
    df_val.loc[never_smokers, "Cigarettes_Per_Day"] = 0
    df_val.loc[never_smokers, "Years_Smoked"] = 0
    df_val.loc[never_smokers, "Pack_Years"] = 0.0
    df_val.loc[never_smokers, "Smoking_Intensity"] = 0
    df_val.loc[never_smokers, "Smoking_Frequency"] = "None"
    
    # Rule 2: Years smoked cannot exceed Age - 10
    too_many_years = df_val["Years_Smoked"] > (df_val["Age"] - 10)
    n_too_many_years = too_many_years.sum()
    df_val.loc[too_many_years, "Years_Smoked"] = np.maximum(0, df_val.loc[too_many_years, "Age"] - 15)
    
    # Rule 3: Re-calculate and validate Pack Years formula
    df_val["Pack_Years"] = (df_val["Cigarettes_Per_Day"] / 20.0) * df_val["Years_Smoked"]
    
    # Rule 4: Oxygen Saturation limits
    df_val["Oxygen_Saturation"] = np.clip(df_val["Oxygen_Saturation"], 50.0, 100.0)
    df_val["Oxygen_Percentage"] = np.clip(df_val["Oxygen_Percentage"], 50.0, 100.0)
    
    # Rule 5: BMI logical limits
    df_val["BMI"] = np.clip(df_val["BMI"], 10.0, 60.0)
    
    # Rule 6: Pulse logical boundaries
    df_val["Heart_Rate"] = np.clip(df_val["Heart_Rate"], 40, 150)
    
    logger.info(f"Clinical audit completed. Never-smoker anomalies corrected: {n_anomalies_smokers} records.")
    logger.info(f"Years smoked vs. Age boundary anomalies corrected: {n_too_many_years} records.")
    logger.info(f"Validated 100% of patient records successfully. No records dropped, all corrected.")
    
    return df_val
