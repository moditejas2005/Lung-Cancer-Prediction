"""
medical_validator.py - Clinical Rules Validator
=================================================
After CTGAN generates synthetic patient data, some records might be
medically impossible. For example, CTGAN might generate a "Never smoker"
who somehow has 10 pack-years — which is contradictory.

This module applies hard medical rules to fix such impossible values.
It does NOT delete records — it corrects them to valid values.

Rules enforced:
  1. Never-smokers must have ALL smoking metrics = 0 / "None"
  2. Years smoked cannot exceed (Age - 10) — a person can't smoke before they're ~10
  3. Recalculate Pack_Years from the corrected values
  4. Oxygen Saturation must be between 50% and 100%
  5. BMI must be between 10 and 60
  6. Heart Rate must be between 40 and 150 bpm
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("MedicalValidator")

def validate_medical_records(df):
    """
    Applies all medical validation rules to the synthetic patient dataset.
    Fixes medically impossible values by correcting them (not by deleting rows).

    Args:
        df: A pandas DataFrame of synthetic patient records from CTGAN

    Returns:
        df_val: A corrected DataFrame where all records follow medical logic
    """
    logger.info("Starting clinical rules medical validation on synthetic dataset...")
    df_val = df.copy()  # Work on a copy — never modify the original data directly
    
    n_original = len(df_val)  # Keep track of how many records we started with
    
    # ── Rule 1: Never-smokers must have zero smoking values ──
    # CTGAN sometimes gives never-smokers non-zero smoking data because
    # it doesn't know the semantic meaning of "Never" in Smoking_Status.
    never_smokers = df_val["Smoking_Status"] == "Never"
    
    # Count how many records violate this rule before fixing
    anomalies_smokers = never_smokers & (
        (df_val["Cigarettes_Per_Day"] > 0) | 
        (df_val["Years_Smoked"] > 0) | 
        (df_val["Pack_Years"] > 0.0) | 
        (df_val["Smoking_Intensity"] > 0) | 
        (df_val["Smoking_Frequency"] != "None")
    )
    n_anomalies_smokers = anomalies_smokers.sum()
    
    # Fix: Force all smoking-related fields to 0/None for never-smokers
    df_val.loc[never_smokers, "Cigarettes_Per_Day"] = 0
    df_val.loc[never_smokers, "Years_Smoked"] = 0
    df_val.loc[never_smokers, "Pack_Years"] = 0.0
    df_val.loc[never_smokers, "Smoking_Intensity"] = 0
    df_val.loc[never_smokers, "Smoking_Frequency"] = "None"
    
    # ── Rule 2: Years smoked cannot exceed Age - 10 ──
    # A 20-year-old cannot have smoked for 30 years — physically impossible.
    too_many_years = df_val["Years_Smoked"] > (df_val["Age"] - 10)
    n_too_many_years = too_many_years.sum()
    # Cap years smoked to (Age - 15), minimum 0
    df_val.loc[too_many_years, "Years_Smoked"] = np.maximum(0, df_val.loc[too_many_years, "Age"] - 15)
    
    # ── Rule 3: Recalculate Pack_Years from corrected values ──
    # Pack_Years = (Cigarettes per day / 20) × Years smoked
    # This ensures consistency after Rules 1 and 2 corrected the values.
    df_val["Pack_Years"] = (df_val["Cigarettes_Per_Day"] / 20.0) * df_val["Years_Smoked"]
    
    # ── Rule 4: Clip Oxygen Saturation to physiologically valid range ──
    # Human blood oxygen saturation must be between 50% (critical) and 100% (perfect)
    df_val["Oxygen_Saturation"] = np.clip(df_val["Oxygen_Saturation"], 50.0, 100.0)
    df_val["Oxygen_Percentage"] = np.clip(df_val["Oxygen_Percentage"], 50.0, 100.0)
    
    # ── Rule 5: Clip BMI to valid range ──
    # BMI below 10 or above 60 is not medically realistic
    df_val["BMI"] = np.clip(df_val["BMI"], 10.0, 60.0)
    
    # ── Rule 6: Clip Heart Rate to valid range ──
    # A resting heart rate below 40 or above 150 is not realistic for a patient
    df_val["Heart_Rate"] = np.clip(df_val["Heart_Rate"], 40, 150)
    
    logger.info(f"Clinical audit completed. Never-smoker anomalies corrected: {n_anomalies_smokers} records.")
    logger.info(f"Years smoked vs. Age boundary anomalies corrected: {n_too_many_years} records.")
    logger.info(f"Validated 100% of patient records successfully. No records dropped, all corrected.")
    
    return df_val
