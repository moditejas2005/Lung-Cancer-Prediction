"""
data_cleaner.py - Dataset Generator & Cleaner
===============================================
This module has two jobs:

  1. generate_base_dataset():
     Creates a realistic fake patient dataset from scratch using statistical rules.
     It simulates things like:
       - Demographics (age, gender)
       - Smoking history (status, intensity, pack-years)
       - Environmental exposure (PM2.5, radon, asbestos)
       - Physiological measurements (BMI, oxygen saturation, heart rate)
       - Symptoms (cough, breathlessness, chest pain)
       - Cancer diagnosis (using a logistic risk formula based on all the above)

  2. clean_dataset():
     Takes the raw generated (or CTGAN synthetic) data and cleans it:
       - Splits "120/80" blood pressure strings into two numeric columns
       - Fills any missing values with median/mode
       - Standardizes column data types
"""

import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataCleaner")

def generate_base_dataset(n_samples=5000, seed=42):
    """
    Creates a realistic synthetic patient dataset from scratch using statistical rules.
    This is the "seed" data that the CTGAN model will learn from.

    Args:
        n_samples : Number of patient records to generate (default 5000)
        seed      : Random seed for reproducibility (default 42)

    Returns:
        df: A pandas DataFrame with one row per patient and all medical features + Diagnosis label
    """
    logger.info(f"Generating base medical dataset with {n_samples} records (Seed: {seed})...")
    np.random.seed(seed)  # Fix random seed so results are the same every run
    
    # ── 1. Demographics ──
    age = np.random.randint(18, 90, size=n_samples)   # Ages 18-90 uniformly distributed
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.48, 0.52])  # Slight female majority
    
    # ── 2. Smoking Profiles ──
    # 50% never smokers, 30% former smokers, 20% current smokers (realistic population ratios)
    smoking_status = np.random.choice(["Never", "Former", "Current"], size=n_samples, p=[0.50, 0.30, 0.20])
    
    # Initialize all smoking values to 0/"None" (they'll be filled for smokers below)
    smoking_intensity = np.zeros(n_samples, dtype=int)
    smoking_frequency = np.array(["None"] * n_samples, dtype=object)
    years_smoked = np.zeros(n_samples, dtype=int)
    cigarettes_per_day = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        if smoking_status[i] == "Never":
            continue  # Leave all values at 0 — never-smokers have no smoking history
        elif smoking_status[i] == "Former":
            # Former smokers: lighter smoking history
            smoking_intensity[i] = np.random.randint(1, 6)   # Intensity 1-5
            smoking_frequency[i] = np.random.choice(["Occasional", "Regular"], p=[0.4, 0.6])
            # Years smoked cannot exceed (Age - 15) — you can't smoke before ~age 15
            max_years = max(1, age[i] - 15)
            years_smoked[i] = np.random.randint(1, max_years + 1)
            cigarettes_per_day[i] = np.random.randint(2, 20)  # 2-20 cigarettes/day
        else:  # Current smoker — heavier smoking history
            smoking_intensity[i] = np.random.randint(3, 11)   # Intensity 3-10 (heavier)
            smoking_frequency[i] = "Regular"
            max_years = max(1, age[i] - 15)
            years_smoked[i] = np.random.randint(3, max_years + 1)
            cigarettes_per_day[i] = np.random.randint(5, 41)  # 5-40 cigarettes/day
            
    # Pack-Years = (Cigarettes per day / 20) × Years smoked
    # This is a standard medical measure of cumulative smoking exposure
    pack_years = (cigarettes_per_day / 20.0) * years_smoked
    
    # ── 3. Environmental Exposure ──
    pm25 = np.random.uniform(5.0, 120.0, size=n_samples)                    # PM2.5 air pollution (μg/m³)
    radon = np.random.uniform(0.1, 15.0, size=n_samples)                    # Radon gas (Becquerels)
    asbestos = np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08])     # 8% of population exposed to asbestos
    
    # ── 4. Body Mass Index (BMI) ──
    # Normal distribution centered at 26.5 (slightly overweight average)
    bmi = np.random.normal(26.5, 5.0, size=n_samples)
    bmi = np.clip(bmi, 15.0, 45.0)  # Clip to realistic range (underweight to severely obese)
    
    # ── 5. Cancer Risk Formula (Logistic Regression) ──
    # This is a sigmoid function that converts a risk score to a probability.
    # Each factor adds to the logit (log-odds) of having cancer:
    #   - Older age → higher risk
    #   - More pack-years → higher risk
    #   - Higher PM2.5 → higher risk
    #   - Higher radon → higher risk
    #   - Asbestos exposure → much higher risk (coefficient 2.2 is very strong)
    #   - Obese BMI (>30) → slightly higher risk
    logit = (
        -5.0 
        + 0.035 * age 
        + 0.075 * pack_years 
        + 0.015 * pm25 
        + 0.12 * radon 
        + 2.2 * asbestos 
        + 0.05 * np.maximum(0, bmi - 30)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    # Add small randomness
    diagnosis = np.random.binomial(1, prob)
    
    # 6. Physiological symptoms dependent on diagnosis and smoking
    oxygen_saturation = np.zeros(n_samples)
    breathlessness = np.zeros(n_samples, dtype=int)
    chronic_cough = np.zeros(n_samples, dtype=int)
    chest_pain = np.zeros(n_samples, dtype=int)
    wheezing = np.zeros(n_samples, dtype=int)
    fatigue_level = np.zeros(n_samples, dtype=int)
    heart_rate = np.zeros(n_samples, dtype=int)
    blood_pressure = []
    
    for i in range(n_samples):
        is_cancer = (diagnosis[i] == 1)
        is_smoker = (smoking_status[i] != "Never")
        
        # Oxygen Saturation: lower if cancer, older age, smoker, or high PM2.5
        o2_base = 98.5
        if is_cancer:
            o2_base -= np.random.uniform(3.0, 12.0)
        if is_smoker:
            o2_base -= np.random.uniform(0.5, 3.0)
        o2_base -= (age[i] / 50.0)
        o2_base -= (pm25[i] / 100.0)
        oxygen_saturation[i] = np.clip(o2_base, 80.0, 100.0)
        
        # Symptoms probability based on clinical correlations
        p_cough = 0.85 if is_cancer else (0.45 if is_smoker else 0.12)
        chronic_cough[i] = np.random.binomial(1, p_cough)
        
        p_breath = 0.75 if is_cancer else (0.35 if (is_smoker or bmi[i] > 32) else 0.10)
        breathlessness[i] = np.random.binomial(1, p_breath)
        
        p_pain = 0.65 if is_cancer else 0.08
        chest_pain[i] = np.random.binomial(1, p_pain)
        
        p_wheeze = 0.60 if is_cancer else (0.35 if is_smoker else 0.08)
        wheezing[i] = np.random.binomial(1, p_wheeze)
        
        # Fatigue
        fatigue_base = np.random.randint(0, 3)
        if is_cancer:
            fatigue_base += np.random.randint(2, 4)
        fatigue_level[i] = min(5, fatigue_base)
        
        # Heart Rate
        hr = int(np.random.normal(72, 10))
        if is_cancer:
            hr += np.random.randint(5, 15)
        heart_rate[i] = np.clip(hr, 50, 120)
        
        # Blood pressure systolic/diastolic
        bp_systolic = int(np.random.normal(120, 12))
        if bmi[i] > 30:
            bp_systolic += int((bmi[i] - 30) * 1.5)
        if age[i] > 60:
            bp_systolic += int((age[i] - 60) * 0.4)
            
        bp_diastolic = int(bp_systolic * 0.65 + np.random.normal(0, 5))
        blood_pressure.append(f"{bp_systolic}/{bp_diastolic}")

    # Oxygen percentage
    oxygen_percentage = oxygen_saturation * np.random.uniform(0.98, 1.0)
    
    # 7. Assemble DataFrame
    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Smoking_Status": smoking_status,
        "Smoking_Intensity": smoking_intensity,
        "Smoking_Frequency": smoking_frequency,
        "Years_Smoked": years_smoked,
        "Cigarettes_Per_Day": cigarettes_per_day,
        "Pack_Years": pack_years,
        "PM25_Level": pm25,
        "Radon_Exposure": radon,
        "Asbestos_Exposure": asbestos,
        "BMI": bmi,
        "Oxygen_Saturation": oxygen_saturation,
        "Oxygen_Percentage": oxygen_percentage,
        "Breathlessness": breathlessness,
        "Chronic_Cough": chronic_cough,
        "Chest_Pain": chest_pain,
        "Wheezing": wheezing,
        "Fatigue_Level": fatigue_level,
        "Heart_Rate": heart_rate,
        "Blood_Pressure": blood_pressure,
        "Diagnosis": diagnosis
    })
    
    return df

def clean_dataset(df):
    """
    Cleans a given medical dataset, handling parsing and types.
    """
    logger.info("Cleaning medical dataset and parsing Blood_Pressure...")
    df_cleaned = df.copy()
    
    # Parse Blood_Pressure into Systolic and Diastolic
    if "Blood_Pressure" in df_cleaned.columns:
        systolics = []
        diastolics = []
        for val in df_cleaned["Blood_Pressure"]:
            try:
                parts = str(val).split("/")
                systolics.append(float(parts[0]))
                diastolics.append(float(parts[1]))
            except Exception:
                systolics.append(120.0)
                diastolics.append(80.0)
                
        df_cleaned["BP_Systolic"] = systolics
        df_cleaned["BP_Diastolic"] = diastolics
        df_cleaned = df_cleaned.drop(columns=["Blood_Pressure"])
    else:
        logger.info("Blood_Pressure column not found; skipping parsing.")
    
    # Standardize types
    df_cleaned["Gender"] = df_cleaned["Gender"].astype(str)
    df_cleaned["Smoking_Status"] = df_cleaned["Smoking_Status"].astype(str)
    df_cleaned["Smoking_Frequency"] = df_cleaned["Smoking_Frequency"].astype(str)
    
    # Fill missing values if any
    for col in df_cleaned.select_dtypes(include=[np.number]).columns:
        if df_cleaned[col].isnull().any():
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            
    for col in df_cleaned.select_dtypes(include=["object", "category"]).columns:
        if df_cleaned[col].isnull().any():
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            
    logger.info("Dataset cleaning completed.")
    return df_cleaned
