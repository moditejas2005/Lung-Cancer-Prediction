# 🫁 Lung Cancer Prediction — Local Demo Guide

A quick, step-by-step guide to set up and run the **Lung Cancer Prediction** web application on any Windows PC for a college demo.

---

## 📋 Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Windows 10 / 11 |
| **Python** | 3.10 or newer |
| **Git** | Installed (for GitHub clone option) |
| **Internet** | Needed only to install packages |

---

## Step 1 — Get the Code

You have two options:

**Option A — USB Drive (No internet needed):**
1. Copy the entire `Lung_Cancer_Prediction` folder to a USB drive.
2. Plug it into the college PC and paste the folder onto the Desktop.

**Option B — Clone from GitHub (Recommended):**
```cmd
cd Desktop
git clone https://github.com/moditejas2005/Lung-Cancer-Prediction.git
cd Lung-Cancer-Prediction
```

---

## Step 2 — Create a Virtual Environment

Open **Command Prompt** inside the `Lung_Cancer_Prediction` folder, then run:

```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your prompt line — this means the virtual environment is active.

---

## Step 3 — Install Dependencies

With `(venv)` active, run:

```cmd
pip install flask pandas numpy scikit-learn xgboost catboost scipy joblib
```

> **Note:** This installs only what's needed to run the web app.  
> The full training pipeline (CTGAN, Optuna, SHAP) is **not required** for the demo — the pre-trained model is already saved in the `models/` folder.

---

## Step 4 — Run the Application

```cmd
python app.py
```

You will see output like:
```
 * Serving Flask app 'app'
 * Debug mode: off
 * Running on http://127.0.0.1:5000
```

Open your browser and go to:  
👉 **http://127.0.0.1:5000** or **http://localhost:5000**

---

## Step 5 — Using the Web App

1. Fill in the patient form (Age, Gender, Smoking History, Symptoms, etc.)
2. Click **"Predict"**
3. The system will show:
   - ✅ **Cancer Risk Level** (High / Low)
   - 📊 **Probability Score** (e.g., 84.3%)
   - 🔍 **Key Risk Factors** contributing to the result

---

## Step 6 — Stop the Server

When your demo is done, press **`Ctrl + C`** in the Command Prompt to stop the server.

---

## 🗂️ Project Structure (Quick Reference)

```
Lung_Cancer_Prediction/
│
├── app.py                  ← Flask web server (main entry point)
├── inference.py            ← Model loading & prediction logic
├── validators.py           ← Clinical input validation rules
├── pipeline_runner.py      ← Full training pipeline (not needed for demo)
│
├── models/                 ← Pre-trained model files (.joblib)
├── templates/              ← HTML pages for the web interface
│
├── augmentation/           ← CTGAN synthetic data generation
├── preprocessing/          ← Data cleaning & feature engineering
├── training/               ← Model training, ensembles & calibration
├── validation/             ← Statistical & medical data validation
├── explainability/         ← SHAP explainability analysis
│
├── data/                   ← Generated datasets & evaluation reports
└── logs/                   ← Pipeline and application log files
```

---

## ⚠️ Common Issues & Fixes

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: flask` | Run `pip install flask` with `(venv)` active |
| `ModuleNotFoundError: xgboost` | Run `pip install xgboost` |
| `Port 5000 already in use` | Kill existing process or use `python app.py --port 5001` |
| Browser shows blank page | Wait 3–5 seconds and refresh; Flask may still be starting |
| `Model file not found` | Ensure the `models/` folder is present with `.joblib` files |

---

Good luck with your demo! 🎓
