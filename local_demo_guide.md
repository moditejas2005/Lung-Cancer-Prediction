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

Depending on whether you just want to run the web app demo or run the entire pipeline from scratch, choose **one** of these options:

### Option A: Web App Demo Only (Fastest & Lightweight)
If you only need to run the web interface (using pre-trained models already in the `models/` folder):
```cmd
pip install flask pandas numpy scikit-learn xgboost catboost scipy joblib
```

### Option B: Full Training Pipeline (Run Everything From Scratch)
If you want to reinstall and run the entire data generation and model training pipeline from the very beginning:
```cmd
pip install -r requirements.txt
```
*(This will install CTGAN, PyTorch, Optuna, SHAP, and all other advanced modeling packages).*

---

## Step 4 — Run the Project

Choose the command below depending on what you want to show:

### 🌐 Option A: Launch the Web App
To run the pre-trained clinical web application interface:
```cmd
python app.py
```
* Go to **http://127.0.0.1:5000** in your browser.

---

### ⚙️ Option B: Run the Full Pipeline From Scratch
If you want to demonstrate how you generated the dataset and trained the models from scratch, run the `pipeline_runner.py` script. 

To run it **quickly** for a live demo (takes ~2 minutes):
```cmd
python pipeline_runner.py --test-mode
```

To run a **full production run** (takes ~30-45 minutes depending on GPU):
```cmd
python pipeline_runner.py
```

#### What `pipeline_runner.py` does automatically:
1. **Generates Base Data:** Simulates raw medical patient distributions.
2. **Cleans Data:** Cleans, formats, and handles missing details.
3. **Feature Engineering:** Computes medical index features (BMI, Smoking Risk, etc.).
4. **CTGAN Data Augmentation:** Trains a Conditional Generative Adversarial Network to generate synthetic records.
5. **Data Validation:** Checks data drift (PSI) and runs statistical and medical safety tests on synthetic data.
6. **Ensemble Model Training:** Trains XGBoost, CatBoost, Random Forest, etc.
7. **Hyperparameter Tuning:** Uses Optuna to find the best model configuration.
8. **Calibration:** Calibrates model probabilities (Isotonic Scaling).
9. **Threshold Optimization:** Balances precision and recall to set the safest classification cutoff.
10. **Explainability:** Calculates global and local explanations via SHAP.
11. **Report Generation:** Outputs Excel evaluation sheets and plots inside the `data/reports/` folder.

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
├── pipeline_runner.py      ← Full training pipeline (runs everything from scratch)
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
