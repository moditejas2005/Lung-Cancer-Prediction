# Local Demo Guide: Running Lung Cancer Predictor

This guide provides step-by-step instructions on how to quickly set up and run the Lung Cancer Predictor project locally on your college PC for your demo today.

## Prerequisites
- A Windows PC (your college PC)
- **Python 3.10** (or similar Python 3 version) installed
- **Git** installed (if you are cloning from GitHub)
- Internet connection to download packages

---

## Step 1: Get the Code onto the College PC

You have two options to get the code onto the college PC:

**Option A (Using a USB Drive):**
1. Copy your entire `Lung_Cancer_Prediction` folder to a USB drive.
2. Plug the USB drive into your college PC and copy the folder to the Desktop.

**Option B (Using GitHub - Recommended):**
1. Open the Command Prompt or Terminal on the college PC.
2. Navigate to the Desktop (or wherever you want the project):
   ```cmd
   cd Desktop
   ```
3. Clone your GitHub repository:
   ```cmd
   git clone https://github.com/moditejas2005/Lung-Cancer-Prediction.git
   ```
4. Enter the downloaded folder:
   ```cmd
   cd Lung-Cancer-Prediction
   ```

---

## Step 2: Set Up the Python Environment

To avoid interfering with any existing Python setups on the college PC, it's best to use a virtual environment.

1. In your Command Prompt (making sure you are inside the `Lung_Cancer_Prediction` folder), create a virtual environment named `venv`:
   ```cmd
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```cmd
   venv\Scripts\activate
   ```
   *(You should see `(venv)` appear at the start of your command prompt line).*

---

## Step 3: Install the Dependencies

Now, we will install only the necessary packages to run the web application (skipping the massive training libraries to save time and space).

1. Ensure your virtual environment is activated `(venv)`.
2. Run the following command to install the lightweight dependencies:
   ```cmd
   pip install pandas numpy scikit-learn xgboost catboost scipy joblib flask
   ```
   *(This will quickly install Flask, Pandas, Scikit-Learn, XGBoost, and CatBoost).*

---

## Step 4: Run the Application

Once the installation finishes, you are ready to start the server!

1. Start the Flask application by running:
   ```cmd
   python app.py
   ```
2. You will see an output that looks like this:
   ```text
    * Serving Flask app 'app'
    * Debug mode: off
    * Running on http://127.0.0.1:5000
   ```
3. Open your web browser (Chrome, Edge, etc.) and go to this exact address:
   **http://127.0.0.1:5000** or **http://localhost:5000**

---

## Step 5: Stop the Server (After Demo)

When your demo is complete, go back to the Command Prompt window where the server is running and press **`Ctrl + C`** on your keyboard to stop it. 

Good luck with your college demo!
