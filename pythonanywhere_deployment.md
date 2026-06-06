# Deployment Guide: PythonAnywhere & GitHub

This guide covers how to deploy the Lung Cancer Prediction application on PythonAnywhere since your code is already pushed to GitHub.

---

## Part 1: Your GitHub Repository

Your code has already been successfully pushed to GitHub!
You can view your repository here: [https://github.com/Adityadubbewar-oss/Lung-Cancer-Prediction](https://github.com/Adityadubbewar-oss/Lung-Cancer-Prediction)

---

## Part 2: Deploying to PythonAnywhere

### Step 1: Set Up PythonAnywhere Account
1. Go to [PythonAnywhere](https://www.pythonanywhere.com/) and log in (or create a free Beginner account if you haven't).
2. If this is a new account, verify your email address.

### Step 2: Open a Bash Console
1. Go to the **Consoles** tab.
2. Under "Start a new console", click **Bash**.

### Step 3: Clone Your Repository
In the Bash console, run the following command to download your code from GitHub:
```bash
git clone https://github.com/Adityadubbewar-oss/Lung-Cancer-Prediction.git
```
*(You will be asked for your username and password. For the password, use your GitHub Personal Access Token (PAT) that you previously used.)*

### Step 4: Create a Virtual Environment
1. Still in the Bash console, navigate to your new project directory:
   ```bash
   cd Lung-Cancer-Prediction
   ```
2. Create a virtual environment using Python 3.10:
   ```bash
   mkvirtualenv --python=/usr/bin/python3.10 myenv
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
*(Note: If you run into issues with numpy/pandas versions, run `pip install "numpy<2.0" --force-reinstall`)*

### Step 5: Set Up the Web App
1. Click the **Web** tab in the PythonAnywhere header.
2. Click the **Add a new web app** button.
3. Click **Next** on the domain name screen.
4. Select **Manual configuration** (do NOT choose Flask).
5. Select **Python 3.10** (or whatever version you used for `myenv`).
6. Click **Next** to finish creating the app.

### Step 6: Configure the WSGI File
1. On your newly created Web app page, scroll down to the **Code** section.
2. In the **Source code** box, enter: `/home/YOUR_PYTHONANYWHERE_USERNAME/Lung-Cancer-Prediction` and click the checkmark to save it. (Replace `YOUR_PYTHONANYWHERE_USERNAME` with your actual PythonAnywhere username, usually it's `Aditya2004` based on your previous deployment).
3. Click on the link next to **WSGI configuration file** (it ends in `_wsgi.py`).
4. Delete EVERYTHING in that file and replace it with:

```python
import sys
import os

# 1. Add your project directory to the sys.path
path = '/home/YOUR_PYTHONANYWHERE_USERNAME/Lung-Cancer-Prediction'
if path not in sys.path:
    sys.path.insert(0, path)

# 2. Import your Flask app
from app import app as application
```
*(Make sure you replace `YOUR_PYTHONANYWHERE_USERNAME` with your actual PythonAnywhere username!)*
5. Click the green **Save** button at the top right, then go back to the Web tab.

### Step 7: Link the Virtual Environment
1. Scroll down to the **Virtualenv** section on the Web tab.
2. Enter the path to your virtual environment: `/home/YOUR_PYTHONANYWHERE_USERNAME/.virtualenvs/myenv` and click the checkmark.

### Step 8: Launch the Site
1. Scroll to the top of the **Web** tab and click the large green **Reload** button.
2. Click the link at the top (e.g., `http://yourusername.pythonanywhere.com`) to view your live site!

If you see an error like "Something went wrong", check the **Error log** link at the bottom of the Web tab.
