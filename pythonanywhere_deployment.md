# Deployment Guide: PythonAnywhere & GitHub

This guide covers how to push your local code to your new GitHub account and then deploy the application on PythonAnywhere.

---

## Part 1: Pushing Code to GitHub

Since you have a new GitHub account (`moditejas2005`), you'll need to manually create the repository first. 

GitHub no longer accepts account passwords for pushing code via the command line. You must use a **Personal Access Token (PAT)**. 

### Step 1: Create a Personal Access Token
1. Log into GitHub and go to **Settings** (click your profile picture in the top right).
2. Scroll to the bottom of the left sidebar and click **Developer settings**.
3. Click **Personal access tokens** -> **Tokens (classic)**.
4. Click **Generate new token** -> **Generate new token (classic)**.
5. In the "Note" field, type something like "PythonAnywhere Deployment".
6. Under "Select scopes", check the box next to **repo** (this gives it permission to push code).
7. Scroll down and click **Generate token**.
8. **IMPORTANT:** Copy the token that appears on the next screen (it starts with `ghp_`). You will not be able to see it again!

### Step 2: Create the Repository on GitHub
1. Go to your GitHub homepage and click the **New** button to create a new repository.
2. Name the repository `Lung-Cancer-Prediction`.
3. Do NOT check any boxes to initialize it with a README, .gitignore, or license. Leave it completely empty.
4. Click **Create repository**.

### Step 3: Push Your Code
Now that your repository is created, open your command prompt or terminal, make sure you are in your project folder (`c:\Users\hp\Desktop\Cancer\Lung_Cancer_Prediction`), and run these exact commands one by one:

```bash
git remote add origin https://github.com/moditejas2005/Lung-Cancer-Prediction.git
git branch -M main
git push -u origin main
```
*(When prompted for your password, paste the Personal Access Token (PAT) you generated in Step 1!)*

Once this is successful, your code is on GitHub! You can now move on to Part 2.

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
git clone https://github.com/moditejas2005/Lung-Cancer-Prediction.git
```
*(For the password, use your GitHub Personal Access Token (PAT) that you generated in Part 1.)*

### Step 4: Create a Virtual Environment
1. Still in the Bash console, navigate to your new project directory:
   ```bash
   cd Lung-Cancer-Prediction
   ```
2. Create a virtual environment using Python 3.10 (WITH system packages to save space):
   ```bash
   mkvirtualenv --python=/usr/bin/python3.10 --system-site-packages myenv
   ```
3. **IMPORTANT - Clear your cache first** to free up space (to avoid the Disk Quota error):
   ```bash
   rm -rf ~/.cache/pip
   ```
4. Install only the specific missing deployment dependencies (we skip pandas/numpy/scikit-learn because PythonAnywhere already has them pre-installed for free!):
   ```bash
   pip install --no-cache-dir xgboost catboost
   ```

### Step 5: Set Up the Web App
1. Click the **Web** tab in the PythonAnywhere header.
2. Click the **Add a new web app** button.
3. Click **Next** on the domain name screen.
4. Select **Manual configuration** (do NOT choose Flask).
5. Select **Python 3.10** (or whatever version you used for `myenv`).
6. Click **Next** to finish creating the app.

### Step 6: Configure the WSGI File
1. On your newly created Web app page, scroll down to the **Code** section.
2. In the **Source code** box, enter: `/home/YOUR_PYTHONANYWHERE_USERNAME/Lung-Cancer-Prediction` and click the checkmark to save it. (Replace `YOUR_PYTHONANYWHERE_USERNAME` with your actual PythonAnywhere username).
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
