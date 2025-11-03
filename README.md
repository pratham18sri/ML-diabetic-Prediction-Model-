# Diabetes Prediction Project

This project trains two ML models (Gaussian Naive Bayes and Logistic Regression) on the PIMA Diabetes dataset and exposes a Flask API to get predictions. A React frontend provides a form to send patient data and show predictions.

Folder structure
```
diabetes_project/
├── models/                     # saved scaler and model pickles (created after training)
├── diabetes.csv                # dataset (script will try to download if missing)
├── train_models.py             # training + evaluation + save models
├── app.py                      # Flask API for prediction
├── requirements.txt
├── .gitignore
└── frontend/
    ├── package.json
    └── src/
        ├── index.js
        ├── App.jsx
        └── PredictionForm.jsx
```

Quick start (Windows PowerShell)

1. Create Python venv and activate

```powershell
python -m venv venv
venv\Scripts\Activate.ps1   # if using PowerShell; or venv\Scripts\activate.bat for cmd
```

2. Install Python deps

```powershell
pip install -r requirements.txt
```

3. Train models (downloads dataset if missing)

```powershell
python train_models.py
```

This creates `models/scaler.pkl`, `models/nb_model.pkl`, and `models/logreg_model.pkl` and saves `roc_curve.png` and `correlation_heatmap.png`.

4. Run Flask API

```powershell
python app.py
```

The API will listen on http://127.0.0.1:5000

5. Frontend

Open a new terminal and go to `frontend`:

```powershell
cd frontend
npm install
npm start
```

Open http://localhost:3000

Notes
- The training script attempts to download a standard diabetes CSV if `diabetes.csv` is not present.
- The Flask `/predict` endpoint expects JSON with the fields: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`. You can optionally provide `model` = "logreg" or "nb" (default: logreg).

If you'd like, I can now run quick checks (syntax linting) or run the training script here — tell me which you'd prefer.