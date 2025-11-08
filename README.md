# Diabetes Prediction Project

## Project Overview
This project implements machine learning models to predict diabetes risk using the PIMA Indians Diabetes Dataset. It trains multiple ML models including Gaussian Naive Bayes and Logistic Regression, exposes them through a Flask API, and provides a React frontend for real-time predictions.

## Models & Performance
- **Models Implemented:**
  - Logistic Regression (with GridSearchCV optimization)
  - Gaussian Naive Bayes
  - Additional models planned: Random Forest, Gradient Boosting

### Model Performance Metrics
Based on our latest evaluation:
- Logistic Regression: ~78.5% accuracy
- Naive Bayes: ~76.2% accuracy
- Cross-validation scores and ROC curves available in `roc_curve.png`

## Dataset Information
- **Source:** PIMA Indians Diabetes Dataset
- **Features:**
  1. Pregnancies: Number of times pregnant
  2. Glucose: Plasma glucose concentration
  3. Blood Pressure: Diastolic blood pressure (mm Hg)
  4. Skin Thickness: Triceps skin fold thickness (mm)
  5. Insulin: 2-Hour serum insulin (mu U/ml)
  6. BMI: Body mass index
  7. Diabetes Pedigree Function: Diabetes pedigree function
  8. Age: Age in years
- **Target:** Diabetes Outcome (0: No Diabetes, 1: Diabetes)
- **Size:** 768 records

## Project Structure
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

## Setup & Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup (Windows PowerShell)

1. Create and activate Python virtual environment:
```powershell
python -m venv venv
venv\Scripts\Activate.ps1   # if using PowerShell
# or use venv\Scripts\activate.bat for cmd
```

2. Install Python dependencies:
```powershell
pip install -r requirements.txt
```

3. Train models (downloads dataset if missing):
```powershell
python train_models.py
```

This creates:
- `models/scaler.pkl`: StandardScaler for feature normalization
- `models/nb_model.pkl`: Trained Naive Bayes model
- `models/logreg_model.pkl`: Trained Logistic Regression model
- `roc_curve.png`: ROC curves comparing model performance
- `correlation_heatmap.png`: Feature correlation visualization

4. Start the Flask API:
```powershell
python app.py
```
The API will be available at http://127.0.0.1:5000

### Frontend Setup

1. Navigate to frontend directory and install dependencies:
```powershell
cd frontend
npm install
```

2. Start the development server:
```powershell
npm start
```
Access the web interface at http://localhost:3000

## API Documentation

### GET /
Returns:
- API status
- List of required features
- Available models

### POST /predict
Make predictions using trained models.

Request body:
```json
{
    "Pregnancies": 2,
    "Glucose": 121,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 79,
    "BMI": 25.6,
    "DiabetesPedigreeFunction": 0.2,
    "Age": 33,
    "model": "logreg"  // Optional, defaults to "logreg"
}
```

Response:
```json
{
    "prediction": "Non-Diabetic",
    "prediction_binary": 0,
    "probability": 0.82,
    "model": "logreg"
}

## Implementation Details

### Data Preprocessing
- Missing value handling: Zeros replaced with mean values for medical measurements
- Feature scaling: StandardScaler applied to all features
- Train/Test split: 80/20 with stratification for balanced classes

### Model Training
- **Logistic Regression:**
  - GridSearchCV for hyperparameter optimization
  - Parameters tuned: C, penalty, solver
  - Best parameters saved with model
  
- **Naive Bayes:**
  - Gaussian NB implementation
  - No hyperparameter tuning needed
  - Handles numerical features well

### Visualizations
- **ROC Curves (`roc_curve.png`):**
  - Compares model performance
  - Shows true positive vs false positive rates
  
- **Correlation Heatmap (`correlation_heatmap.png`):**
  - Visualizes feature relationships
  - Helps identify potential feature importance

## Development Notes
- Dataset automatically downloads if `diabetes.csv` is missing
- All models and scaler are persisted using joblib
- Frontend uses React hooks for state management
- CORS enabled for local development

## Future Improvements
1. Add more models (Random Forest, XGBoost)
2. Implement k-fold cross-validation
3. Add feature importance analysis
4. Expand frontend visualizations
5. Add model retraining endpoint
6. Containerize application

## Contributing
Pull requests welcome! Please ensure:
- Code passes linting
- New features include tests
- Documentation is updated