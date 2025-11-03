"""train_models.py

Trains Gaussian Naive Bayes and Logistic Regression on the PIMA diabetes dataset.
If `diabetes.csv` doesn't exist in the project root, the script will try to download a common copy.
Saves scaler and models into `models/`.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    # xgboost may not be installed; the training runner will install it when needed
    XGBClassifier = None

# Settings
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "diabetes.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# If data not present, try to download a public copy
DATA_URLS = [
    # A commonly used copy with headers
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    # fallback (columns in a different format sometimes)
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
]

if not os.path.exists(DATA_PATH):
    print("diabetes.csv not found locally. Attempting to download a public copy...")
    success = False
    for url in DATA_URLS:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            text = r.text
            # If second URL (no headers), add headers to make compatible
            if "Pregnancies" not in text.splitlines()[0]:
                headers = "Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome\n"
                text = headers + text
            with open(DATA_PATH, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Downloaded dataset from {url}")
            success = True
            break
        except Exception as e:
            print(f"Download failed from {url}: {e}")
    if not success:
        raise SystemExit("Could not download diabetes dataset. Please place diabetes.csv in project root and rerun.")

# 1. Load data
print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# Validate expected columns
expected_cols = set(["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"])
if not expected_cols.issubset(df.columns):
    raise SystemExit("Dataset is missing expected columns. Please provide a PIMA-formatted diabetes.csv")

# 2. Data cleaning: replace zeros in certain columns with mean (common PIMA preprocessing)
cols_replace_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_replace_zero:
    if col in df.columns:
        zero_mask = df[col] == 0
        if zero_mask.any():
            mean_val = df.loc[~zero_mask, col].mean()
            df.loc[zero_mask, col] = mean_val
            print(f"Replaced zeros in {col} with mean: {mean_val:.3f}")

print('\nMissing values after cleaning:\n', df.isnull().sum())

# 3. EDA summary and heatmap
print('\nData summary:\n', df.describe())
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
heatmap_path = os.path.join(ROOT, "correlation_heatmap.png")
plt.savefig(heatmap_path)
print(f"Saved correlation heatmap to {heatmap_path}")
plt.close()

# 4. Feature scaling
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Saved scaler to {scaler_path}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Train models
# Baseline: Naive Bayes (fast)
nb = GaussianNB()
print("Training Gaussian Naive Bayes...")
nb.fit(X_train, y_train)

# Hyperparameter tuning for Logistic Regression using GridSearchCV
print("Tuning Logistic Regression with GridSearchCV...")
lr = LogisticRegression(max_iter=1000, solver="liblinear")
lr_param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_grid = GridSearchCV(lr, lr_param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
lr_grid.fit(X_train, y_train)
logreg_best = lr_grid.best_estimator_
print(f"Best LogisticRegression params: {lr_grid.best_params_}")

# Add RandomForest and tune it — often improves accuracy on tabular data
print("Tuning RandomForest with GridSearchCV...")
rf = RandomForestClassifier(random_state=42)
rf_param_grid = {"n_estimators": [100, 200], "max_depth": [None, 6, 12], "min_samples_split": [2, 5]}
rf_grid = GridSearchCV(rf, rf_param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
print(f"Best RandomForest params: {rf_grid.best_params_}")

# Try Gradient Boosting (often strong on tabular data)
print("Tuning GradientBoosting with GridSearchCV...")
gb = GradientBoostingClassifier(random_state=42)
gb_param_grid = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
gb_grid = GridSearchCV(gb, gb_param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
gb_grid.fit(X_train, y_train)
gb_best = gb_grid.best_estimator_
print(f"Best GradientBoosting params: {gb_grid.best_params_}")

# Try XGBoost (if available) — often performs well on tabular data
if XGBClassifier is not None:
    print("Tuning XGBoost with GridSearchCV...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_param_grid = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
    xgb_grid = GridSearchCV(xgb, xgb_param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    xgb_best = xgb_grid.best_estimator_
    print(f"Best XGBoost params: {xgb_grid.best_params_}")
else:
    xgb_best = None
    print("XGBoost not installed; skipping XGBoost tuning. Install xgboost to enable.")

# Predictions on test set
y_pred_nb = nb.predict(X_test)
y_pred_lr = logreg_best.predict(X_test)
y_pred_rf = rf_best.predict(X_test)
y_pred_gb = gb_best.predict(X_test)
if xgb_best is not None:
    y_pred_xgb = xgb_best.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
else:
    y_pred_xgb = None
    acc_xgb = None

acc_nb = accuracy_score(y_test, y_pred_nb)
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_gb = accuracy_score(y_test, y_pred_gb)
print(f"\nNaive Bayes Accuracy: {acc_nb:.4f}")
print(f"Logistic Regression (tuned) Accuracy: {acc_lr:.4f}")
print(f"Random Forest (tuned) Accuracy: {acc_rf:.4f}")
print(f"Gradient Boosting (tuned) Accuracy: {acc_gb:.4f}")
if acc_xgb is not None:
    print(f"XGBoost (tuned) Accuracy: {acc_xgb:.4f}")

# Save models using joblib for better compatibility
nb_path = os.path.join(MODEL_DIR, "nb_model.pkl")
logreg_path = os.path.join(MODEL_DIR, "logreg_model.pkl")
rf_path = os.path.join(MODEL_DIR, "rf_model.pkl")
joblib.dump(nb, nb_path)
joblib.dump(logreg_best, logreg_path)
joblib.dump(rf_best, rf_path)
gb_path = os.path.join(MODEL_DIR, "gb_model.pkl")
joblib.dump(gb_best, gb_path)
if xgb_best is not None:
    xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    joblib.dump(xgb_best, xgb_path)
    print(f"Saved XGBoost model to {xgb_path}")
print(f"Saved models to {nb_path}, {logreg_path} and {rf_path}")

# 6. Evaluation
cm_nb = confusion_matrix(y_test, y_pred_nb)
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("\nNaive Bayes Confusion Matrix:\n", cm_nb)
print("\nLogistic Regression Confusion Matrix:\n", cm_lr)

print("\nNaive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))

# ROC
y_prob_nb = nb.predict_proba(X_test)[:, 1]
# For some sklearn versions GaussianNB doesn't implement predict_proba for certain cases; guard it
try:
    y_prob_lr = logreg_best.predict_proba(X_test)[:, 1]
except Exception:
    # fallback to decision_function
    y_prob_lr = logreg_best.decision_function(X_test)

fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
auc_nb = auc(fpr_nb, tpr_nb)
auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(6, 5))
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC={auc_nb:.3f})')
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
roc_path = os.path.join(ROOT, "roc_curve.png")
plt.savefig(roc_path)
print(f"Saved ROC curve to {roc_path}")
plt.close()

print("\nModel training complete. Models saved in /models folder.")
