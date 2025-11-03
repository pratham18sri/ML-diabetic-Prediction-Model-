"""Flask API for Diabetes Prediction

Loads scaler and models from models/ and exposes /predict endpoint.
Accepts JSON with the 8 features. Optional 'model' field: 'logreg' or 'nb'. Default: logreg.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np

app = Flask(__name__)
CORS(app)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")

# Load artifacts
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
nb_path = os.path.join(MODEL_DIR, "nb_model.pkl")
logreg_path = os.path.join(MODEL_DIR, "logreg_model.pkl")

if not os.path.exists(scaler_path) or not os.path.exists(nb_path) or not os.path.exists(logreg_path):
    raise SystemExit("Required model files not found in models/. Run train_models.py first.")

try:
    scaler = joblib.load(scaler_path)
    nb = joblib.load(nb_path)
    logreg = joblib.load(logreg_path)
except Exception as e:
    raise SystemExit(f"Error loading models: {e}")

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "Diabetes prediction API", "features": FEATURES})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400
    # allow form-data like access
    try:
        model_choice = data.get("model", "logreg").lower() if isinstance(data, dict) else "logreg"
    except Exception:
        model_choice = "logreg"

    try:
        values = [float(data[f]) for f in FEATURES]
    except Exception as e:
        return jsonify({"error": f"Invalid input or missing fields. Required: {FEATURES}. Detail: {e}"}), 400

    X = scaler.transform([values])

    if model_choice == "nb":
        pred = int(nb.predict(X)[0])
        try:
            prob = float(nb.predict_proba(X)[0][1])
        except Exception:
            prob = None
    else:
        pred = int(logreg.predict(X)[0])
        try:
            prob = float(logreg.predict_proba(X)[0][1])
        except Exception:
            prob = None

    result_label = "Non-Diabetic" if pred == 0 else "Diabetic - High Risk"
    response = {
        "prediction": result_label,
        "prediction_binary": pred,
        "confidence": round(prob * 100, 2) if prob is not None else None,
        "model": model_choice
    }
    return jsonify(response)

if __name__ == "__main__":
    # Bind to 0.0.0.0 so the backend is reachable from the frontend when
    # the dev server is accessed via a network IP (e.g. http://10.x.x.x:3000).
    # For local-only use you can also open the frontend at http://localhost:3000.
    app.run(host="0.0.0.0", port=5000, debug=True)
