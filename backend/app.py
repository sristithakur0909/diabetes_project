"""
Diabetic Prediction System — Backend API
Naviotech Solution Pvt. Ltd. — ML Internship
Author: Sristi Thakur

Serves the trained Logistic Regression model behind a simple REST API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
FEATURE_ORDER = joblib.load(os.path.join(BASE_DIR, "feature_order.joblib"))


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "diabetic-prediction-api"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row = [float(data[f]) for f in FEATURE_ORDER]
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric"}), 400

    X = np.array(row).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0][1])

    return jsonify({
        "prediction": "Diabetic" if pred == 1 else "Not Diabetic",
        "probability_diabetic": round(proba, 4),
        "input_features": dict(zip(FEATURE_ORDER, row)),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
