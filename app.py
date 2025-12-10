import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Path to your model inside the repo
MODEL_DIR = "model"
BUNDLE_FILENAME = "ids_ensemble.joblib"
BUNDLE_PATH = os.path.join(MODEL_DIR, BUNDLE_FILENAME)

print("Initializing IDS Ensemble API...")

# 🔹 Load the model bundle directly from the repo
print(f"Loading model bundle from {BUNDLE_PATH} ...")
bundle = joblib.load(BUNDLE_PATH)

model_bin = bundle["bin"]            # Stage A (binary classifier)
model_mul = bundle["mul"]            # Stage B (multi-class classifier)
feature_names = bundle["features"]   # List of feature names expected
best_thr = bundle["best_threshold"]  # Tuned threshold

print(f"Loaded bundle with {len(feature_names)} features.")
print(f"Stage A threshold: {best_thr:.3f}")


@app.route("/")
def home():
    return "IDS Ensemble API is running."


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True) or {}

    # 1) Build feature vector in the correct order expected by the model
    x_list = []
    for fname in feature_names:
        val = data.get(fname, 0.0)
        try:
            val = float(val)
        except Exception:
            val = 0.0
        x_list.append(val)

    # 2) Use a DataFrame so ColumnTransformer sees the correct column names
    row_df = pd.DataFrame([x_list], columns=feature_names)

    # 3) Stage A: benign vs attack (probability)
    proba_attack = model_bin.predict_proba(row_df)[0, 1]
    is_attack = int(proba_attack >= best_thr)

    risk_level = "benign"
    attack_type = None

    # 4) Stage B: attack type prediction (only if Stage A says "attack")
    if is_attack == 1:
        risk_level = "attack"
        attack_type = model_mul.predict(row_df)[0]

    return jsonify({
        "risk_level": risk_level,
        "prob_attack": float(proba_attack),
        "attack_type": attack_type,
        "used_threshold": float(best_thr),
        "received_features": list(data.keys())
    })


if __name__ == "__main__":
    # For local testing; on Render you still use: gunicorn app:app
    app.run(host="0.0.0.0", port=5000, debug=True)
