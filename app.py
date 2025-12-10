from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import requests 

app = Flask(__name__)

# ================== CONFIG ==================


MODEL_DIR = "model"
BUNDLE_FILENAME = "ids_ensemble.joblib"
BUNDLE_PATH = os.path.join(MODEL_DIR, BUNDLE_FILENAME)

# 👉 Put your actual Google Drive FILE ID here
# Example share link:
#   https://drive.google.com/file/d/1AbCDefGhIjKlMnOpQRsTuVWxyz12345/view?usp=sharing
# FILE ID = 1AbCDefGhIjKlMnOpQRsTuVWxyz12345
GDRIVE_FILE_ID = "PUT_YOUR_FILE_ID_HERE"

# ============================================

def download_model_if_needed():
    """Download the model from Google Drive if it's not already present."""
    # Ensure directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(BUNDLE_PATH):
        print(f"[MODEL] Found existing {BUNDLE_PATH}, skipping download.")
        return

    print(f"[MODEL] {BUNDLE_PATH} not found. Downloading from Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(BUNDLE_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"[MODEL] Download complete: {BUNDLE_PATH}")
    except Exception as e:
        print(f"[MODEL] ERROR while downloading model: {e}")
        raise


print("Initializing IDS Ensemble API...")


download_model_if_needed()


print(f"Loading model bundle from {BUNDLE_PATH} ...")
bundle = joblib.load(BUNDLE_PATH)

model_bin = bundle["bin"]        
model_mul = bundle["mul"]        
feature_names = bundle["features"]
best_thr = bundle["best_threshold"]

print(f"Loaded bundle with {len(feature_names)} features.")
print(f"Stage A threshold: {best_thr:.3f}")


@app.route("/")
def home():
    return "IDS Ensemble API is running."


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True) or {}

    x_list = []
    for fname in feature_names:
        val = data.get(fname, 0.0)
        try:
            val = float(val)
        except Exception:
            val = 0.0
        x_list.append(val)

    x = np.array(x_list, dtype=float).reshape(1, -1)

    proba_attack = model_bin.predict_proba(x)[0, 1]
    is_attack = int(proba_attack >= best_thr)

    risk_level = "benign"
    attack_type = None

    if is_attack == 1:
        risk_level = "attack"
        attack_type = model_mul.predict(x)[0]

    return jsonify({
        "risk_level": risk_level,
        "prob_attack": float(proba_attack),
        "attack_type": attack_type,
        "used_threshold": float(best_thr),
        "received_features": list(data.keys())
    })


if __name__ == "__main__":
  
    app.run(host="0.0.0.0", port=5000, debug=True)
