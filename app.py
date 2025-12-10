import os
import requests
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_DIR = "model"
BUNDLE_FILENAME = "ids_ensemble.joblib"
BUNDLE_PATH = os.path.join(MODEL_DIR, BUNDLE_FILENAME)

# 👉 Your actual Google Drive file ID
GDRIVE_FILE_ID = "1aytzw8S6L4gkXUc5zHBpavp9JgvOHGrr"


def download_model_if_needed():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(BUNDLE_PATH):
        print(f"[MODEL] Found existing {BUNDLE_PATH}, skipping download.")
        return

    print(f"[MODEL] {BUNDLE_PATH} not found. Downloading from Google Drive...")

    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"

    # 1) First request
    response = session.get(base_url, params={"id": GDRIVE_FILE_ID}, stream=True)
    if response.status_code != 200:
        print("[MODEL] First request status:", response.status_code)
        raise RuntimeError("Failed initial download request to Google Drive")

    # 2) Check if Google added a confirm token (for large files)
    def _get_confirm_token(resp):
        for key, value in resp.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    token = _get_confirm_token(response)

    if token:
        print("[MODEL] Got Google Drive confirm token, requesting again...")
        response = session.get(
            base_url,
            params={"id": GDRIVE_FILE_ID, "confirm": token},
            stream=True
        )
        if response.status_code != 200:
            print("[MODEL] Confirmed request status:", response.status_code)
            raise RuntimeError("Failed confirmed download request to Google Drive")

    # 3) Save to disk
    with open(BUNDLE_PATH, "wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"[MODEL] Download complete: {BUNDLE_PATH}")

    # 4) Sanity check: make sure it's not an HTML error page
    size = os.path.getsize(BUNDLE_PATH)
    print(f"[MODEL] Downloaded file size: {size} bytes")

    with open(BUNDLE_PATH, "rb") as f:
        head = f.read(200)

    if head.startswith(b"<!DOCTYPE html") or head.startswith(b"<html"):
        raise RuntimeError(
            "Downloaded file looks like HTML, not a model. "
            "Check FILE ID and sharing permissions (must be 'Anyone with the link')."
        )


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
