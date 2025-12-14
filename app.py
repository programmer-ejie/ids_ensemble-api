import os
import json
import time
import joblib
import pandas as pd
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)


MODEL_DIR = "model"
BUNDLE_FILENAME = "ids_ensemble.joblib"
BUNDLE_PATH = os.path.join(MODEL_DIR, BUNDLE_FILENAME)

print("Initializing IDS Ensemble API...")
print(f"Loading model bundle from {BUNDLE_PATH} ...")
bundle = joblib.load(BUNDLE_PATH)

model_bin = bundle["bin"]
model_mul = bundle["mul"]
feature_names = bundle["features"]
best_thr = float(bundle["best_threshold"])

print(f"Loaded bundle with {len(feature_names)} features.")
print(f"Stage A threshold: {best_thr:.3f}")


LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, "predictions.jsonl")


LARAVEL_LOG_URL = os.getenv("LARAVEL_LOG_URL", "").strip()
LARAVEL_API_KEY = os.getenv("LARAVEL_API_KEY", "").strip()
LARAVEL_TIMEOUT = float(os.getenv("LARAVEL_TIMEOUT", "8"))


def append_log(record: dict):
   
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[LOG] Failed to write log: {e}")


def forward_to_laravel(record: dict):
   
    if not LARAVEL_LOG_URL:
        return  

    headers = {"Content-Type": "application/json"}
    if LARAVEL_API_KEY:
        headers["X-API-KEY"] = LARAVEL_API_KEY

    try:
        r = requests.post(LARAVEL_LOG_URL, json=record, headers=headers, timeout=LARAVEL_TIMEOUT)
        if not r.ok:
            print(f"[LARAVEL] HTTP {r.status_code}: {r.text[:200]}")
        else:
            print("[LARAVEL] log saved ✅")
    except Exception as e:
        print(f"[LARAVEL] forward error: {e}")


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


    row_df = pd.DataFrame([x_list], columns=feature_names)

    proba_attack = float(model_bin.predict_proba(row_df)[0, 1])
    is_attack = int(proba_attack >= best_thr)

    risk_level = "benign"
    attack_type = None

    if is_attack == 1:
        risk_level = "attack"
        attack_type = str(model_mul.predict(row_df)[0])

    result = {
        "risk_level": risk_level,
        "prob_attack": proba_attack,
        "attack_type": attack_type,
        "used_threshold": best_thr,
    }

   
    record = {
        "ts": int(time.time()),
        **data,     
        **result  
    }


    append_log(record)
    forward_to_laravel(record)

    return jsonify({
        **result,
        "received_features": list(data.keys())
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
