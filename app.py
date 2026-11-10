import os, json, time, math, hashlib
import joblib
import pandas as pd
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
BUNDLE_FILENAME = os.getenv("IDS_MODEL_FILENAME", "cicids78_demo.joblib").strip()
BUNDLE_PATH = os.path.join(MODEL_DIR, BUNDLE_FILENAME)

print("Initializing IDS Ensemble API...")
print(f"Loading model bundle from {BUNDLE_PATH} ...")
bundle = joblib.load(BUNDLE_PATH)
with open(BUNDLE_PATH, "rb") as model_file:
    model_sha256 = hashlib.sha256(model_file.read()).hexdigest()

model_bin = bundle["bin"]
model_mul = bundle["mul"]

# Normalize expected feature names
feature_names = [f.strip() for f in bundle["features"]]
best_thr = float(bundle["best_threshold"])
if not 0 <= best_thr <= 1:
    raise ValueError("Model threshold must be between zero and one")
attack_column = list(model_bin.classes_).index(1)
if model_bin.named_steps["pre"] is model_mul.named_steps["pre"]:
    raise ValueError("Stage A and Stage B must have independent preprocessors")

def model_metadata():
    return {
        "model_filename": BUNDLE_FILENAME,
        "model_sha256": model_sha256,
        "schema_version": bundle.get("schema_version", "unknown"),
        "used_threshold": best_thr,
        "deployment_profile": "experimental-demo",
    }

print(f"Loaded bundle with {len(feature_names)} features.")
print(f"Stage A threshold: {best_thr:.3f}")
print("First 10 expected features:", feature_names[:10])

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

# ✅ NEW: expose expected feature list (so your sender can match exactly)
@app.route("/api/features", methods=["GET"])
def features():
    return jsonify({
        "count": len(feature_names),
        "features": feature_names,
        **model_metadata(),
    })

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "invalid_json_object"}), 400

    # Normalize incoming keys
    data = {str(k).strip(): v for k, v in data.items()}

    # ✅ NEW: accept sender snake_case keys and map them into CIC-style keys (if model uses those)
    aliases = {
        "dst_port": "destination_port",
        "flow_pkts_s": "flow_packets_s",
        "tot_fwd_pkts": "total_fwd_packets",
        "tot_bwd_pkts": "total_backward_packets",
        "tot_fwd_bytes": "total_length_of_fwd_packets",
        "tot_bwd_bytes": "total_length_of_bwd_packets",
        "fwd_pkt_len_mean": "fwd_packet_length_mean",
        "bwd_pkt_len_mean": "bwd_packet_length_mean",
    }
    for src_key, dst_key in aliases.items():
        if src_key in data and dst_key not in data:
            data[dst_key] = data[src_key]

    # Debug counters
    present = sum(1 for f in feature_names if f in data)
    missing = len(feature_names) - present

    # Optional server-side debug in Render logs (comment out later)
    if missing > 0:
        missing_names = [f for f in feature_names if f not in data]
        return jsonify({"error": "missing_model_features",
                        "missing_expected_features": missing_names,
                        **model_metadata()}), 422

    # Build model input in exact order
    x_list = []
    for fname in feature_names:
        val = data[fname]
        try:
            val = float(val)
            if not math.isfinite(val):
                raise ValueError("Non-finite model feature")
        except (ValueError, TypeError, OverflowError):
            return jsonify({"error": "invalid_model_feature", "feature": fname}), 422
        x_list.append(val)

    row_df = pd.DataFrame([x_list], columns=feature_names)

    proba_attack = float(model_bin.predict_proba(row_df)[0, attack_column])
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

    record = {"ts": int(time.time()), **data, **result}
    append_log(record)
    forward_to_laravel(record)

    return jsonify({
        **result,
        "present_expected_features": present,
        "missing_expected_features": missing,
        "received_features_count": len(data.keys()),
        "api_version": "demo-v4",
        **model_metadata(),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
