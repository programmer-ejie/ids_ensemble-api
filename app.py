
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


BUNDLE_PATH = "model/ids_ensemble.joblib"

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
