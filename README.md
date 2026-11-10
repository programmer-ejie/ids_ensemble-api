# ids_ensemble-api

The default model is `model/cicids78_demo.joblib`, a 78-feature experimental
candidate using the retrained Stage A and Histogram Gradient Boosting Stage B.
Its SHA-256 is
`73893343b1d6f8b84b86dba6dc66e61f5a456d1472aa10859310aa04ff9e982e`.
It did not meet all validation requirements and is intended for demonstration.
All 70 current full-feature demo fixtures are detected as attacks; 65 receive
the selected attack type. This is not independent accuracy validation.

Set `IDS_MODEL_FILENAME` only if intentionally selecting another file under
`model/`. Leave it unset to use the demo candidate. `/api/features` exposes
the loaded filename, checksum, threshold, feature list, and schema.

Render start command: `gunicorn app:app`. The local command is `python app.py`.
Updating local files does not change a running Render deployment until deployed.

The API rejects missing or invalid model features with HTTP 422 instead of
inventing zero values. The demo sends all 78 features; the current NetFlow
collector supplies fewer and is not compatible with this model without richer
measurements. Block and Recover are unaffected.

Run local checks with `python -m unittest discover -s tests -v`. Tests disable
logging and the Laravel callback; the 70-sample check uses the parent Laravel
project's fixture file and is skipped in a standalone clone without those data.
