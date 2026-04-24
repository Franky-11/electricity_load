import os


TIMEZONE = "Europe/Berlin"
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")

PARAM_PATH = os.path.join(ARTIFACTS_DIR, "sarima_params.npz")
SPEC_PATH = os.path.join(ARTIFACTS_DIR, "sarima_spec.json")
VAL_PATH = os.path.join(ARTIFACTS_DIR, "val_sarima_latest.json")
METRICS_CSV = os.path.join(ARTIFACTS_DIR, "metrics.csv")

REPO_PARAM_FALLBACK = "sarima_params.npz"
REPO_SPEC_FALLBACK = "sarima_spec.json"
