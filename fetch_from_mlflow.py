import os
import sys
import mlflow
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import recall_score
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# -------------------------- Configuration --------------------------
MLFLOW_TRACKING_URI = "http://35.188.33.106:5000"
MODEL_NAME = "heart-disease-logistic-regression"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
DOWNLOAD_DIR = BASE_DIR / "downloaded_models"
LOCAL_MODEL_PATH = DOWNLOAD_DIR / "model.joblib"

DOWNLOAD_DIR.mkdir(exist_ok=True)

# -------------------------- Fetch Functions --------------------------
def fetch_latest_model(client, name):
    """Fetch latest registered model version from MLflow."""
    print(f"[INFO] Searching for latest model: {name}")
    try:
        versions = client.search_model_versions(
            filter_string=f"name='{name}'",
            order_by=["version_number DESC"],
            max_results=1
        )
        if not versions:
            raise RuntimeError(f"No versions found for model '{name}'")

        latest = versions[0]
        print(
            f"[INFO] Found model v{latest.version} | "
            f"Stage: {latest.current_stage} | Run ID: {latest.run_id}"
        )
        return latest

    except Exception as e:
        raise RuntimeError(f"Failed to fetch model metadata: {e}")

def load_model_from_registry(version):
    """Load model directly from MLflow registry."""
    try:
        model_uri = f"models:/{MODEL_NAME}/{version.version}"
        print(f"[INFO] Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("[INFO] Model loaded successfully")
        return model
    except MlflowException as e:
        raise RuntimeError(f"MLflow error while loading model: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading model: {e}")

def download_model_artifacts(version):
    """Download model artifacts locally."""
    try:
        artifact_uri = f"models:/{MODEL_NAME}/{version.version}"
        print(f"[INFO] Downloading artifacts from: {artifact_uri}")
        path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri,
            dst_path=str(DOWNLOAD_DIR)
        )
        print(f"[INFO] Artifacts downloaded to: {path}")
        return path
    except Exception as e:
        raise RuntimeError(f"Failed to download artifacts: {e}")

# -------------------------- Validation Logic (from test_model.py) --------------------------
def run_model_tests(model):
    """Run model tests equivalent to test_model.py"""
    print("[INFO] Running model validation checks...")

    if not DATA_PATH.exists():
        raise RuntimeError("data.csv not found for validation")

    data = pd.read_csv(DATA_PATH)

    # Same preprocessing as train.py
    data = data.dropna().copy()
    data["gender"] = data["gender"].map({"male": 1, "female": 0})
    data["target"] = data["target"].map({"yes": 1, "no": 0})

    required_features = [
        "age", "gender", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ]

    for col in required_features:
        if col not in data.columns:
            raise RuntimeError(f"Missing required feature: {col}")

    X = data.drop("target", axis=1)
    y = data["target"]

    preds = model.predict(X)

    if len(preds) != len(X):
        raise RuntimeError("Prediction length mismatch")

    recall = recall_score(y, preds, pos_label=1)
    print(f"[INFO] Recall score: {recall:.4f}")

    if recall < 0.80:
        raise RuntimeError(f"Recall below threshold: {recall}")

    print("[INFO] All validation checks passed ✅")

# -------------------------- Main Entry --------------------------
def main():
    print(f"[INFO] Setting MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    version = fetch_latest_model(client, MODEL_NAME)

    model = load_model_from_registry(version)

    download_model_artifacts(version)

    run_model_tests(model)

    print("[SUCCESS] Model fetched, validated, and ready for use 🎉")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[CRITICAL] {e}")
        sys.exit(1)
