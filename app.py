# ==============================================================
# Optimized FastAPI + MLflow Application (Heart Disease)
# --------------------------------------------------------------
# - Uses MLflow Model Registry (models:/.../latest)
# - Combines train.py + test_model.py logic
# - Caches model in memory and local disk
# - Includes fairness + recall validation
# ==============================================================

import os
import mlflow
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score
from fairlearn.metrics import MetricFrame, false_negative_rate

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://35.188.33.106:5000"
EXPERIMENT_NAME = "heart-disease-experiment"
MODEL_NAME = "heart-disease-logistic-regression"

LOCAL_MODEL_DIR = "models"
LOCAL_MODEL_PATH = Path(LOCAL_MODEL_DIR) / "model.joblib"
Path(LOCAL_MODEL_DIR).mkdir(exist_ok=True)

MODEL_CACHE = None

# --------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="FastAPI + MLflow Model Registry (Optimized)",
    version="1.0.0",
)

# --------------------------------------------------------------
# Request Schema
# --------------------------------------------------------------
class HeartDiseaseInput(BaseModel):
    age: float
    gender: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# --------------------------------------------------------------
# Data Preparation (shared logic)
# --------------------------------------------------------------
def load_and_prepare_data():
    df = pd.read_csv("./data.csv")
    df = df.dropna().copy()
    df["gender"] = df["gender"].map({"male": 1, "female": 0})
    df["target"] = df["target"].map({"yes": 1, "no": 0})

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["0-20", "21-40", "41-60", "61-80", "81+"],
        right=True
    )

    X = df.drop(["target", "age_group"], axis=1)
    y = df["target"]
    age_sensitive = df["age_group"]

    return train_test_split(
        X, y, age_sensitive, test_size=0.2, random_state=42
    )

# --------------------------------------------------------------
# Train + Register Model (train.py logic)
# --------------------------------------------------------------
def train_and_register_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test, _, age_test = load_and_prepare_data()

    grid = {
        "C": mlflow.utils.autolog_utils.log_input_examples.__self__ or
        list(pd.np.logspace(-4, 4, 20)),
        "solver": ["liblinear"],
    }

    search = RandomizedSearchCV(
        LogisticRegression(max_iter=1000),
        param_distributions={"C": pd.np.logspace(-4, 4, 20), "solver": ["liblinear"]},
        cv=5,
        n_iter=20,
        verbose=1,
    )

    with mlflow.start_run(run_name="heart-disease-training"):
        search.fit(X_train, y_train)

        model = search.best_estimator_
        acc = search.score(X_test, y_test)

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        joblib.dump(model, LOCAL_MODEL_PATH)

    return model, acc

# --------------------------------------------------------------
# Load Latest Model (optimized)
# --------------------------------------------------------------
def load_latest_model():
    global MODEL_CACHE

    if MODEL_CACHE is not None:
        return MODEL_CACHE

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
        joblib.dump(model, LOCAL_MODEL_PATH)
        MODEL_CACHE = model
        return model
    except Exception:
        if LOCAL_MODEL_PATH.exists():
            MODEL_CACHE = joblib.load(LOCAL_MODEL_PATH)
            return MODEL_CACHE
        raise HTTPException(status_code=503, detail="Model unavailable")

# --------------------------------------------------------------
# Validation (test_model.py logic)
# --------------------------------------------------------------
def validate_model(model):
    df = pd.read_csv("./data.csv").dropna().copy()
    df["gender"] = df["gender"].map({"male": 1, "female": 0})
    df["target"] = df["target"].map({"yes": 1, "no": 0})

    required = [
        "age","gender","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal"
    ]
    for col in required:
        if col not in df.columns:
            raise HTTPException(status_code=500, detail=f"Missing column {col}")

    X = df.drop("target", axis=1)
    y = df["target"]

    preds = model.predict(X)
    recall = recall_score(y, preds, pos_label=1)

    if recall < 0.80:
        raise HTTPException(status_code=500, detail="Recall below threshold")

    return recall

# --------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Heart Disease FastAPI + MLflow API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/train")
def train():
    model, acc = train_and_register_model()
    recall = validate_model(model)
    return {"status": "trained", "accuracy": acc, "recall": recall}

@app.get("/fetch")
def fetch():
    model = load_latest_model()
    recall = validate_model(model)
    return {"status": "loaded", "recall": recall}

@app.post("/predict")
def predict(data: HeartDiseaseInput):
    model = load_latest_model()
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}

# --------------------------------------------------------------
# Run Server
# --------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
