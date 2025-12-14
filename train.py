import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, false_negative_rate

# --------------------------------------------------
# MLflow Configuration
# --------------------------------------------------
MLFLOW_TRACKING_URI = "http://35.188.33.106:5000"
EXPERIMENT_NAME = "heart-disease-experiment"
REGISTERED_MODEL_NAME = "heart-disease-logistic-regression"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv("./data.csv")

# Drop missing values
cleaned_df = df.dropna().copy()

# Encode categorical columns
cleaned_df["gender"] = cleaned_df["gender"].map({"male": 1, "female": 0})
cleaned_df["target"] = cleaned_df["target"].map({"yes": 1, "no": 0})

# --------------------------------------------------
# Bucket AGE into 20-year bins (Sensitive attribute)
# --------------------------------------------------
cleaned_df["age_group"] = pd.cut(
    cleaned_df["age"],
    bins=[0, 20, 40, 60, 80, 100],
    labels=["0-20", "21-40", "41-60", "61-80", "81+"],
    right=True
)

# --------------------------------------------------
# Split features and target
# --------------------------------------------------
X = cleaned_df.drop(["target", "age_group"], axis=1)
y = cleaned_df["target"]
age_sensitive = cleaned_df["age_group"]

np.random.seed(42)

X_train, X_test, y_train, y_test, age_train, age_test = train_test_split(
    X, y, age_sensitive, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Train model with hyperparameter tuning
# --------------------------------------------------
log_reg_grid = {
    "C": np.logspace(-4, 4, 20),
    "solver": ["liblinear"]
}

rs_log_reg = RandomizedSearchCV(
    LogisticRegression(max_iter=1000),
    param_distributions=log_reg_grid,
    cv=5,
    n_iter=20,
    verbose=1
)

# --------------------------------------------------
# MLflow Run (Training + Registration)
# --------------------------------------------------
with mlflow.start_run(run_name="logistic-regression-heart-disease"):

    rs_log_reg.fit(X_train, y_train)

    best_model = rs_log_reg.best_estimator_
    test_accuracy = rs_log_reg.score(X_test, y_test)

    print("Best params:", rs_log_reg.best_params_)
    print("Test accuracy:", test_accuracy)

    # ---- Log parameters & metrics ----
    mlflow.log_params(rs_log_reg.best_params_)
    mlflow.log_metric("accuracy", test_accuracy)

    # ---- Save local model artifact ----
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    model_path = artifacts_dir / "model.joblib"
    joblib.dump(best_model, model_path)

    print(f"\nModel saved successfully at: {model_path}")

    mlflow.log_artifact(str(model_path), artifact_path="local_model")

    # ---- Register model in MLflow ----
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME
    )

# --------------------------------------------------
# Fairness Evaluation using Fairlearn
# --------------------------------------------------
y_pred = best_model.predict(X_test)

metrics = {
    "accuracy": accuracy_score,
    "false_negative_rate": false_negative_rate
}

mf = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=age_test
)

print("\n=== Fairness Metrics by Age Group ===")
print(mf.by_group)

print("\n=== Overall Metrics ===")
print(mf.overall)
