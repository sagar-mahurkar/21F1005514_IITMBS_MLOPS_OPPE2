import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Load data
df = pd.read_csv("./data.csv")

# Drop missing values
cleaned_df = df.dropna().copy()

# Encode categorical columns
cleaned_df["gender"] = cleaned_df["gender"].map({"male": 1, "female": 0})
cleaned_df["target"] = cleaned_df["target"].map({"yes": 1, "no": 0})

# Split features and target
X = cleaned_df.drop("target", axis=1)
y = cleaned_df["target"]

np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Hyperparameter search
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

rs_log_reg.fit(X_train, y_train)

print("Best params:", rs_log_reg.best_params_)
print("Test accuracy:", rs_log_reg.score(X_test, y_test))
