import unittest
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import recall_score

class TestHeartDiseaseModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load model and data once for all tests"""
        base_dir = Path(__file__).resolve().parent

        cls.model_path = base_dir / "artifacts" / "model.joblib"
        cls.data_path = base_dir / "data.csv"
        cls.target_column = "target"

        # --- Existence checks ---
        if not cls.model_path.exists():
            raise RuntimeError(f"Model not found at {cls.model_path}")

        if not cls.data_path.exists():
            raise RuntimeError(f"Data not found at {cls.data_path}")

        # --- Load model and data ---
        cls.model = joblib.load(cls.model_path)
        cls.data = pd.read_csv(cls.data_path)

        # --- Same preprocessing as train.py ---
        cls.data = cls.data.dropna().copy()
        cls.data["gender"] = cls.data["gender"].map({"male": 1, "female": 0})
        cls.data["target"] = cls.data["target"].map({"yes": 1, "no": 0})

    # --------------------------------------------------
    # Test 1: Data integrity
    # --------------------------------------------------
    def test_required_columns_exist(self):
        required_features = [
            "age",
            "gender",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]

        for col in required_features:
            self.assertIn(
                col,
                self.data.columns,
                f"Missing required feature: {col}"
            )

    # --------------------------------------------------
    # Test 2: Model recall threshold (healthcare critical)
    # --------------------------------------------------
    def test_model_recall_threshold(self):
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]

        predictions = self.model.predict(X)

        # Recall for positive class (heart disease = 1)
        recall = recall_score(y, predictions, pos_label=1)

        self.assertGreaterEqual(
            recall,
            0.80,
            f"Model recall too low: {recall}"
        )

    # --------------------------------------------------
    # Test 3: Model prediction shape sanity check
    # --------------------------------------------------
    def test_prediction_shape(self):
        X = self.data.drop(self.target_column, axis=1)
        preds = self.model.predict(X)

        self.assertEqual(
            len(preds),
            len(X),
            "Number of predictions does not match number of samples"
        )

if __name__ == "__main__":
    unittest.main()
