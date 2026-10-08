import json
import os
import unittest

import joblib
import pandas as pd


DATASET_PATH = "placement_data.csv"
MODEL_PATH = "placement_model.pkl"
METRICS_PATH = "metrics.json"


class TestMLPipeline(unittest.TestCase):

    # -------------------------------------------------
    # Test 1: Dataset exists
    # -------------------------------------------------

    def test_dataset_created(self):
        self.assertTrue(
            os.path.exists(DATASET_PATH),
            "Dataset file was not found."
        )

    # -------------------------------------------------
    # Test 2: Dataset has required columns
    # -------------------------------------------------

    def test_dataset_columns(self):
        data = pd.read_csv(DATASET_PATH)

        required_columns = {
            "cgpa",
            "placement_exam_marks",
            "placed"
        }

        self.assertTrue(
            required_columns.issubset(
                set(data.columns)
            ),
            "Required dataset columns are missing."
        )

    # -------------------------------------------------
    # Test 3: Dataset is not empty
    # -------------------------------------------------

    def test_dataset_not_empty(self):
        data = pd.read_csv(DATASET_PATH)

        self.assertGreater(
            len(data),
            0,
            "Dataset is empty."
        )

    # -------------------------------------------------
    # Test 4: Model exists
    # -------------------------------------------------

    def test_model_created(self):
        self.assertTrue(
            os.path.exists(MODEL_PATH),
            "Trained model was not created."
        )

    # -------------------------------------------------
    # Test 5: Metrics file exists
    # -------------------------------------------------

    def test_metrics_created(self):
        self.assertTrue(
            os.path.exists(METRICS_PATH),
            "Metrics file was not created."
        )

    # -------------------------------------------------
    # Test 6: Accuracy is valid
    # -------------------------------------------------

    def test_accuracy_is_valid(self):

        with open(
            METRICS_PATH,
            "r"
        ) as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(
            accuracy,
            0.0
        )

        self.assertLessEqual(
            accuracy,
            1.0
        )

    # -------------------------------------------------
    # Test 7: Model can make prediction
    # -------------------------------------------------

    def test_model_prediction(self):

        model = joblib.load(
            MODEL_PATH
        )

        sample = pd.DataFrame([
            {
                "cgpa": 8.0,
                "placement_exam_marks": 70
            }
        ])

        prediction = model.predict(
            sample
        )[0]

        self.assertIn(
            int(prediction),
            [0, 1]
        )


if __name__ == "__main__":
    unittest.main()
