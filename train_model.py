import json
import os

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_PATH = "placement_data.csv"
MODEL_PATH = "placement_model.pkl"
METRICS_PATH = "metrics.json"

FEATURES = [
    "cgpa",
    "placement_exam_marks"
]

TARGET = "placed"


# ---------------------------------------------------------
# Load and validate dataset
# ---------------------------------------------------------

def load_dataset():
    print("Loading dataset...")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    data = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))
    print("Columns:", list(data.columns))

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("Dataset is empty.")

    if data[required_columns].isnull().any().any():
        raise ValueError(
            "Dataset contains missing values in required columns."
        )

    return data


# ---------------------------------------------------------
# Train ML model
# ---------------------------------------------------------

def train_model():
    print("\nStarting ML training pipeline...")
    print("--------------------------------")

    # Load dataset
    data = load_dataset()

    # Display basic dataset information
    print("\nDataset information:")
    print(data[FEATURES + [TARGET]].describe())

    print("\nTarget distribution:")
    print(data[TARGET].value_counts())

    # Separate input features and target
    X = data[FEATURES]
    y = data[TARGET]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records :", len(X_test))

    # ML pipeline
    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    # Train model
    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("Model training completed.")

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))

    print("\nConfusion Matrix:")
    print(matrix)

    # Save trained model
    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved as {MODEL_PATH}"
    )

    # Save metrics
    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET
    }

    with open(
        METRICS_PATH,
        "w"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"Metrics saved as {METRICS_PATH}"
    )

    return accuracy


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

if __name__ == "__main__":
    train_model()
