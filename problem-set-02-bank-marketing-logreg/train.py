"""
Trains a Logistic Regression model to predict term-deposit subscription.

Usage:
    python train.py --data data/bank-full.csv
"""

import argparse
import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from preprocessing import prepare_data

MODEL_PATH = "./models/logreg_pipeline.joblib"


def main(data_path: str):
    os.makedirs("./models", exist_ok=True)

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(data_path)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",  # target is ~88/12 imbalanced
        )),
    ])

    # Small grid over regularization strength/type — not exhaustive,
    # but enough to show the choice was validated rather than guessed.
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    }

    search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1
    )
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

    joblib.dump(search.best_estimator_, MODEL_PATH)
    joblib.dump((X_test, y_test), "./models/test_split.joblib")

    print(f"Saved trained pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to bank-full.csv (or bank.csv)")
    args = parser.parse_args()
    main(args.data)
