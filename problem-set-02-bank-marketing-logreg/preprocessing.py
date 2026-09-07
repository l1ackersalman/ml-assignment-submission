"""
Loads and preprocesses the UCI Bank Marketing dataset.

Handles:
- the ';'-delimited CSV format the UCI release ships in
- the 'unknown' sentinel used throughout categorical columns
- the well-known 'duration' leakage column (see note below)
- one-hot encoding of categoricals + scaling of numerics
- train/test split with stratification (target is imbalanced ~88/12)
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "y"

NUMERIC_COLS = [
    "age", "balance", "day", "campaign", "pdays", "previous",
]
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome",
]

# 'duration' (last contact duration, in seconds) is excluded by default.
# The dataset documentation itself flags it as target leakage: duration
# is only known *after* the call ends, and a duration of 0 means y='no'
# by definition. Including it inflates offline accuracy/AUC but produces
# a model that can't actually be used for pre-call prediction, which is
# the bank's real use case. Set INCLUDE_DURATION=True only if you
# explicitly want the (unrealistic) upper-bound benchmark.
INCLUDE_DURATION = False


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [c.strip().strip('"') for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.strip('"')
    return df


def build_feature_pipeline():
    numeric_cols = list(NUMERIC_COLS)
    if INCLUDE_DURATION:
        numeric_cols = numeric_cols + ["duration"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_COLS),
        ]
    )
    return preprocessor, numeric_cols


def prepare_data(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    df = load_raw(csv_path)

    df[TARGET_COL] = (df[TARGET_COL].str.lower() == "yes").astype(int)

    preprocessor, numeric_cols = build_feature_pipeline()
    feature_cols = numeric_cols + CATEGORICAL_COLS

    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor
