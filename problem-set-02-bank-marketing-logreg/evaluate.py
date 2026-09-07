"""
Evaluates the trained logistic regression pipeline on the held-out test
split, reports metrics, and saves a coefficient importance chart (which
features push predictions toward "will subscribe").

Usage:
    python evaluate.py
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

MODEL_PATH = "./models/logreg_pipeline.joblib"
TEST_SPLIT_PATH = "./models/test_split.joblib"


def main():
    pipeline = joblib.load(MODEL_PATH)
    X_test, y_test = joblib.load(TEST_SPLIT_PATH)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    print("=== Classification report ===")
    print(classification_report(y_test, y_pred, target_names=["no", "yes"]))

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("=== Confusion matrix ===")
    print(cm)

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("./models/roc_curve.png", dpi=150)

    # --- Feature importance (logistic regression coefficients) ---
    clf = pipeline.named_steps["clf"]
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    coefs = clf.coef_[0]

    order = np.argsort(np.abs(coefs))[::-1][:15]
    top_features = feature_names[order]
    top_coefs = coefs[order]

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    colors = ["#2a9d8f" if c > 0 else "#e76f51" for c in top_coefs]
    ax2.barh(top_features[::-1], top_coefs[::-1], color=colors[::-1])
    ax2.set_xlabel("Coefficient (log-odds impact)")
    ax2.set_title("Top 15 features by |coefficient|")
    fig2.tight_layout()
    fig2.savefig("./models/feature_importance.png", dpi=150)

    print("Saved roc_curve.png and feature_importance.png to ./models/")


if __name__ == "__main__":
    main()
