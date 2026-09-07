"""
Loads the trained model and reports test-set metrics: accuracy,
precision, recall, F1, AUC, and a confusion matrix. Also saves a PNG
of the confusion matrix and the ROC curve for the README / report.

Usage:
    python evaluate.py
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt

import config
from data_pipeline import get_datasets


def main():
    _, _, test_ds = get_datasets()

    model = tf.keras.models.load_model(config.MODEL_PATH)

    y_true, y_prob = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0).ravel()
        y_prob.extend(preds.tolist())
        y_true.extend(labels.numpy().ravel().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    print("=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    print("=== Confusion matrix ===")
    print(cm)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], config.CLASS_NAMES)
    ax.set_yticks([0, 1], config.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("./models/confusion_matrix.png", dpi=150)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    fig2.tight_layout()
    fig2.savefig("./models/roc_curve.png", dpi=150)

    print("Saved confusion_matrix.png and roc_curve.png to ./models/")


if __name__ == "__main__":
    main()
