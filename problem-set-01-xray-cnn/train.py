"""
Trains the pneumonia CNN and saves the best checkpoint + training curves.

Usage:
    python train.py
    XRAY_DATA_ROOT=/path/to/chest_xray python train.py   # custom data path
"""

import json
import os

import tensorflow as tf

import config
from data_pipeline import get_datasets, compute_class_weights
from model import build_model


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    train_ds, val_ds, _ = get_datasets()
    class_weights = compute_class_weights()
    print(f"Class weights (to offset imbalance): {class_weights}")

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    with open(config.HISTORY_PATH, "w") as f:
        json.dump(history.history, f, indent=2)

    print(f"Best model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
