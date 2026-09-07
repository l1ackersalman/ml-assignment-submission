"""
A compact CNN for binary chest X-ray classification (NORMAL vs PNEUMONIA).

Design notes:
- 4 conv blocks with increasing filter depth (32->256), each followed by
  BatchNorm + MaxPool, is enough capacity for 180x180 grayscale-ish
  radiographs without the overfitting risk a deeper net would carry on
  only ~5,200 training images.
- Dropout before the dense head fights overfitting given the dataset's
  small size and class imbalance.
- Sigmoid output + binary crossentropy since this is a two-class problem
  (kept as 1 unit rather than 2-unit softmax for simplicity).
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import config


def build_model():
    model = models.Sequential([
        layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
