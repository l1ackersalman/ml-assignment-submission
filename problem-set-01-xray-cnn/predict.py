"""
Run the trained model on a single X-ray image.

Usage:
    python predict.py path/to/xray.jpeg
"""

import sys

import numpy as np
import tensorflow as tf

import config


def predict(image_path):
    model = tf.keras.models.load_model(config.MODEL_PATH)

    img = tf.keras.utils.load_img(
        image_path, target_size=(config.IMG_HEIGHT, config.IMG_WIDTH)
    )
    arr = tf.keras.utils.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr, verbose=0)[0][0])
    label = config.CLASS_NAMES[1] if prob >= 0.5 else config.CLASS_NAMES[0]
    confidence = prob if prob >= 0.5 else 1 - prob

    print(f"Prediction: {label}  (confidence: {confidence:.2%})")
    return label, confidence


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])
