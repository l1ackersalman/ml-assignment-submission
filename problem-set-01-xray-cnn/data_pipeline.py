"""
Builds tf.data pipelines for train/val/test splits.

The pneumonia dataset is class-imbalanced (roughly 3:1 PNEUMONIA:NORMAL
in train), so we also compute class weights here rather than silently
letting the model learn a biased prior.
"""

import tensorflow as tf
from tensorflow.keras import layers

import config


def _make_dataset(directory, shuffle, augment=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        color_mode="rgb",
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        seed=config.SEED,
    )

    normalization = layers.Rescaling(1.0 / 255)
    ds = ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        augmenter = tf.keras.Sequential([
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomContrast(0.1),
        ])
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


def get_datasets():
    train_ds = _make_dataset(config.TRAIN_DIR, shuffle=True, augment=True)
    val_ds = _make_dataset(config.VAL_DIR, shuffle=False, augment=False)
    test_ds = _make_dataset(config.TEST_DIR, shuffle=False, augment=False)
    return train_ds, val_ds, test_ds


def compute_class_weights():
    """
    Returns a {0: weight, 1: weight} dict so the loss function penalises
    mistakes on the minority class (NORMAL) more heavily than the
    majority class (PNEUMONIA). Uses inverse-frequency weighting.
    """
    import os

    counts = {}
    for idx, cls in enumerate(config.CLASS_NAMES):
        cls_dir = os.path.join(config.TRAIN_DIR, cls)
        counts[idx] = len([f for f in os.listdir(cls_dir) if not f.startswith(".")])

    total = sum(counts.values())
    n_classes = len(counts)
    weights = {idx: total / (n_classes * count) for idx, count in counts.items()}
    return weights
