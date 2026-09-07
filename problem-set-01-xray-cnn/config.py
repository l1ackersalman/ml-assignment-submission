"""
Central configuration for the pneumonia X-ray classifier.
Keeping these in one place makes it easy to tweak hyperparameters
without hunting through every script.
"""

import os

# ---- Paths -----------------------------------------------------------
# Expected layout after you unzip the Kaggle "chest_xray" dataset:
#   DATA_ROOT/train/{NORMAL,PNEUMONIA}/*.jpeg
#   DATA_ROOT/val/{NORMAL,PNEUMONIA}/*.jpeg
#   DATA_ROOT/test/{NORMAL,PNEUMONIA}/*.jpeg
DATA_ROOT = os.environ.get("XRAY_DATA_ROOT", "./data/chest_xray")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_cnn.keras")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

# ---- Image / training hyperparameters --------------------------------
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
