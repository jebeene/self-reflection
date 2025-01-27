import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.pth")
LOG_FILE_PATH = os.path.join(BASE_DIR, "logs", "app.log")

# Emotion Labels
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Face Detection Parameters
SCALE_FACTOR = 1.3
MIN_NEIGHBORS = 5

# Training Configuration
BATCH_SIZE = 128
EPOCHS = 80
LEARNING_RATE = 0.001

# Device Configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
