import cv2
import torch
import numpy as np

def preprocess_face(face, device):
    """Preprocesses a detected face for model inference."""
    face = cv2.resize(face, (48, 48))  # Resize to match model input size
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    face = np.expand_dims(face, axis=0)  # Add channel dimension
    face = torch.tensor(face, dtype=torch.float32).to(device) / 255.0  # Normalize and convert to tensor
    return face