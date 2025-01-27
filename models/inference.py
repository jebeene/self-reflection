import torch
import cv2
import numpy as np
from models.emotion_cnn import EmotionCNN
from emotion_detector.detector import detect_faces
from emotion_detector.utils import preprocess_face
import config

# Load trained model
model = EmotionCNN(num_classes=len(config.EMOTION_LABELS))
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
model.to(config.DEVICE)
model.eval()


def predict_emotion(frame):
    """Detects faces in a frame and predicts emotions."""
    faces, gray = detect_faces(frame)
    emotions = []

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_tensor = preprocess_face(face, config.DEVICE)
        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)
        emotion = config.EMOTION_LABELS[predicted.item()]
        emotions.append((emotion, (x, y, w, h)))

    return emotions


def start_webcam():
    """Starts real-time emotion detection using webcam."""
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        emotions = predict_emotion(frame)

        for emotion, (x, y, w, h) in emotions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_webcam()
