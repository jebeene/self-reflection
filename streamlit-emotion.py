import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace

st.set_page_config(layout="wide")
st.title("🎭 Real-Time Emotion Detection")

# Initialize webcam
cap = cv2.VideoCapture(0)
FRAME_WINDOW = st.image([])

EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "happy": (0, 255, 255),
    "sad": (255, 0, 0),
    "surprise": (255, 165, 0),
    "neutral": (192, 192, 192)
}

st.markdown("Press `Stop` to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame")
        break

    # Flip horizontally for natural webcam view
    frame = cv2.flip(frame, 1)

    # Run DeepFace analysis
    try:
        analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        emotions = analysis[0]["emotion"]
        dominant_emotion = max(emotions, key=emotions.get)
        color = EMOTION_COLORS.get(dominant_emotion, (255, 255, 255))

        # Draw emotion overlay
        label = f"{dominant_emotion.upper()}: {emotions[dominant_emotion]:.2f}%"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    except Exception as e:
        print("DeepFace error:", e)

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()