import cv2
import threading
import numpy as np
from deepface import DeepFace

# Emotion-to-color mapping
EMOTION_COLORS = {
    "angry": (0, 0, 255),   # Red
    "disgust": (0, 128, 0),  # Green
    "fear": (128, 0, 128),   # Purple
    "happy": (0, 255, 255),  # Yellow
    "sad": (255, 0, 0),      # Blue
    "surprise": (255, 165, 0), # Orange
    "neutral": (192, 192, 192) # Gray
}

# Load Haarcascade for lightweight face detection
haarcascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarcascade_path)

if face_cascade.empty():
    print("[ERROR] Could not load Haarcascade file. Exiting...")
    exit()

# Initialize webcam (reduce resolution for performance)
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # Reduce width
cap.set(4, 240)  # Reduce height

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Webcam started. Press 'q' to exit.")

# Shared data
current_emotions = {}  # Store emotion percentages
processing = False  # Prevent multiple DeepFace calls
smooth_color = np.zeros(3, dtype=np.float32)  # Smoothed color (NumPy array)
alpha = 0.1  # Smoothing factor (lower = smoother transitions)

def blend_colors(emotions):
    """Blend colors based on emotion percentages and ensure NumPy format."""
    blended_color = np.zeros(3, dtype=np.float32)
    total_weight = sum(emotions.values())

    if total_weight == 0:
        return np.array([0, 0, 0], dtype=np.float32)  # Default black if no emotion detected

    for emotion, confidence in emotions.items():
        color = np.array(EMOTION_COLORS.get(emotion, (255, 255, 255)), dtype=np.float32)  # Convert to NumPy array
        weight = confidence / total_weight
        blended_color += weight * color

    return blended_color  # Ensure NumPy array format

def analyze_emotion(frame):
    """Run DeepFace emotion analysis in a separate thread."""
    global current_emotions, processing
    try:
        analysis = DeepFace.analyze(
            frame,
            actions=['emotion'],
            detector_backend='opencv',  # Use OpenCV backend (lighter than RetinaFace)
            enforce_detection=False,
            align=False  # Disable alignment for better speed on Raspberry Pi
        )

        emotions = analysis[0]['emotion']
        current_emotions = {emotion: round(conf, 2) for emotion, conf in emotions.items()}  # Store percentages
        print(f"[RESULT] Detected Emotions: {current_emotions}")

    except Exception as e:
        print(f"[ERROR] DeepFace analysis failed: {e}")

    processing = False  # Allow next detection

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces (smaller scaleFactor improves performance)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) > 0:
        print(f"[INFO] Detected {len(faces)} face(s).")

        # Run emotion analysis if not already processing
        if not processing:
            processing = True
            threading.Thread(target=analyze_emotion, args=(frame.copy(),), daemon=True).start()

    else:
        print("[INFO] No face detected.")

    # Get blended color based on emotions
    target_color = blend_colors(current_emotions)

    # Smooth transition using Exponential Moving Average (EMA)
    smooth_color = alpha * target_color + (1 - alpha) * smooth_color  # EMA Formula

    # Convert to integer RGB tuple
    overlay_color = tuple(map(int, smooth_color))

    # Apply semi-transparent overlay with the detected color
    overlay = np.full(frame.shape, overlay_color, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Display detected emotions with confidence levels
    y_offset = 30
    for emotion, confidence in current_emotions.items():
        text = f"{emotion.capitalize()}: {confidence:.2f}%"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25

    # Show the frame
    cv2.imshow('Emotion Analysis', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam and all resources released.")