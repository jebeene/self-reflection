import cv2
import threading
import numpy as np
from deepface import DeepFace

# === SETTINGS ===
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
HEADER_HEIGHT = 200  # Adjusted to fit portrait layout

# Emotion-to-color mapping
EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "happy": (0, 255, 255),
    "sad": (255, 0, 0),
    "surprise": (255, 165, 0),
    "neutral": (192, 192, 192)
}
emotion_positions = {
    "angry": (0.35, 0.40),
    "disgust": (0.35, 0.60),
    "fear": (0.35, 0.80),
    "happy": (0.74, 0.18),
    "sad": (0.74, 0.42),
    "surprise": (0.74, 0.68),
    "neutral": (0.74, 0.92),
}

# Load Haarcascade
haarcascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarcascade_path)

if face_cascade.empty():
    print("[ERROR] Could not load Haarcascade file. Exiting...")
    exit()
def resize_and_letterbox_header(img, target_width, target_height):
    """Resize the header to fit target width, pad top/bottom to match height."""
    h, w = img.shape[:2]
    scale = target_width / w
    new_w = target_width
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if new_h < target_height:
        pad = target_height - new_h
        top_pad = pad // 2
        bottom_pad = pad - top_pad

        # Use the average color of the top-left corner as background fill
        bg_color = img[0, 0].tolist()
        padded = cv2.copyMakeBorder(
            resized,
            top_pad, bottom_pad, 0, 0,
            borderType=cv2.BORDER_CONSTANT,
            value=bg_color
        )
        return padded
    else:
        return resized[:target_height, :, :]
# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

# Load header image
header = cv2.imread("header.png")
if header is None:
    print("[ERROR] Header image not found. Exiting...")
    exit()

print("[INFO] Webcam started. Press 'q' to exit.")

current_emotions = {}
processing = False
smooth_color = np.zeros(3, dtype=np.float32)
alpha = 0.1

def blend_colors(emotions):
    blended_color = np.zeros(3, dtype=np.float32)
    total_weight = sum(emotions.values())
    if total_weight == 0:
        return blended_color
    for emotion, confidence in emotions.items():
        color = np.array(EMOTION_COLORS.get(emotion, (255, 255, 255)), dtype=np.float32)
        weight = confidence / total_weight
        blended_color += weight * color
    return blended_color

def analyze_emotion(frame):
    global current_emotions, processing
    try:
        analysis = DeepFace.analyze(
            frame,
            actions=['emotion'],
            detector_backend='opencv',
            enforce_detection=False,
            align=False
        )
        emotions = analysis[0]['emotion']
        current_emotions = {emotion: round(conf, 2) for emotion, conf in emotions.items()}
        print(f"[RESULT] Detected Emotions: {current_emotions}")
    except Exception as e:
        print(f"[ERROR] DeepFace analysis failed: {e}")
    processing = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    # === Rotate and resize frame for portrait ===
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # Compute aspect-ratio preserving scale
    h, w = frame.shape[:2]
    scale = max(TARGET_WIDTH / w, (TARGET_HEIGHT - HEADER_HEIGHT) / h)
    resized_w, resized_h = int(w * scale), int(h * scale)
    resized_frame = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    # Center crop to target dimensions
    x_start = (resized_w - TARGET_WIDTH) // 2
    y_start = (resized_h - (TARGET_HEIGHT - HEADER_HEIGHT)) // 2
    frame = resized_frame[y_start:y_start + (TARGET_HEIGHT - HEADER_HEIGHT), x_start:x_start + TARGET_WIDTH]
    # Resize header to match portrait width
    header_resized = resize_and_letterbox_header(header, TARGET_WIDTH, HEADER_HEIGHT)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) > 0:
        if not processing:
            processing = True
            threading.Thread(target=analyze_emotion, args=(frame.copy(),), daemon=True).start()

    target_color = blend_colors(current_emotions)
    smooth_color = alpha * target_color + (1 - alpha) * smooth_color
    overlay_color = tuple(map(int, smooth_color))

    overlay = np.full(frame.shape, overlay_color, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    header_copy = header_resized.copy()
    for emotion, (x_ratio, y_ratio) in emotion_positions.items():
        if emotion in current_emotions:
            x = int(x_ratio * TARGET_WIDTH)
            y = int(y_ratio * HEADER_HEIGHT)
            text = f"{current_emotions[emotion]:.2f}%"
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            cv2.putText(header_copy, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    combined_frame = np.vstack((header_copy, frame))

    # === Fullscreen Portrait Display ===
    cv2.namedWindow('Emotion Analysis', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Emotion Analysis', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Emotion Analysis', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam and all resources released.")