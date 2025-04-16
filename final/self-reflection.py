import cv2
import threading
import numpy as np
from deepface import DeepFace
from tkinter import Tk, Canvas
from PIL import Image, ImageTk

# === Settings ===
# These are the dimensions before rotation:
TARGET_WIDTH = 540       # width of header and video feed (portrait arrangement)
HEADER_HEIGHT = 200      # header height
VIDEO_HEIGHT = 760       # video feed height
TARGET_HEIGHT = HEADER_HEIGHT + VIDEO_HEIGHT  # 960

# After rotating the combined image 90° clockwise,
# the dimensions swap. Use these for the Tkinter window:
ROTATED_WIDTH = TARGET_HEIGHT   # 960
ROTATED_HEIGHT = TARGET_WIDTH   # 540

alpha = 0.1  # Smoothing factor

# Load Haar cascade for face detection
haarcascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarcascade_path)

EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "happy": (0, 255, 255),
    "sad": (255, 0, 0),
    "surprise": (255, 165, 0),
    "neutral": (192, 192, 192)
}

# Emotion bar positions in header.png (in percentages)
emotion_positions_bar = {
    "angry": (0.21, 0.40),
    "disgust": (0.24, 0.57),
    "fear": (0.17, 0.74),
    "happy": (0.72, 0.23),
    "sad": (0.67, 0.40),
    "surprise": (0.77, 0.57),
    "neutral": (0.76, 0.74),
}

# === GUI Setup ===
root = Tk()
root.title("Emotion Detection")
# Instead of fullscreen we force the window to the size of the rotated image.
# (When displayed on a portrait monitor via proper OS/hardware rotation,
#  the image will appear correct.)
canvas = Canvas(root)
canvas.pack(fill="both", expand=True)

# === Load and prepare Header Image ===
header_img = cv2.imread("header.png")
if header_img is None:
    raise FileNotFoundError("Header image not found.")

def resize_and_pad_header(img, width, height):
    h, w = img.shape[:2]
    scale = width / w
    new_w = width
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if new_h < height:
        top_pad = (height - new_h) // 2
        bottom_pad = height - new_h - top_pad
        bg_color = img[0, 0].tolist()
        padded = cv2.copyMakeBorder(resized, top_pad, bottom_pad, 0, 0,
                                    cv2.BORDER_CONSTANT, value=bg_color)
        return padded
    else:
        return resized[:height, :, :]

header_resized = resize_and_pad_header(header_img, TARGET_WIDTH, HEADER_HEIGHT)

# === Camera Setup ===
cap = cv2.VideoCapture(0)

# Shared state
current_emotions = {}
smooth_color = np.zeros(3, dtype=np.float32)
processing = False

def crop_center(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = max(target_w / w, target_h / h)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
    resized_h, resized_w = resized.shape[:2]
    x = (resized_w - target_w) // 2
    y = (resized_h - target_h) // 2
    return resized[y:y+target_h, x:x+target_w]

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

def detect_emotion(frame):
    global current_emotions, processing
    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        emotions = result[0]["emotion"]
        current_emotions = {emotion: round(conf, 2) for emotion, conf in emotions.items()}
    except Exception as e:
        print("[DeepFace] Error:", e)
        current_emotions.clear()
    processing = False

def update_frame():
    global processing, smooth_color
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    # Crop frame to the dimensions for video feed (from the camera)
    frame = crop_center(frame, TARGET_WIDTH, VIDEO_HEIGHT)

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # Process emotion only if a face is detected and not already processing
    if len(faces) > 0 and not processing:
        processing = True
        threading.Thread(target=detect_emotion, args=(frame.copy(),), daemon=True).start()
    elif len(faces) == 0:
        current_emotions.clear()

    # Color blend overlay
    target_color = blend_colors(current_emotions)
    smooth_color = alpha * target_color + (1 - alpha) * smooth_color
    overlay_color = tuple(map(int, smooth_color))
    overlay = np.full(frame.shape, overlay_color, dtype=np.uint8)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # Prepare header with emotion bar overlays
    header_display = header_resized.copy()
    BAR_WIDTH = 100
    BAR_HEIGHT = 12
    BORDER_COLOR = (255, 255, 255)  # White border

    for emotion, (x_ratio, y_ratio) in emotion_positions_bar.items():
        if emotion in current_emotions:
            confidence = current_emotions[emotion]
            x = int(x_ratio * TARGET_WIDTH)
            y = int(y_ratio * HEADER_HEIGHT)
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))

            # Draw dark background bar
            cv2.rectangle(header_display, (x, y), (x + BAR_WIDTH, y + BAR_HEIGHT),
                          (50, 50, 50), thickness=-1)
            # Draw filled bar portion based on confidence
            fill_width = int((confidence / 100) * BAR_WIDTH)
            cv2.rectangle(header_display, (x, y), (x + fill_width, y + BAR_HEIGHT),
                          color, thickness=-1)
            # Draw white border
            cv2.rectangle(header_display, (x, y), (x + BAR_WIDTH, y + BAR_HEIGHT),
                          BORDER_COLOR, thickness=1)

    # Stack header above video feed (combined image is 540x960, portrait)
    combined = np.vstack((header_display, frame))

    # Rotate 90° clockwise; the rotated image now has dimensions 960x540.
    rotated = cv2.rotate(combined, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Resize explicitly to fill the Tkinter canvas
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    resized_rotated = cv2.resize(rotated, (canvas_width, canvas_height))
    combined_rgb = cv2.cvtColor(resized_rotated, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(combined_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor="nw", image=imgtk)
    canvas.imgtk = imgtk  # keep a reference so it isn’t garbage-collected

    root.after(10, update_frame)

# Start loop
update_frame()
root.mainloop()
cap.release()