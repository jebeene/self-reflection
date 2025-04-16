import cv2
import threading
import numpy as np
from deepface import DeepFace
from tkinter import Tk, Canvas
from PIL import Image, ImageTk
import time

# === Settings ===
TARGET_WIDTH = 540       # Width of header and video feed
HEADER_HEIGHT = 200      # Header height
VIDEO_HEIGHT = 760       # Video feed height
TARGET_HEIGHT = HEADER_HEIGHT + VIDEO_HEIGHT  # Combined height

# After rotation, dimensions swap:
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
canvas = Canvas(root)
canvas.pack(fill="both", expand=True)
# Create a placeholder for the image; this ID will be reused during updates.
canvas_image_id = None

# === Load and Prepare Header Image ===
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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Shared state for emotions and color smoothing
current_emotions = {}
smooth_color = np.zeros(3, dtype=np.float32)
processing = False
frame_count = 0

# Shared frame buffer and a lock for thread-safety
latest_frame = None
frame_lock = threading.Lock()
stop_capture = False  # For clean thread termination

def capture_frames():
    """ Continuously read frames from the camera in a separate thread. """
    global latest_frame, stop_capture
    while not stop_capture:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
        else:
            # Slight delay if frame reading fails
            time.sleep(0.01)

# Start the camera capture thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

def crop_center(frame, target_w, target_h):
    """ Resize frame so that it covers target dimensions and then crop the center. """
    h, w = frame.shape[:2]
    scale = max(target_w / w, target_h / h)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    resized_h, resized_w = resized.shape[:2]
    x = (resized_w - target_w) // 2
    y = (resized_h - target_h) // 2
    return resized[y:y+target_h, x:x+target_w]

def blend_colors(emotions):
    """ Blend colors based on emotion confidences. """
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
    """ Perform emotion detection in a separate thread to avoid blocking. """
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
    """ Main GUI update loop: retrieves latest frame, processes it, and updates the canvas. """
    global smooth_color, frame_count, processing, canvas_image_id

    with frame_lock:
        if latest_frame is None:
            root.after(30, update_frame)
            return
        # Work on a copy of the latest captured frame
        frame = latest_frame.copy()

    # Crop frame to dimensions for video feed
    frame = crop_center(frame, TARGET_WIDTH, VIDEO_HEIGHT)

    # Convert to grayscale and detect faces (detection on a smaller image is faster)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    frame_count += 1
    if frame_count % 5 == 0 and len(faces) > 0 and not processing:
        processing = True
        # Use a new thread for emotion analysis to not block the update loop
        threading.Thread(target=detect_emotion, args=(frame.copy(),), daemon=True).start()
    elif len(faces) == 0:
        current_emotions.clear()
        frame_count = 0

    # Update the blended overlay color based on current detected emotions
    target_color = blend_colors(current_emotions)
    smooth_color = alpha * target_color + (1 - alpha) * smooth_color
    overlay_color = tuple(map(int, smooth_color))
    overlay = np.full(frame.shape, overlay_color, dtype=np.uint8)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # Create header overlay with emotion confidence bars
    header_display = header_resized.copy()
    BAR_WIDTH = 100
    BAR_HEIGHT = 12
    BORDER_COLOR = (255, 255, 255)
    for emotion, (x_ratio, y_ratio) in emotion_positions_bar.items():
        if emotion in current_emotions:
            confidence = current_emotions[emotion]
            x = int(x_ratio * TARGET_WIDTH)
            y = int(y_ratio * HEADER_HEIGHT)
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            # Draw background bar, filled portion, and border
            cv2.rectangle(header_display, (x, y), (x + BAR_WIDTH, y + BAR_HEIGHT),
                          (50, 50, 50), thickness=-1)
            fill_width = int((confidence / 100) * BAR_WIDTH)
            cv2.rectangle(header_display, (x, y), (x + fill_width, y + BAR_HEIGHT),
                          color, thickness=-1)
            cv2.rectangle(header_display, (x, y), (x + BAR_WIDTH, y + BAR_HEIGHT),
                          BORDER_COLOR, thickness=1)

    # Stack header above video feed and rotate the combined image
    combined = np.vstack((header_display, frame))
    rotated = cv2.rotate(combined, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Resize to fill the canvas dimensions, falling back to the rotated size if needed
    canvas_width = canvas.winfo_width() or ROTATED_WIDTH
    canvas_height = canvas.winfo_height() or ROTATED_HEIGHT
    resized_rotated = cv2.resize(rotated, (canvas_width, canvas_height), interpolation=cv2.INTER_LINEAR)
    combined_rgb = cv2.cvtColor(resized_rotated, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(combined_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    # Update or create the image on the canvas
    if canvas_image_id is None:
        canvas_image_id = canvas.create_image(0, 0, anchor="nw", image=imgtk)
    else:
        canvas.itemconfig(canvas_image_id, image=imgtk)
    # Keep a reference to avoid garbage collection
    canvas.imgtk = imgtk

    # Schedule the next frame update (approximately 33 fps)
    root.after(30, update_frame)

# Start the GUI update loop
update_frame()

def on_closing():
    """ Cleanly shut down the capture thread and close resources. """
    global stop_capture
    stop_capture = True
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()