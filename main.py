import cv2
import torch
from emotion_detector.detector import detect_faces
from emotion_detector.utils import preprocess_face
from models.emotion_cnn import EmotionCNN
from led_controller.controller import LEDController
import config


def map_emotion_to_color(emotion):
    """Maps detected emotion to corresponding LED color."""
    emotion_colors = {
        "Angry": "red",
        "Sad": "blue",
        "Happy": "green",
        "Surprise": "yellow",
        "Neutral": "white",
        "Fear": "purple",
        "Disgust": "orange"
    }
    return emotion_colors.get(emotion, "off")


# Initialize components
DEVICE = config.DEVICE
model = EmotionCNN().to(DEVICE)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=DEVICE))
model.eval()
controller = LEDController()


def run_real_time_analysis():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces, gray = detect_faces(frame)
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_tensor = preprocess_face(face, DEVICE)
            with torch.no_grad():
                outputs = model(face_tensor)
                _, predicted = torch.max(outputs, 1)
            emotion = config.EMOTION_LABELS[predicted.item()]
            color = map_emotion_to_color(emotion)
            controller.set_color(color)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    controller.set_color("off")  # Turn off LEDs when exiting


if __name__ == "__main__":
    run_real_time_analysis()