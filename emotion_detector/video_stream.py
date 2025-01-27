# emotion_detector/video_stream.py

import cv2
import logging

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            logging.error("Could not open video device.")
            raise ValueError("Could not open video device")
        else:
            logging.info("Video stream started.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            logging.warning("Failed to read frame from video stream.")
            return None
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
            logging.info("Video stream released.")
