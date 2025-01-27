# tests/test_detector.py

import unittest
import os
from emotion_detector.detector import EmotionDetector

class TestEmotionDetector(unittest.TestCase):
    def setUp(self):
        self.tflite_model_path = "models/emotion_model.tflite"
        if not os.path.exists(self.tflite_model_path):
            self.skipTest(f"TensorFlow Lite model not found at {self.tflite_model_path}.")
        self.detector = EmotionDetector(self.tflite_model_path)

    def test_model_loading(self):
        self.assertIsNotNone(self.detector.interpreter, "Model interpreter should not be None.")

    def test_emotion_detection(self):
        # This test requires a valid video stream with a detectable face.
        # For automated testing, you might mock the VideoStream.
        emotion = self.detector.get_emotion()
        # Since we can't predict the emotion, just check if it's a string or None
        self.assertTrue(isinstance(emotion, str) or emotion is None, "Emotion should be a string or None.")

    def tearDown(self):
        self.detector.release()

if __name__ == '__main__':
    unittest.main()
