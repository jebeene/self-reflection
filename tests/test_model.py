# tests/test_model.py

import unittest
import os
import torch
from models.emotion_cnn import EmotionCNN  # Import the centralized model
from models.inference import predict_emotion
import config

class TestModel(unittest.TestCase):
    def setUp(self):
        """Set up the model and ensure a test image exists."""
        self.model_path = config.MODEL_PATH
        self.test_image_path = "datasets/fer2013/test/Happy/00000_1.jpg"  # Example image path
        if not os.path.exists(self.model_path):
            self.skipTest(f"Model not found at {self.model_path}.")
        if not os.path.exists(self.test_image_path):
            self.skipTest(f"Test image not found at {self.test_image_path}.")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = EmotionCNN(num_classes=config.NUM_CLASSES).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def test_model_loading(self):
        """Test if the model loads correctly."""
        self.assertIsNotNone(self.model, "Model should not be None.")

    def test_predict_emotion(self):
        """Test emotion prediction on a single image."""
        # Capture the printed output
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output
        predict_emotion(self.test_image_path)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue().strip()
        self.assertIn("✅ Predicted Emotion:", output, "Prediction output not found.")

    def tearDown(self):
        """Clean up after tests."""
        pass

if __name__ == '__main__':
    unittest.main()