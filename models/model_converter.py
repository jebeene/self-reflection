import torch
import config
from models.emotion_cnn import EmotionCNN

# Load trained model
model = EmotionCNN(num_classes=len(config.EMOTION_LABELS))
model.load_state_dict(torch.load(config.MODEL_PATH, map_location="cpu"))
model.eval()

def convert_to_torchscript():
    """Converts the trained model to TorchScript format for deployment."""
    scripted_model = torch.jit.script(model)
    torchscript_path = config.MODEL_PATH.replace(".pth", ".pt")
    scripted_model.save(torchscript_path)
    print(f"✅ TorchScript model saved at {torchscript_path}")

def convert_to_onnx():
    """Converts the trained model to ONNX format for deployment."""
    onnx_path = config.MODEL_PATH.replace(".pth", ".onnx")
    dummy_input = torch.randn(1, 1, 48, 48)
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"✅ ONNX model saved at {onnx_path}")

if __name__ == "__main__":
    convert_to_torchscript()
    convert_to_onnx()
