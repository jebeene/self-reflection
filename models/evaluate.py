import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.emotion_cnn import EmotionCNN
import config

# Load test dataset
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.ImageFolder(root=config.BASE_DIR + "/datasets/fer2013/test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# Load trained model
model = EmotionCNN(num_classes=len(config.EMOTION_LABELS)).to(config.DEVICE)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
model.eval()

# Define loss function
criterion = nn.CrossEntropyLoss()


def evaluate_model():
    """Evaluates the model on the test dataset."""
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss /= total
    test_acc = 100 * correct / total
    print(f"✅ Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    evaluate_model()
