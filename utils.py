# utils.py
import io
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# SimpleCNN must match the model you trained (same architecture & names)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 digits

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# transforms: convert uploaded image to 1x28x28 tensor (0..1)
_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),  # scales to [0,1]
])

def transform_pil_image(pil_image: Image.Image):
    """
    Convert a PIL image to a batch tensor shape (1,1,28,28).
    """
    tensor = _transform(pil_image)  # shape [1,28,28]
    tensor = tensor.unsqueeze(0)     # shape [1,1,28,28]
    return tensor

def load_model(path: str, device=None):
    """
    Load the SimpleCNN model from path. Returns model on device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    state = torch.load(path, map_location=device)
    # if saved with `model.state_dict()`, use load_state_dict
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, device

def predict_from_pil(pil_image: Image.Image, model, device):
    """
    Returns (predicted_class:int, confidence:float)
    """
    tensor = transform_pil_image(pil_image).to(device)
    with torch.no_grad():
        outputs = model(tensor)  # shape [1,10]
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return int(pred.item()), float(conf.item())
