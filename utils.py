# utils.py
import io
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Define the same model used for training
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Transform the image before prediction
_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

def transform_pil_image(pil_image: Image.Image):
    tensor = _transform(pil_image)
    return tensor.unsqueeze(0)

def load_model(path: str, device=None):
    """
    Robust loader for PyTorch 2.6+, allowing safe loading of old checkpoints.
    """
    import torch.serialization
    from torch.serialization import add_safe_globals

    # ✅ Allow PyTorch to trust this Net class
    add_safe_globals([Net])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Force old-style loading (fixes the warning)
        state = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        print("❌ Model load failed:", e)
        raise

    model = Net().to(device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    print("✅ Model loaded successfully!")
    return model, device

def predict_from_pil(pil_image: Image.Image, model, device):
    tensor = transform_pil_image(pil_image).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return int(pred.item()), float(conf.item())
