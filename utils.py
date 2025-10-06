# utils.py
import io
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Define the same model architecture used in training
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
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.LogSoftmax(dim=1)(x)

# Transform for uploaded images
_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),  # scales pixel values to [0,1]
])

def transform_pil_image(pil_image: Image.Image):
    """
    Convert PIL image to a (1,1,28,28) tensor for inference.
    """
    tensor = _transform(pil_image)
    tensor = tensor.unsqueeze(0)
    return tensor

def load_model(path: str, device=None):
    """
    Safe model loader compatible with PyTorch 2.6+.
    """
    import torch.serialization
    # ✅ Allowlist the Net class for safe deserialization
    torch.serialization.add_safe_globals([Net])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Try loading safely first
        state = torch.load(path, map_location=device)
    except Exception as e:
        print("⚠️ Safe load failed, retrying with weights_only=False:", e)
        state = torch.load(path, map_location=device, weights_only=False)

    # Case 1: full model object
    if isinstance(state, nn.Module):
        model = state.to(device)
        print("✅ Loaded full model object")

    # Case 2: checkpoint dictionary
    elif isinstance(state, dict) and "state_dict" in state:
        model = Net().to(device)
        model.load_state_dict(state["state_dict"])
        print("✅ Loaded model from checkpoint dict")

    # Case 3: plain state_dict
    else:
        model = Net().to(device)
        model.load_state_dict(state)
        print("✅ Loaded model from state_dict")

    model.eval()
    return model, device

def predict_from_pil(pil_image: Image.Image, model, device):
    """
    Make prediction from PIL image using trained model.
    Returns (predicted_class:int, confidence:float)
    """
    tensor = transform_pil_image(pil_image).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return int(pred.item()), float(conf.item())
