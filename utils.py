# utils.py
import io
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Must match Net from train_model.py
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

# Image preprocessing for MNIST digits
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
    Load the trained Net model from path (PyTorch >=2.6 fix).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Explicitly allow your Net class as safe
    torch.serialization.add_safe_globals([Net])

    # ✅ Tell torch.load to ignore weights_only restriction
    state = torch.load(path, map_location=device, weights_only=False)

    model = Net().to(device)
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
