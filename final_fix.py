# final_fix.py
import torch
import torch.nn as nn

# Define the same Net class you used
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

# --- Fix part ---
# Load the model file you already have (bypass PyTorch 2.6 restriction)
old_model = torch.load("mnist_cnn_fixed.pth", map_location="cpu", weights_only=False)

# If it's a full model -> extract state_dict
if isinstance(old_model, nn.Module):
    torch.save(old_model.state_dict(), "mnist_cnn_clean.pth")
    print("✅ Saved clean weights as mnist_cnn_clean.pth")

# If it’s already a state_dict, just resave to be safe
elif isinstance(old_model, dict):
    torch.save(old_model, "mnist_cnn_clean.pth")
    print("✅ Already state_dict — re-saved as mnist_cnn_clean.pth")

else:
    print("❌ Something unexpected in your checkpoint")
