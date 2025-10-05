# fix_model.py
import torch
from utils import Net  # import your model class

# Load the old checkpoint (the one saved with torch.save(model))
model = torch.load("mnist_cnn.pth", map_location="cpu", weights_only=False)

# If it's already a Net, grab the state_dict
if isinstance(model, Net):
    torch.save(model.state_dict(), "mnist_cnn_fixed.pth")
    print("✅ Saved as mnist_cnn_fixed.pth using state_dict")
else:
    print("❌ The file did not contain a Net model.")
