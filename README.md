# MNIST Digit Classification API (PyTorch + FastAPI)

This project demonstrates how to deploy a custom-trained PyTorch model using FastAPI.

## 🔹 Model
- A **SimpleCNN** trained on the MNIST dataset (handwritten digits 0–9).
- Achieves ~97% test accuracy.
- Model weights are stored in `mnist_cnn.pth`.

## 🔹 Features
- Upload an image (digit) via Swagger UI or API.
- The API returns the predicted digit (0–9) and confidence score.
- Lightweight, runs on CPU or GPU.

## 🔹 How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn main:app --reload
