# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import utils

app = FastAPI(title="MNIST Digit Inference API")

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to your saved model
MODEL_PATH = "mnist_cnn_clean.pth"

# load model at startup
try:
    model, device = utils.load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH} on device {device}")
except Exception as e:
    print("Failed to load model:", e)
    model, device = None, None


@app.get("/")
def read_root():
    return {"status": "ok", "message": "MNIST prediction server. Use POST /predict"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    pred_class, confidence = utils.predict_from_pil(pil_image, model, device)
    return JSONResponse(content={"prediction": pred_class, "confidence": round(confidence, 4)})
