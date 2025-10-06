from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import utils


app = FastAPI(title="MNIST Digit Recognition App")

# Serve static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Path to your saved model
MODEL_PATH = "mnist_cnn_clean.pth"

# Load model on startup
try:
    model, device = utils.load_model(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH} on device {device}")
except Exception as e:
    print("❌ Failed to load model:", e)
    model, device = None, None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/", response_class=HTMLResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):
    """Handles image upload + prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        pred_class, confidence = utils.predict_from_pil(pil_image, model, device)

        result = {
            "filename": file.filename,
            "prediction": pred_class,
            "confidence": round(confidence * 100, 2),
        }
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": result},
        )
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=400, detail="Invalid image file")
