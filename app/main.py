"""
FastAPI application serving the pneumonia classifier.

Model weights are downloaded from Hugging Face Hub at container startup
rather than baked into the Docker image, keeping the image small and the
model file out of version control. Set MODEL_REPO_ID and MODEL_FILENAME
as environment variables at deploy time.
"""

import io
import os
import logging

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download

from app.model import load_model, predict, THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "muby26/chest-xray-pneumonia-resnet50")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "resnet50_pneumonia.pt")
MAX_UPLOAD_SIZE_MB = 10

app = FastAPI(title="Chest X-Ray Pneumonia Classifier")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None  # populated at startup


@app.on_event("startup")
def startup_load_model():
    global model
    logger.info(f"Downloading model weights from {MODEL_REPO_ID}/{MODEL_FILENAME}")
    try:
        weights_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        model = load_model(weights_path, device)
        logger.info("Model loaded successfully.")
    except Exception as e:
        # Fail loudly at startup rather than serving requests with no model.
        logger.error(f"Model failed to load: {e}")
        raise


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Try again shortly.")

    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_SIZE_MB}MB limit.")

    try:
        probability = predict(model, io.BytesIO(contents), device)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Inference failed unexpectedly.")

    return {
        "probability": round(probability, 4),
        "threshold": THRESHOLD,
        "label": "PNEUMONIA" if probability >= THRESHOLD else "NORMAL"
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


# Serve the frontend last, so /predict and /health above take priority
# over the static file catch-all.
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
