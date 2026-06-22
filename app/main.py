import io
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.inference import mask_to_png_base64, predict_mask
from app.model_loader import load_config, load_model

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, source = load_model(cfg, device)

    state["cfg"] = cfg
    state["device"] = device
    state["model"] = model
    state["model_source"] = source

    yield
    state.clear()


app = FastAPI(title="PSPNet CamVid Segmentation API", lifespan=lifespan)


class PredictResponse(BaseModel):
    inference_time_ms: float
    width: int
    height: int
    mask_png_base64: str


@app.get("/health")
def health():
    return {"status": "ok", "model_source": state.get("model_source"), "device": str(state.get("device"))}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode uploaded file as an image.")

    start = time.perf_counter()
    mask = predict_mask(state["model"], image, state["cfg"], state["device"])
    inference_time_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        inference_time_ms=round(inference_time_ms, 2),
        width=image.width,
        height=image.height,
        mask_png_base64=mask_to_png_base64(mask),
    )
