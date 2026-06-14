import sys
import os
sys.path.append('src')

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import numpy as np
import io

from disease_detector import predict

app = FastAPI(title="Crop Disease Detection API")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict/disease")
async def predict_disease(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img = img.resize((64, 64))
        img_array = np.array(img)
        label = predict(img_array)
        return JSONResponse({
            "status": "success",
            "disease": label
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

@app.get("/health")
def health():
    return {"status": "healthy"}