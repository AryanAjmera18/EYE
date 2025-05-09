from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
from PIL import Image
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import io
import logging

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EyeDiseaseAPI")

# === FastAPI App Init ===
app = FastAPI(title="Eye Disease Detection API (ONNX + Drift + Prometheus)")
Instrumentator().instrument(app).expose(app)

# === Drift Metric Setup ===
drift_mean = Gauge("image_drift_mean", "Mean pixel value of input image")
drift_std = Gauge("image_drift_std", "Standard deviation of input image")

# === Model Path ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "..", "Artifacts", "serving_models", "model.onnx")
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

# === Load ONNX Model ===
try:
    ort_session = ort.InferenceSession(MODEL_PATH)
    logger.info(f"✅ Loaded ONNX model from {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load ONNX model from {MODEL_PATH}: {e}")

# === Define Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# === Class Labels ===
CLASS_NAMES = [
    "Cataract", "Diabetic Retinopathy", "Glaucoma",
    "Hypertension", "Macular Degeneration", "Normal",
    "Pathological Myopia", "Retinitis Pigmentosa",
    "Retinoblastoma", "Retinal Detachment"
]

@app.get("/")
def read_root():
    return {"message": "🧠 Eye Disease Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG formats are supported.")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # === Drift Detection Logging ===
        image_np = np.array(image) / 255.0
        mean, std = image_np.mean(), image_np.std()
        drift_mean.set(mean)
        drift_std.set(std)
        logger.info(f"[DRIFT] Input image stats — Mean: {mean:.4f}, Std: {std:.4f}")

        # === Inference ===
        input_tensor = transform(image).unsqueeze(0).numpy()
        inputs = {ort_session.get_inputs()[0].name: input_tensor}
        outputs = ort_session.run(None, inputs)
        output = outputs[0]

        pred_idx = int(output.argmax())
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(torch.softmax(torch.tensor(output), dim=1)[0][pred_idx])

        return JSONResponse({
            "predicted_class": pred_label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")