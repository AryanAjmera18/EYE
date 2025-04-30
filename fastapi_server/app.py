from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
import io
import os

app = FastAPI(title="Eye Disease Detection API")

# === Load model ===
MODEL_PATH = r"D:\MlopsProj\Artifacts\04_29_2025_18_13_54\model_trainer\trained_model\model.pkl"

try:
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)

    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# === Define transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# === Define class labels ===
CLASS_NAMES = [
    "Cataract", "Diabetic Retinopathy", "Glaucoma",
    "Hypertension", "Macular Degeneration", "Normal",
    "Pathological Myopia", "Retinitis Pigmentosa",
    "Retinoblastoma", "Retinal Detachment"
]

@app.get("/")
def root():
    return {"message": "Eye Disease Detection API is up!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images are supported.")

    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = output.argmax(dim=1).item()
            pred_label = CLASS_NAMES[pred_idx]
            confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

        return JSONResponse({
            "predicted_class": pred_label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
