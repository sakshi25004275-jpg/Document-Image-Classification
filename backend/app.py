from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
import os

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = load_model("document_classifier.keras")

class_names = [
    'handwritten',
    'specification',
    'email',
    'news article',
    'form',
    'scientific publication',
    'questionnaire',
    'presentation',
    'memo',
    'budget',
    'advertisement',
    'scientific report',
    'letter',
    'invoice',
    'resume',
    'file folder'
]


@app.get("/")
def home():
    return {"message": "Document Classification API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = "uploaded_file.jpg"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = image.load_img(
        file_path,
        target_size=(224,224)
    )

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    result = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)*100)

    return {
        "prediction": result,
        "confidence": confidence
    }
