from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="API Segmentation Cityscapes")

IMG_SIZE = (256, 256)

# Chargement du modèle
model = tf.keras.models.load_model(
    "best_model_unet_mini.h5",
    compile=False
)

# Palette de couleurs pour les 8 classes
COLORS = np.array([
    [128, 64, 128],   # Route
    [244, 35, 232],   # Trottoir
    [70, 70, 70],     # Bâtiment
    [102, 102, 156],  # Mur
    [190, 153, 153],  # Clôture
    [153, 153, 153],  # Poteau
    [107, 142, 35],   # Végétation
    [0, 0, 142],      # Véhicule
], dtype=np.uint8)

@app.get("/")
def home():
    return {
        "message": "API segmentation Cityscapes opérationnelle"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Lecture de l'image envoyée
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize(IMG_SIZE)

    # Prétraitement
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prédiction
    prediction = model.predict(image_array, verbose=0)

    # Classe prédite pour chaque pixel
    mask = np.argmax(prediction[0], axis=-1)

    # Conversion du masque en couleurs
    mask_color = COLORS[mask]

    # Conversion en PNG
    mask_image = Image.fromarray(mask_color)

    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png"
    )