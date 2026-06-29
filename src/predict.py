import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
from config import BASELINE_MODEL

model = tf.keras.models.load_model(BASELINE_MODEL)
# Class names (must match your dataset)
class_names = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia"
]

# Image path
img_path = r"C:\Users\Wariz\OneDrive\Documents\LungVision-AI\sample.jpg"

# Load image
img = image.load_img(img_path, target_size=(224,224))

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]

confidence = np.max(prediction) * 100

print("\nPrediction:", predicted_class)
print(f"Confidence: {confidence:.2f}%")