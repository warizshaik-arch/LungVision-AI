import tensorflow as tf
from pathlib import Path

model_path = Path("models/baseline/lungvision_cnn.keras")

print("Loading model...")
model = tf.keras.models.load_model(model_path)

print("Saving model...")
model.save(model_path)

print("Done! Model re-saved successfully.")