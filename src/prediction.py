import tensorflow as tf
import numpy as np
from config import BASELINE_MODEL

# Load model once
model = tf.keras.models.load_model(BASELINE_MODEL)


def predict(image):
    """
    Predict disease from a PIL image.
    Returns:
        prediction (numpy array)
        predicted_index (int)
        confidence (float)
    """

    img = image.resize((224, 224))

    img = np.array(img).astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction) * 100)

    return prediction, predicted_index, confidence