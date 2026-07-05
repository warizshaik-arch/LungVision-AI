import tensorflow as tf
import numpy as np
from config import BASELINE_MODEL

model = tf.keras.models.load_model(BASELINE_MODEL)

feature_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer("conv2d_2").output
)

dummy = np.random.rand(1,224,224,3).astype("float32")

features = feature_model.predict(dummy)

print(features.shape)