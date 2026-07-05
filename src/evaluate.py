import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from config import DATASET_PATH, BASELINE_MODEL

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Get class names automatically
CLASS_NAMES = val_ds.class_names
print("Classes:", CLASS_NAMES)

# Load model
model = tf.keras.models.load_model(BASELINE_MODEL)

# Predict
predictions = model.predict(val_ds)
y_pred = np.argmax(predictions, axis=1)

# True labels
y_true = np.concatenate(
    [labels.numpy() for _, labels in val_ds],
    axis=0
)

# Report
print("\nClassification Report\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES
))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()